"""The batch review stage: bulk accept/reject/flag over the bootstrap predictions.

The review queue (crops with a prediction file and neither a label file nor a
manual-queue marker) is served whole in the stage's composed order and shown
as a grid-sized batch; the user marks individual crops and commits the whole
batch in one request:

- **accept** (every unmarked crop) promotes each prediction to a YOLO label
  file, exactly like the review stage's accept (see
  :func:`tools.labeler.stages.review.decide`); an empty prediction accepts to
  an empty label (the crop is confirmed as background/ignored).
- **reject** (a plain click) deletes only the prediction; the crop stays
  non-annotated and returns to the annotate queue.
- **needs manual adjustment** (a shift-click) deletes the prediction and
  flags the crop for the manual queue, exactly like the review stage's flag
  decision (see :func:`tools.labeler.stages.review.mark_manual`).

Every decision goes through those review-stage functions, so batch review is
a second view over the very same queue and artifacts: decisions are
destructive and immediate, labels are written before predictions are deleted,
and existing labels are never overwritten.

The queue's composed order is the ``confidence`` ranking (highest
lowest-box-confidence first) with empty predictions sprinkled in randomly at
:data:`EMPTY_SPRINKLE_RATE` of the served slots, so predicted backgrounds
keep being confirmed at a steady rate while the high-confidence mass is
cleared; malformed prediction files come last, in natural order.
``GET /list`` serves the composed order whole — the client displays its first
entries as the current batch — and re-composes it on every fetch, so moving
the batch-size slider never reshuffles the grid.
"""

from __future__ import annotations

import random

from fastapi import APIRouter, Request

from tools.labeler.dataset import Dataset
from tools.labeler.errors import ApiError
from tools.labeler.rankings import find_ranking
from tools.labeler.stages.base import Stage
from tools.labeler.stages.review import decide, mark_manual
from tools.labeler.web import read_json_body

#: Target share of empty predictions (predicted backgrounds) in the served
#: queue: each merged slot draws from the empty pool with this probability
#: while both the scored and the empty pools still have entries.
EMPTY_SPRINKLE_RATE = 0.25

#: Decisions one commit may carry (the grid's batch slider caps at 32).
MAX_DECISIONS = 64

#: The commit actions, one per mark kind.
ACTIONS = ("accept", "reject", "manual")


def entry_with_boxes(ds: Dataset, entry: dict[str, object]) -> dict[str, object]:
    """Re-read one queue entry with its prediction boxes inlined.

    The grid shows every batch's boxes at once, so the listing carries them
    in the entry instead of serving them per crop. The result is
    ``{path, boxes, empty, error, rank_note}``: `boxes` is the parsed
    prediction (None for a malformed file), `empty` flags a predicted
    background, `error` carries the parse message (None for a sound file),
    and `rank_note` is the ranking's explanation of the placement, if any.

    Raises:
        ApiError: If the prediction is missing (the entry was decided since
            the queue scan) — the listing re-syncs the queue from disk then.
    """
    rel = str(entry["path"])
    if entry["error"] is not None:
        return {
            "path": rel,
            "boxes": None,
            "empty": False,
            "error": entry["error"],
            "rank_note": entry.get("rank_note"),
        }
    boxes = ds.read_prediction(rel)  # strict; missing or malformed since the scan
    return {
        "path": rel,
        "boxes": boxes,
        "empty": not boxes,
        "error": None,
        "rank_note": entry.get("rank_note"),
    }


def sprinkle(
    scored: list[dict[str, object]],
    empties: list[dict[str, object]],
    malformed: list[dict[str, object]],
    rate: float = EMPTY_SPRINKLE_RATE,
) -> list[dict[str, object]]:
    """Merge `empties` randomly into `scored` at `rate` density.

    The empty pool is consumed in random order and every merged slot takes
    the next empty entry with probability `rate` while both pools still have
    entries, so the expected share of empty predictions in any served prefix
    is `rate` (individual batches vary naturally around it). When one pool
    runs out, the rest of the other follows verbatim — leftover empties keep
    their random order. `malformed` entries are appended after both, in their
    given order: they are unscoreable and can never be auto-accepted, so they
    wait at the tail for an explicit decision.
    """
    pool = list(empties)
    random.shuffle(pool)
    merged: list[dict[str, object]] = []
    si = ei = 0
    while si < len(scored) or ei < len(pool):
        take_empty = ei < len(pool) and (si >= len(scored) or random.random() < rate)
        if take_empty:
            merged.append(pool[ei])
            ei += 1
        else:
            merged.append(scored[si])
            si += 1
    merged.extend(malformed)
    return merged


def apply_decision(ds: Dataset, rel: str, action: str) -> dict[str, object]:
    """Apply one crop's accept/reject/manual decision, capturing failures.

    The three actions delegate to the review stage's decision functions
    (:func:`~tools.labeler.stages.review.decide` and
    :func:`~tools.labeler.stages.review.mark_manual`), so a batch decision is
    byte-identical to the same decision taken one crop at a time. An
    :class:`~tools.labeler.errors.ApiError` (missing prediction, malformed
    file, missing crop image) is captured as the result's `error` instead of
    raised, so one failing crop never aborts the rest of the batch.

    Returns:
        ``{path, action, ok, saved, already_annotated, error}`` — `saved` is
        the number of label boxes written (accept), `already_annotated`
        reports a stale prediction dropped for a crop annotated in the
        meantime, and `error` is the failure message (None on success).
    """
    try:
        if action == "accept":
            outcome = decide(ds, rel, True)
        elif action == "reject":
            outcome = decide(ds, rel, False)
        else:
            outcome = mark_manual(ds, rel)
    except ApiError as exc:
        return {
            "path": rel,
            "action": action,
            "ok": False,
            "saved": 0,
            "already_annotated": False,
            "error": exc.message,
        }
    return {
        "path": rel,
        "action": action,
        "ok": True,
        "saved": int(outcome.get("saved", 0)),
        "already_annotated": bool(outcome.get("already_annotated", False)),
        "error": None,
    }


def create_stage(ds: Dataset) -> Stage:
    """Build the batch review stage over dataset `ds`."""
    router = APIRouter()

    @router.get("/list")
    async def listing() -> dict[str, object]:
        """Respond with the composed batch queue and the stale prediction count.

        The queue is the review queue in the ``confidence`` ranking's order
        (highest lowest-box-confidence first) with its empty predictions
        sprinkled in randomly at ``EMPTY_SPRINKLE_RATE`` and its malformed
        files last; entries carry their prediction boxes inline (null for a
        malformed file) so the grid renders a whole batch from one response.
        ``total - remaining`` predictions are stale and ignored.
        """
        queue, stale = ds.queue_and_stale()
        ordered, _ = find_ranking("confidence").order(ds, queue)
        scored: list[dict[str, object]] = []
        empties: list[dict[str, object]] = []
        malformed: list[dict[str, object]] = []
        for e in ordered:
            try:
                entry = entry_with_boxes(ds, e)
            except ApiError as exc:  # decided since the queue scan: treat as malformed
                entry = {
                    **e,
                    "boxes": None,
                    "empty": False,
                    "error": exc.message,
                }
            if entry["error"] is not None:
                malformed.append(entry)
            elif entry["empty"]:
                empties.append(entry)
            else:
                scored.append(entry)
        entries = sprinkle(scored, empties, malformed)
        return {
            "predictions": entries,
            "total": len(entries) + stale,
            "remaining": len(entries),
        }

    @router.post("/decision")
    async def post_decision(request: Request) -> dict[str, object]:
        """Commit one batch: apply an accept/reject/manual decision per crop.

        The body is ``{"decisions": [{"path", "action"}]}`` with 1..64
        distinct paths and `action` one of accept/reject/manual; the response
        holds one result per decision, in request order (see
        :func:`apply_decision`). Decisions touch only their own crop's files,
        and a failing crop is reported in its result instead of aborting the
        batch.
        """
        payload = await read_json_body(request)
        raw = payload.get("decisions")
        if not isinstance(raw, list) or not raw:
            raise ApiError(400, "missing 'decisions'")
        if len(raw) > MAX_DECISIONS:
            raise ApiError(400, f"too many decisions (max {MAX_DECISIONS})")
        decisions: list[tuple[str, str]] = []
        seen: set[str] = set()
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ApiError(400, f"decision {i}: expected an object")
            rel = item.get("path")
            action = item.get("action")
            if not isinstance(rel, str):
                raise ApiError(400, f"decision {i}: missing 'path'")
            if action not in ACTIONS:
                raise ApiError(400, f"decision {i}: unknown action {action!r}")
            if rel in seen:
                raise ApiError(400, f"duplicate path: {rel}")
            seen.add(rel)
            decisions.append((rel, action))
        return {
            "results": [apply_decision(ds, rel, action) for rel, action in decisions]
        }

    return Stage(
        id="batch-review",
        title="batch review",
        description=(
            "Batch grid over bootstrap predictions: accept many at a time, "
            "click to reject or flag for manual adjustment; highest-confidence "
            "first with predicted backgrounds sprinkled in."
        ),
        router=router,
    )
