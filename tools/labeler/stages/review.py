"""The review stage: accept/reject/flag review over the bootstrap predictions.

Per crop of the queue (crops with a prediction file and neither a label file
nor a manual-queue marker):

- **accept** promotes the prediction to a YOLO label file (confidence column
  dropped, byte-compatible with the annotate stage's writer) and deletes the
  prediction; an empty prediction accepts to an empty label (background).
- **reject** deletes only the prediction; the crop stays non-annotated and
  returns to the manual annotate queue.
- **needs manual adjustment** deletes the prediction and flags the crop for
  the manual queue (a marker under the manual root), which the annotate
  stage serves first; the prediction is discarded unparsed, so malformed
  prediction files can be flagged too.

Decisions are destructive and immediate — there is no undo. Wrong predictions
are cheap to regenerate (``scripts/bootstrap_annotations.py``), and wrong
labels can be fixed in the annotate stage.

The queue's presentation order is pluggable (``tools/labeler/rankings.py``):
``GET /list`` serves it in a chosen ranking's order (rare labels first, ...),
and decisions are identical in every ranking.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from tools.labeler.dataset import Box, Dataset
from tools.labeler.errors import ApiError
from tools.labeler.rankings import RANKINGS, find_ranking, summaries
from tools.labeler.stages.base import Stage
from tools.labeler.web import read_json_body, require_arg


def decide(ds: Dataset, rel: str, accept: bool) -> dict[str, object]:
    """Apply an accept/reject decision for the prediction mirroring `rel`.

    Accept writes the label file first and deletes the prediction second, so
    an interrupted accept leaves a stale prediction (pruned by the next
    bootstrap run) instead of losing a label. An existing label file is never
    overwritten: the crop was annotated in the meantime, so only the stale
    prediction is dropped. Reject just discards the prediction; the crop stays
    unannotated.

    The decision touches only the decided crop's files — no dataset rescan —
    so its cost is independent of the dataset size. The queue length after
    the decision is the client's own list minus the decided entry (re-synced
    via ``GET /list`` when a decision fails).

    Returns:
        ``{ok, accepted, saved, already_annotated}`` — `saved` is the number
        of label boxes written (accept).

    Raises:
        ApiError: If the prediction is missing, or accept is refused for a
            malformed prediction or a missing crop image.
    """
    pred = ds.prediction_path_for(rel)
    if not pred.is_file():
        raise ApiError(404, f"no such prediction: {rel}")
    label = ds.label_path_for(rel)
    already = label.is_file()
    saved = 0
    if accept and not already:
        image = ds.image_path(rel)
        if not image.is_file():
            raise ApiError(404, f"no such image: {rel}")
        boxes = ds.read_prediction(rel)  # strict: refuses malformed files
        ds.write_labels(
            rel,
            [
                Box(cls=b["cls"], cx=b["cx"], cy=b["cy"], w=b["w"], h=b["h"])
                for b in boxes
            ],
        )
        saved = len(boxes)
    pred.unlink()
    return {
        "ok": True,
        "accepted": accept,
        "saved": saved,
        "already_annotated": already,
    }


def mark_manual(ds: Dataset, rel: str) -> dict[str, object]:
    """Flag the crop mirroring `rel` as needing manual adjustment.

    The manual-queue marker is written **before** the prediction is deleted,
    so an interrupted mark leaves the crop flagged and its prediction merely
    ignored (counted stale by :meth:`Dataset.queue_and_stale`) instead of
    losing the flag. The prediction is discarded without parsing, so a
    malformed prediction file can be flagged too (unlike accept). As with
    accept, an existing label file is never touched: the crop was annotated
    in the meantime, so only the stale prediction is dropped and no marker is
    written.

    The decision touches only the decided crop's files — no dataset rescan.

    Returns:
        ``{ok, marked, already_annotated}`` — `marked` is False only for the
        already-annotated outcome.

    Raises:
        ApiError: If the prediction is missing, or the crop image is missing
            for a crop that still needs flagging.
    """
    pred = ds.prediction_path_for(rel)
    if not pred.is_file():
        raise ApiError(404, f"no such prediction: {rel}")
    label = ds.label_path_for(rel)
    already = label.is_file()
    if not already:
        image = ds.image_path(rel)
        if not image.is_file():
            raise ApiError(404, f"no such image: {rel}")
        ds.mark_manual(rel)
    pred.unlink()
    return {
        "ok": True,
        "marked": not already,
        "already_annotated": already,
    }


def create_stage(ds: Dataset) -> Stage:
    """Build the review stage over dataset `ds`."""
    router = APIRouter()

    @router.get("/list")
    async def listing(
        ranking: str | None = Query(default=None),
    ) -> dict[str, object]:
        """Respond with the review queue and the stale prediction count.

        `ranking` names a ranking from ``GET /rankings`` (default:
        ``natural``); it only changes presentation order — queue membership
        is identical in every ranking. Entries always carry `rank_note` (the
        ranking's one-line explanation of the placement, null when it has
        none), and `label_errors` lists the label files the ranking could not
        parse while scoring (only rankings that read label files report any).
        """
        r = find_ranking(ranking or "natural")
        queue, stale = ds.queue_and_stale()
        ordered, label_errors = r.order(ds, queue)
        entries = [{**e, "rank_note": e.get("rank_note")} for e in ordered]
        return {
            "predictions": entries,
            "total": len(entries) + stale,
            "remaining": len(entries),
            "ranking": r.id,
            "label_errors": label_errors,
        }

    @router.get("/rankings")
    async def get_rankings() -> dict[str, object]:
        """Every queue ranking (drives the review workspace's selector)."""
        return {"rankings": summaries(RANKINGS)}

    @router.get("/prediction")
    async def get_prediction(
        path: str | None = Query(default=None),
    ) -> dict[str, object]:
        """The prediction boxes (normalised, with confidence) for one crop."""
        rel = require_arg(path, "path")
        return {"path": rel, "boxes": ds.read_prediction(rel)}

    @router.post("/decision")
    async def post_decision(request: Request) -> dict[str, object]:
        """Apply an accept/reject for one prediction."""
        payload = await read_json_body(request)
        rel = payload.get("path")
        if not isinstance(rel, str):
            raise ApiError(400, "missing 'path'")
        accept = payload.get("accept")
        if not isinstance(accept, bool):
            raise ApiError(400, "missing 'accept'")
        return decide(ds, rel, accept)

    @router.post("/manual")
    async def post_manual(request: Request) -> dict[str, object]:
        """Flag one prediction's crop as needing manual adjustment."""
        payload = await read_json_body(request)
        rel = payload.get("path")
        if not isinstance(rel, str):
            raise ApiError(400, "missing 'path'")
        return mark_manual(ds, rel)

    return Stage(
        id="review",
        title="review",
        description=(
            "Accept/reject/flag pass over bootstrap predictions; "
            "accepts become YOLO label files, flags join the manual queue."
        ),
        router=router,
    )
