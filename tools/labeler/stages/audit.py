"""The audit stage: model-vs-label disagreement review over annotated crops.

Per crop of the queue (crops holding both a label file and an audit
prediction file that disagree):

- **keep_label** (``l``) deletes only the audit prediction: the annotation is
  confirmed as correct.
- **take_prediction** (``p``) overwrites the label with the model's boxes
  (confidence column dropped, byte-compatible with the annotate stage's
  writer) and deletes the audit prediction; an empty prediction writes an
  empty label (background/ignored).
- **discard_both** (``x``) deletes the label first and the audit prediction
  second, returning the crop to the manual annotate queue.

Every decision consumes the audit prediction file, so the queue empties as
the user works and re-running ``scripts/audit_annotations.py`` starts a fresh
round. Decisions are destructive and immediate — there is no undo; a wrongly
discarded label is re-annotated in the annotate stage. The label is written
before the prediction is deleted, so an interrupted take_prediction keeps a
still-comparing snapshot instead of losing work. A broken side can never be
accepted: keep_label refuses a malformed label, take_prediction refuses a
malformed prediction.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from tools.labeler.compare import compare_boxes
from tools.labeler.dataset import Box, Dataset
from tools.labeler.errors import ApiError
from tools.labeler.stages.base import Stage
from tools.labeler.web import read_json_body, require_arg

ACTIONS = ("keep_label", "take_prediction", "discard_both")


def decide(ds: Dataset, rel: str, action: str) -> dict[str, object]:
    """Apply one three-way audit decision for the crop `rel`.

    ``keep_label`` deletes only the audit prediction; ``take_prediction``
    writes the predicted boxes as the label (after validation, before the
    prediction is deleted); ``discard_both`` deletes the label first and the
    prediction second. All three leave the crop without an audit prediction,
    which is what ends its review — the next audit run may queue it again.

    Args:
        ds: The dataset to decide on.
        rel: Crop path (relative to the images root).
        action: One of ``keep_label``, ``take_prediction``, ``discard_both``.

    Returns:
        ``{ok, action, saved}`` — `saved` is the number of label boxes
        written (take_prediction). The decision touches only the decided
        crop's files — no dataset rescan — so its cost is independent of the
        dataset size; the queue length after the decision is the client's own
        list minus the decided entry (re-synced via ``GET /list`` when a
        decision fails).

    Raises:
        ApiError: If the audit prediction is missing, `action` is unknown,
            or the action refuses its side: keep_label on a missing or
            malformed label, take_prediction on a malformed prediction or a
            missing crop image.
    """
    if action not in ACTIONS:
        raise ApiError(400, f"unknown action: {action!r}")
    pred = ds.audit_prediction_path_for(rel)
    if not pred.is_file():
        raise ApiError(404, f"no such audit prediction: {rel}")
    label = ds.label_path_for(rel)
    saved = 0
    if action == "keep_label":
        if not label.is_file():
            raise ApiError(404, f"no such label: {rel}")
        ds.read_labels(rel)  # strict: refuses a malformed label
        pred.unlink()
    elif action == "take_prediction":
        image = ds.image_path(rel)
        if not image.is_file():
            raise ApiError(404, f"no such image: {rel}")
        boxes = ds.read_audit_prediction(rel)  # strict: refuses malformed files
        ds.write_labels(
            rel,
            [
                Box(cls=b["cls"], cx=b["cx"], cy=b["cy"], w=b["w"], h=b["h"])
                for b in boxes
            ],
        )
        saved = len(boxes)
        pred.unlink()
    else:  # discard_both
        if label.is_file():
            label.unlink()
        pred.unlink()
    return {"ok": True, "action": action, "saved": saved}


def create_stage(ds: Dataset) -> Stage:
    """Build the audit stage over dataset `ds`."""
    router = APIRouter()

    @router.get("/list")
    async def listing() -> dict[str, object]:
        """Respond with the audit queue and the number of compared crops."""
        queue, compared = ds.audit_queue()
        return {
            "audits": queue,
            "total": compared,
            "remaining": len(queue),
        }

    @router.get("/compare")
    async def get_compare(
        path: str | None = Query(default=None),
    ) -> dict[str, object]:
        """Both box sets of one crop and their disagreement issues."""
        rel = require_arg(path, "path")
        if not ds.label_path_for(rel).is_file():
            raise ApiError(404, f"no such label: {rel}")
        labels = ds.read_labels(rel)
        preds = ds.read_audit_prediction(rel)
        return {
            "path": rel,
            "labels": labels,
            "predictions": preds,
            "issues": compare_boxes(labels, preds),
        }

    @router.post("/decision")
    async def post_decision(request: Request) -> dict[str, object]:
        """Apply a keep-label / take-prediction / discard-both decision."""
        payload = await read_json_body(request)
        rel = payload.get("path")
        if not isinstance(rel, str):
            raise ApiError(400, "missing 'path'")
        action = payload.get("action")
        if not isinstance(action, str) or action not in ACTIONS:
            raise ApiError(400, "missing or unknown 'action'")
        return decide(ds, rel, action)

    return Stage(
        id="audit",
        title="audit",
        description=(
            "Compare model predictions against annotations; resolve each "
            "disagreement by keeping the label, taking the prediction, or "
            "discarding both."
        ),
        router=router,
    )
