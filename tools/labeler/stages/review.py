"""The review stage: accept/reject review over the bootstrap predictions.

Ported from ``tools/review.py``. Per crop of the queue (crops with a
prediction file and no label file):

- **accept** promotes the prediction to a YOLO label file (confidence column
  dropped, byte-compatible with the annotate stage's writer) and deletes the
  prediction; an empty prediction accepts to an empty label (background).
- **reject** deletes only the prediction; the crop stays non-annotated and
  returns to the manual annotate queue.

Decisions are destructive and immediate — there is no undo. Wrong predictions
are cheap to regenerate (``scripts/bootstrap_annotations.py``), and wrong
labels can be fixed in the annotate stage.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from tools.labeler.dataset import Box, Dataset
from tools.labeler.errors import ApiError
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
    queue, _stale = ds.queue_and_stale()
    return {
        "ok": True,
        "accepted": accept,
        "saved": saved,
        "already_annotated": already,
        "remaining": len(queue),
    }


def create_stage(ds: Dataset) -> Stage:
    """Build the review stage over dataset `ds`."""
    router = APIRouter()

    @router.get("/list")
    async def listing() -> dict[str, object]:
        """Respond with the review queue and the stale prediction count."""
        queue, stale = ds.queue_and_stale()
        return {
            "predictions": queue,
            "total": len(queue) + stale,
            "remaining": len(queue),
        }

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

    return Stage(
        id="review",
        title="review",
        description=(
            "Accept/reject pass over bootstrap predictions; "
            "accepts become YOLO label files."
        ),
        router=router,
    )
