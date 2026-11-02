"""The annotate stage: manual bounding-box editing over the crop queue.

Exposes the queue listing, label read/write, and validation with the response
shapes the UI depends on: ``{path, done, has_labels}`` listing entries and the
``{ok, saved, has_labels}`` save result. Label files are written in the strict
YOLO on-disk format of ``tools/labeler/dataset.py``.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from tools.labeler.dataset import Dataset
from tools.labeler.errors import ApiError
from tools.labeler.stages.base import Stage
from tools.labeler.web import read_json_body, require_arg


def create_stage(ds: Dataset) -> Stage:
    """Build the annotate stage over dataset `ds`."""
    router = APIRouter()

    @router.get("/list")
    async def listing() -> dict[str, object]:
        """Respond with every crop's path and its labeling status."""
        images: list[dict[str, object]] = []
        done = 0
        for rel in ds.list_images():
            lp = ds.label_path_for(rel)
            exists = lp.is_file()
            done += int(exists)
            images.append(
                {
                    "path": rel,
                    "done": exists,
                    "has_labels": exists and lp.stat().st_size > 0,
                }
            )
        return {"images": images, "total": len(images), "done": done}

    @router.get("/labels")
    async def get_labels(path: str | None = Query(default=None)) -> dict[str, object]:
        """The crop's current boxes (normalised 0-1), empty when unlabeled."""
        rel = require_arg(path, "path")
        return {"path": rel, "boxes": ds.read_labels(rel)}

    @router.post("/labels")
    async def post_labels(request: Request) -> dict[str, object]:
        """Validate and save boxes for one crop; an empty list marks it empty."""
        payload = await read_json_body(request)
        rel = payload.get("path")
        if not isinstance(rel, str):
            raise ApiError(400, "missing 'path'")
        image = ds.image_path(rel)
        if not image.is_file():
            raise ApiError(404, f"no such image: {rel}")
        boxes = ds.validate_boxes(payload.get("boxes", []))
        ds.write_labels(rel, boxes)
        return {"ok": True, "saved": len(boxes), "has_labels": bool(boxes)}

    return Stage(
        id="annotate",
        title="annotate",
        description=(
            "Draw character bounding boxes on the crop queue; "
            "writes YOLO label files."
        ),
        router=router,
    )
