"""The summary stage: read-only overview of the dataset as it stands.

No queue, no actions, no artifacts: ``GET /overview`` aggregates the labeling
state over the images and labels trees — progress totals, the labeled boxes
per class (including never-labeled classes) next to the crops marked empty,
and one row per subject (source page) with the full per-class breakdown. The
page never writes anything; malformed label files and label files mirroring
no image are reported in the response instead of being counted.
"""

from __future__ import annotations

from fastapi import APIRouter

from tools.labeler.dataset import Dataset
from tools.labeler.stages.base import Stage


def create_stage(ds: Dataset) -> Stage:
    """Build the summary stage over dataset `ds`."""
    router = APIRouter()

    @router.get("/overview")
    async def overview() -> dict[str, object]:
        """The dataset's aggregate labeling state (shape of `Dataset.summary`).

        A scan route like the `/list` routes: the images and labels trees are
        walked per request, so the summary always reflects the files as they
        are (every response is `Cache-Control: no-store` anyway).
        """
        return ds.summary()

    return Stage(
        id="summary",
        title="summary",
        description=(
            "Read-only overview of the dataset: progress totals, "
            "class and per-subject label distributions."
        ),
        router=router,
    )
