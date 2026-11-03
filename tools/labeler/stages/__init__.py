"""Stage registry: every workflow stage the labeler serves.

Adding a stage is one import plus one entry in :func:`build_stages`, plus one
frontend workspace in ``frontend/src/stages/registry.ts`` under the same id.
"""

from __future__ import annotations

from tools.labeler.dataset import Dataset
from tools.labeler.stages import annotate, audit, review
from tools.labeler.stages.base import Stage, summaries

__all__ = ["Stage", "build_stages", "summaries"]


def build_stages(ds: Dataset) -> list[Stage]:
    """Construct every stage over dataset `ds`, in menu order."""
    return [
        annotate.create_stage(ds),
        review.create_stage(ds),
        audit.create_stage(ds),
    ]
