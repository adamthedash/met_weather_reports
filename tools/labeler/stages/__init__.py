"""Stage registry for the labeler tool."""

from tools.labeler.stages.base import REGISTRY, Stage, register, summaries

__all__ = ["REGISTRY", "Stage", "register", "summaries"]
