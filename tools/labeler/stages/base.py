"""Stage seam: a stage is one workflow type served by the labeler UI.

A *stage* (a pipeline stage) bundles three things:

- the **queue** of items (crops) it works over and their per-item status,
- the **actions** a user performs on an item (edit boxes, accept/reject, ...),
- the **artifacts** it consumes and produces (mirrored sidecar files under a
  dataset root, e.g. YOLO label or prediction files).

Each stage contributes an API router (mounted under ``/api/<id>``) on the
backend and a workspace component (registered under the same id in
``frontend/src/stages/registry.ts``) on the frontend. Adding a stage touches
only ``stages/__init__.py`` (one entry in ``build_stages``) and that frontend
registry; the shared engine and core routes stay unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fastapi import APIRouter


@dataclass(frozen=True)
class Stage:
    """One workflow stage: its static description and API router.

    The router's routes are mounted at ``/api/<id>``; the matching frontend
    workspace registers under the same id.
    """

    id: str
    title: str
    description: str
    router: APIRouter


def summaries(stages: Sequence[Stage]) -> list[dict[str, str]]:
    """Every stage as JSON-safe ``{id, title, description}`` dicts."""
    return [
        {"id": s.id, "title": s.title, "description": s.description} for s in stages
    ]
