"""Stage seam: a stage is one workflow type served by the labeler UI.

A *stage* (a pipeline stage) bundles three things:

- the **queue** of items (crops) it works over and their per-item status,
- the **actions** a user performs on an item (edit boxes, accept/reject, ...),
- the **artifacts** it consumes and produces (mirrored sidecar files under a
  dataset root, e.g. YOLO label or prediction files).

Each stage registers an API router (mounted under ``/api/<id>``) on the backend
and a workspace component (registered under the same id in
``frontend/src/stages/registry.ts``) on the frontend. Adding a stage touches
only those two files; the shared engine and core routes stay unchanged.

This module holds the registry contract only; the annotate and review stages
land with the dataset layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    """Static description of one workflow stage."""

    id: str
    title: str
    description: str


REGISTRY: list[Stage] = []


def register(stage: Stage) -> None:
    """Add `stage` to the registry.

    Raises:
        ValueError: If another stage with the same id is already registered.
    """
    if any(s.id == stage.id for s in REGISTRY):
        raise ValueError(f"duplicate stage id: {stage.id!r}")
    REGISTRY.append(stage)


def summaries() -> list[dict[str, str]]:
    """Every registered stage as JSON-safe ``{id, title, description}`` dicts."""
    return [
        {"id": s.id, "title": s.title, "description": s.description} for s in REGISTRY
    ]
