"""Pluggable rankings: presentation order over the review stage's queue.

A *ranking* decides in which order the review queue is served, so targeted
review passes (rare labels first, low confidence first, ...) can be added
without touching the review engine: one :class:`Ranking` entry in
:data:`RANKINGS` is all it takes — ``GET /api/review/rankings`` and the
review workspace's selector pick it up automatically. A ranking only
reorders the queue: queue membership and the accept/reject decisions are
identical in every ranking.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tools.labeler.dataset import Dataset, natural_key
from tools.labeler.errors import ApiError

#: Queue ordering over the review entries.
#:
#: Receives the dataset and the queue entries (``{"path", "boxes", "empty",
#: "error"}`` as built by :meth:`Dataset.queue_and_stale`) and returns the
#: ``(entries, label_errors)`` pair: the same entries reordered, each
#: optionally enriched with a ``rank_note`` explaining its placement, and the
#: ``{"path", "error"}`` dataset files the ranking could not parse (empty for
#: rankings that read no extra files).
OrderFn = Callable[
    [Dataset, list[dict[str, object]]],
    tuple[list[dict[str, object]], list[dict[str, object]]],
]


@dataclass(frozen=True)
class Ranking:
    """One queue ranking: its static description and ordering function.

    The review workspace shows `title` in its selector and `description`
    alongside it; `order` performs the ranking itself (see :data:`OrderFn`).
    """

    id: str
    title: str
    description: str
    order: OrderFn


def natural_order(
    ds: Dataset, entries: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Keep the queue in its natural path order — the default ranking.

    Returns:
        The ``(entries, label_errors)`` pair: the entries unchanged and an
        empty error list.
    """
    return list(entries), []


def rare_labels_order(
    ds: Dataset, entries: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Order the queue by the rarity of its rarest predicted label.

    Rarity is the labeled box count of a class across the labels tree
    (:meth:`Dataset.label_class_counts`); a never-labeled class counts 0 and
    is therefore the rarest. Each crop is scored by its rarest predicted
    class and the queue is served rarest first, so predictions containing
    labels that are rare in the accepted annotations come first. Crops with
    no parseable predicted class (predicted backgrounds, malformed
    prediction files) come last, in natural order; ties among scored crops
    keep the natural order too.

    Returns:
        The ``(entries, label_errors)`` pair: the reordered entries (each
        scored crop enriched with a ``rank_note`` naming its rarest predicted
        label) and the malformed label files skipped while counting.
    """
    counts, label_errors = ds.label_class_counts()
    scored: list[tuple[tuple[int, list[int | str]], dict[str, object]]] = []
    unscored: list[dict[str, object]] = []
    for entry in entries:
        rel = str(entry["path"])
        boxes = []
        if entry["error"] is None and not entry["empty"]:
            try:
                boxes = ds.read_prediction(rel)
            except ApiError:
                pass  # the entry was decided or corrupted since the queue scan
        if boxes:
            rarest = min(boxes, key=lambda b: counts[b["cls"]])
            count = counts[rarest["cls"]]
            name = ds.classes[rarest["cls"]]
            entry = {
                **entry,
                "rank_note": f"rarest predicted label: {name} ×{count} in labels",
            }
            scored.append(((count, natural_key(rel)), entry))
        else:
            unscored.append(entry)
    scored.sort(key=lambda pair: pair[0])
    ordered = [entry for _, entry in scored]
    unscored.sort(key=lambda e: natural_key(str(e["path"])))
    return ordered + unscored, label_errors


RANKINGS: list[Ranking] = [
    Ranking(
        id="natural",
        title="natural order",
        description="The queue's natural path order (source page, row, column).",
        order=natural_order,
    ),
    Ranking(
        id="rare_labels",
        title="rare labels first",
        description=(
            "Predictions whose rarest predicted label is rarest in the "
            "accepted annotations first; predicted backgrounds last."
        ),
        order=rare_labels_order,
    ),
]


def find_ranking(ranking_id: str) -> Ranking:
    """Look up a queue ranking by id.

    Raises:
        ApiError: 400 if no ranking with that id is registered.
    """
    for ranking in RANKINGS:
        if ranking.id == ranking_id:
            return ranking
    valid = ", ".join(r.id for r in RANKINGS)
    raise ApiError(400, f"unknown ranking {ranking_id!r} (valid: {valid})")


def summaries(rankings: Sequence[Ranking]) -> list[dict[str, str]]:
    """Every ranking as JSON-safe ``{id, title, description}`` dicts."""
    return [
        {"id": r.id, "title": r.title, "description": r.description}
        for r in rankings
    ]
