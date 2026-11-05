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
from heapq import heapify, heappop, heappush

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


@dataclass(frozen=True)
class _GapCandidate:
    """One scored queue entry with its per-class predicted box statistics.

    `key` is the entry's natural sort key, `box_counts` counts the entry's
    predicted boxes per class, and `conf_sums` sums their confidences per
    class; both lists are indexed by class id.
    """

    entry: dict[str, object]
    key: list[int | str]
    box_counts: list[int]
    conf_sums: list[float]


def fill_gaps_order(
    ds: Dataset, entries: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Order the queue by how much each prediction fills the annotation gap.

    Every predicted box scores its confidence times ``1 / (1 + n)`` where
    `n` is its class's labeled box count across the labels tree
    (:meth:`Dataset.label_class_counts`) — a never-labeled class weighs the
    most — and a crop's score is the sum over its boxes, so covering several
    under-annotated classes, or more boxes of them, outranks covering
    well-annotated ones. The order is built greedily: each pick's boxes are
    counted as filled gap (the per-class counts grow as if the crop had been
    accepted) before the next crop is scored, so the queue works down what
    is still missing instead of repeating an already-covered gap. Crops with
    no parseable predicted class (predicted backgrounds, malformed
    prediction files) come last, in natural order; ties among scored crops
    keep the natural order too.

    Returns:
        The ``(entries, label_errors)`` pair: the reordered entries (each
        scored crop enriched with a ``rank_note`` naming the classes it
        fills and its pick score) and the malformed label files skipped
        while counting.
    """
    counts, label_errors = ds.label_class_counts()
    n_classes = len(ds.classes)
    scored: list[_GapCandidate] = []
    unscored: list[dict[str, object]] = []
    for entry in entries:
        rel = str(entry["path"])
        boxes = []
        if entry["error"] is None and not entry["empty"]:
            try:
                boxes = ds.read_prediction(rel)
            except ApiError:
                pass  # the entry was decided or corrupted since the queue scan
        if not boxes:
            unscored.append(entry)
            continue
        box_counts = [0] * n_classes
        conf_sums = [0.0] * n_classes
        for box in boxes:
            box_counts[box["cls"]] += 1
            conf_sums[box["cls"]] += box["conf"]
        scored.append(
            _GapCandidate(entry, natural_key(rel), box_counts, conf_sums)
        )

    projected = list(counts)  # labeled counts + the boxes picked so far

    def score_of(candidate: _GapCandidate) -> float:
        """Gap-filling score of `candidate` at the projected class counts."""
        return sum(
            conf / (1 + projected[cls])
            for cls, conf in enumerate(candidate.conf_sums)
        )

    # Lazy max-heap on (-score, natural order): scores only ever fall while
    # the greedy pass runs (counts only grow), so a popped tuple holding a
    # stale score is refreshed and re-pushed instead of picked.
    heap: list[tuple[float, list[int | str], int]] = [
        (-score_of(candidate), candidate.key, i)
        for i, candidate in enumerate(scored)
    ]
    heapify(heap)
    picked = [False] * len(scored)
    ordered: list[dict[str, object]] = []
    while heap:
        neg, _, i = heappop(heap)
        if picked[i]:
            continue
        candidate = scored[i]
        score = score_of(candidate)
        if -neg != score:  # stale: an earlier pick changed the class counts
            heappush(heap, (-score, candidate.key, i))
            continue
        picked[i] = True
        filled = [c for c in range(n_classes) if candidate.box_counts[c]]
        filled.sort(
            key=lambda c: (-candidate.conf_sums[c] / (1 + projected[c]), c)
        )
        names = ", ".join(
            f"{candidate.box_counts[c]}×{ds.classes[c]}" for c in filled
        )
        ordered.append(
            {
                **candidate.entry,
                "rank_note": f"fills gaps: {names} (score {score:.2f})",
            }
        )
        for c in filled:
            projected[c] += candidate.box_counts[c]  # the pick fills c's gap
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
    Ranking(
        id="fill_gaps",
        title="fill the gaps first",
        description=(
            "Predictions that most reduce the annotation gap first: each "
            "predicted box scores its confidence × 1/(1 + its class's "
            "labeled count), and each pick's boxes are counted as filled "
            "gap before the next is chosen; predicted backgrounds last."
        ),
        order=fill_gaps_order,
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
