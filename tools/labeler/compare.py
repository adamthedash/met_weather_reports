"""Model-vs-label comparison: pair a crop's boxes and report disagreements.

Pure geometry over normalised 0-1 YOLO boxes: label boxes and predicted
boxes are paired greedily by descending IoU (a pair only forms when the two
boxes overlap at all), and every leftover box or mismatched pair becomes an
:class:`Issue`. ``MIN_IOU`` is the overlap below which a matched pair counts
as badly localised. This module knows nothing about files; ``Dataset.audit_queue``
applies it per crop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tools.labeler.dataset import Box, PredBox

MIN_IOU = 0.5

IssueKind = Literal[
    "unmatched_label", "unmatched_pred", "class_mismatch", "weak_overlap"
]


class Issue(TypedDict):
    """One model-vs-label disagreement of a crop.

    ``label`` / ``pred`` are indices into the compared box lists (None for
    the side an issue does not involve), and ``iou`` is the overlap of a
    matched pair (None for unmatched boxes).
    """

    kind: IssueKind
    label: int | None
    pred: int | None
    iou: float | None


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two normalised YOLO boxes.

    Args:
        a: First box (``cls``, ``cx``, ``cy``, ``w``, ``h``; extra keys such
            as a prediction's ``conf`` are ignored).
        b: Second box.

    Returns:
        The IoU in ``[0, 1]``: 1 for identical boxes, 0 for boxes that do not
        overlap (or degenerate boxes with no area).
    """
    ax1 = a["cx"] - a["w"] / 2
    ay1 = a["cy"] - a["h"] / 2
    ax2 = a["cx"] + a["w"] / 2
    ay2 = a["cy"] + a["h"] / 2
    bx1 = b["cx"] - b["w"] / 2
    by1 = b["cy"] - b["h"] / 2
    bx2 = b["cx"] + b["w"] / 2
    by2 = b["cy"] + b["h"] / 2
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    inter = max(inter_w, 0.0) * max(inter_h, 0.0)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0.0 else 0.0


def compare_boxes(
    labels: Sequence[Box], preds: Sequence[PredBox], min_iou: float = MIN_IOU
) -> list[Issue]:
    """Pair label and prediction boxes and list the crop's disagreements.

    Pairs are formed greedily by descending IoU, taking each box at most
    once. The result is empty when the two sets agree: every box matched,
    matched classes equal, and every matched overlap at least `min_iou`.
    Geometry-only differences below `min_iou` do count as disagreements
    (``weak_overlap``), so a badly drawn or badly localised box is surfaced
    even when its class is right.

    Args:
        labels: The crop's annotated boxes (normalised 0-1).
        preds: The crop's predicted boxes (normalised 0-1).
        min_iou: Overlap a matched pair needs to count as well localised.

    Returns:
        One :class:`Issue` per disagreement: ``unmatched_label`` /
        ``unmatched_pred`` for boxes no pair claimed, ``class_mismatch`` for
        a matched pair with different classes, and ``weak_overlap`` for a
        matched pair below `min_iou` (one pair can produce both of the last
        two). Issues are ordered by label index, then by prediction index
        for the unmatched predictions.

    Raises:
        ValueError: If `min_iou` is outside ``[0, 1]``.
    """
    if not 0.0 <= min_iou <= 1.0:
        raise ValueError(f"min_iou must be between 0 and 1, got {min_iou}")

    candidates = sorted(
        (
            (iou(label, pred), li, pi)
            for li, label in enumerate(labels)
            for pi, pred in enumerate(preds)
        ),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    matched: dict[int, tuple[int, float]] = {}
    used_preds: set[int] = set()
    for value, li, pi in candidates:
        if value <= 0.0 or li in matched or pi in used_preds:
            continue
        matched[li] = (pi, value)
        used_preds.add(pi)

    issues: list[Issue] = []
    for li, label in enumerate(labels):
        pair = matched.get(li)
        if pair is None:
            issues.append(
                Issue(kind="unmatched_label", label=li, pred=None, iou=None)
            )
            continue
        pi, value = pair
        if label["cls"] != preds[pi]["cls"]:
            issues.append(
                Issue(kind="class_mismatch", label=li, pred=pi, iou=value)
            )
        if value < min_iou:
            issues.append(
                Issue(kind="weak_overlap", label=li, pred=pi, iou=value)
            )
    for pi in range(len(preds)):
        if pi not in used_preds:
            issues.append(
                Issue(kind="unmatched_pred", label=None, pred=pi, iou=None)
            )
    return issues
