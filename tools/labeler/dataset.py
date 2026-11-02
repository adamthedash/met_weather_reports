"""Dataset helpers: class names, image listing, and YOLO label file I/O.

The dataset is an explicit :class:`Dataset` value object so each workflow
stage can declare its own roots. Parsing is strict (malformed label files are
reported, never skipped), ``validate_boxes`` clamps boxes to the image, and
the on-disk format (``<class> <cx> <cy> <w> <h>`` normalised 0-1, 6 decimals)
feeds ``scripts/prepare_dataset.py`` and ultralytics training unchanged.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import yaml

from tools.labeler.errors import ApiError

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_BOXES = 64
_NUM_RE = re.compile(r"(\d+)")


class Box(TypedDict):
    """A YOLO bounding box in normalised 0-1 coordinates."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float


class PredBox(TypedDict):
    """A predicted YOLO bounding box in normalised 0-1 coordinates."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float


def load_classes(yaml_path: Path) -> list[str]:
    """Read class names from the int-keyed ``names:`` mapping of a YOLO data.yaml.

    ``data.yaml`` is the single source of truth for classes; nothing falls
    back to defaults.

    Raises:
        ValueError: If the file is missing, is not valid YAML, or has no
            usable ``names:`` mapping (non-mapping or empty, non-integer or
            duplicate keys, non-string names).
    """
    if not yaml_path.is_file():
        raise ValueError(f"data.yaml not found: {yaml_path}")
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{yaml_path}: invalid YAML: {exc}") from exc
    raw = data.get("names") if isinstance(data, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{yaml_path}: expected a non-empty 'names:' mapping")
    names: dict[int, str] = {}
    for key, value in raw.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise ValueError(f"{yaml_path}: names key {key!r} is not an integer")
        try:
            idx = int(key)
        except ValueError as exc:
            raise ValueError(
                f"{yaml_path}: names key {key!r} is not an integer"
            ) from exc
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(
                f"{yaml_path}: names[{key!r}] must be a string, got {type(value).__name__}"
            )
        if idx in names:
            raise ValueError(f"{yaml_path}: duplicate names index {idx}")
        names[idx] = str(value)
    return [names[i] for i in sorted(names)]


def natural_key(rel: str) -> list[int | str]:
    """Sort key so 10_2.png sorts before 10_10.png."""
    return [int(p) if p.isdigit() else p.lower() for p in _NUM_RE.split(rel)]


def safe_path(root: Path, rel: str) -> Path:
    """Resolve `rel` under `root`, rejecting traversal and absolute paths.

    Raises:
        ApiError: If `rel` is empty, absolute, or escapes `root`.
    """
    if not rel or "\\" in rel:
        raise ApiError(400, f"bad path: {rel!r}")
    parts = Path(rel)
    if parts.is_absolute() or ".." in parts.parts:
        raise ApiError(400, f"bad path: {rel!r}")
    resolved = (root / parts).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ApiError(400, f"bad path: {rel!r}")
    return resolved


@dataclass(frozen=True)
class Dataset:
    """One dataset: crop images root, label files root, and class names.

    Label files mirror the images tree (``<stem>.txt`` in the same subdirs)
    under ``labels_root``; prediction files (the bootstrap script's model
    output) mirror it under ``predictions_root``. Semantics shared with the
    training pipeline:

    - label file exists (any content) → crop is done/annotated
    - empty label file → done but nothing to label (background/ignored)
    - non-empty → crop has labeled characters
    """

    images_root: Path
    labels_root: Path
    predictions_root: Path
    classes: tuple[str, ...]

    def image_path(self, rel: str) -> Path:
        """Absolute path of crop `rel` under the images root.

        Raises:
            ApiError: If `rel` escapes the images root.
        """
        return safe_path(self.images_root, rel)

    def label_path_for(self, rel: str) -> Path:
        """Labels-root mirror of a crop path: <stem>.txt in the same subdirs.

        Raises:
            ApiError: If `rel` escapes the labels root.
        """
        p = Path(rel)
        return safe_path(
            self.labels_root, (p.parent / (p.stem + ".txt")).as_posix()
        )

    def list_images(self) -> list[str]:
        """All crop images under the images root, relative posix paths, natural order."""
        rels = [
            p.relative_to(self.images_root).as_posix()
            for p in self.images_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        return sorted(rels, key=natural_key)

    def read_labels(self, rel: str) -> list[Box]:
        """Read the YOLO label file mirroring `rel`; empty when absent or unlabeled.

        Raises:
            ApiError: If a label line has the wrong field count or an
                out-of-range class.
        """
        boxes: list[Box] = []
        path = self.label_path_for(rel)
        if not path.is_file():
            return boxes
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ApiError(
                    400,
                    f"{path.name} line {lineno}: expected 5 fields, got {len(parts)}",
                )
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
            if not 0 <= cls < len(self.classes):
                raise ApiError(
                    400, f"{path.name} line {lineno}: class {cls} out of range"
                )
            boxes.append({"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h})
        return boxes

    def validate_boxes(self, raw: object) -> list[Box]:
        """Validate raw JSON boxes, clamping each to the image, and return them.

        Raises:
            ApiError: If `raw` is not a list of well-formed boxes within the
                class range and the image bounds.
        """
        if not isinstance(raw, list):
            raise ApiError(400, "'boxes' must be a list")
        if len(raw) > MAX_BOXES:
            raise ApiError(400, f"too many boxes (max {MAX_BOXES})")
        boxes: list[Box] = []
        for i, b in enumerate(raw):
            try:
                cls = int(b["cls"])
                cx, cy, w, h = (float(b[k]) for k in ("cx", "cy", "w", "h"))
            except (KeyError, TypeError, ValueError) as exc:
                raise ApiError(400, f"box {i}: expected cls, cx, cy, w, h") from exc
            if not 0 <= cls < len(self.classes):
                raise ApiError(
                    400,
                    f"box {i}: class {cls} out of range 0..{len(self.classes) - 1}",
                )
            if not all(math.isfinite(v) for v in (cx, cy, w, h)):
                raise ApiError(400, f"box {i}: non-finite coordinates")
            if w <= 0 or h <= 0:
                raise ApiError(400, f"box {i}: width and height must be positive")
            # clamp to the image, keep the box valid
            x1 = min(max(cx - w / 2, 0.0), 1.0)
            y1 = min(max(cy - h / 2, 0.0), 1.0)
            x2 = min(max(cx + w / 2, 0.0), 1.0)
            y2 = min(max(cy + h / 2, 0.0), 1.0)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                raise ApiError(400, f"box {i}: lies outside the image")
            boxes.append(
                {
                    "cls": cls,
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                    "w": x2 - x1,
                    "h": y2 - y1,
                }
            )
        return boxes

    def write_labels(self, rel: str, boxes: list[Box]) -> Path:
        """Write `boxes` to the YOLO label file mirroring `rel`; return its path.

        Raises:
            ApiError: If `rel` escapes the labels root.
        """
        path = self.label_path_for(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
            for b in boxes
        ]
        path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        return path

    def prediction_path_for(self, rel: str) -> Path:
        """Predictions-root mirror of a crop path: <stem>.txt in the same subdirs.

        Same contract as ``scripts/bootstrap_annotations.py:prediction_path_for``.

        Raises:
            ApiError: If `rel` escapes the predictions root.
        """
        p = Path(rel)
        return safe_path(
            self.predictions_root, (p.parent / (p.stem + ".txt")).as_posix()
        )

    def read_prediction(self, rel: str) -> list[PredBox]:
        """Read the prediction file mirroring `rel`; empty for a predicted background.

        Lines hold six fields (``cls cx cy w h conf``, normalised 0-1) as
        written by ``scripts/bootstrap_annotations.py:write_prediction``.
        Parsing is strict: a malformed line is reported, never skipped, so
        corruption can not slip into the dataset as a label. Prediction files
        are therefore intentionally not loadable as YOLO labels; only accepted
        files under ``labels_root`` are.

        Raises:
            ApiError: If the prediction is missing, a line has the wrong field
                count or non-numeric fields, a class id is out of range, or a
                value is non-finite or outside 0-1.
        """
        path = self.prediction_path_for(rel)
        if not path.is_file():
            raise ApiError(404, f"no such prediction: {rel}")
        boxes: list[PredBox] = []
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ApiError(
                    400,
                    f"{path.name} line {lineno}: expected 6 fields, got {len(parts)}",
                )
            try:
                cls = int(parts[0])
                cx, cy, w, h, conf = (float(v) for v in parts[1:])
            except ValueError as exc:
                raise ApiError(
                    400, f"{path.name} line {lineno}: non-numeric field"
                ) from exc
            if not 0 <= cls < len(self.classes):
                raise ApiError(
                    400, f"{path.name} line {lineno}: class {cls} out of range"
                )
            if not all(
                math.isfinite(v) and 0.0 <= v <= 1.0 for v in (cx, cy, w, h, conf)
            ):
                raise ApiError(
                    400,
                    f"{path.name} line {lineno}: values must be finite and in 0-1",
                )
            boxes.append(
                {"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf}
            )
        return boxes

    def queue_and_stale(self) -> tuple[list[dict[str, object]], int]:
        """Return the review queue and the number of stale prediction files.

        The queue holds every crop that has a prediction file and no label
        file, in natural order — exactly the crops the last bootstrap run
        predicted. Prediction files whose crop is annotated or gone are stale
        (pruned by ``scripts/bootstrap_annotations.py`` on its next run) and
        never queue.

        Each entry is ``{"path", "boxes", "empty", "error"}``: `boxes` counts
        the predicted boxes (None for a malformed file), `empty` flags a
        predicted background crop, and `error` carries the parse message of a
        malformed file (None otherwise).

        Returns:
            The ``(queue, stale_count)`` pair.
        """
        queue: list[dict[str, object]] = []
        queued: set[str] = set()
        for rel in self.list_images():
            if self.label_path_for(rel).is_file():
                continue  # already annotated (possibly since the bootstrap run)
            if not self.prediction_path_for(rel).is_file():
                continue  # no prediction for this crop
            queued.add(rel)
            try:
                boxes = self.read_prediction(rel)
                entry: dict[str, object] = {
                    "path": rel,
                    "boxes": len(boxes),
                    "empty": not boxes,
                    "error": None,
                }
            except ApiError as exc:
                entry = {
                    "path": rel,
                    "boxes": None,
                    "empty": False,
                    "error": exc.message,
                }
            queue.append(entry)
        on_disk = {
            p.relative_to(self.predictions_root).with_suffix("").as_posix()
            for p in self.predictions_root.rglob("*.txt")
        }
        queued_stems = {Path(rel).with_suffix("").as_posix() for rel in queued}
        return queue, len(on_disk - queued_stems)
