"""Dataset helpers: class names, class presentation, image listing, and YOLO label file I/O.

The dataset is an explicit :class:`Dataset` value object so each workflow
stage can declare its own roots. Parsing is strict (malformed label files are
reported, never skipped), ``validate_boxes`` clamps boxes to the image, and
the on-disk format (``<class> <cx> <cy> <w> <h>`` normalised 0-1, 6 decimals)
feeds ``scripts/prepare_dataset.py`` and ultralytics training unchanged.
Class names come from ``data.yaml`` and class presentation (display glyph,
keyboard shortcut) from the sibling ``classes.yaml``, so the tooling is not
hard-wired to any dataset's class set.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict

import yaml

from tools.labeler.compare import compare_boxes
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


class LabelProblem(TypedDict):
    """A label file the summary could not count: `{path, error}`."""

    path: str
    error: str


class SummaryTotals(TypedDict):
    """Dataset-wide crop and box totals of :meth:`Dataset.summary`."""

    crops: int
    annotated: int
    empty: int
    unlabeled: int
    boxes: int


class SummarySubject(TypedDict):
    """One subject row of :meth:`Dataset.summary`.

    The crop counts of one subject (a source page's crop folder — or the
    page's own image in a flat dataset) plus its labeled boxes per class
    (``class_counts`` parallels ``classes``).
    """

    id: str
    crops: int
    annotated: int
    empty: int
    unlabeled: int
    boxes: int
    class_counts: list[int]


class _StrictLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that rejects duplicate mapping keys.

    PyYAML silently keeps the last of duplicate keys, which would quietly
    drop or renumber a class; strict parsing must reject the file instead.
    """


def _construct_mapping_strict(
    loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    """Mapping constructor that raises on a duplicate key (see :class:`_StrictLoader`)."""
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:  # unhashable key
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"unhashable key {key!r}",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_strict
)


def _load_yaml(yaml_path: Path, what: str) -> object:
    """Parse one YAML file, rejecting duplicate mapping keys.

    Args:
        yaml_path: The YAML file to read.
        what: Human-readable role of the file, used in error messages.

    Returns:
        The parsed document.

    Raises:
        ValueError: If the file is missing or is not valid YAML.
    """
    if not yaml_path.is_file():
        raise ValueError(f"{what} not found: {yaml_path}")
    try:
        return yaml.load(yaml_path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"{yaml_path}: invalid YAML: {exc}") from exc


def load_classes(yaml_path: Path) -> list[str]:
    """Read class names from the int-keyed ``names:`` mapping of a YOLO data.yaml.

    ``data.yaml`` is the single source of truth for classes; nothing falls
    back to defaults.

    Raises:
        ValueError: If the file is missing, is not valid YAML (including a
            duplicate mapping key), or has no usable ``names:`` mapping
            (non-mapping or empty, non-integer or duplicate keys,
            non-string names).
    """
    data = _load_yaml(yaml_path, "data.yaml")
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


@dataclass(frozen=True)
class ClassDisplay:
    """One class's presentation: display glyph and keyboard shortcut.

    Sourced from the dataset's ``classes.yaml`` (see :func:`load_class_map`).
    The frontend draws `glyph` on box chips and in readings and binds `key`
    in the annotate stage; `key` is None for a click-only class.
    """

    name: str
    glyph: str
    key: str | None


# KeyboardEvent.key values the annotate stage binds to its own actions
# (frontend/src/stages/AnnotateWorkspace.svelte): navigation (a, d,
# ArrowLeft, ArrowRight), save & next (Enter), mark empty (Backspace),
# delete box (Delete), undo (z), view mode (f), zoom ([ ]), cancel (Escape).
# A class shortcut must not collide with any of them; modified chords like
# Ctrl+S are no collision since class keys fire unmodified.
RESERVED_KEYS = frozenset(
    {
        "a",
        "d",
        "f",
        "z",
        "[",
        "]",
        "Enter",
        "Backspace",
        "Delete",
        "Escape",
        "ArrowLeft",
        "ArrowRight",
    }
)


def load_class_map(yaml_path: Path, class_names: Sequence[str]) -> list[ClassDisplay]:
    """Read per-class presentation (glyph, keyboard shortcut) from ``classes.yaml``.

    ``classes.yaml`` sits alongside ``data.yaml`` and is required: it is the
    single source of truth for how each class displays and keys in the
    labeler, so the application is not hard-wired to any dataset's class
    names. It is a class-name-keyed mapping of ``{glyph, key}`` entries
    (``key: null`` makes the class click-only):

    .. code-block:: yaml

        dash: {glyph: "-", key: "-"}
        tr:   {glyph: "tr", key: "t"}
        x:    {glyph: "x",  key: "x"}

    Parsing is strict (same contract as :func:`load_classes`): the entry set
    must match `class_names` exactly and nothing is silently ignored,
    defaulted, or overridden.

    Args:
        yaml_path: The ``classes.yaml`` file to read.
        class_names: The dataset's class names (from :func:`load_classes`),
            defining the required entry set.

    Returns:
        One :class:`ClassDisplay` per class, in `class_names` order.

    Raises:
        ValueError: If the file is missing, is not valid YAML (including a
            duplicate mapping key), has no usable mapping, has entries not
            matching `class_names` exactly (missing or unknown names),
            defines a malformed `glyph` or `key`, or binds two classes to
            one key or a class to a reserved labeler action key.
    """
    data = _load_yaml(yaml_path, "classes.yaml")
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{yaml_path}: expected a non-empty class mapping")

    def text_field(value: object, field: str, name: str) -> str:
        """One entry's glyph/key: a non-empty string (an int renders as its digits)."""
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(
                f"{yaml_path}: {field} for class {name!r} must be a string, "
                f"got {type(value).__name__}"
            )
        text = str(value)
        if not text:
            raise ValueError(f"{yaml_path}: {field} for class {name!r} must not be empty")
        return text

    entries: dict[str, dict[object, object]] = {}
    for raw_name, raw_entry in data.items():
        if isinstance(raw_name, bool) or not isinstance(raw_name, (int, str)):
            raise ValueError(
                f"{yaml_path}: class name key {raw_name!r} is not a string"
            )
        name = str(raw_name)
        if name in entries:
            raise ValueError(f"{yaml_path}: duplicate entry for class {name!r}")
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"{yaml_path}: entry for class {name!r} must be a mapping"
            )
        if "glyph" not in raw_entry or "key" not in raw_entry:
            raise ValueError(
                f"{yaml_path}: entry for class {name!r} must define 'glyph' and 'key'"
            )
        unknown = sorted(str(k) for k in raw_entry if k not in ("glyph", "key"))
        if unknown:
            raise ValueError(
                f"{yaml_path}: entry for class {name!r} has unknown fields: "
                + ", ".join(unknown)
            )
        entries[name] = raw_entry

    missing = [n for n in class_names if n not in entries]
    unknown = sorted(n for n in entries if n not in set(class_names))
    if missing or unknown:
        problems = []
        if missing:
            problems.append("missing entries for: " + ", ".join(missing))
        if unknown:
            problems.append("unknown classes: " + ", ".join(unknown))
        raise ValueError(
            f"{yaml_path}: does not match the data.yaml class names — "
            + "; ".join(problems)
        )

    display: list[ClassDisplay] = []
    bound: dict[str, str] = {}  # keyboard key -> class name
    for name in class_names:
        entry = entries[name]
        glyph = text_field(entry["glyph"], "glyph", name)
        raw_key = entry["key"]
        key: str | None = None
        if raw_key is not None:
            key = text_field(raw_key, "key", name)
            if key in RESERVED_KEYS:
                raise ValueError(
                    f"{yaml_path}: key {key!r} for class {name!r} collides "
                    "with a labeler action key"
                )
            if key in bound:
                raise ValueError(
                    f"{yaml_path}: key {key!r} is bound to both "
                    f"{bound[key]!r} and {name!r}"
                )
            bound[key] = name
        display.append(ClassDisplay(name=name, glyph=glyph, key=key))
    return display


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
    """One dataset: crop images root, label files root, class names and presentation.

    `id` is the dataset's short name (the data.yaml directory's name, e.g.
    ``digits`` or ``tables``): stable per dataset, it namespaces the UI's
    persisted state and is reported by ``GET /api/health``.

    `classes` is the ``data.yaml`` class set (the id → name mapping label
    files index); `display` is the parallel ``classes.yaml`` presentation
    (glyph, keyboard shortcut) the frontend renders and keys by.

    Label files mirror the images tree (``<stem>.txt`` in the same subdirs)
    under ``labels_root``; prediction files mirror it under two roots:
    ``predictions_root`` (the bootstrap script's model output for unlabeled
    crops) and ``audit_root`` (the audit script's model output for every
    annotated crop, compared against the labels). Manual-queue markers mirror it under
    ``manual_root``: one empty ``<stem>.txt`` per crop flagged in review as
    needing manual adjustment (the prediction was discarded). Semantics
    shared with the training pipeline:

    - label file exists (any content) → crop is done/annotated
    - empty label file → done but nothing to label (background/ignored)
    - non-empty → crop has labeled characters
    - manual marker + no label file → crop is in the manual queue: excluded
      from the review queue, listed first in annotate
    """

    id: str
    images_root: Path
    labels_root: Path
    manual_root: Path
    predictions_root: Path
    audit_root: Path
    classes: tuple[str, ...]
    display: tuple[ClassDisplay, ...]

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

    def manual_path_for(self, rel: str) -> Path:
        """Manual-root mirror of a crop path: <stem>.txt in the same subdirs.

        Same mirror convention as :meth:`label_path_for`; the file is an
        empty marker (its existence is all that matters).

        Raises:
            ApiError: If `rel` escapes the manual root.
        """
        p = Path(rel)
        return safe_path(
            self.manual_root, (p.parent / (p.stem + ".txt")).as_posix()
        )

    def is_manual(self, rel: str) -> bool:
        """Whether crop `rel` carries a manual-queue marker.

        Raises:
            ApiError: If `rel` escapes the manual root.
        """
        return self.manual_path_for(rel).is_file()

    def mark_manual(self, rel: str) -> Path:
        """Flag crop `rel` as needing manual adjustment; return the marker path.

        The marker is the manual queue's membership: written by the review
        stage's *needs manual adjustment* decision and cleared by
        :meth:`write_labels` once the crop is annotated.

        Raises:
            ApiError: If `rel` escapes the manual root.
        """
        path = self.manual_path_for(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def clear_manual(self, rel: str) -> None:
        """Remove crop `rel`'s manual-queue marker, if any (no-op when absent).

        Raises:
            ApiError: If `rel` escapes the manual root.
        """
        path = self.manual_path_for(rel)
        if path.is_file():
            path.unlink()

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

        Parsing is strict (see :meth:`read_label_file`).

        Raises:
            ApiError: If a label line has the wrong field count or
                non-numeric fields, or a class id is out of range.
        """
        path = self.label_path_for(rel)
        if not path.is_file():
            return []
        return self.read_label_file(rel, path)

    def read_label_file(self, rel: str, path: Path) -> list[Box]:
        """Strictly parse one five-column YOLO label file.

        Each line is ``<class> <cx> <cy> <w> <h>`` in normalised 0-1
        coordinates. Parsing is strict: a malformed line is reported, never
        skipped, so corruption can not slip into the dataset as a label.

        Args:
            rel: Crop path the file mirrors; used in error messages.
            path: The label file to parse.

        Returns:
            The label boxes in file order; empty for an empty label file.

        Raises:
            ApiError: If the file is missing, a line has the wrong field
                count or non-numeric fields, or a class id is out of range.
        """
        if not path.is_file():
            raise ApiError(404, f"no such label: {rel}")
        boxes: list[Box] = []
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
            try:
                cls = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
            except ValueError as exc:
                raise ApiError(
                    400, f"{path.name} line {lineno}: non-numeric field"
                ) from exc
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

        Also clears the crop's manual-queue marker: a labeled crop (including
        an empty label — marked empty/ignored) is done and never needs manual
        adjustment again, so the manual queue and ``labels_root`` never
        overlap.

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
        self.clear_manual(rel)
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

    def audit_prediction_path_for(self, rel: str) -> Path:
        """Audit-root mirror of a crop path: <stem>.txt in the same subdirs.

        Same contract as ``scripts/audit_annotations.py:prediction_path_for``.

        Raises:
            ApiError: If `rel` escapes the audit root.
        """
        p = Path(rel)
        return safe_path(
            self.audit_root, (p.parent / (p.stem + ".txt")).as_posix()
        )

    def read_prediction(self, rel: str) -> list[PredBox]:
        """Read the prediction file mirroring `rel`; empty for a predicted background.

        Lines hold six fields (``cls cx cy w h conf``, normalised 0-1) as
        written by ``scripts/bootstrap_annotations.py:write_prediction``;
        parsing is strict (see :meth:`read_prediction_file`). Prediction
        files are therefore intentionally not loadable as YOLO labels; only
        accepted files under ``labels_root`` are.

        Raises:
            ApiError: If the prediction is missing or malformed.
        """
        return self.read_prediction_file(rel, self.prediction_path_for(rel))

    def read_audit_prediction(self, rel: str) -> list[PredBox]:
        """Read the audit prediction file mirroring `rel`; empty for a predicted background.

        Same format and strictness as :meth:`read_prediction`, read from the
        audit tree written by ``scripts/audit_annotations.py``.

        Raises:
            ApiError: If the audit prediction is missing or malformed.
        """
        return self.read_prediction_file(rel, self.audit_prediction_path_for(rel))

    def read_prediction_file(self, rel: str, path: Path) -> list[PredBox]:
        """Strictly parse one six-column prediction file.

        Each line is ``cls cx cy w h conf`` in normalised 0-1 coordinates.
        Parsing is strict: a malformed line is reported, never skipped, so
        corruption can not slip into the dataset as a label.

        Args:
            rel: Crop path the file mirrors; used in error messages.
            path: The prediction file to parse.

        Returns:
            The predicted boxes in file order; empty for a predicted
            background crop.

        Raises:
            ApiError: If the file is missing, a line has the wrong field
                count or non-numeric fields, a class id is out of range, or a
                value is non-finite or outside 0-1.
        """
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

        The queue holds every crop that has a prediction file, no label file,
        and no manual-queue marker, in natural order — exactly the crops the
        last bootstrap run predicted. Prediction files whose crop is
        annotated, flagged for manual adjustment, or gone are stale (pruned
        by ``scripts/bootstrap_annotations.py`` on its next run) and never
        queue.

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
            if self.is_manual(rel):
                continue  # flagged for manual adjustment: claimed by annotate
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

    def label_class_counts(self) -> tuple[list[int], list[dict[str, object]]]:
        """Count labeled boxes per class across the labels tree.

        Used by queue rankings (``tools/labeler/rankings.py``) to measure
        label rarity. Every label file under ``labels_root`` is parsed with
        the strict :meth:`read_label_file` format; a malformed file
        contributes nothing and is reported instead of silently skewing the
        counts (it is also surfaced by the annotate and audit stages).

        Returns:
            The ``(counts, skipped)`` pair: `counts` holds one box count per
            class index (0 for never-labeled classes), `skipped` holds
            ``{"path", "error"}`` entries for label files that failed to
            parse.
        """
        counts = [0] * len(self.classes)
        skipped: list[dict[str, object]] = []
        for path in sorted(self.labels_root.rglob("*.txt")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.labels_root).as_posix()
            try:
                boxes = self.read_label_file(rel, path)
            except ApiError as exc:
                skipped.append({"path": rel, "error": exc.message})
                continue
            for b in boxes:
                counts[b["cls"]] += 1
        return counts, skipped

    def audit_queue(self) -> tuple[list[dict[str, object]], int]:
        """Return the audit queue and the number of compared crops.

        The queue holds every crop that has both a label file and an audit
        prediction file and whose two box sets disagree per
        :func:`tools.labeler.compare.compare_boxes`, in natural order. Crops
        whose sets agree are compared but not queued; crops missing either
        file are not compared at all (decided, or not covered by the last
        ``scripts/audit_annotations.py`` run).

        Each entry is ``{"path", "label_boxes", "pred_boxes", "issues",
        "error"}``: `label_boxes` / `pred_boxes` count the boxes of each side
        (None for a malformed file — the side that failed is the one with the
        None count), `issues` counts the disagreements (None when a side is
        malformed), and `error` carries the parse message of a malformed file
        (None otherwise). A malformed file queues with its error so the user
        must resolve it; the audit stage's actions refuse the broken side.

        Returns:
            The ``(queue, compared)`` pair: the crops to review and the
            number of crops holding both a label and an audit prediction.
        """
        queue: list[dict[str, object]] = []
        compared = 0
        for rel in self.list_images():
            if not self.label_path_for(rel).is_file():
                continue  # not annotated: nothing to compare against
            if not self.audit_prediction_path_for(rel).is_file():
                continue  # decided this round, or never predicted
            compared += 1
            labels: list[Box] | None
            preds: list[PredBox] | None
            errors: list[str] = []
            try:
                labels = self.read_labels(rel)
            except ApiError as exc:
                labels = None
                errors.append(exc.message)
            try:
                preds = self.read_audit_prediction(rel)
            except ApiError as exc:
                preds = None
                errors.append(exc.message)
            entry: dict[str, object]
            if labels is not None and preds is not None:
                issues = compare_boxes(labels, preds)
                if not issues:
                    continue  # model and label agree
                entry = {
                    "path": rel,
                    "label_boxes": len(labels),
                    "pred_boxes": len(preds),
                    "issues": len(issues),
                    "error": None,
                }
            else:
                entry = {
                    "path": rel,
                    "label_boxes": len(labels) if labels is not None else None,
                    "pred_boxes": len(preds) if preds is not None else None,
                    "issues": None,
                    "error": "; ".join(errors),
                }
            queue.append(entry)
        return queue, compared

    def summary(self) -> dict[str, object]:
        """Aggregate the dataset's current labeling state for the summary page.

        One image-driven pass over :meth:`list_images` — the same crop set the
        annotate stage lists and ``scripts/prepare_dataset.py`` trains on.
        Every crop is grouped under its **subject**: the first path component
        of its relative path (``images/<source-id>/…``, so a second page's
        ``<subject>_2`` folder is a subject of its own), or its own file stem
        for a crop sitting directly under the images root (in a flat dataset
        each image is a source page of its own).
        The mirrored label file then decides the crop's state with the shared
        dataset semantics: absent → `unlabeled`, empty → `annotated` and
        `empty`, non-empty → `annotated` with its boxes counted per class.

        A malformed label file counts as `annotated` (its file exists, the
        shared done-semantics) but contributes no boxes and no `empty` count;
        it is reported in `label_errors` instead of silently skewing the
        counts. A label file mirroring no image — what
        ``scripts/prepare_dataset.py`` rejects as inconsistent — is listed in
        `orphan_labels` and excluded from every count.

        Returns:
            ``{classes, totals, class_counts, subjects, label_errors,
            orphan_labels}``: `classes` is the data.yaml class set; `totals`
            holds ``{crops, annotated, empty, unlabeled, boxes}`` (where
            `annotated` includes `empty`); `class_counts` holds one labeled
            box count per class (0 for never-labeled classes); `subjects`
            holds one :class:`SummarySubject` per subject in
            :func:`natural_key` order whose rows sum exactly to `totals` and
            `class_counts`; `label_errors` holds :class:`LabelProblem`
            entries for malformed label files; `orphan_labels` lists the
            label files mirroring no image.
        """
        n_cls = len(self.classes)
        totals = SummaryTotals(crops=0, annotated=0, empty=0, unlabeled=0, boxes=0)
        class_counts = [0] * n_cls
        subjects: dict[str, SummarySubject] = {}
        label_errors: list[LabelProblem] = []
        image_stems: set[str] = set()
        for rel in self.list_images():
            p = Path(rel)
            image_stems.add(p.with_suffix("").as_posix())
            subject_id = p.parts[0] if len(p.parts) > 1 else p.stem
            row = subjects.setdefault(
                subject_id,
                SummarySubject(
                    id=subject_id,
                    crops=0,
                    annotated=0,
                    empty=0,
                    unlabeled=0,
                    boxes=0,
                    class_counts=[0] * n_cls,
                ),
            )
            row["crops"] += 1
            totals["crops"] += 1
            label_file = self.label_path_for(rel)
            if not label_file.is_file():
                row["unlabeled"] += 1
                totals["unlabeled"] += 1
                continue
            row["annotated"] += 1
            totals["annotated"] += 1
            try:
                boxes = self.read_label_file(rel, label_file)
            except ApiError as exc:
                label_errors.append(
                    {
                        "path": label_file.relative_to(self.labels_root).as_posix(),
                        "error": exc.message,
                    }
                )
                continue
            if not boxes:
                row["empty"] += 1
                totals["empty"] += 1
                continue
            row["boxes"] += len(boxes)
            totals["boxes"] += len(boxes)
            for b in boxes:
                row["class_counts"][b["cls"]] += 1
                class_counts[b["cls"]] += 1
        orphan_labels = sorted(
            (
                path.relative_to(self.labels_root).as_posix()
                for path in self.labels_root.rglob("*.txt")
                if path.is_file()
                and path.relative_to(self.labels_root).with_suffix("").as_posix()
                not in image_stems
            ),
            key=natural_key,
        )
        return {
            "classes": list(self.classes),
            "totals": totals,
            "class_counts": class_counts,
            "subjects": [subjects[sid] for sid in sorted(subjects, key=natural_key)],
            "label_errors": label_errors,
            "orphan_labels": orphan_labels,
        }
