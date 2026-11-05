#!/usr/bin/env python3
"""Recombine the per-patch labels of weather-report pages into per-image labels.

Usage:

    pixi run python scripts/recombine_labels.py

For every ``<source-id>/`` folder of per-patch YOLO labels under ``--labels``
(by default ``datasets/digits/labels``, the tree the annotation tool writes --
see ``tools/labeler/README.md``) the script maps each labeled character box
from its cell crop back onto the page and writes one per-page YOLO label file
``<out>/<source-id>.txt`` (by default ``data/page_labels``).

The mapping is the reassembly ``scripts/preprocess_pages.py`` saved the
geometry for: ``<qc>/<source-id>/grid.json`` records, per exported patch, the
exact page-space rectangle the patch was cropped from (``cells[].rect``, with
``cells[].file`` naming the patch) and the page image size
(``image.width_px`` / ``image.height_px``). A patch-normalised box is scaled
by the crop rectangle and normalised by the page size; no patch pixels are
read, so the mapping is independent of how the patches are stored. Patch
labels are parsed strictly per the annotation tool's label contract (five
columns ``<class> <cx> <cy> <w> <h>``) and the output uses the tool's label
formatting, so class ids pass through unchanged and the per-page files are
byte-compatible with crop labels. Only crops with a label file contribute: a
grid cell without one is an unlabeled patch and is skipped, an empty label
file is an annotated background patch and contributes no boxes.

Neighbouring patches overlap (every crop is padded by ``EXPAND`` in
``scripts/preprocess_pages.py``), so one character can be labeled on two
patches. Such near-duplicate boxes are kept as labeled -- the per-page file
mirrors the annotations, it does not clean them -- but the duplicate pairs
(same class, different patches, IoU >= ``MIN_IOU``, paired greedily by
descending IoU as in ``tools/labeler/compare.py``) are counted in the log and
run summary so the overlap stays visible.

One run recombines every source under ``--labels``. A source whose labels
cannot be recombined (missing or malformed ``grid.json``, a label file
mirroring no grid cell, a malformed label line, a class id outside the
``--data-yaml`` class set) fails as a whole and gets no output file: it is
logged, listed in the run summary and makes the run exit non-zero. Label
files outside the ``<source-id>/<row>_<col>.txt`` layout have no page
geometry and are listed individually the same way. Expected errors skip to
the next source; unexpected errors abort the run immediately so bugs surface.
A source without any label file writes nothing and is counted in the run
summary. Outputs are overwritten wholesale on every run; stale outputs of
sources no longer present under ``--labels`` are left in place.

Run activity is logged to stderr and appended to ``<out>/recombine.log``:
initialisation (class loading, label scan), every per-source pipeline step
and written artifact at DEBUG, one line per completed source and one warning
per failed source. ``--log-level`` narrows the verbosity (``DEBUG``, the
default, shows everything); the run summary on stdout is not affected.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS_DIR = REPO_ROOT / "datasets" / "digits" / "labels"
DEFAULT_QC_DIR = REPO_ROOT / "data" / "qc"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "page_labels"
DEFAULT_DATA_YAML = REPO_ROOT / "datasets" / "digits" / "data.yaml"

# Overlap at which two same-class boxes count as one character labeled on two
# patches. Same contract as ``tools/labeler/compare.py:MIN_IOU``.
MIN_IOU = 0.5

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"

logger = logging.getLogger(__name__)


class Box(TypedDict):
    """One character box in normalised 0-1 coordinates.

    The annotation tool's ``Box`` (``tools/labeler/dataset.py``): the same
    five fields the label files hold, here in page space after the mapping.
    """

    cls: int
    cx: float
    cy: float
    w: float
    h: float


class GridImage(TypedDict):
    """The page-image block of ``grid.json`` consumed by the reassembly.

    A subset of ``scripts/preprocess_pages.py:GridImage``: only the page
    dimensions are needed to normalise the mapped boxes.
    """

    width_px: float
    height_px: float


class GridCell(TypedDict):
    """One exported patch entry of ``grid.json``.

    The fields of ``scripts/preprocess_pages.py:GridCell`` the reassembly
    consumes: which patch the entry describes and the page-space rectangle it
    was cropped from (``EXPAND`` padding applied, rounded outward, clipped to
    the page).
    """

    row: int
    col: int
    file: str  # patch filename, "<row>_<col>.png"
    rect: list[int]  # [left, upper, right, lower] crop rectangle in page pixels


class GridMetadata(TypedDict):
    """The subset of a ``grid.json`` table-fitting result the reassembly consumes.

    The same document as ``scripts/preprocess_pages.py:GridMetadata``; the
    fitted grid geometry (spacing, phase, grid lines) plays no part in
    mapping patch boxes back onto the page.
    """

    source_id: str
    image: GridImage
    cells: list[GridCell]


class PlacedBox(NamedTuple):
    """One recombined box: its page-space box and the patch it was labeled on.

    The patch identity lets :func:`count_duplicate_pairs` tell apart the
    genuine several-characters-per-patch case from one character labeled on
    two overlapping patches.
    """

    patch: str
    box: Box


class SourceSummary(NamedTuple):
    """Patch and box counts of one recombined source page."""

    source_id: str
    n_patches: int  # exported grid cells
    n_labeled: int  # patches whose label file holds boxes
    n_empty: int  # patches whose label file is empty/ignored
    n_unlabeled: int  # patches without a label file
    n_boxes: int  # boxes written to the page label file
    n_dups: int  # near-duplicate cross-patch box pairs (kept, counted)


# ---------------------------------------------------------------------------
# logging


def setup_logging(level_name: str, log_file: Path) -> None:
    """Configure run logging: stderr plus an appended ``log_file``.

    Both handlers receive identical records with UTC timestamps; the log file
    is appended to, so reruns extend the same trail. Same contract as
    ``scripts/preprocess_pages.py:setup_logging``.

    Args:
        level_name: Logging level name (``DEBUG``..``CRITICAL``). ``DEBUG``,
            the ``--log-level`` default, logs everything including every
            pipeline step and written artifact.
        log_file: Log file to append to; the parent directory is created as
            needed.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    formatter.converter = time.gmtime
    root = logging.getLogger()
    root.setLevel(level_name)
    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    for handler in (stderr_handler, file_handler):
        handler.setFormatter(formatter)
        root.addHandler(handler)


# ---------------------------------------------------------------------------
# dataset helpers


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


def load_classes(yaml_path: Path) -> list[str]:
    """Read class names from the int-keyed ``names:`` mapping of a YOLO data.yaml.

    ``data.yaml`` is the single source of truth for classes; nothing falls
    back to defaults. Same contract as
    ``scripts/bootstrap_annotations.py:load_classes`` (and thereby
    ``tools/labeler.dataset.load_classes``).

    Args:
        yaml_path: Path to the dataset's ``data.yaml``.

    Returns:
        Class names indexed by class id.

    Raises:
        ValueError: If the file is missing, is not valid YAML (including a
            duplicate mapping key), or has no usable ``names:`` mapping
            (non-mapping or empty, non-integer or duplicate keys,
            non-string names).
    """
    if not yaml_path.is_file():
        raise ValueError(f"data.yaml not found: {yaml_path}")
    try:
        data = yaml.load(yaml_path.read_text(encoding="utf-8"), Loader=_StrictLoader)
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
            raise ValueError(f"{yaml_path}: names key {key!r} is not an integer") from exc
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"{yaml_path}: names[{key!r}] must be a string, got {type(value).__name__}")
        if idx in names:
            raise ValueError(f"{yaml_path}: duplicate names index {idx}")
        names[idx] = str(value)
    return [names[i] for i in sorted(names)]


# ---------------------------------------------------------------------------
# label and grid I/O


def repo_relative(path: Path) -> str:
    """Render ``path`` as a POSIX path relative to the repository root.

    Args:
        path: Any filesystem path.

    Returns:
        The repo-relative POSIX path, or the absolute POSIX path when ``path``
        lies outside the repository (e.g. a ``--labels`` directory elsewhere).
    """
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved.as_posix()
    return relative.as_posix()


def parse_label_file(path: Path, n_classes: int) -> list[Box]:
    """Strictly parse one five-column YOLO label file.

    Each line is ``<class> <cx> <cy> <w> <h>`` in normalised 0-1
    coordinates; blank lines and ``#`` comments are skipped. Parsing is
    strict: a malformed line is reported, never skipped, so corruption cannot
    slip into the per-page output. Same contract as
    ``tools/labeler/dataset.Dataset.read_label_file``, raising ``ValueError``
    instead of an API error.

    Args:
        path: The label file to parse.
        n_classes: Size of the dataset's class set (``data.yaml``); valid
            class ids are ``0 .. n_classes - 1``.

    Returns:
        The label boxes in file order; empty for an empty (or comment-only)
        label file.

    Raises:
        ValueError: If the file is missing, a line has the wrong field count
            or non-numeric fields, or a class id is out of range.
    """
    if not path.is_file():
        raise ValueError(f"no such label: {path}")
    boxes: list[Box] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path.name} line {lineno}: expected 5 fields, got {len(parts)}")
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path.name} line {lineno}: non-numeric field") from exc
        if not 0 <= cls < n_classes:
            raise ValueError(f"{path.name} line {lineno}: class {cls} out of range")
        boxes.append(Box(cls=cls, cx=cx, cy=cy, w=w, h=h))
    return boxes


def load_grid_metadata(path: Path) -> GridMetadata:
    """Read one page's ``grid.json`` and validate the fields reassembly consumes.

    Validates the ``GridMetadata`` subset this script maps through: a
    ``source_id``, a page size with positive dimensions, and a non-empty
    ``cells`` list whose entries carry an integer ``row``/``col``, a bare
    patch ``file`` name, and a well-formed crop ``rect`` -- all unique per
    page, so one label file maps to exactly one grid cell. The rest of the
    table-fitting result is ignored.

    Args:
        path: The page's ``grid.json``.

    Returns:
        The page's consumed grid metadata.

    Raises:
        ValueError: If the file is missing, is not valid JSON, or any
            consumed field is missing or malformed (including a duplicate
            ``file``/stem or ``(row, col)`` cell).
    """
    if not path.is_file():
        raise ValueError(f"grid.json not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object")

    source_id = raw.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"{path}: 'source_id' must be a non-empty string")

    image = raw.get("image")
    if not isinstance(image, dict):
        raise ValueError(f"{path}: 'image' must be a mapping")
    dims: dict[str, float] = {}
    for name in ("width_px", "height_px"):
        value = image.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: image.{name} must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path}: image.{name} must be positive")
        dims[name] = float(value)

    cells = raw.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{path}: 'cells' must be a non-empty list")
    grid_cells: list[GridCell] = []
    seen_files: set[str] = set()
    seen_stems: set[str] = set()
    seen_positions: set[tuple[int, int]] = set()
    for i, raw_cell in enumerate(cells):
        if not isinstance(raw_cell, dict):
            raise ValueError(f"{path}: cells[{i}] must be a mapping")
        row = raw_cell.get("row")
        col = raw_cell.get("col")
        for name, value in (("row", row), ("col", col)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{path}: cells[{i}].{name} must be an integer")
        file = raw_cell.get("file")
        if not isinstance(file, str) or not file or Path(file).name != file:
            raise ValueError(f"{path}: cells[{i}].file must be a bare filename")
        rect = raw_cell.get("rect")
        if (
            not isinstance(rect, list)
            or len(rect) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) for v in rect)
        ):
            raise ValueError(
                f"{path}: cells[{i}].rect must be [left, upper, right, lower] integers"
            )
        left, upper, right, lower = rect
        if not (right > left and lower > upper):
            raise ValueError(
                f"{path}: cells[{i}].rect is not [left, upper, right, lower]: {rect}"
            )
        if file in seen_files or Path(file).stem in seen_stems:
            raise ValueError(f"{path}: duplicate cells file {file!r}")
        if (row, col) in seen_positions:
            raise ValueError(f"{path}: duplicate cell ({row}, {col})")
        seen_files.add(file)
        seen_stems.add(Path(file).stem)
        seen_positions.add((row, col))
        grid_cells.append(GridCell(row=row, col=col, file=file, rect=rect))

    return GridMetadata(
        source_id=source_id,
        image=GridImage(width_px=dims["width_px"], height_px=dims["height_px"]),
        cells=grid_cells,
    )


def write_page_labels(path: Path, boxes: Sequence[Box]) -> None:
    """Write one page's recombined boxes as a per-page YOLO label file.

    Byte-compatible with the annotation tool's crop labels (the formatting of
    ``tools/labeler/dataset.Dataset.write_labels``): one
    ``<class> <cx> <cy> <w> <h>`` line per box at six decimals, a trailing
    newline when non-empty and none for an empty file. Any existing file is
    overwritten.

    Args:
        path: Destination ``<source-id>.txt``; the parent directory is
            created as needed.
        boxes: The page's boxes in output order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
        for b in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# recombination


def map_box_to_page(
    box: Box, rect: Sequence[int], width_px: float, height_px: float
) -> Box:
    """Map one patch-normalised label box onto the page.

    The patch was cropped from the page rectangle ``rect``, so the box's edges
    scale by the rectangle's size and translate by its origin; the result is
    normalised by the page size. The mapping is exact: boxes are never
    clamped, so a box that stuck out past a patch edge sticks out past the
    page edge here just as far.

    Args:
        box: The label box in patch-normalised coordinates.
        rect: The patch's crop rectangle ``[left, upper, right, lower]`` in
            page pixels (``grid.json``'s ``cells[].rect``).
        width_px: Page image width in pixels.
        height_px: Page image height in pixels.

    Returns:
        The same box in page-normalised coordinates.
    """
    left, upper, right, lower = rect
    patch_w = right - left
    patch_h = lower - upper
    x1 = left + (box["cx"] - box["w"] / 2.0) * patch_w
    x2 = left + (box["cx"] + box["w"] / 2.0) * patch_w
    y1 = upper + (box["cy"] - box["h"] / 2.0) * patch_h
    y2 = upper + (box["cy"] + box["h"] / 2.0) * patch_h
    return Box(
        cls=box["cls"],
        cx=(x1 + x2) / 2.0 / width_px,
        cy=(y1 + y2) / 2.0 / height_px,
        w=(x2 - x1) / width_px,
        h=(y2 - y1) / height_px,
    )


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two normalised YOLO boxes.

    Same contract as ``tools/labeler/compare.py:iou`` (which pairs a crop's
    label and prediction boxes; here it pairs the same character's boxes on
    two overlapping patches).

    Args:
        a: First box (``cls``, ``cx``, ``cy``, ``w``, ``h``).
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


def count_duplicate_pairs(placed: Sequence[PlacedBox], min_iou: float = MIN_IOU) -> int:
    """Count the near-duplicate box pairs a page's overlapping patches produced.

    Neighbouring patches overlap, so one character can be labeled on two
    patches; those duplicates are kept in the output and only counted here.
    Two boxes form a duplicate pair when they come from different patches,
    share a class and overlap at least ``min_iou``; pairs are formed greedily
    by descending IoU, taking each box at most once (the pairing contract of
    ``tools/labeler/compare.py``).

    Args:
        placed: The page's recombined boxes, each tagged with the patch it
            was labeled on.
        min_iou: Overlap two boxes need to count as one character.

    Returns:
        The number of duplicate pairs formed.

    Raises:
        ValueError: If `min_iou` is outside ``[0, 1]``.
    """
    if not 0.0 <= min_iou <= 1.0:
        raise ValueError(f"min_iou must be between 0 and 1, got {min_iou}")
    candidates: list[tuple[float, int, int]] = []
    for i, a in enumerate(placed):
        for j in range(i + 1, len(placed)):
            b = placed[j]
            if a.patch == b.patch or a.box["cls"] != b.box["cls"]:
                continue
            value = iou(a.box, b.box)
            if value >= min_iou:
                candidates.append((value, i, j))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    used: set[int] = set()
    pairs = 0
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs += 1
    return pairs


def scan_labels(labels_root: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    """Discover the sources and unmappable label files under ``labels_root``.

    A label file belongs to a source iff it lies exactly at
    ``<source-id>/<name>.txt``; the returned sources map each ``<source-id>``
    to its label files in sorted key order (subfolders without label files
    are sources too -- they recombine to nothing). Every other ``.txt`` file
    (at the root or nested deeper) mirrors no grid patch and has no page
    geometry to be mapped back with; it is returned separately so the run can
    report it.

    Args:
        labels_root: Root of the per-patch labels tree.

    Returns:
        ``(sources, unmappable)``: source id -> sorted label files, and the
        label files outside the ``<source-id>/<name>.txt`` layout, sorted.

    Raises:
        ValueError: If ``labels_root`` is missing.
    """
    if not labels_root.is_dir():
        raise ValueError(f"labels directory not found: {labels_root}")
    sources: dict[str, list[Path]] = {
        p.name: [] for p in sorted(labels_root.iterdir()) if p.is_dir()
    }
    unmappable: list[Path] = []
    for path in sorted(labels_root.rglob("*.txt")):
        rel = path.relative_to(labels_root)
        if len(rel.parts) == 2:
            sources.setdefault(rel.parts[0], []).append(path)
        else:
            unmappable.append(path)
    return sources, unmappable


def process_source(
    source_id: str,
    label_files: list[Path],
    qc_root: Path,
    out_dir: Path,
    n_classes: int,
) -> SourceSummary:
    """Recombine one page's per-patch labels into its per-page label file.

    Loads ``<qc>/<source-id>/grid.json``, maps every labeled box of every
    exported patch back onto the page and writes ``<out>/<source-id>.txt``.
    Label files must mirror the grid's patch set one-to-one: a grid cell
    without a label file is an unlabeled patch (skipped and counted), a label
    file without a grid cell is a data inconsistency and fails the whole
    source before anything is written.

    Args:
        source_id: The page's id (its ``<source-id>`` folder name).
        label_files: The source's label files; every one must mirror a grid
            cell.
        qc_root: Root of the ``qc/`` tree holding ``<source-id>/grid.json``.
        out_dir: Output root for the per-page label files.
        n_classes: Size of the dataset's class set (``data.yaml``).

    Returns:
        The source's ``SourceSummary``.

    Raises:
        ValueError: If ``grid.json`` is missing or invalid, a label file is
            malformed or holds an out-of-range class id, or a label file
            mirrors no grid cell.
        OSError: If a file cannot be read or written.
    """
    logger.debug("processing %s", source_id)
    metadata = load_grid_metadata(qc_root / source_id / "grid.json")
    by_stem = {path.stem: path for path in label_files}

    placed: list[PlacedBox] = []
    n_empty = 0
    n_unlabeled = 0
    for cell in metadata["cells"]:
        stem = Path(cell["file"]).stem
        label_path = by_stem.pop(stem, None)
        if label_path is None:
            n_unlabeled += 1
            continue
        boxes = parse_label_file(label_path, n_classes)
        if not boxes:
            n_empty += 1
        for box in boxes:
            placed.append(
                PlacedBox(
                    patch=stem,
                    box=map_box_to_page(
                        box,
                        cell["rect"],
                        metadata["image"]["width_px"],
                        metadata["image"]["height_px"],
                    ),
                )
            )
    if by_stem:
        raise ValueError(
            "label files match no grid cell: "
            + ", ".join(sorted(p.name for p in by_stem.values()))
        )

    n_dups = count_duplicate_pairs(placed)
    out_path = out_dir / f"{source_id}.txt"
    write_page_labels(out_path, [p.box for p in placed])
    logger.debug("%s: wrote %s", source_id, repo_relative(out_path))
    return SourceSummary(
        source_id=source_id,
        n_patches=len(metadata["cells"]),
        n_labeled=len(label_files) - n_empty,
        n_empty=n_empty,
        n_unlabeled=n_unlabeled,
        n_boxes=len(placed),
        n_dups=n_dups,
    )


# ---------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the label recombination script."""
    parser = argparse.ArgumentParser(
        description="Recombine the per-patch YOLO labels of weather-report pages into per-page label files.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help="root of the per-patch labels tree (default: %(default)s)",
    )
    parser.add_argument(
        "--qc",
        type=Path,
        default=DEFAULT_QC_DIR,
        help="root of the qc/ tree holding <source-id>/grid.json (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output root for the per-page label files (default: %(default)s)",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="dataset data.yaml naming the classes (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="logging verbosity; DEBUG (the default) logs every pipeline step and written artifact",
    )
    return parser.parse_args()


def main() -> None:
    """Recombine every source under ``--labels`` and print the run summary.

    Logging is configured first from ``--log-level`` (stderr plus the appended
    ``<out>/recombine.log``); the run summary still goes to stdout. Sources
    that fail with an expected error are logged and skipped and no output is
    written for them; the run summary lists every failure and the run exits 1
    if there was any. Unexpected errors abort the run immediately so bugs
    surface.

    Raises:
        SystemExit: If the arguments are invalid, the class set cannot be
            loaded, the labels tree cannot be scanned, or any source or label
            file failed.
    """
    args = parse_args()
    labels_root: Path = args.labels.resolve()
    qc_root: Path = args.qc.resolve()
    out_dir: Path = args.out.resolve()
    setup_logging(args.log_level, out_dir / "recombine.log")

    try:
        classes = load_classes(args.data_yaml)
    except ValueError as exc:
        logger.error("cannot load classes: %s", exc)
        raise SystemExit(f"cannot load classes: {exc}") from exc
    logger.info("%d classes from %s", len(classes), repo_relative(args.data_yaml))

    try:
        sources, unmappable = scan_labels(labels_root)
    except ValueError as exc:
        logger.error("cannot scan %s: %s", repo_relative(labels_root), exc)
        raise SystemExit(f"cannot scan {labels_root}: {exc}") from exc
    logger.info("%d sources under %s", len(sources), repo_relative(labels_root))
    logger.info("writing %s", repo_relative(out_dir))

    failed: list[tuple[str, str]] = []
    for path in unmappable:
        reason = (
            "not a <source-id>/<row>_<col>.txt patch label; "
            "no page geometry to map it back with"
        )
        logger.warning("label %s failed: %s", repo_relative(path), reason)
        failed.append((repo_relative(path), reason))

    summaries: list[SourceSummary] = []
    n_no_labels = 0
    for source_id, label_files in sources.items():
        if not label_files:
            n_no_labels += 1
            logger.info("%s: no label files", source_id)
            continue
        try:
            summary = process_source(
                source_id, label_files, qc_root, out_dir, len(classes)
            )
        except (ValueError, OSError) as exc:
            logger.warning("source %s failed: %s", source_id, exc)
            failed.append((source_id, str(exc)))
            continue
        summaries.append(summary)
        logger.info(
            "%s: %d patches (%d labeled / %d empty / %d unlabeled), "
            "%d boxes (%d duplicate pairs)",
            summary.source_id,
            summary.n_patches,
            summary.n_labeled,
            summary.n_empty,
            summary.n_unlabeled,
            summary.n_boxes,
            summary.n_dups,
        )

    total_boxes = sum(s.n_boxes for s in summaries)
    total_dups = sum(s.n_dups for s in summaries)
    total_unlabeled = sum(s.n_unlabeled for s in summaries)
    total_empty = sum(s.n_empty for s in summaries)
    run_summary = (
        f"{len(sources)} sources ({n_no_labels} without labels) / "
        f"{len(summaries)} page labels / "
        f"{total_boxes} boxes ({total_dups} duplicate pairs, "
        f"{total_unlabeled} unlabeled patches, {total_empty} empty patches) "
        f"-> {out_dir} ({len(failed)} failed)"
    )
    logger.info(run_summary)
    print(run_summary)
    for label, reason in failed:
        print(f"  {label}: {reason}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
