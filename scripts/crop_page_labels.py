#!/usr/bin/env python3
"""Re-crop page-space annotations onto a patch set's exported cell crops.

Usage:

    pixi run python scripts/crop_page_labels.py

For every ``<source-id>.txt`` of page-space YOLO labels under ``--pages`` (by
default ``data/v2_patches/page_labels``, the tree
``scripts/recombine_labels.py`` writes) the script maps each labeled character
box down onto the patch crops described by ``<qc>/<source-id>/table_grid.json``
(by default ``data/v2_patches/qc``, the geometry
``scripts/preprocess_pages.py`` writes) and writes one per-patch YOLO label
file ``<out>/<source-id>/<row>_<col>.txt`` (by default
``data/v2_patches/labels``), named after the patch it mirrors.

The mapping is the inverse of ``scripts/recombine_labels.py:map_box_to_page``:
a page-normalised box is scaled by the page size and intersected with the
patch's page-space crop rectangle (``cells[].rect``); the intersection, clipped
to the crop rectangle, is normalised by the crop's size. Neighbouring patches
overlap (every crop is padded), so a box near a boundary is clipped into every
patch it touches and can appear on several patches. A clipped copy that retains
less than ``--min-retained-area`` of the box's original area (half by default)
is omitted from that patch's label: the per-patch files keep the copies where
at least half the character is visible instead of mirroring near-duplicates of
truncated fragments. Surviving copies that still appear on several patches are
kept as labeled and counted in the log and run summary, together with the
discarded partial copies, so the duplication stays visible. A box that
intersects no patch at all (an annotation outside the exported grid) and an
intersection that clips to zero area -- or below the six-decimal precision the
labels are written at -- are dropped and counted the same way. Page labels are
parsed strictly per the annotation tool's label contract (five columns
``<class> <cx> <cy> <w> <h>``) and the output uses the tool's label formatting,
so class ids pass through unchanged and the per-patch files are byte-compatible
with hand-written crop labels.

An empty label file says "annotated and confirmed to hold nothing", unlike an
unlabeled patch; that state is re-derived from the page's confirmed-empty
sidecar ``<pages>/<source-id>.empty.json`` (the ``scripts/recombine_labels.py``
empty regions, in page-normalised coordinates). A patch that never received a
box copy is written as an empty label file exactly when its nominal cell centre
-- the kept grid line intersection its ``col``/``row`` names -- lies inside one
of those regions. A patch that received any box copy is never written as empty,
and a patch whose only copies were discarded as partial is not written either:
it held part of an annotated character, so it gets no file and stays unlabeled
(unknown, never assumed empty), even when its centre lies in an empty region.
No class is involved and the class set is untouched. A source without its
sidecar has no confirmed-empty regions and therefore never marks a patch empty;
the missing sidecar is warned about.

One run re-crops every source under ``--pages``. The output root is deleted and
recreated at the start of every run, so no label file can survive from an older
grid or a previous run; the run log is therefore kept next to the output root
(``<out>/../crop_labels.log``, appended). A source whose labels cannot be
re-cropped (missing or malformed ``table_grid.json``, a patch image missing
from ``--patches``, a malformed label line, a class id outside the
``--data-yaml`` class set, a malformed empty sidecar) fails as a whole and gets
no output files: it is logged, listed in the run summary and makes the run exit
non-zero. A source with no grid geometry (the page has no patches in this patch
set) has nothing to map onto: it is skipped, counted and does not fail the
run. Label files outside the ``<source-id>/<name>.txt`` layout have no page
identity and are listed individually as failures; an empty sidecar without its
label file is an orphan and is only warned about. Expected errors skip to the
next source; unexpected errors abort the run immediately so bugs surface.

Run activity is logged to stderr and appended to ``<out>/../crop_labels.log``:
initialisation (class loading, the retained-area threshold, label scan), every
per-source pipeline step and written artifact at DEBUG, one line per completed
source (including the partial copies discarded), and warnings for skipped or
failed sources and for page boxes with no kept copy. ``--log-level`` narrows
the verbosity (``DEBUG``, the default, shows everything); the run summary on
stdout is not affected.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAGES_DIR = REPO_ROOT / "data" / "v2_patches" / "page_labels"
DEFAULT_QC_DIR = REPO_ROOT / "data" / "v2_patches" / "qc"
DEFAULT_PATCHES_DIR = REPO_ROOT / "data" / "v2_patches" / "patches"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "v2_patches" / "labels"
DEFAULT_DATA_YAML = REPO_ROOT / "datasets" / "digits" / "data.yaml"

# The per-page geometry written by ``scripts/preprocess_pages.py``.
GRID_FILENAME = "table_grid.json"

# Minimum fraction of a page box's original area a patch copy must retain to be
# written to that patch's label file. Overridable with --min-retained-area.
DEFAULT_MIN_RETAINED_AREA = 0.5

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"

logger = logging.getLogger(__name__)


class Box(TypedDict):
    """One character box in normalised 0-1 coordinates.

    The annotation tool's ``Box`` (``tools/labeler/dataset.py``): the same five
    fields the label files hold, here in patch space after the mapping.
    """

    cls: int
    cx: float
    cy: float
    w: float
    h: float


class Rect(TypedDict):
    """One region rectangle in normalised 0-1 page coordinates.

    The annotation tool's ``Box`` (``tools/labeler/dataset.py``) minus the
    class: a confirmed-empty region has no class.
    """

    cx: float
    cy: float
    w: float
    h: float


class ClippedCopy(NamedTuple):
    """One page box clipped to a patch crop, with how much of it survived.

    ``retained_area`` is the fraction of the original box's area the clipped
    copy keeps (``1.0`` when the box lies wholly inside the crop); the
    retained-area threshold uses it to omit heavily truncated copies.
    """

    box: Box
    retained_area: float


class GridLine(TypedDict):
    """One kept grid line of ``table_grid.json``.

    The line's grid index (the ``col``/``row`` cell-naming convention) and its
    page-pixel position anchor the nominal cell centres the empty marking
    tests.
    """

    index: int  # grid line index k
    position: float  # line position in page pixels


class GridImage(TypedDict):
    """The page-image block of ``table_grid.json`` the mapping normalises by.

    Only the page dimensions are needed to scale page-normalised boxes to page
    pixels and back.
    """

    width_px: float
    height_px: float


class GridCell(TypedDict):
    """One exported patch entry of ``table_grid.json``.

    Which patch the entry describes and the page-space rectangle it was cropped
    from (``EXPAND`` padding applied, rounded outward, clipped to the page).
    """

    row: int
    col: int
    file: str  # patch filename, "<row>_<col>.png"
    rect: list[int]  # [left, upper, right, lower] crop rectangle in page pixels


class GridMetadata(TypedDict):
    """The subset of a ``table_grid.json`` grid-geometry result the mapping consumes.

    The same subset as ``scripts/recombine_labels.py:GridMetadata``. The fitted
    grid geometry (phase, grid line residuals) plays no part in mapping page
    boxes onto patches; ``columns``/``rows`` anchor the nominal cell centres
    the empty marking tests.
    """

    source_id: str
    image: GridImage
    spacing_x: float
    spacing_y: float
    columns: list[GridLine]
    rows: list[GridLine]
    cells: list[GridCell]


class SourceSummary(NamedTuple):
    """Patch and box counts of one re-cropped source page."""

    source_id: str
    n_patches: int  # exported grid cells
    n_labeled: int  # patches written with at least one box copy
    n_empty: int  # patches written as empty label files (sidecar regions)
    n_unlabeled: int  # patches with no label file
    n_boxes: int  # box copies written to patch label files
    n_duplicate_copies: int  # surviving copies beyond the first, per page box
    n_dropped: int  # page boxes left with no surviving copy
    n_partial_discarded: int  # copies discarded for retaining under the threshold


# ---------------------------------------------------------------------------
# logging


def setup_logging(level_name: str, log_file: Path) -> None:
    """Configure run logging: stderr plus an appended ``log_file``.

    Both handlers receive identical records with UTC timestamps; the log file
    is appended to, so reruns extend the same trail. Same contract as
    ``scripts/recombine_labels.py:setup_logging``.

    Args:
        level_name: Logging level name (``DEBUG``..``CRITICAL``). ``DEBUG``,
            the ``--log-level`` default, logs everything including every
            pipeline step and written artifact.
        log_file: Log file to append to; the parent directory is created as
            needed. It must lie outside the output root, which every run
            deletes.
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
    ``scripts/recombine_labels.py:load_classes`` (and thereby
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


def repo_relative(path: Path) -> str:
    """Render ``path`` as a POSIX path relative to the repository root.

    Args:
        path: Any filesystem path.

    Returns:
        The repo-relative POSIX path, or the absolute POSIX path when ``path``
        lies outside the repository (e.g. a ``--pages`` directory elsewhere).
    """
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved.as_posix()
    return relative.as_posix()


# ---------------------------------------------------------------------------
# label, sidecar and grid I/O


def parse_label_file(path: Path, n_classes: int) -> list[Box]:
    """Strictly parse one five-column YOLO label file.

    Each line is ``<class> <cx> <cy> <w> <h>`` in normalised 0-1
    coordinates; blank lines and ``#`` comments are skipped. Parsing is
    strict: a malformed line is reported, never skipped, so corruption cannot
    slip into the per-patch output. Same contract as
    ``scripts/recombine_labels.py:parse_label_file``.

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
            raise ValueError(
                f"{path.name} line {lineno}: expected 5 fields, got {len(parts)}"
            )
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path.name} line {lineno}: non-numeric field") from exc
        if not 0 <= cls < n_classes:
            raise ValueError(f"{path.name} line {lineno}: class {cls} out of range")
        boxes.append(Box(cls=cls, cx=cx, cy=cy, w=w, h=h))
    return boxes


def parse_empty_regions(path: Path, source_id: str) -> list[Rect]:
    """Strictly parse one ``<source-id>.empty.json`` confirmed-empty sidecar.

    The document is ``{"source_id": ..., "empty": [{"cx", "cy", "w", "h"},
    ...]}`` in page-normalised coordinates (the
    ``scripts/recombine_labels.py`` empty sidecar). Parsing is strict in the
    style of :func:`load_grid_metadata`: a malformed record is reported, never
    skipped, and the document's ``source_id`` must name the page it sits next
    to.

    Args:
        path: The sidecar to parse.
        source_id: The page's id the sidecar must name.

    Returns:
        The confirmed-empty region rectangles in file order.

    Raises:
        ValueError: If the file is not valid JSON, is not a JSON object, its
            ``source_id`` does not match, ``empty`` is not a list, or a record
            is malformed (non-mapping, non-numeric or non-finite field,
            non-positive width or height).
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if raw.get("source_id") != source_id:
        raise ValueError(
            f"{path}: source_id {raw.get('source_id')!r} does not match {source_id!r}"
        )
    entries = raw.get("empty")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'empty' must be a list")
    rects: list[Rect] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: empty[{i}] must be a mapping")
        values: dict[str, float] = {}
        for name in ("cx", "cy", "w", "h"):
            value = entry.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{path}: empty[{i}].{name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{path}: empty[{i}].{name} must be finite")
            values[name] = float(value)
        if values["w"] <= 0.0 or values["h"] <= 0.0:
            raise ValueError(f"{path}: empty[{i}] must have positive width and height")
        rects.append(
            Rect(cx=values["cx"], cy=values["cy"], w=values["w"], h=values["h"])
        )
    return rects


def parse_grid_lines(path: Path, name: str, raw: object) -> list[GridLine]:
    """Strictly parse one ``table_grid.json`` axis's kept grid lines.

    Each entry is a ``GridLine``: an integer ``index`` (grid line index ``k``,
    the cell ``col``/``row`` naming convention) and a finite numeric
    ``position`` in page pixels. Parsing is strict in the style of
    :func:`load_grid_metadata`: a malformed entry is reported, never skipped,
    and indices are unique per axis so one cell name resolves to one line.

    Args:
        path: The ``table_grid.json`` being parsed (for error messages).
        name: The axis's field name (``columns`` or ``rows``).
        raw: The field's raw JSON value.

    Returns:
        The kept grid lines in file order.

    Raises:
        ValueError: If the value is not a non-empty list, an entry is
            malformed (non-mapping, non-integer ``index``, non-numeric or
            non-finite ``position``), or an index is duplicated.
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: '{name}' must be a non-empty list")
    lines: list[GridLine] = []
    seen: set[int] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: {name}[{i}] must be a mapping")
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{path}: {name}[{i}].index must be an integer")
        position = entry.get("position")
        if isinstance(position, bool) or not isinstance(position, (int, float)):
            raise ValueError(f"{path}: {name}[{i}].position must be a number")
        if not math.isfinite(position):
            raise ValueError(f"{path}: {name}[{i}].position must be finite")
        if index in seen:
            raise ValueError(f"{path}: duplicate {name} index {index}")
        seen.add(index)
        lines.append(GridLine(index=index, position=float(position)))
    return lines


def load_grid_metadata(path: Path) -> GridMetadata:
    """Read one page's ``table_grid.json`` and validate the fields the mapping consumes.

    Validates the ``GridMetadata`` subset this script maps through: a
    ``source_id``, a page size with positive dimensions, positive
    ``spacing_x``/``spacing_y``, non-empty ``columns``/``rows`` kept grid
    lines (integer ``index``, finite ``position``, unique per axis), and a
    non-empty ``cells`` list whose entries carry an integer ``row``/``col``
    naming a kept grid line, a bare patch ``file`` name, and a well-formed
    crop ``rect`` -- all unique per page, so one label file mirrors exactly
    one grid cell. The rest of the grid geometry is ignored. Same contract as
    ``scripts/recombine_labels.py:load_grid_metadata``.

    Args:
        path: The page's ``table_grid.json``.

    Returns:
        The page's consumed grid metadata.

    Raises:
        ValueError: If the file is missing, is not valid JSON, or any
            consumed field is missing or malformed (including a duplicate
            ``file``/stem or ``(row, col)`` cell, or a cell whose
            ``row``/``col`` matches no kept grid line).
    """
    if not path.is_file():
        raise ValueError(f"{GRID_FILENAME} not found: {path}")
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

    spacings: dict[str, float] = {}
    for name in ("spacing_x", "spacing_y"):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: '{name}' must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path}: '{name}' must be positive")
        spacings[name] = float(value)

    columns = parse_grid_lines(path, "columns", raw.get("columns"))
    rows = parse_grid_lines(path, "rows", raw.get("rows"))
    column_positions = {line["index"]: line["position"] for line in columns}
    row_positions = {line["index"]: line["position"] for line in rows}

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
        if col not in column_positions:
            raise ValueError(f"{path}: cells[{i}].col {col} matches no kept grid line")
        if row not in row_positions:
            raise ValueError(f"{path}: cells[{i}].row {row} matches no kept grid line")
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
        spacing_x=spacings["spacing_x"],
        spacing_y=spacings["spacing_y"],
        columns=columns,
        rows=rows,
        cells=grid_cells,
    )


def write_patch_labels(path: Path, boxes: Sequence[Box]) -> None:
    """Write one patch's boxes as a per-patch YOLO label file.

    Byte-compatible with the annotation tool's crop labels (the formatting of
    ``tools/labeler/dataset.Dataset.write_labels``): one
    ``<class> <cx> <cy> <w> <h>`` line per box at six decimals, a trailing
    newline when non-empty and none for an empty file. Any existing file is
    overwritten.

    Args:
        path: Destination ``<row>_<col>.txt``; the parent directory is
            created as needed.
        boxes: The patch's boxes in output order; empty writes an empty label
            file (the patch is annotated and confirmed to hold nothing).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
        for b in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# re-cropping


def map_box_to_patch(
    box: Box, rect: Sequence[int], width_px: float, height_px: float
) -> ClippedCopy | None:
    """Map one page-normalised label box onto a patch crop, clipped to the crop.

    The patch was cropped from the page rectangle ``rect``, so the box is
    intersected with that rectangle and the intersection's edges translate by
    the rectangle's origin and scale by its size. Unlike
    ``scripts/recombine_labels.py:map_box_to_page`` the box is clipped to the
    crop: what the patch does not contain is not labeled on it. The retained
    area of the copy is returned alongside the box so the caller can decide
    whether the visible part is worth labeling.

    Args:
        box: The label box in page-normalised coordinates.
        rect: The patch's crop rectangle ``[left, upper, right, lower]`` in
            page pixels (``table_grid.json``'s ``cells[].rect``).
        width_px: Page image width in pixels.
        height_px: Page image height in pixels.

    Returns:
        The clipped box in patch-normalised coordinates and the fraction of the
        original box's area it retains, or ``None`` when the box does not
        intersect the crop, the intersection has zero area, or the clipped box
        degenerates to zero width or height at the label files' six-decimal
        precision (which would write a zero-size box the annotation tool
        rejects).
    """
    left, upper, right, lower = rect
    patch_w = right - left
    patch_h = lower - upper
    x1 = (box["cx"] - box["w"] / 2.0) * width_px
    x2 = (box["cx"] + box["w"] / 2.0) * width_px
    y1 = (box["cy"] - box["h"] / 2.0) * height_px
    y2 = (box["cy"] + box["h"] / 2.0) * height_px
    ix1 = max(x1, left)
    ix2 = min(x2, right)
    iy1 = max(y1, upper)
    iy2 = min(y2, lower)
    if not (ix2 > ix1 and iy2 > iy1):
        return None
    cx = ((ix1 + ix2) / 2.0 - left) / patch_w
    cy = ((iy1 + iy2) / 2.0 - upper) / patch_h
    w = (ix2 - ix1) / patch_w
    h = (iy2 - iy1) / patch_h
    if round(w, 6) == 0.0 or round(h, 6) == 0.0:
        return None
    retained_area = (ix2 - ix1) * (iy2 - iy1) / ((x2 - x1) * (y2 - y1))
    return ClippedCopy(
        box=Box(cls=box["cls"], cx=cx, cy=cy, w=w, h=h),
        retained_area=retained_area,
    )


def grid_line_position(lines: Sequence[GridLine], index: int, name: str) -> float:
    """Look up one kept grid line's page-pixel position by its grid index.

    Args:
        lines: One axis's kept grid lines (``table_grid.json`` ``columns`` or
            ``rows``).
        index: The grid line index to find (a cell's ``col`` or ``row``).
        name: The cell field's name, for the error message.

    Returns:
        The line's position in page pixels.

    Raises:
        ValueError: If no kept line has that index (``load_grid_metadata``
            rejects such pages up front; this is a defensive check).
    """
    for line in lines:
        if line["index"] == index:
            return line["position"]
    raise ValueError(f"cell {name} {index} matches no kept grid line")


def nominal_cell_centre(cell: GridCell, metadata: GridMetadata) -> tuple[float, float]:
    """Return one grid cell's nominal centre in page pixels.

    The cell sits on the kept grid line intersection its ``col``/``row``
    names, so the centre is that intersection
    (``scripts/preprocess_pages.py:build_grid_metadata`` geometry). This is
    the point the empty marking tests against the page's confirmed-empty
    regions.

    Args:
        cell: The grid cell entry (uses ``row`` and ``col``).
        metadata: The page's consumed grid geometry.

    Returns:
        ``(x, y)`` of the nominal cell centre in page pixels.

    Raises:
        ValueError: If the cell's ``row`` or ``col`` matches no kept grid
            line (``load_grid_metadata`` rejects such pages up front).
    """
    x = grid_line_position(metadata["columns"], cell["col"], "col")
    y = grid_line_position(metadata["rows"], cell["row"], "row")
    return x, y


def point_in_rect(
    point: tuple[float, float], rect: Rect, width_px: float, height_px: float
) -> bool:
    """Test whether a page-pixel point lies inside a normalised page rectangle.

    Args:
        point: ``(x, y)`` in page pixels.
        rect: Region rectangle in page-normalised coordinates.
        width_px: Page image width in pixels.
        height_px: Page image height in pixels.

    Returns:
        ``True`` when the point lies inside the rectangle's bounds (edges
        included).
    """
    px, py = point
    x1 = (rect["cx"] - rect["w"] / 2.0) * width_px
    x2 = (rect["cx"] + rect["w"] / 2.0) * width_px
    y1 = (rect["cy"] - rect["h"] / 2.0) * height_px
    y2 = (rect["cy"] + rect["h"] / 2.0) * height_px
    return x1 <= px <= x2 and y1 <= py <= y2


def scan_page_labels(
    pages_root: Path,
) -> tuple[list[tuple[str, Path]], list[Path], list[Path]]:
    """Discover the annotated sources and stray files under ``pages_root``.

    A label file belongs to a source iff it lies exactly at
    ``<source-id>.txt``; every other ``.txt`` file (nested deeper) has no page
    identity and no page geometry to be mapped with, so it is returned
    separately and the run reports it. A ``<source-id>.empty.json`` sidecar is
    paired iff it sits next to a discovered label file; anything else is an
    orphan and is only warned about.

    Args:
        pages_root: Root of the page-space label tree.

    Returns:
        ``(sources, unmappable, orphan_sidecars)``: the sources as
        ``(source id, label file)`` in sorted key order, the label files
        outside the ``<source-id>.txt`` layout (sorted), and the empty sidecars
        without their label file (sorted).

    Raises:
        ValueError: If ``pages_root`` is missing.
    """
    if not pages_root.is_dir():
        raise ValueError(f"page labels directory not found: {pages_root}")
    sources: list[tuple[str, Path]] = []
    unmappable: list[Path] = []
    for path in sorted(pages_root.rglob("*.txt")):
        relative = path.relative_to(pages_root)
        if len(relative.parts) == 1:
            sources.append((path.stem, path))
        else:
            unmappable.append(path)
    paired = {f"{stem}.empty.json" for stem, _ in sources}
    orphan_sidecars = [
        path
        for path in sorted(pages_root.rglob("*.empty.json"))
        if path.relative_to(pages_root).as_posix() not in paired
    ]
    return sources, unmappable, orphan_sidecars


def process_source(
    source_id: str,
    label_path: Path,
    sidecar_path: Path,
    qc_root: Path,
    patches_root: Path,
    out_dir: Path,
    n_classes: int,
    min_retained_area: float,
) -> SourceSummary:
    """Re-crop one page's page-space labels onto its patch grid's crops.

    Loads ``<qc>/<source-id>/table_grid.json``, maps every page box onto every
    patch crop it intersects (clipped), drops the copies that retain less than
    ``min_retained_area`` of the box's original area, and writes
    ``<out>/<source-id>/<row>_<col>.txt`` for every patch that received a
    surviving copy plus an empty label file for every patch that never received
    a box copy and whose nominal cell centre lies in a confirmed-empty sidecar
    region. A patch whose only copies were dropped as partial keeps no file: it
    held part of a character, so it stays unlabeled rather than empty. All
    other patches get no file and stay unlabeled.

    Args:
        source_id: The page's id (its ``<source-id>.txt`` name).
        label_path: The page's page-space label file.
        sidecar_path: The page's ``<source-id>.empty.json``; missing means no
            confirmed-empty regions.
        qc_root: Root of the ``qc/`` tree holding ``<source-id>/table_grid.json``.
        patches_root: Root of the patches tree; every grid cell's patch must
            exist under ``<patches>/<source-id>/``.
        out_dir: Output root for the per-patch label files.
        n_classes: Size of the dataset's class set (``data.yaml``).
        min_retained_area: Minimum fraction of a page box's original area a
            patch copy must retain to be written (``--min-retained-area``);
            copies below it are omitted from that patch.

    Returns:
        The source's ``SourceSummary``.

    Raises:
        ValueError: If ``table_grid.json`` is missing or invalid, a patch
            image is missing, the label file is malformed or holds an
            out-of-range class id, or the empty sidecar is malformed.
        OSError: If a file cannot be read or written.
    """
    logger.debug("processing %s", source_id)
    metadata = load_grid_metadata(qc_root / source_id / GRID_FILENAME)
    for cell in metadata["cells"]:
        patch_path = patches_root / source_id / cell["file"]
        if not patch_path.is_file():
            raise ValueError(f"patch image missing: {repo_relative(patch_path)}")

    width_px = metadata["image"]["width_px"]
    height_px = metadata["image"]["height_px"]
    boxes = parse_label_file(label_path, n_classes)
    if sidecar_path.is_file():
        empty_rects = parse_empty_regions(sidecar_path, source_id)
    else:
        # No sidecar: the page has no confirmed-empty regions, so no patch is
        # ever marked empty. A missing sidecar is not a failure -- the boxes
        # can still be re-cropped -- but it is never silent.
        logger.warning(
            "%s: no %s, no confirmed-empty regions", source_id, sidecar_path.name
        )
        empty_rects = []

    placed: list[list[Box]] = [[] for _ in metadata["cells"]]
    kept: set[int] = set()
    fragment_cells: set[int] = set()
    n_partial_discarded = 0
    for i, cell in enumerate(metadata["cells"]):
        for index, box in enumerate(boxes):
            copy = map_box_to_patch(box, cell["rect"], width_px, height_px)
            if copy is None:
                continue
            if copy.retained_area < min_retained_area:
                fragment_cells.add(i)
                n_partial_discarded += 1
                continue
            placed[i].append(copy.box)
            kept.add(index)
    n_copies = sum(len(copies) for copies in placed)
    n_duplicate_copies = n_copies - len(kept)
    n_dropped = len(boxes) - len(kept)

    out_root = out_dir / source_id
    n_labeled = 0
    n_empty = 0
    for i, (cell, copies) in enumerate(zip(metadata["cells"], placed)):
        stem = Path(cell["file"]).stem
        out_path = out_root / f"{stem}.txt"
        if copies:
            write_patch_labels(out_path, copies)
            n_labeled += 1
            continue
        if i in fragment_cells:
            # Held only copies too partial to label: unknown, never empty.
            continue
        centre = nominal_cell_centre(cell, metadata)
        if any(
            point_in_rect(centre, rect, width_px, height_px) for rect in empty_rects
        ):
            write_patch_labels(out_path, [])
            n_empty += 1
    n_unlabeled = len(metadata["cells"]) - n_labeled - n_empty
    logger.debug("%s: wrote %s", source_id, repo_relative(out_root))
    if n_dropped:
        logger.warning(
            "%s: %d page boxes dropped (no kept copy), %d copies kept",
            source_id,
            n_dropped,
            n_copies,
        )
    if n_duplicate_copies:
        logger.debug(
            "%s: %d duplicate copies across overlapping patches",
            source_id,
            n_duplicate_copies,
        )
    return SourceSummary(
        source_id=source_id,
        n_patches=len(metadata["cells"]),
        n_labeled=n_labeled,
        n_empty=n_empty,
        n_unlabeled=n_unlabeled,
        n_boxes=n_copies,
        n_duplicate_copies=n_duplicate_copies,
        n_dropped=n_dropped,
        n_partial_discarded=n_partial_discarded,
    )


# ---------------------------------------------------------------------------
# CLI


def guard_out_dir(out_dir: Path, protected: Sequence[Path]) -> None:
    """Refuse an output root that would delete an input or the repository.

    Every run deletes ``--out`` wholesale, so an output root that contains (or
    is) a protected path -- the repository root, the input trees, the class
    file, the run log -- would destroy the data it reads. The check runs
    before anything is written.

    Args:
        out_dir: The resolved output root.
        protected: Paths that must survive the run.

    Raises:
        SystemExit: If any protected path lies at or inside ``out_dir``.
    """
    resolved = out_dir.resolve()
    for path in protected:
        other = path.resolve()
        if resolved == other or resolved in other.parents:
            raise SystemExit(
                f"--out {resolved} would delete {other}; refusing to clear it"
            )


def retained_area_fraction(value: str) -> float:
    """Parse and validate a ``--min-retained-area`` fraction.

    Args:
        value: The raw command-line value.

    Returns:
        The fraction as a float in ``(0, 1]``.

    Raises:
        argparse.ArgumentTypeError: If the value is not a finite number greater
            than 0 and at most 1.
    """
    try:
        fraction = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from exc
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise argparse.ArgumentTypeError(
            f"must be greater than 0 and at most 1, got {value!r}"
        )
    return fraction


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the annotation re-cropping script."""
    parser = argparse.ArgumentParser(
        description="Re-crop page-space YOLO labels onto a patch set's exported cell crops.",
    )
    parser.add_argument(
        "--pages",
        type=Path,
        default=DEFAULT_PAGES_DIR,
        help="root of the page-space label tree (default: %(default)s)",
    )
    parser.add_argument(
        "--qc",
        type=Path,
        default=DEFAULT_QC_DIR,
        help="root of the qc/ tree holding <source-id>/table_grid.json (default: %(default)s)",
    )
    parser.add_argument(
        "--patches",
        type=Path,
        default=DEFAULT_PATCHES_DIR,
        help="root of the patches tree holding <source-id>/<row>_<col>.png (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output root for the per-patch label files; deleted and recreated every run (default: %(default)s)",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="dataset data.yaml naming the classes (default: %(default)s)",
    )
    parser.add_argument(
        "--min-retained-area",
        type=retained_area_fraction,
        default=DEFAULT_MIN_RETAINED_AREA,
        help=(
            "fraction of a box's original area a patch copy must retain to be "
            "written; smaller copies are discarded (default: %(default)s)"
        ),
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
    """Re-crop every source under ``--pages`` and print the run summary.

    Logging is configured first from ``--log-level`` (stderr plus the appended
    ``<out>/../crop_labels.log``); the run summary still goes to stdout. The
    output root is then deleted and recreated, so no label file survives from
    an older grid or a previous run. The retained-area threshold is logged and
    applied to every box copy. Sources that fail with an expected error
    are logged and skipped and no output is written for them; sources without
    grid geometry are skipped without failing; the run summary lists every
    failure and the run exits 1 if there was any. Unexpected errors abort the
    run immediately so bugs surface.

    Raises:
        SystemExit: If the arguments are invalid, the output root is unsafe to
            clear, the class set cannot be loaded, the page-label tree cannot
            be scanned, or any source or label file failed.
    """
    args = parse_args()
    pages_root: Path = args.pages.resolve()
    qc_root: Path = args.qc.resolve()
    patches_root: Path = args.patches.resolve()
    out_dir: Path = args.out.resolve()
    log_file = out_dir.parent / "crop_labels.log"
    guard_out_dir(
        out_dir,
        [REPO_ROOT, pages_root, qc_root, patches_root, args.data_yaml, log_file],
    )
    setup_logging(args.log_level, log_file)

    if out_dir.exists():
        if not out_dir.is_dir():
            raise SystemExit(f"--out is not a directory: {out_dir}")
        shutil.rmtree(out_dir)
        logger.info("cleared %s", repo_relative(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        classes = load_classes(args.data_yaml)
    except ValueError as exc:
        logger.error("cannot load classes: %s", exc)
        raise SystemExit(f"cannot load classes: {exc}") from exc
    logger.info("%d classes from %s", len(classes), repo_relative(args.data_yaml))
    logger.info(
        "retained-area threshold %.3f (smaller patch copies discarded)",
        args.min_retained_area,
    )

    try:
        sources, unmappable, orphan_sidecars = scan_page_labels(pages_root)
    except ValueError as exc:
        logger.error("cannot scan %s: %s", repo_relative(pages_root), exc)
        raise SystemExit(f"cannot scan {pages_root}: {exc}") from exc
    logger.info("%d sources under %s", len(sources), repo_relative(pages_root))
    logger.info("writing %s", repo_relative(out_dir))

    failed: list[tuple[str, str]] = []
    for path in unmappable:
        reason = "not a <source-id>.txt page label; no page identity to re-crop it with"
        logger.warning("label %s failed: %s", repo_relative(path), reason)
        failed.append((repo_relative(path), reason))

    for path in orphan_sidecars:
        logger.warning(
            "empty sidecar %s has no label file, ignored", repo_relative(path)
        )

    summaries: list[SourceSummary] = []
    n_skipped = 0
    for source_id, label_path in sources:
        sidecar_path = label_path.with_name(f"{source_id}.empty.json")
        grid_path = qc_root / source_id / GRID_FILENAME
        if not grid_path.is_file():
            n_skipped += 1
            logger.warning(
                "%s: no %s, skipped (no patches in this patch set)",
                source_id,
                repo_relative(grid_path),
            )
            continue
        try:
            summary = process_source(
                source_id,
                label_path,
                sidecar_path,
                qc_root,
                patches_root,
                out_dir,
                len(classes),
                args.min_retained_area,
            )
        except (ValueError, OSError) as exc:
            logger.warning("source %s failed: %s", source_id, exc)
            failed.append((source_id, str(exc)))
            continue
        summaries.append(summary)
        logger.info(
            "%s: %d patches (%d labeled / %d empty / %d unlabeled), "
            "%d boxes (%d duplicate copies, %d dropped, %d partial copies discarded)",
            summary.source_id,
            summary.n_patches,
            summary.n_labeled,
            summary.n_empty,
            summary.n_unlabeled,
            summary.n_boxes,
            summary.n_duplicate_copies,
            summary.n_dropped,
            summary.n_partial_discarded,
        )

    total_labeled = sum(s.n_labeled for s in summaries)
    total_empty = sum(s.n_empty for s in summaries)
    total_unlabeled = sum(s.n_unlabeled for s in summaries)
    total_boxes = sum(s.n_boxes for s in summaries)
    total_dups = sum(s.n_duplicate_copies for s in summaries)
    total_dropped = sum(s.n_dropped for s in summaries)
    total_partial = sum(s.n_partial_discarded for s in summaries)
    run_summary = (
        f"{len(sources)} sources ({n_skipped} without new geometry) / "
        f"{len(summaries)} sources re-cropped / "
        f"{total_labeled} labeled patches, {total_empty} empty patches, "
        f"{total_unlabeled} unlabeled patches / "
        f"{total_boxes} boxes ({total_dups} duplicate copies, {total_dropped} dropped, "
        f"{total_partial} partial copies discarded) "
        f"-> {out_dir} ({len(failed)} failed, "
        f"{len(orphan_sidecars)} orphan sidecars ignored)"
    )
    logger.info(run_summary)
    print(run_summary)
    for label, reason in failed:
        print(f"  {label}: {reason}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
