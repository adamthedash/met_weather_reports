#!/usr/bin/env python3
"""Cut weather-report pages into daily-table cell patches from table labels.

Usage:

    pixi run python scripts/preprocess_table_patches.py

For every labeled page under ``--dataset`` (by default ``datasets/tables``, the
full-page table dataset curated with the in-repo labeler) the pipeline reads
the page's YOLO table-region labels, takes the single ``daily`` box and cuts
the page into the guaranteed 12 x 31 grid of equal-sized cells. Patches are
written to ``<out>/patches/<source-id>/<row>_<col>.png`` (by default
``data/table_patches/patches``) and the daily grid's geometry to
``<out>/qc/<source-id>/table_grid.json``. A QC render,
``<out>/qc/<source-id>/grid_overlay.jpg``, draws the day x month cell
boundaries and the ``daily`` box on the page so the division can be checked
by eye against the printed table.

The ``daily`` box spans 12 month columns (January left to December right) and
31 day rows (day 1 top to day 31 bottom), so ``row`` is ``day - 1`` and ``col``
is ``month - 1``. The box is divided evenly: cell ``(row, col)`` is the
``1/12`` x ``1/31`` sub-rectangle at that position. Every cell is then cropped
with ``EXPAND`` of its width/height added on every side (10% per side, the same
padding as the detector pipeline in ``scripts/preprocess_pages.py``), rounded
outward and clipped to the page, so neighbouring patches overlap and boundary
characters are not clipped. The patch naming and geometry expected by
``datasets/digits/images`` and ``scripts/recombine_labels.py`` are preserved.

``table_grid.json`` uses the schema of the detector pipeline's ``grid.json``
(``scripts/preprocess_pages.py:GridMetadata``): the page image, ``spacing_x`` /
``spacing_y`` (the nominal cell size), the kept grid lines (nominal cell
centres, indexed 0-11 and 0-30), the first nominal cell centre as the phase,
and one ``cells`` entry per exported patch with the padded crop rectangle it
was cut from. ``scripts/recombine_labels.py`` reads it to map per-patch labels
back onto the page and to record confirmed-empty cells, so the reassembly works
unchanged apart from the filename.

Page handling follows the other pipeline scripts. Pages without a label file
are skipped and counted. A labelled page fails when its label file is malformed,
has no ``daily`` box or more than one, or the box is degenerate; expected
failures are logged and skipped, and the run exits non-zero. Orphan label files
(no matching page image) and duplicate page stems also fail the run. Unexpected
errors abort immediately so bugs surface. The pipeline is read-only towards its
input dataset and never writes into ``datasets/``; copying the patches into
``datasets/digits/images/`` stays a manual step.

Only the ``daily`` class is used. The ``month_totals``, ``year_total`` and
``days_with_rainfall`` boxes are parsed (so corrupt labels still fail) but their
cells are not cut.

Run activity is logged to stderr and appended to ``<out>/preprocess.log``:
initialisation, every per-page step and written artifact at DEBUG, one line per
completed page and one warning per failed page. ``--log-level`` narrows the
verbosity (``DEBUG``, the default, shows everything); the run summary on stdout
is not affected.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import NamedTuple, TypedDict

import yaml
import yaml.constructor
import yaml.resolver
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "tables"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "table_patches"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# The dataset's daily table is a fixed month x day grid: 12 month columns and
# 31 day rows of equal-sized cells.
DAILY_CLASS_NAME = "daily"
DAILY_COLS = 12
DAILY_ROWS = 31
EXPAND = 0.1

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"
GRID_FILENAME = "table_grid.json"
OVERLAY_FILENAME = "grid_overlay.jpg"

# QC render styling: red cell boundaries, a heavier blue outline for the
# hand-drawn daily box, and a quality high enough to judge alignment.
GRID_LINE_COLOUR = (255, 0, 0)
DAILY_BOX_COLOUR = (0, 102, 255)
GRID_LINE_WIDTH = 2
DAILY_BOX_WIDTH = 4
OVERLAY_JPEG_QUALITY = 90

logger = logging.getLogger(__name__)


class Box(TypedDict):
    """One table-region box in normalised 0-1 page coordinates."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float


class GridLine(TypedDict):
    """One kept grid line of a ``table_grid.json`` axis."""

    index: int  # grid line index k (the cell's row/col)
    position: float  # line position in page pixels


class GridImage(TypedDict):
    """Page image identity in ``table_grid.json``: the geometry's coordinate space."""

    path: str  # repo-relative POSIX path of the page scan
    width_px: float
    height_px: float


class GridCell(TypedDict):
    """One exported patch in ``table_grid.json``."""

    row: int
    col: int
    file: str  # patch filename, "<row>_<col>.png"
    rect: list[int]  # [left, upper, right, lower] padded crop rectangle


class GridMetadata(TypedDict):
    """Schema of a ``<out>/qc/<source-id>/table_grid.json`` file.

    The same document as the detector pipeline's ``GridMetadata`` in
    ``scripts/preprocess_pages.py``. All coordinates are page pixels; ``cells``
    is in export order (row-major, matching ``patches/<source-id>/``).
    ``phase`` is the first nominal cell centre so the kept line indices are the
    0-based cell indices.
    """

    source_id: str
    image: GridImage
    spacing_x: float
    spacing_y: float
    phase_x: float
    phase_y: float
    rms_x: float
    rms_y: float
    columns: list[GridLine]
    rows: list[GridLine]
    cells: list[GridCell]


class NominalCell(NamedTuple):
    """One daily-grid cell before padding, in page pixels."""

    row: int
    col: int
    x1: float
    y1: float
    x2: float
    y2: float


class CellPatch(NamedTuple):
    """One exported patch: its cell, the padded crop rectangle and written file."""

    row: int
    col: int
    rect: tuple[int, int, int, int]  # [left, upper, right, lower] in page pixels
    path: Path


class PageSummary(NamedTuple):
    """Patch count of one processed page."""

    source_id: str
    n_patches: int


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
    for noisy in ("PIL",):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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
    ``scripts/recombine_labels.py:load_classes``.

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
        lies outside the repository (e.g. a ``--dataset`` directory elsewhere).
    """
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved.as_posix()
    return relative.as_posix()


def label_path_for(image: Path, dataset_dir: Path) -> Path:
    """Map one page image to its mirrored label file.

    The dataset's ``images`` tree is mirrored onto its ``labels`` tree with the
    suffix changed to ``.txt``, matching the labeler and
    ``scripts/prepare_dataset.py``.

    Args:
        image: Page image under ``<dataset>/images``.
        dataset_dir: Dataset root containing ``images`` and ``labels``.

    Returns:
        The mirrored ``<dataset>/labels/<...>.txt`` path (it may not exist).
    """
    relative = image.relative_to(dataset_dir / "images")
    return (dataset_dir / "labels" / relative).with_suffix(".txt")


def scan_pages(dataset_dir: Path) -> list[Path]:
    """Collect the page images of a table dataset and validate their labels.

    Every image under ``<dataset>/images`` is a page; nothing is excluded.
    Pages are sorted by path so batch runs are deterministic. A page without a
    label file is returned too (the caller skips it); an orphan label file is a
    curation inconsistency and rejects the whole scan.

    Args:
        dataset_dir: Dataset root containing ``images`` and ``labels``.

    Returns:
        Page image paths sorted by path.

    Raises:
        ValueError: If the ``images`` or ``labels`` tree is missing, holds no
            page images, two pages share a stem (they would map to the same
            ``<source-id>`` output folder), or a label file mirrors no image.
    """
    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"
    if not images_root.is_dir():
        raise ValueError(f"images directory not found: {images_root}")
    if not labels_root.is_dir():
        raise ValueError(f"labels directory not found: {labels_root}")

    pages = sorted(
        p
        for p in images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not pages:
        raise ValueError(f"no page images found under {images_root}")

    seen: dict[str, Path] = {}
    for page in pages:
        if page.stem in seen:
            raise ValueError(
                f"duplicate <source-id> {page.stem!r}: {seen[page.stem]} and {page}"
            )
        seen[page.stem] = page

    expected = {page.relative_to(images_root).with_suffix(".txt") for page in pages}
    orphans = [
        path
        for path in sorted(labels_root.rglob("*.txt"))
        if path.relative_to(labels_root) not in expected
    ]
    if orphans:
        raise ValueError(
            "label files without a matching image: "
            + ", ".join(repo_relative(p) for p in orphans)
        )
    return pages


# ---------------------------------------------------------------------------
# label parsing


def parse_label_file(path: Path, n_classes: int) -> list[Box]:
    """Strictly parse one five-column YOLO label file.

    Each line is ``<class> <cx> <cy> <w> <h>`` in normalised 0-1
    coordinates; blank lines and ``#`` comments are skipped. Parsing is
    strict: a malformed line is reported, never skipped. Same contract as
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


def find_daily_box(boxes: list[Box], daily_cls: int) -> Box:
    """Return the page's single ``daily`` table box.

    Args:
        boxes: The page's parsed label boxes.
        daily_cls: The ``daily`` class id from ``data.yaml``.

    Returns:
        The page's ``daily`` box.

    Raises:
        ValueError: If the page has no ``daily`` box (or the label file is
            empty), or more than one (the daily grid must be unambiguous).
    """
    daily = [box for box in boxes if box["cls"] == daily_cls]
    if not daily:
        raise ValueError(f"no {DAILY_CLASS_NAME} box in label file")
    if len(daily) > 1:
        raise ValueError(
            f"{len(daily)} {DAILY_CLASS_NAME} boxes in label file, expected exactly one"
        )
    return daily[0]


# ---------------------------------------------------------------------------
# cell geometry and cropping


def daily_box_to_pixels(
    box: Box, width_px: float, height_px: float
) -> tuple[float, float, float, float]:
    """Convert a normalised ``daily`` box to page-pixel bounds.

    Args:
        box: The page's ``daily`` box in normalised 0-1 page coordinates.
        width_px: Page image width in pixels.
        height_px: Page image height in pixels.

    Returns:
        ``(x1, y1, x2, y2)`` of the box in page pixels.

    Raises:
        ValueError: If the resulting box has no positive width or height.
    """
    x1 = (box["cx"] - box["w"] / 2.0) * width_px
    y1 = (box["cy"] - box["h"] / 2.0) * height_px
    x2 = (box["cx"] + box["w"] / 2.0) * width_px
    y2 = (box["cy"] + box["h"] / 2.0) * height_px
    if not (x2 > x1 and y2 > y1):
        raise ValueError(f"degenerate {DAILY_CLASS_NAME} box: {box}")
    return x1, y1, x2, y2


def build_nominal_cells(
    bounds: tuple[float, float, float, float],
) -> list[NominalCell]:
    """Divide the daily table bounds into the 12 x 31 grid of equal cells.

    Cells are emitted row-major: row 0 left to right across columns 0-11, then
    row 1, and so on. Row 0 is the top (day 1) and column 0 the left (January).

    Args:
        bounds: ``(x1, y1, x2, y2)`` of the ``daily`` box in page pixels.

    Returns:
        One nominal (unpadded) cell per grid position, in row-major order.

    Raises:
        ValueError: If the bounds are degenerate or the nominal cell size is
            not positive.
    """
    x1, y1, x2, y2 = bounds
    cell_w = (x2 - x1) / DAILY_COLS
    cell_h = (y2 - y1) / DAILY_ROWS
    if not (cell_w > 0.0 and cell_h > 0.0):
        raise ValueError(f"degenerate daily cell size: {cell_w} x {cell_h} px")
    return [
        NominalCell(
            row=row,
            col=col,
            x1=x1 + col * cell_w,
            y1=y1 + row * cell_h,
            x2=x1 + (col + 1) * cell_w,
            y2=y1 + (row + 1) * cell_h,
        )
        for row in range(DAILY_ROWS)
        for col in range(DAILY_COLS)
    ]


def export_cell_patches(
    image_path: Path,
    cells: list[NominalCell],
    out_dir: Path,
    expand: float = EXPAND,
) -> list[CellPatch]:
    """Crop and save one image patch for every nominal cell.

    Each cell is expanded by ``expand`` of its width/height on every side,
    rounded outward, clipped to the page, cropped and written to
    ``out_dir / f"{row}_{col}.png"``. This is the padding contract of
    ``scripts/preprocess_pages.py:export_cell_patches``.

    Args:
        image_path: Page image to crop from.
        cells: Nominal cells from ``build_nominal_cells``.
        out_dir: Destination directory, created as needed.
        expand: Relative padding added on every side of a cell.

    Returns:
        One ``CellPatch`` per written file, in ``cells`` order, carrying the
        exact padded crop rectangle used.

    Raises:
        ValueError: If ``expand`` is negative or clipping collapses a cell
            rectangle.
        OSError: If the page image cannot be read or a patch cannot be written.
    """
    if expand < 0.0:
        raise ValueError("expand must be non-negative")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[CellPatch] = []
    with Image.open(image_path) as img:
        for cell in cells:
            pad_x = expand * (cell.x2 - cell.x1)
            pad_y = expand * (cell.y2 - cell.y1)
            left = max(math.floor(cell.x1 - pad_x), 0)
            upper = max(math.floor(cell.y1 - pad_y), 0)
            right = min(math.ceil(cell.x2 + pad_x), img.width)
            lower = min(math.ceil(cell.y2 + pad_y), img.height)
            if not (right > left and lower > upper):
                raise ValueError(
                    f"cell ({cell.row}, {cell.col}) is outside the page: "
                    f"rect {[left, upper, right, lower]}"
                )
            patch: Image.Image = img.crop((left, upper, right, lower))
            out_path = out_dir / f"{cell.row}_{cell.col}.png"
            patch.save(out_path)
            written.append(
                CellPatch(
                    row=cell.row,
                    col=cell.col,
                    rect=(left, upper, right, lower),
                    path=out_path,
                )
            )
    return written


def build_grid_metadata(
    source_id: str,
    image_path: Path,
    width_px: float,
    height_px: float,
    bounds: tuple[float, float, float, float],
    patches: list[CellPatch],
) -> GridMetadata:
    """Assemble one page's daily-grid geometry in the ``GridMetadata`` schema.

    Grid line positions are the nominal cell centres: ``columns[c]`` sits at
    ``x1 + (c + 0.5) * spacing_x`` and ``rows[r]`` at
    ``y1 + (r + 0.5) * spacing_y``. The phase is the first centre, so the kept
    indices are exactly the 0-based cell indices and
    ``scripts/recombine_labels.py:nominal_cell_rect`` recovers each unpadded
    cell from ``spacing`` and the line positions.

    Args:
        source_id: The page's ``<source-id>`` (its image file stem).
        image_path: Page image the patches were cut from.
        width_px: Page image width in pixels.
        height_px: Page image height in pixels.
        bounds: ``(x1, y1, x2, y2)`` of the ``daily`` box in page pixels.
        patches: Exported patch records from ``export_cell_patches``.

    Returns:
        The page's grid geometry: plain Python numbers throughout
        (JSON-serialisable), ``cells`` in ``patches`` order (row-major).
    """
    x1, y1, x2, y2 = bounds
    spacing_x = (x2 - x1) / DAILY_COLS
    spacing_y = (y2 - y1) / DAILY_ROWS
    columns = [
        GridLine(index=col, position=x1 + (col + 0.5) * spacing_x)
        for col in range(DAILY_COLS)
    ]
    rows = [
        GridLine(index=row, position=y1 + (row + 0.5) * spacing_y)
        for row in range(DAILY_ROWS)
    ]
    return GridMetadata(
        source_id=source_id,
        image=GridImage(
            path=repo_relative(image_path),
            width_px=width_px,
            height_px=height_px,
        ),
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        phase_x=columns[0]["position"],
        phase_y=rows[0]["position"],
        rms_x=0.0,
        rms_y=0.0,
        columns=columns,
        rows=rows,
        cells=[
            GridCell(
                row=patch.row,
                col=patch.col,
                file=patch.path.name,
                rect=list(patch.rect),
            )
            for patch in patches
        ],
    )


def write_grid_metadata(metadata: GridMetadata, path: Path) -> None:
    """Write ``metadata`` to ``path`` as pretty-printed JSON with a trailing newline.

    Args:
        metadata: Page grid geometry in the ``GridMetadata`` schema.
        path: Destination file; the parent directory is created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# QC renders


def clip_pixel(value: float, limit: int) -> int:
    """Round a page coordinate to the nearest drawable pixel.

    Page coordinates can fall just outside the bitmap (a hand-drawn box that
    touches or slightly overshoots the page edge), and JPEG drawing must stay
    inside the canvas, so the rounded value is clamped to
    ``0 .. limit - 1``.

    Args:
        value: Coordinate in page pixels.
        limit: Canvas extent in pixels along this axis.

    Returns:
        The nearest pixel index, clamped to the canvas.
    """
    return min(max(round(value), 0), limit - 1)


def render_grid_overlay(
    image_path: Path,
    bounds: tuple[float, float, float, float],
    out_path: Path,
) -> None:
    """Draw the daily grid's cell boundaries on the page (``grid_overlay.jpg``).

    The page is rendered at native resolution with one red line per cell
    boundary of the 12 x 31 grid -- the 11 internal vertical and 30 internal
    horizontal dividers plus the outer edge -- and the hand-drawn ``daily``
    box outlined in blue. Lines are placed at the same pixel coordinates the
    patches are cut from, so the render shows exactly the division in use and
    lets the box be checked against the printed table by eye.

    Args:
        image_path: Page image to render.
        bounds: ``(x1, y1, x2, y2)`` of the ``daily`` box in page pixels.
        out_path: Destination ``.jpg`` path; the parent directory is created.

    Raises:
        OSError: If the page image cannot be read or the overlay cannot be
            written.
    """
    x1, y1, x2, y2 = bounds
    cell_w = (x2 - x1) / DAILY_COLS
    cell_h = (y2 - y1) / DAILY_ROWS
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        overlay = img.convert("RGB")
    width, height = overlay.size
    draw = ImageDraw.Draw(overlay)

    # Vertical dividers span the daily box's height, horizontal ones its width,
    # so the lines describe the table region rather than the whole page.
    for col in range(DAILY_COLS + 1):
        x = clip_pixel(x1 + col * cell_w, width)
        draw.line(
            [(x, clip_pixel(y1, height)), (x, clip_pixel(y2, height))],
            fill=GRID_LINE_COLOUR,
            width=GRID_LINE_WIDTH,
        )
    for row in range(DAILY_ROWS + 1):
        y = clip_pixel(y1 + row * cell_h, height)
        draw.line(
            [(clip_pixel(x1, width), y), (clip_pixel(x2, width), y)],
            fill=GRID_LINE_COLOUR,
            width=GRID_LINE_WIDTH,
        )

    draw.rectangle(
        (
            clip_pixel(x1, width),
            clip_pixel(y1, height),
            clip_pixel(x2, width),
            clip_pixel(y2, height),
        ),
        outline=DAILY_BOX_COLOUR,
        width=DAILY_BOX_WIDTH,
    )
    overlay.save(out_path, format="JPEG", quality=OVERLAY_JPEG_QUALITY)


# ---------------------------------------------------------------------------
# pipeline


def process_page(
    image_path: Path,
    label_path: Path,
    patches_root: Path,
    qc_root: Path,
    n_classes: int,
    daily_cls: int,
) -> PageSummary:
    """Cut one page's daily-table cells and write its grid geometry.

    Writes ``<patches>/<source-id>/<row>_<col>.png``,
    ``<qc>/<source-id>/table_grid.json`` and ``<qc>/<source-id>/grid_overlay.jpg``;
    the geometry is written after the patches, so it always describes the patch
    set on disk, and the QC render is written last, so its presence means the
    page completed.

    Args:
        image_path: Page image to process.
        label_path: The page's mirrored table-label file.
        patches_root: Root of the patches tree.
        qc_root: Root of the qc tree.
        n_classes: Size of the dataset's class set (``data.yaml``).
        daily_cls: The ``daily`` class id from ``data.yaml``.

    Returns:
        The page's ``PageSummary``.

    Raises:
        ValueError: If the label file is malformed, has no single ``daily``
            box, or the daily box is degenerate.
        OSError: If the page image cannot be read or an artifact cannot be
            written.
    """
    source_id = image_path.stem
    logger.debug("processing %s", repo_relative(image_path))

    boxes = parse_label_file(label_path, n_classes)
    daily = find_daily_box(boxes, daily_cls)

    with Image.open(image_path) as img:
        width_px = float(img.width)
        height_px = float(img.height)
    logger.debug("%s: page %.0f x %.0f px", source_id, width_px, height_px)
    bounds = daily_box_to_pixels(daily, width_px, height_px)
    cells = build_nominal_cells(bounds)
    logger.debug(
        "%s: daily box %.1f x %.1f px, %d cells of %.1f x %.1f px",
        source_id,
        bounds[2] - bounds[0],
        bounds[3] - bounds[1],
        len(cells),
        (bounds[2] - bounds[0]) / DAILY_COLS,
        (bounds[3] - bounds[1]) / DAILY_ROWS,
    )
    patches = export_cell_patches(image_path, cells, patches_root / source_id)
    logger.debug(
        "%s: wrote %d patches to %s",
        source_id,
        len(patches),
        repo_relative(patches_root / source_id),
    )
    metadata = build_grid_metadata(
        source_id, image_path, width_px, height_px, bounds, patches
    )

    grid_path = qc_root / source_id / GRID_FILENAME
    write_grid_metadata(metadata, grid_path)
    logger.debug("%s: wrote %s", source_id, repo_relative(grid_path))

    overlay_path = qc_root / source_id / OVERLAY_FILENAME
    render_grid_overlay(image_path, bounds, overlay_path)
    logger.debug("%s: wrote %s", source_id, repo_relative(overlay_path))
    return PageSummary(source_id=source_id, n_patches=len(patches))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the table-patch pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Cut weather-report pages into daily-table cell patches using the "
            "dataset's ground-truth table labels."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="table dataset root with images/ and labels/ (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output root holding patches/ and qc/ (default: %(default)s)",
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
    """Process every labelled page under ``--dataset`` and print the run summary.

    Logging is configured first from ``--log-level`` (stderr plus the appended
    ``<out>/preprocess.log``); the run summary still goes to stdout. Pages that
    fail with an expected error are logged and skipped; the run summary lists
    the failed pages and the run exits 1 if any page failed. Unexpected errors
    abort the run immediately so bugs surface.

    Raises:
        SystemExit: If the arguments are invalid, the class set or dataset
            cannot be loaded, the ``daily`` class is missing, or any page
            failed.
    """
    args = parse_args()
    dataset_dir: Path = args.dataset.resolve()
    out_dir: Path = args.out.resolve()
    setup_logging(args.log_level, out_dir / "preprocess.log")

    try:
        classes = load_classes(dataset_dir / "data.yaml")
    except ValueError as exc:
        logger.error("cannot load classes: %s", exc)
        raise SystemExit(f"cannot load classes: {exc}") from exc
    if DAILY_CLASS_NAME not in classes:
        logger.error(
            "%s has no %r class (classes: %s)",
            repo_relative(dataset_dir / "data.yaml"),
            DAILY_CLASS_NAME,
            ", ".join(classes),
        )
        raise SystemExit(
            f"{dataset_dir / 'data.yaml'} has no {DAILY_CLASS_NAME!r} class"
        )
    daily_cls = classes.index(DAILY_CLASS_NAME)
    logger.info(
        "%d classes from %s (daily = %d)",
        len(classes),
        repo_relative(dataset_dir / "data.yaml"),
        daily_cls,
    )

    try:
        pages = scan_pages(dataset_dir)
    except ValueError as exc:
        logger.error("cannot scan %s: %s", repo_relative(dataset_dir), exc)
        raise SystemExit(f"cannot scan {dataset_dir}: {exc}") from exc

    patches_root = out_dir / "patches"
    qc_root = out_dir / "qc"
    logger.info(
        "writing %s and %s",
        repo_relative(patches_root),
        repo_relative(qc_root),
    )

    total_patches = 0
    n_unlabeled = 0
    failed: list[tuple[str, str]] = []
    for page in pages:
        label_path = label_path_for(page, dataset_dir)
        if not label_path.is_file():
            n_unlabeled += 1
            logger.debug("%s: no label file, skipped", page.stem)
            continue
        try:
            summary = process_page(
                page,
                label_path,
                patches_root,
                qc_root,
                len(classes),
                daily_cls,
            )
        except (ValueError, OSError) as exc:
            logger.warning("page %s failed: %s", repo_relative(page), exc)
            failed.append((repo_relative(page), str(exc)))
            continue
        total_patches += summary.n_patches
        logger.info("%s: %d patches", summary.source_id, summary.n_patches)

    run_summary = (
        f"{len(pages)} pages ({n_unlabeled} unlabeled skipped) / "
        f"{total_patches} patches -> {patches_root} ({len(failed)} failed)"
    )
    logger.info(run_summary)
    print(run_summary)
    for label, reason in failed:
        print(f"  {label}: {reason}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
