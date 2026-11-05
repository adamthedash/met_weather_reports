#!/usr/bin/env python3
"""Detect the table cells of weather-report pages and export one crop per cell.

Usage:

    pixi run python scripts/preprocess_pages.py

For every page image under ``--data`` (by default ``data/images``, the image
folder written by ``scripts/scrape_zooniverse.py``) the pipeline runs
PaddleOCR table cell detection, reconstructs the page's cell grid (spacing via
an FFT over the detected box centres, phase via a circular mean) and crops one
patch per grid cell to ``patches/<source-id>/<row>_<col>.png`` -- the naming
and geometry expected by ``datasets/digits/images`` (see
``tools/labeler/README.md``). ``<source-id>`` is the page image's file stem,
used as-is: the subject id for scraped pages (``103801209``, or
``103801209_2`` for a subject's second page), so the ``patches/`` and ``qc/``
folders are organised by subject id.

Per-page intermediates, results and QC renders go to ``qc/<source-id>/``:

- ``result.json`` -- the raw PaddleOCR detection result
- ``paddle.jpg`` -- raw image with the PaddleOCR detections overlaid
- ``cell_dist.jpg`` -- cell width/height distributions
- ``grid_phase.jpg`` -- FFT power spectra of the box centres
- ``inferred.jpg`` -- raw image with the inferred grid boxes overlaid
- ``grid.json`` -- the fitted table grid and every patch's crop rectangle

``grid.json`` is the machine-readable result of the table fitting: grid
spacing and phase per axis, the kept grid lines, and -- per exported patch --
the exact page-space rectangle it was cut from. It is what later reassembly
uses to map per-patch inference results back onto the page. It is written
together with the patches, so a page rejected before cropping (or one whose
export failed) has no ``grid.json`` either.

Pages are processed one at a time. A page that cannot be processed with an
expected error (no usable cells, an unreadable image, an unwritable artifact)
is logged and skipped, and the run continues with the next page; failed pages
are listed in the run summary and make the run exit non-zero. Unexpected
errors abort the run immediately so bugs surface. As a sanity check on the
reconstruction, a page whose estimated grid is smaller than ``MIN_GRID_COLS``
columns or ``MIN_GRID_ROWS`` rows is rejected before any patch is cropped: a
register page is a day x month grid, so a smaller grid means the
reconstruction went wrong. Artifacts already written for a page, including
those of a failed one, stay on disk. The pipeline is read-only towards its
input and never writes into ``datasets/``; moving ``patches/<source-id>/``
into ``datasets/digits/images/`` stays a manual step.

Run activity is logged to stderr and appended to ``<out>/preprocess.log``:
initialisation (model loading, page scan), every per-page pipeline step and
written artifact at DEBUG, one line per completed page and one warning per
failed page. ``--log-level`` narrows the verbosity (``DEBUG``, the default,
shows everything); the run summary on stdout is not affected.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import NamedTuple, TypedDict

import matplotlib

matplotlib.use("Agg")  # the pipeline renders QC JPGs headlessly

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from paddleocr import TableCellsDetection
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data/images"
DEFAULT_OUT_DIR = REPO_ROOT / "data"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

MODEL_NAME = "RT-DETR-L_wireless_table_cell_det"
SIZE_TOL = 0.2
EXPAND = 0.1
MIN_PERIOD_FACTOR = 0.7
MAX_PERIOD_FACTOR = 3.0
# A register page is a day x month grid, so a plausible reconstruction has at
# least 12 columns (Jan-Dec) and 31 rows (days).
MIN_GRID_COLS = 12
MIN_GRID_ROWS = 31

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"

logger = logging.getLogger(__name__)


class Box(TypedDict):
    """A single detected object (e.g. a table cell) from PaddleOCR."""

    cls_id: int
    label: str
    score: float
    coordinate: list[float]  # [x1, y1, x2, y2]


class PaddleOcrResult(TypedDict):
    """Schema of a PaddleOCR result file (e.g. ``qc/<source-id>/result.json``)."""

    input_path: str
    page_index: int | None
    boxes: list[Box]


class GridLine(TypedDict):
    """One kept grid line of a fitted axis in ``grid.json``."""

    index: int  # grid line index k
    position: float  # line position in page pixels


class GridImage(TypedDict):
    """Page image identity in ``grid.json``: the coordinate space of the geometry."""

    path: str  # repo-relative POSIX path, as in PaddleOcrResult.input_path
    width_px: float
    height_px: float


class GridCell(TypedDict):
    """One exported patch in ``grid.json``."""

    row: int
    col: int
    file: str  # patch filename, "<row>_<col>.png"
    rect: list[int]  # [left, upper, right, lower] crop rectangle in page pixels


class GridMetadata(TypedDict):
    """Schema of a ``qc/<source-id>/grid.json`` table-fitting result file.

    All coordinates are page pixels and ``cells`` is in the export order
    (sorted by column then row), matching ``patches/<source-id>/`` on disk.
    Together with the patch filenames this is everything reassembly needs to
    map per-patch inference results back onto the page.
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


class CellPatch(NamedTuple):
    """One exported patch: its grid cell, crop rectangle and written file."""

    row: int
    col: int
    rect: tuple[int, int, int, int]  # [left, upper, right, lower] in page pixels
    path: Path


class AxisFit(NamedTuple):
    """The fitted grid geometry of one axis, as produced by the table fitting."""

    spacing: float  # grid spacing in pixels
    phase: float  # grid phase in pixels
    rms: float  # residual RMS in pixels
    indices: np.ndarray  # kept grid line indices
    lines: np.ndarray  # kept grid line positions in pixels


class PageSummary(NamedTuple):
    """Grid geometry and patch count of one processed page."""

    source_id: str
    n_rows: int
    n_cols: int
    spacing_x: float
    spacing_y: float
    rms_x: float
    rms_y: float
    n_patches: int


# ---------------------------------------------------------------------------
# logging


def setup_logging(level_name: str, log_file: Path) -> None:
    """Configure run logging: stderr plus an appended ``log_file``.

    Both handlers receive identical records with UTC timestamps; the log file
    is appended to, so reruns extend the same trail. The ``matplotlib``,
    ``ppocr``, ``paddlex`` and ``onnxruntime`` loggers are quieted to WARNING
    so font-matching and model-stack chatter do not drown the run's own
    records.

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
    for noisy in ("matplotlib", "ppocr", "paddlex", "onnxruntime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# cell detection


def repo_relative(path: Path) -> str:
    """Render ``path`` as a POSIX path relative to the repository root.

    Args:
        path: Any filesystem path.

    Returns:
        The repo-relative POSIX path, or the absolute POSIX path when ``path``
        lies outside the repository (e.g. a ``--data`` directory elsewhere).
    """
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved.as_posix()
    return relative.as_posix()


def detect_cells(
    model: TableCellsDetection, image_path: Path, threshold: float
) -> PaddleOcrResult:
    """Run table cell detection on one page image.

    Args:
        model: The shared ``TableCellsDetection`` model instance.
        image_path: Page image to detect on, read from disk as-is.
        threshold: Minimum confidence for a returned box.

    Returns:
        The page's detections in the ``PaddleOcrResult`` schema: ``input_path``
        relative to the repository root, plain Python numbers throughout
        (JSON-serialisable).

    Raises:
        ValueError: If the model returns anything other than exactly one
            result for the page, or a result / box does not have the expected
            shape.
    """
    outputs = list(model.predict(str(image_path), threshold=threshold, batch_size=1))
    if len(outputs) != 1:
        raise ValueError(f"expected 1 detection result, got {len(outputs)}")

    payload = outputs[0].json
    raw = payload.get("res") if isinstance(payload, dict) else None
    if not isinstance(raw, dict) or not isinstance(raw.get("boxes"), list):
        shape = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"unexpected detection result shape: {shape}")

    boxes: list[Box] = []
    for i, raw_box in enumerate(raw["boxes"]):
        try:
            boxes.append(
                Box(
                    cls_id=int(raw_box["cls_id"]),
                    label=str(raw_box["label"]),
                    score=float(raw_box["score"]),
                    coordinate=[float(v) for v in raw_box["coordinate"]],
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"detection box {i} is malformed: {raw_box}") from exc

    page_index = raw.get("page_index")
    return PaddleOcrResult(
        input_path=repo_relative(image_path),
        page_index=int(page_index) if page_index is not None else None,
        boxes=boxes,
    )


def write_result_json(result: PaddleOcrResult, path: Path) -> None:
    """Write ``result`` to ``path`` as pretty-printed JSON with a trailing newline.

    Args:
        result: Page detection result in the ``PaddleOcrResult`` schema.
        path: Destination file; the parent directory is created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def load_page_boxes(result: PaddleOcrResult) -> pl.DataFrame:
    """Build the per-box geometry frame for one page's detections.

    One row per detected box with its corner coordinates (``[x1, y1, x2, y2]``),
    width, height and aspect ratio.

    Args:
        result: The page's ``PaddleOcrResult``.

    Returns:
        DataFrame with columns ``x1``, ``y1``, ``x2``, ``y2``, ``box_width``,
        ``box_height`` and ``aspect_ratio``.

    Raises:
        ValueError: If the page has no boxes at all or a box is not a valid
            ``[x1, y1, x2, y2]`` rectangle.
    """
    records: list[dict[str, float]] = []
    for i, box in enumerate(result["boxes"]):
        coordinate = box["coordinate"]
        if len(coordinate) != 4:
            raise ValueError(
                f"box {i} coordinate is not [x1, y1, x2, y2]: {coordinate}"
            )
        x1, y1, x2, y2 = coordinate
        if not (x2 > x1 and y2 > y1):
            raise ValueError(f"box {i} is not [x1, y1, x2, y2]: {coordinate}")
        records.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box_width": x2 - x1,
                "box_height": y2 - y1,
            }
        )

    if not records:
        raise ValueError("no boxes detected")

    boxes = pl.DataFrame(
        records,
        schema={
            "x1": pl.Float64,
            "y1": pl.Float64,
            "x2": pl.Float64,
            "y2": pl.Float64,
            "box_width": pl.Float64,
            "box_height": pl.Float64,
        },
    )
    return boxes.with_columns(
        (pl.col("box_width") / pl.col("box_height")).alias("aspect_ratio")
    )


def median_box_size(boxes: pl.DataFrame) -> tuple[float, float]:
    """Median detected box width and height of one page.

    Args:
        boxes: Per-box frame from ``load_page_boxes``.

    Returns:
        ``(median_width, median_height)`` in pixels.
    """
    widths = boxes["box_width"].to_numpy()
    heights = boxes["box_height"].to_numpy()
    return float(np.median(widths)), float(np.median(heights))


def filter_boxes_by_median_size(
    boxes: pl.DataFrame, med_w: float, med_h: float, tol: float = SIZE_TOL
) -> pl.DataFrame:
    """Keep the boxes that plausibly belong to the uniform cell grid.

    A box is kept when its width and its height both lie within ``tol`` of the
    page's median box size (the grid cells of a printed table are uniform;
    headers and spurious detections are not).

    Args:
        boxes: Per-box frame from ``load_page_boxes``.
        med_w: Median box width in pixels.
        med_h: Median box height in pixels.
        tol: Relative tolerance around the median size (0.2 = +/-20%).

    Returns:
        The kept rows of ``boxes``.

    Raises:
        ValueError: If ``tol`` is negative or no box survives the filter.
    """
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    kept = boxes.filter(
        pl.col("box_width").is_between(med_w * (1.0 - tol), med_w * (1.0 + tol)),
        pl.col("box_height").is_between(med_h * (1.0 - tol), med_h * (1.0 + tol)),
    )
    if kept.height == 0:
        raise ValueError(
            f"no boxes within median size +/-{tol:.0%} of {med_w:.1f} x {med_h:.1f} px"
        )
    return kept


# ---------------------------------------------------------------------------
# grid reconstruction


def estimate_spacing_fft(
    positions: np.ndarray,
    min_period: float,
    max_period: float,
    bin_size: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate the dominant spacing of 1-D point positions using an FFT.

    The positions are rasterised into a per-``bin_size`` occupancy signal,
    mean-subtracted and Hann-windowed. The spacing is the period of the
    strongest spectral peak whose period lies within
    ``[min_period, max_period]`` (this band keeps harmonics and
    sub-harmonics out of the running). The peak frequency is refined
    between FFT bins with parabolic interpolation.

    Args:
        positions: Point positions along one axis (box centres).
        min_period: Smallest period considered, in pixels.
        max_period: Largest period considered, in pixels.
        bin_size: Occupancy histogram bin width in pixels.

    Returns:
        ``(spacing, frequencies, power)`` on the FFT frequency grid.

    Raises:
        ValueError: If fewer than two positions are given, the arguments do
            not satisfy ``0 < bin_size < min_period < max_period``, or no FFT
            bin falls inside the period band.
    """
    pos: np.ndarray = np.asarray(positions, dtype=float)
    if pos.size < 2:
        raise ValueError("Need at least two positions to estimate a spacing")
    if not (0.0 < bin_size < min_period < max_period):
        raise ValueError("Need 0 < bin_size < min_period < max_period")

    edges: np.ndarray = np.arange(pos.min(), pos.max() + bin_size, bin_size)
    occupancy: np.ndarray = np.histogram(pos, bins=edges)[0].astype(float)
    signal: np.ndarray = occupancy - occupancy.mean()
    windowed: np.ndarray = signal * np.hanning(signal.size)

    power: np.ndarray = np.abs(np.fft.rfft(windowed))
    freqs: np.ndarray = np.fft.rfftfreq(signal.size, d=bin_size)

    periods: np.ndarray = np.full_like(freqs, np.inf)
    np.divide(1.0, freqs, out=periods, where=freqs > 0)
    band: np.ndarray = (periods >= min_period) & (periods <= max_period)
    if not band.any():
        raise ValueError("No FFT bin falls inside the period band")

    peak_idx: int = int(np.argmax(np.where(band, power, -np.inf)))
    delta: float = 0.0
    if 0 < peak_idx < power.size - 1:
        denom: float = float(
            power[peak_idx - 1] - 2.0 * power[peak_idx] + power[peak_idx + 1]
        )
        if denom != 0.0:
            delta = 0.5 * float(power[peak_idx - 1] - power[peak_idx + 1]) / denom
    peak_freq: float = float(freqs[peak_idx]) + delta * float(freqs[1] - freqs[0])

    return 1.0 / peak_freq, freqs, power


def fit_grid_axis(
    positions: np.ndarray, spacing: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a 1-D grid of given ``spacing`` to point positions.

    The phase is estimated as the circular mean of the positions taken
    modulo the spacing: every position is mapped to an angle
    ``theta = 2*pi*x/spacing`` on the unit circle, and the mean direction
    gives the grid offset. Each position is then assigned to its nearest
    grid line ``phase + k * spacing``.

    Args:
        positions: Point positions along one axis (box centres).
        spacing: Grid spacing along the axis, in pixels.

    Returns:
        ``(phase, indices, residuals)`` where ``phase`` lies in
        ``[0, spacing)``, ``indices`` are the grid line indices ``k``, and
        ``residuals`` are the deviations from the fitted grid lines.

    Raises:
        ValueError: If fewer than two positions are given or ``spacing`` is
            not positive.
    """
    pos: np.ndarray = np.asarray(positions, dtype=float)
    if pos.size < 2:
        raise ValueError("Need at least two positions to fit a grid")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")

    angles: np.ndarray = 2.0 * np.pi * pos / spacing
    mean_angle: float = float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))
    phase: float = mean_angle % (2.0 * np.pi) * spacing / (2.0 * np.pi)

    indices: np.ndarray = np.rint((pos - phase) / spacing).astype(int)
    residuals: np.ndarray = pos - (phase + indices * spacing)
    return phase, indices, residuals


def grid_line_support(
    line_positions: np.ndarray, box_starts: np.ndarray, box_ends: np.ndarray
) -> np.ndarray:
    """Mask of grid lines that overlap at least one real box extent.

    A line at position ``p`` overlaps a box when the box spans ``p``
    (``start <= p <= end``). Lines with no overlapping box lie outside the
    region covered by detected cells and are dropped, turning the
    infinitely repeating grid into finite bounds.

    Args:
        line_positions: Candidate grid line positions.
        box_starts: Per-box lower edge along the axis.
        box_ends: Per-box upper edge along the axis.

    Returns:
        Boolean mask over ``line_positions``.

    Raises:
        ValueError: If no grid lines are given or the box extents are empty or
            of mismatched length.
    """
    lines: np.ndarray = np.asarray(line_positions, dtype=float)
    starts: np.ndarray = np.asarray(box_starts, dtype=float)
    ends: np.ndarray = np.asarray(box_ends, dtype=float)
    if lines.size == 0:
        raise ValueError("No grid lines given")
    if starts.size == 0 or starts.size != ends.size:
        raise ValueError("Need matching, non-empty box extents")

    overlaps: np.ndarray = (starts[None, :] <= lines[:, None]) & (
        lines[:, None] <= ends[None, :]
    )
    return np.asarray(overlaps.any(axis=1))


def kept_grid_lines(
    spacing: float,
    phase: float,
    extent: float,
    box_starts: np.ndarray,
    box_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the grid lines of one axis that fall inside the detected table.

    Candidate lines ``phase + k * spacing`` are generated across
    ``[0, extent)`` and pruned with ``grid_line_support`` so only lines
    overlapping at least one detected box survive; these are the table bounds.

    Args:
        spacing: Grid spacing of the axis in pixels.
        phase: Grid phase of the axis in pixels (from ``fit_grid_axis``).
        extent: Image width (x axis) or height (y axis) in pixels.
        box_starts: Per-box lower edge along the axis.
        box_ends: Per-box upper edge along the axis.

    Returns:
        ``(indices, lines)`` of the supported grid lines, in increasing order.

    Raises:
        ValueError: If no candidate line is supported by a detected box.
    """
    indices: np.ndarray = np.arange(
        int(np.ceil(-phase / spacing)),
        int(np.floor((extent - phase) / spacing)) + 1,
    )
    candidates: np.ndarray = phase + indices * spacing
    keep: np.ndarray = grid_line_support(candidates, box_starts, box_ends)
    if not keep.any():
        raise ValueError("no grid line is supported by a detected box")
    return indices[keep], candidates[keep]


def build_grid_boxes(
    grid_x_lines: np.ndarray,
    grid_y_lines: np.ndarray,
    k_x: np.ndarray,
    k_y: np.ndarray,
    spacing_x: float,
    spacing_y: float,
) -> pl.DataFrame:
    """Build one cell box per kept grid intersection.

    Each cell is centred on an intersection of kept x/y grid lines and sized
    ``spacing_x x spacing_y``. ``col``/``row`` are the grid line indices, which
    also name the exported patch (``<row>_<col>.png``).

    Args:
        grid_x_lines: Supported grid line positions along x.
        grid_y_lines: Supported grid line positions along y.
        k_x: Grid line indices along x.
        k_y: Grid line indices along y.
        spacing_x: Grid spacing along x in pixels.
        spacing_y: Grid spacing along y in pixels.

    Returns:
        DataFrame with columns ``col``, ``cx``, ``row``, ``cy``, ``x1``, ``y1``,
        ``x2``, ``y2``, ``box_width`` and ``box_height``, sorted by column
        then row.
    """
    x_frame = pl.DataFrame({"col": k_x.astype(int), "cx": grid_x_lines})
    y_frame = pl.DataFrame({"row": k_y.astype(int), "cy": grid_y_lines})
    return (
        x_frame.join(y_frame, how="cross")
        .sort("col", "row")
        .with_columns(
            (pl.col("cx") - spacing_x / 2).alias("x1"),
            (pl.col("cy") - spacing_y / 2).alias("y1"),
            (pl.col("cx") + spacing_x / 2).alias("x2"),
            (pl.col("cy") + spacing_y / 2).alias("y2"),
        )
        .with_columns(
            (pl.col("x2") - pl.col("x1")).alias("box_width"),
            (pl.col("y2") - pl.col("y1")).alias("box_height"),
        )
    )


def export_cell_patches(
    boxes: pl.DataFrame, image_path: Path, out_dir: Path, expand: float = EXPAND
) -> list[CellPatch]:
    """Crop and save an image patch for every cell box.

    Each box is expanded by ``expand`` of its size on every side (10% per
    side grows the box by 20% in each dimension), clipped to the image
    bounds, cropped from ``image_path`` and written to
    ``out_dir / f"{row}_{col}.png"``.

    Args:
        boxes: Grid cell boxes from ``build_grid_boxes`` (needs ``row``,
            ``col``, ``x1``, ``y1``, ``x2``, ``y2``, ``box_width``,
            ``box_height``).
        image_path: Page image to crop from.
        out_dir: Destination directory, created as needed.
        expand: Relative padding added on every side of a box.

    Returns:
        One ``CellPatch`` per written file: the grid cell's ``row``/``col``,
        the exact crop rectangle used (``expand`` padding applied, rounded
        outward, clipped to the image bounds) and the written path, in
        ``boxes`` iteration order (sorted by column then row).

    Raises:
        ValueError: If ``expand`` is negative.
    """
    if expand < 0.0:
        raise ValueError("expand must be non-negative")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[CellPatch] = []
    with Image.open(image_path) as img:
        for cell in boxes.iter_rows(named=True):
            pad_x: float = expand * float(cell["box_width"])
            pad_y: float = expand * float(cell["box_height"])
            left: int = max(int(np.floor(cell["x1"] - pad_x)), 0)
            upper: int = max(int(np.floor(cell["y1"] - pad_y)), 0)
            right: int = min(int(np.ceil(cell["x2"] + pad_x)), img.width)
            lower: int = min(int(np.ceil(cell["y2"] + pad_y)), img.height)
            patch: Image.Image = img.crop((left, upper, right, lower))
            out_path: Path = out_dir / f"{cell['row']}_{cell['col']}.png"
            patch.save(out_path)
            written.append(
                CellPatch(
                    row=int(cell["row"]),
                    col=int(cell["col"]),
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
    fit_x: AxisFit,
    fit_y: AxisFit,
    patches: list[CellPatch],
) -> GridMetadata:
    """Assemble one page's table-fitting results in the ``GridMetadata`` schema.

    The metadata captures what the patch files alone cannot: where the fitted
    grid sits on the page and, per exported patch, the exact rectangle it was
    cropped from. Together with the patch filenames (``<row>_<col>.png``, the
    same names the inference outputs mirror) this is everything reassembly
    needs to map per-patch results back onto the page.

    Args:
        source_id: The page's ``<source-id>`` (its image file stem).
        image_path: Page image the grid was fitted on.
        width_px: Page image width in pixels.
        height_px: Page image height in pixels.
        fit_x: Fitted grid geometry of the x axis.
        fit_y: Fitted grid geometry of the y axis.
        patches: Exported patch records from ``export_cell_patches``, in
            column-then-row order.

    Returns:
        The page's ``GridMetadata``: plain Python numbers throughout
        (JSON-serialisable), ``cells`` in ``patches`` order.
    """
    return GridMetadata(
        source_id=source_id,
        image=GridImage(
            path=repo_relative(image_path),
            width_px=width_px,
            height_px=height_px,
        ),
        spacing_x=fit_x.spacing,
        spacing_y=fit_y.spacing,
        phase_x=fit_x.phase,
        phase_y=fit_y.phase,
        rms_x=fit_x.rms,
        rms_y=fit_y.rms,
        columns=[
            GridLine(index=int(index), position=float(line))
            for index, line in zip(fit_x.indices, fit_x.lines)
        ],
        rows=[
            GridLine(index=int(index), position=float(line))
            for index, line in zip(fit_y.indices, fit_y.lines)
        ],
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


def write_grid_json(metadata: GridMetadata, path: Path) -> None:
    """Write ``metadata`` to ``path`` as pretty-printed JSON with a trailing newline.

    Args:
        metadata: Page table-fitting results in the ``GridMetadata`` schema.
        path: Destination file; the parent directory is created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# QC renders


def save_fig(fig: plt.Figure, out_path: Path) -> None:
    """Save ``fig`` to ``out_path`` as JPG and close it to free its memory.

    Args:
        fig: The matplotlib figure to write.
        out_path: Destination ``.jpg`` path; parent directories are created.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="jpg", bbox_inches="tight")
    plt.close(fig)


def render_detection_overlay(
    image_path: Path, boxes: pl.DataFrame, out_path: Path
) -> None:
    """Overlay the raw PaddleOCR detections on the page (``paddle.jpg``).

    One red outline per detection, before any size filtering, so the detection
    stage can be judged on its own.

    Args:
        image_path: Page image the boxes were detected on.
        boxes: Per-box frame from ``load_page_boxes`` (all detections).
        out_path: Destination ``.jpg`` path.
    """
    img: np.ndarray = plt.imread(image_path)
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.imshow(img)
    for row in boxes.iter_rows(named=True):
        ax.add_patch(
            Rectangle(
                (row["x1"], row["y1"]),
                row["box_width"],
                row["box_height"],
                fill=False,
                edgecolor="red",
                linewidth=1.0,
            )
        )
    ax.set_title(f"{image_path.name} — {boxes.height} PaddleOCR detections")
    ax.set_xlim(0.0, img.shape[1])
    ax.set_ylim(img.shape[0], 0.0)
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, out_path)


def render_cell_distribution(
    widths: np.ndarray,
    heights: np.ndarray,
    med_w: float,
    med_h: float,
    source_id: str,
    out_path: Path,
) -> None:
    """Render the box width/height distribution of a page (``cell_dist.jpg``).

    The main heatmap is bracketed by per-axis histograms on the top and right,
    sharing their x/y limits with the main axes so the panels stay aligned.
    Median width and height are drawn as red dashed reference lines, and the
    legend/annotation box sits in the top-left of the main axes.

    Args:
        widths: Detected box widths in pixels (all detections).
        heights: Detected box heights in pixels (all detections).
        med_w: Median width reference line.
        med_h: Median height reference line.
        source_id: Page id, used in the title.
        out_path: Destination ``.jpg`` path.
    """
    x_max: float = max(float(np.percentile(widths, 99)), 1.0)
    y_max: float = max(float(np.percentile(heights, 99)), 1.0)
    x_edges: np.ndarray = np.linspace(0.0, x_max, 70)
    y_edges: np.ndarray = np.linspace(0.0, y_max, 70)

    fig: plt.Figure = plt.figure(figsize=(10, 8))
    gs: GridSpec = GridSpec(
        2,
        2,
        width_ratios=(4.2, 1.1),
        height_ratios=(1.1, 4.2),
        wspace=0.0,
        hspace=0.0,
        figure=fig,
    )
    # The sharex/sharey kwargs below keep the marginal axes locked to the main
    # axes' extents, so the panels remain pixel-aligned (identical gridspec
    # cells + shared axis groups).
    ax_main: plt.Axes = fig.add_subplot(gs[1, 0])
    ax_top: plt.Axes = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right: plt.Axes = fig.add_subplot(gs[1, 1], sharey=ax_main)

    _, _, _, im = ax_main.hist2d(
        widths,
        heights,
        bins=[x_edges, y_edges],
        norm=LogNorm(vmin=1),
        cmap="viridis",
    )
    ax_main.set_xlabel("Box width (pixels)")
    ax_main.set_ylabel("Box height (pixels)")
    ax_main.set_xlim(0.0, x_edges[-1])
    ax_main.set_ylim(0.0, y_edges[-1])
    ax_main.grid(alpha=0.2, linewidth=0.5)

    ax_main.axvline(med_w, color="red", ls="--", lw=1.2, alpha=0.9)
    ax_main.axhline(med_h, color="red", ls="--", lw=1.2, alpha=0.9)
    ax_top.axvline(med_w, color="red", ls="--", lw=1.2, alpha=0.9)
    ax_right.axhline(med_h, color="red", ls="--", lw=1.2, alpha=0.9)
    ax_main.text(
        0.02,
        0.98,
        f"median width = {med_w:.1f} px\nmedian height = {med_h:.1f} px\nn = {widths.size:,} cells",
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.75,
            edgecolor="gray",
        ),
    )

    ax_top.hist(
        widths,
        bins=x_edges.tolist(),
        color="steelblue",
        edgecolor="white",
        linewidth=0.3,
    )
    ax_top.set_ylabel("count")
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(alpha=0.2, linewidth=0.5)
    ax_top.set_title(
        f"PaddleOCR table cell boxes — width/height 2D histogram "
        f"({widths.size:,} cells, {source_id})"
    )

    ax_right.hist(
        heights,
        bins=y_edges.tolist(),
        orientation="horizontal",
        color="seagreen",
        edgecolor="white",
        linewidth=0.3,
    )
    ax_right.set_xlabel("count")
    ax_right.tick_params(labelleft=False)
    ax_right.grid(alpha=0.2, linewidth=0.5)

    save_fig(fig, out_path)


def render_spacing_spectra(
    spectra: dict[str, tuple[float, np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    """Render the FFT spectra of box-centre positions (``grid_phase.jpg``).

    ``spectra`` maps an axis name (``"x"`` / ``"y"``) to
    ``(spacing, frequencies, power)``. The horizontal axis shows the period
    so the detected spacing can be read off directly.

    Args:
        spectra: Per-axis FFT output from ``estimate_spacing_fft``.
        out_path: Destination ``.jpg`` path.
    """
    fig: plt.Figure
    axes: np.ndarray
    fig, axes = plt.subplots(
        1, len(spectra), figsize=(6.5 * len(spectra), 4), squeeze=False
    )
    # axes.flat (not axes): a (1, n) array yields one row per iteration, which
    # would zip away every axis after the first.
    for ax, (axis_name, (spacing, freqs, power)) in zip(axes.flat, spectra.items()):
        periods: np.ndarray = np.full_like(freqs, np.inf)
        np.divide(1.0, freqs, out=periods, where=freqs > 0)
        keep: np.ndarray = np.isfinite(periods) & (periods <= 4.0 * spacing)
        ax.plot(periods[keep], power[keep], lw=0.8, color="steelblue")
        ax.axvline(
            spacing,
            color="red",
            ls="--",
            lw=1.2,
            label=f"estimated spacing = {spacing:.1f} px",
        )
        ax.set_xlabel(f"Period along {axis_name} (pixels)")
        ax.set_ylabel("FFT power")
        ax.set_title(f"Box-centre positions — {axis_name} axis spectrum")
        ax.legend()
        ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    save_fig(fig, out_path)


def render_inferred_overlay(
    image_path: Path, generated: pl.DataFrame, out_path: Path
) -> None:
    """Overlay the inferred grid boxes on the page (``inferred.jpg``).

    Only the inferred boxes are drawn (the detections are covered by
    ``paddle.jpg``); this render answers whether the reconstructed grid
    aligns with the printed table.

    Args:
        image_path: Page image the grid was fitted on.
        generated: Grid cell boxes from ``build_grid_boxes``.
        out_path: Destination ``.jpg`` path.
    """
    img: np.ndarray = plt.imread(image_path)
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.imshow(img)
    for row in generated.iter_rows(named=True):
        ax.add_patch(
            Rectangle(
                (row["x1"], row["y1"]),
                row["box_width"],
                row["box_height"],
                fill=False,
                edgecolor="deepskyblue",
                linewidth=1.0,
                alpha=0.8,
            )
        )
    ax.set_title(f"{image_path.name} — {generated.height} inferred grid boxes")
    ax.set_xlim(0.0, img.shape[1])
    ax.set_ylim(img.shape[0], 0.0)
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# pipeline


def scan_pages(data_dir: Path) -> list[Path]:
    """Collect the page images under ``data_dir``, one page per image file.

    Every image file found recursively is a page; nothing is excluded. The
    pages are sorted by path so batch runs are deterministic.

    Args:
        data_dir: Directory to scan.

    Returns:
        Page image paths sorted by path.

    Raises:
        ValueError: If ``data_dir`` is missing, holds no image files, or two
            images share a stem (they would map to the same ``<source-id>``
            output folders).
    """
    if not data_dir.is_dir():
        raise ValueError(f"data directory not found: {data_dir}")
    pages = sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not pages:
        raise ValueError(f"no page images found under {data_dir}")
    seen: dict[str, Path] = {}
    for page in pages:
        if page.stem in seen:
            raise ValueError(
                f"duplicate <source-id> {page.stem!r}: {seen[page.stem]} and {page}"
            )
        seen[page.stem] = page
    return pages


def process_page(
    model: TableCellsDetection,
    image_path: Path,
    patches_root: Path,
    qc_root: Path,
    threshold: float,
) -> PageSummary:
    """Run the detect -> grid -> crop pipeline for one page.

    Writes ``patches/<source-id>/<row>_<col>.png`` and the QC folder
    ``qc/<source-id>/`` (``result.json``, ``paddle.jpg``, ``cell_dist.jpg``,
    ``grid_phase.jpg``, ``inferred.jpg``, ``grid.json``). ``grid.json`` holds
    the table-fitting results and is written after the patches are exported,
    so it always describes the patch set on disk; a page rejected by the
    minimum-grid check writes neither.

    Args:
        model: The shared ``TableCellsDetection`` model instance.
        image_path: Page image to process.
        patches_root: Root of the ``patches/`` tree.
        qc_root: Root of the ``qc/`` tree.
        threshold: Detection confidence threshold.

    Returns:
        The page's ``PageSummary``.

    Raises:
        ValueError: If the page has no usable cells: nothing detected, a
            malformed box, no box within the median size band, too few box
            centres to fit a grid, no usable FFT peak, no supported grid
            line on an axis, or an estimated grid smaller than
            ``MIN_GRID_COLS`` x ``MIN_GRID_ROWS``.
        OSError: If the page image cannot be read or an artifact cannot be
            written.
    """
    source_id: str = image_path.stem
    logger.debug("processing %s", repo_relative(image_path))
    qc_dir: Path = qc_root / source_id
    qc_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        width_px: float = float(img.width)
        height_px: float = float(img.height)

    result: PaddleOcrResult = detect_cells(model, image_path, threshold)
    write_result_json(result, qc_dir / "result.json")
    logger.debug("%s: wrote %s", source_id, repo_relative(qc_dir / "result.json"))

    boxes: pl.DataFrame = load_page_boxes(result)
    logger.debug("%s: %d boxes detected", source_id, boxes.height)
    render_detection_overlay(image_path, boxes, qc_dir / "paddle.jpg")
    logger.debug("%s: wrote %s", source_id, repo_relative(qc_dir / "paddle.jpg"))

    med_w, med_h = median_box_size(boxes)
    logger.debug("%s: median box size %.1f x %.1f px", source_id, med_w, med_h)
    render_cell_distribution(
        boxes["box_width"].to_numpy(),
        boxes["box_height"].to_numpy(),
        med_w,
        med_h,
        source_id,
        qc_dir / "cell_dist.jpg",
    )
    logger.debug("%s: wrote %s", source_id, repo_relative(qc_dir / "cell_dist.jpg"))
    filtered: pl.DataFrame = filter_boxes_by_median_size(boxes, med_w, med_h)
    logger.debug(
        "%s: %d of %d boxes within median size band",
        source_id,
        filtered.height,
        boxes.height,
    )

    centers: pl.DataFrame = filtered.with_columns(
        ((pl.col("x1") + pl.col("x2")) / 2).alias("cx"),
        ((pl.col("y1") + pl.col("y2")) / 2).alias("cy"),
    )
    spacing_x, freqs_x, power_x = estimate_spacing_fft(
        centers["cx"].to_numpy(),
        min_period=MIN_PERIOD_FACTOR * med_w,
        max_period=MAX_PERIOD_FACTOR * med_w,
    )
    spacing_y, freqs_y, power_y = estimate_spacing_fft(
        centers["cy"].to_numpy(),
        min_period=MIN_PERIOD_FACTOR * med_h,
        max_period=MAX_PERIOD_FACTOR * med_h,
    )
    logger.debug("%s: spacing %.1f x %.1f px", source_id, spacing_x, spacing_y)
    render_spacing_spectra(
        {"x": (spacing_x, freqs_x, power_x), "y": (spacing_y, freqs_y, power_y)},
        qc_dir / "grid_phase.jpg",
    )
    logger.debug("%s: wrote %s", source_id, repo_relative(qc_dir / "grid_phase.jpg"))

    phase_x, _, residual_x = fit_grid_axis(centers["cx"].to_numpy(), spacing_x)
    phase_y, _, residual_y = fit_grid_axis(centers["cy"].to_numpy(), spacing_y)
    rms_x = float(np.sqrt(np.mean(residual_x**2)))
    rms_y = float(np.sqrt(np.mean(residual_y**2)))

    k_x_kept, grid_x_lines = kept_grid_lines(
        spacing_x,
        phase_x,
        width_px,
        filtered["x1"].to_numpy(),
        filtered["x2"].to_numpy(),
    )
    k_y_kept, grid_y_lines = kept_grid_lines(
        spacing_y,
        phase_y,
        height_px,
        filtered["y1"].to_numpy(),
        filtered["y2"].to_numpy(),
    )

    logger.debug(
        "%s: grid %d cols x %d rows, residual RMS %.2f x %.2f px",
        source_id,
        grid_x_lines.size,
        grid_y_lines.size,
        rms_x,
        rms_y,
    )
    grid_boxes: pl.DataFrame = build_grid_boxes(
        grid_x_lines, grid_y_lines, k_x_kept, k_y_kept, spacing_x, spacing_y
    )
    render_inferred_overlay(image_path, grid_boxes, qc_dir / "inferred.jpg")
    logger.debug("%s: wrote %s", source_id, repo_relative(qc_dir / "inferred.jpg"))

    if grid_x_lines.size < MIN_GRID_COLS or grid_y_lines.size < MIN_GRID_ROWS:
        raise ValueError(
            f"grid too small: {grid_x_lines.size} cols x {grid_y_lines.size} rows "
            f"(need at least {MIN_GRID_COLS} cols x {MIN_GRID_ROWS} rows)"
        )

    patches: list[CellPatch] = export_cell_patches(
        grid_boxes, image_path, patches_root / source_id
    )

    metadata: GridMetadata = build_grid_metadata(
        source_id,
        image_path,
        width_px,
        height_px,
        fit_x=AxisFit(spacing_x, phase_x, rms_x, k_x_kept, grid_x_lines),
        fit_y=AxisFit(spacing_y, phase_y, rms_y, k_y_kept, grid_y_lines),
        patches=patches,
    )
    write_grid_json(metadata, qc_dir / "grid.json")
    logger.debug("%s: wrote %s", source_id, repo_relative(qc_dir / "grid.json"))

    return PageSummary(
        source_id=source_id,
        n_rows=grid_y_lines.size,
        n_cols=grid_x_lines.size,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        rms_x=rms_x,
        rms_y=rms_y,
        n_patches=len(patches),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Detect the table cells of weather-report page scans and export one crop per cell.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory of page images (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output root holding patches/ and qc/ (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="minimum detection confidence (default: %(default)s)",
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
    """Process every page under ``--data`` and print the run summary.

    Logging is configured first from ``--log-level`` (stderr plus the appended
    ``<out>/preprocess.log``); the run summary still goes to stdout. Pages
    that fail with an expected error are logged and skipped; the run summary
    lists the failed pages and the run exits 1 if any page failed. Unexpected
    errors abort the run immediately so bugs surface.

    Raises:
        SystemExit: If the arguments are invalid, the input cannot be
            scanned, the model cannot be loaded, or any page failed.
    """
    args = parse_args()
    data_dir: Path = args.data.resolve()
    out_dir: Path = args.out.resolve()
    setup_logging(args.log_level, out_dir / "preprocess.log")

    if not 0.0 <= args.threshold <= 1.0:
        logger.error("--threshold must be between 0 and 1, got %s", args.threshold)
        raise SystemExit(f"--threshold must be between 0 and 1, got {args.threshold}")

    try:
        pages: list[Path] = scan_pages(data_dir)
    except ValueError as exc:
        logger.error("cannot scan %s: %s", repo_relative(data_dir), exc)
        raise SystemExit(f"cannot scan {data_dir}: {exc}") from exc
    logger.info("%d pages under %s", len(pages), repo_relative(data_dir))

    patches_root: Path = out_dir / "patches"
    qc_root: Path = out_dir / "qc"
    logger.info(
        "writing %s and %s (threshold %.2f)",
        repo_relative(patches_root),
        repo_relative(qc_root),
        args.threshold,
    )

    logger.info("loading model %s (engine=onnxruntime)", MODEL_NAME)
    try:
        model = TableCellsDetection(model_name=MODEL_NAME, engine="onnxruntime")
    except Exception as exc:
        logger.error("cannot load model %s: %s", MODEL_NAME, exc)
        raise SystemExit(f"cannot load model {MODEL_NAME}: {exc}") from exc
    logger.info("model ready")

    total_patches: int = 0
    failed: list[tuple[str, str]] = []
    for page in pages:
        try:
            summary = process_page(model, page, patches_root, qc_root, args.threshold)
        except (ValueError, OSError) as exc:
            logger.warning("page %s failed: %s", repo_relative(page), exc)
            failed.append((repo_relative(page), str(exc)))
            continue
        total_patches += summary.n_patches
        logger.info(
            "%s: %d cols x %d rows (spacing %.1f x %.1f px, "
            "residual RMS %.2f x %.2f px), %d patches",
            summary.source_id,
            summary.n_cols,
            summary.n_rows,
            summary.spacing_x,
            summary.spacing_y,
            summary.rms_x,
            summary.rms_y,
            summary.n_patches,
        )

    run_summary = (
        f"{len(pages)} pages / {total_patches} patches -> {patches_root} "
        f"({len(failed)} failed)"
    )
    logger.info(run_summary)
    print(run_summary)
    for label, reason in failed:
        print(f"  {label}: {reason}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
