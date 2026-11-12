#!/usr/bin/env python3
"""Remove the cropped duplicate boxes from the v2 per-page YOLO labels.

Usage:

    pixi run python scripts/dedupe_page_labels.py
    pixi run python scripts/dedupe_page_labels.py --dry-run

For every page label file ``<pages>/<source-id>.txt`` under ``--pages`` (by
default ``data/v2_patches/page_labels``, the tree
``scripts/recombine_labels.py`` writes) the script finds the boxes that are
cropped copies of another box on the same page and removes them, rewriting the
file in place with the surviving boxes in their original order.

A box is a cropped copy when another box of the same class has at least as
much area and contains at least ``--min-containment`` of the smaller box's
area (``0.95``, 95%, by default). That is exactly the copy the
crop/recombine round trip leaves behind:
``scripts/crop_page_labels.py:map_box_to_patch`` clips a page box to a padded
patch, and ``scripts/recombine_labels.py:map_box_to_page`` maps that clipped
box back inside the page box it came from.
``scripts/recombine_labels.py:merge_duplicate_boxes`` only collapses
near-identical pairs (IoU >= 0.8 from different patches), so a copy clipped
below that overlap stays in the page labels; this script removes it. A pair's
containment ratio is measured by area, which is exact in normalised
coordinates because the page's width and height cancel; boxes are never
clamped to the page.

Only the outermost box of a nest is kept: every box with a containing
same-class box is removed, transitively, so an original plus one or more
cropped copies becomes the original alone. The kept box is never resized,
merged or clamped -- the script removes copies rather than building an
enclosing box. When two boxes have exactly the same area, the earlier line is
treated as the original and the later line as the copy. Different classes are
never compared: a class id is a box's identity. Page labels carry no patch
identity, so a same-class box fully inside another is treated as a copy even
when both were annotated on the same patch; that is the deliberate page-level
view.

Boxes with a non-positive or non-finite area never take part in a pair (they
cannot contain or be contained meaningfully) and are left untouched with a
warning. The script needs no page images and no ``table_grid.json``: it reads
label files and the class set only.

A page with no cropped copies keeps its label file byte-for-byte untouched.
Otherwise the file is rewritten with the standard label formatting (five
columns at six decimals, a trailing newline when non-empty; the formatting of
``tools/labeler/dataset.Dataset.write_labels``), so comments and blank lines
are not preserved on a rewritten file (generated page labels contain none). A
rewritten file is never empty: every removed box has a surviving container, so
at least one box remains and a non-empty label can never become a
confirmed-empty one. Running the script again changes nothing.

``--dry-run`` reports exactly what would be removed and writes no files at
all: no label file is rewritten and no log file is created, so log output goes
to stderr only. A normal run logs to stderr and appends ``<pages>/dedupe.log``.

Parsing is strict, as in the other pipeline scripts: each line is
``<class> <cx> <cy> <w> <h>`` with a class id in the ``--data-yaml`` class
set. A malformed line or an out-of-range class fails its source, whose file is
left untouched, while the other sources still run; nested ``.txt`` files are
not page labels and are reported and left untouched. The run summary on stdout
lists failures and the run exits
1 if there was any; a source without cropped copies does not fail. Unexpected
errors abort the run immediately so bugs surface. Re-running
``scripts/recombine_labels.py`` overwrites the page labels wholesale and can
reintroduce the duplicates, so run this script after recombining.
"""

from __future__ import annotations

import argparse
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
DEFAULT_PAGES_DIR = REPO_ROOT / "data" / "v2_patches" / "page_labels"
DEFAULT_DATA_YAML = REPO_ROOT / "datasets" / "digits" / "data.yaml"

# Fraction of a smaller box's area that must lie inside a larger same-class box
# for the smaller box to count as a cropped copy. Overridable with
# --min-containment.
DEFAULT_MIN_CONTAINMENT = 0.95

LOG_FILENAME = "dedupe.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"

logger = logging.getLogger(__name__)


class Box(TypedDict):
    """One character box in normalised 0-1 coordinates.

    The annotation tool's ``Box`` (``tools/labeler/dataset.py``): the same
    five fields the page label files hold.
    """

    cls: int
    cx: float
    cy: float
    w: float
    h: float


class LabelBox(NamedTuple):
    """One parsed label box and the physical line it was read from.

    The line number makes the removal report point at the exact label line, so
    a duplicate can be checked against the source file without searching for
    its coordinates.
    """

    line: int
    box: Box


class SourceSummary(NamedTuple):
    """Box counts of one deduped page label."""

    source_id: str
    n_before: int  # boxes parsed from the label file
    n_removed: int  # cropped duplicates removed
    n_after: int  # boxes remaining (or that would remain in a dry run)
    changed: bool  # whether boxes were removed (would be, in a dry run)


# ---------------------------------------------------------------------------
# logging


def setup_logging(level_name: str, log_file: Path | None) -> None:
    """Configure run logging: stderr plus, on a normal run, an appended file.

    Both handlers receive identical records with UTC timestamps; the log file
    is appended to, so reruns extend the same trail. A dry run passes
    ``None``, which registers only the stderr handler and therefore writes no
    file at all.

    Args:
        level_name: Logging level name (``DEBUG``..``CRITICAL``). ``DEBUG``,
            the ``--log-level`` default, logs everything including every
            removed box.
        log_file: Log file to append to, or ``None`` for a dry run; the parent
            directory is created as needed.
    """
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    formatter.converter = time.gmtime
    root = logging.getLogger()
    root.setLevel(level_name)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    for handler in handlers:
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
# label I/O


def parse_label_file(path: Path, n_classes: int) -> list[LabelBox]:
    """Strictly parse one five-column page YOLO label file.

    Each line is ``<class> <cx> <cy> <w> <h>`` in normalised 0-1
    coordinates; blank lines and ``#`` comments are skipped. Parsing is
    strict: a malformed line is reported, never skipped, so corruption cannot
    be silently rewritten away. Same contract as
    ``scripts/recombine_labels.py:parse_label_file``, additionally returning
    each box's physical line number for the removal report.

    Args:
        path: The label file to parse.
        n_classes: Size of the dataset's class set (``data.yaml``); valid
            class ids are ``0 .. n_classes - 1``.

    Returns:
        The label boxes in file order, each with its physical line number;
        empty for an empty (or comment-only) label file.

    Raises:
        ValueError: If the file is missing, a line has the wrong field count
            or non-numeric fields, or a class id is out of range.
    """
    if not path.is_file():
        raise ValueError(f"no such label: {path}")
    boxes: list[LabelBox] = []
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
        boxes.append(LabelBox(line=lineno, box=Box(cls=cls, cx=cx, cy=cy, w=w, h=h)))
    return boxes


def write_page_labels(path: Path, boxes: Sequence[Box]) -> None:
    """Write one page's boxes as a per-page YOLO label file.

    Byte-compatible with the annotation tool's crop labels (the formatting of
    ``tools/labeler/dataset.Dataset.write_labels``, as in
    ``scripts/recombine_labels.py:write_page_labels``): one
    ``<class> <cx> <cy> <w> <h>`` line per box at six decimals, a trailing
    newline when non-empty and none for an empty file. Any existing file is
    overwritten.

    Args:
        path: Destination ``<source-id>.txt``; the parent directory is
            created as needed.
        boxes: The page's surviving boxes in output order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
        for b in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# duplicate detection


def box_area(box: Box) -> float:
    """Return the area of one normalised box.

    Args:
        box: A box (``cls``, ``cx``, ``cy``, ``w``, ``h``).

    Returns:
        ``w * h``, which may be zero, negative or non-finite for a malformed
        box; callers use :func:`has_usable_area` to filter those out.
    """
    return box["w"] * box["h"]


def has_usable_area(box: Box) -> bool:
    """Test whether a box can take part in a containment pair.

    A box with a non-positive or non-finite area cannot contain or be
    contained meaningfully, so it is never treated as a duplicate or as an
    original.

    Args:
        box: A box (``cls``, ``cx``, ``cy``, ``w``, ``h``).

    Returns:
        ``True`` when the area is finite and greater than zero.
    """
    area = box_area(box)
    return math.isfinite(area) and area > 0.0


def containment_fraction(inner: Box, outer: Box) -> float:
    """Return how much of ``inner`` lies inside ``outer``, as an area fraction.

    The intersection of the two axis-aligned boxes is divided by ``inner``'s
    area: ``1.0`` when ``inner`` lies wholly inside ``outer``, ``0.0`` when
    they do not overlap. Both boxes are normalised page coordinates, so the
    fraction is exact (the page width and height cancel); boxes are never
    clamped to the page, so a box that sticks out past the page edge is
    measured as written.

    Args:
        inner: The box whose contained fraction is measured.
        outer: The candidate container box.

    Returns:
        The contained fraction in ``[0, 1]``, or ``0.0`` when ``inner`` has no
        positive area or either box has non-finite coordinates.
    """
    ix1 = inner["cx"] - inner["w"] / 2.0
    iy1 = inner["cy"] - inner["h"] / 2.0
    ix2 = inner["cx"] + inner["w"] / 2.0
    iy2 = inner["cy"] + inner["h"] / 2.0
    ox1 = outer["cx"] - outer["w"] / 2.0
    oy1 = outer["cy"] - outer["h"] / 2.0
    ox2 = outer["cx"] + outer["w"] / 2.0
    oy2 = outer["cy"] + outer["h"] / 2.0
    if not all(math.isfinite(v) for v in (ix1, iy1, ix2, iy2, ox1, oy1, ox2, oy2)):
        return 0.0
    area = (ix2 - ix1) * (iy2 - iy1)
    if area <= 0.0:
        return 0.0
    inter_w = min(ix2, ox2) - max(ix1, ox1)
    inter_h = min(iy2, oy2) - max(iy1, oy1)
    intersection = max(inter_w, 0.0) * max(inter_h, 0.0)
    return intersection / area


def duplicate_copies(
    boxes: Sequence[Box], min_containment: float
) -> list[tuple[int, int, float]]:
    """Find the cropped duplicate boxes of one page.

    A box is a duplicate when another same-class box has at least as much area
    and contains at least ``min_containment`` of it. Every such box is
    returned -- including transitive copies -- so the surviving boxes are the
    outermost boxes of each nest. When two boxes have exactly equal area, the
    earlier one is the original and the later one is the duplicate; the
    descent-by-area order below implements that tie-break for free.

    Only boxes with a usable area participate; a malformed box is never
    removed and never acts as a container. The scan is grouped by class (a
    class id is a box's identity) and walks each group largest first, so only
    previously seen boxes of the same class, which all have at least as much
    area, need to be tested.

    Args:
        boxes: One page's boxes, in file order.
        min_containment: Minimum fraction of the smaller box's area that must
            lie inside the larger box, in ``(0, 1]``.

    Returns:
        ``(removed, container, fraction)`` records sorted by the removed
        box's index: which box is a duplicate, which box keeps it contained
        and how much of it that container holds. Empty when the page has no
        duplicates.

    Raises:
        ValueError: If ``min_containment`` is outside ``(0, 1]``.
    """
    if not 0.0 < min_containment <= 1.0:
        raise ValueError(
            f"min_containment must be greater than 0 and at most 1, got {min_containment}"
        )
    by_class: dict[int, list[int]] = {}
    for index, box in enumerate(boxes):
        if has_usable_area(box):
            by_class.setdefault(box["cls"], []).append(index)

    copies: list[tuple[int, int, float]] = []
    for indices in by_class.values():
        indices.sort(key=lambda index: (-box_area(boxes[index]), index))
        for position, index in enumerate(indices):
            inner = boxes[index]
            best: tuple[int, float] | None = None
            for candidate in indices[:position]:
                fraction = containment_fraction(inner, boxes[candidate])
                if fraction >= min_containment and (
                    best is None or fraction > best[1]
                ):
                    best = (candidate, fraction)
            if best is not None:
                copies.append((index, best[0], best[1]))
    copies.sort(key=lambda record: record[0])
    return copies


# ---------------------------------------------------------------------------
# deduping


def scan_label_files(pages_root: Path) -> tuple[list[tuple[str, Path]], list[Path]]:
    """Discover the page label files and the stray label files under ``pages_root``.

    A page label is a top-level ``<source-id>.txt``. Every other ``.txt`` file
    (nested deeper) is not a page label and is returned separately so the run
    can report it; nothing under the root is touched except the discovered
    page label files.

    Args:
        pages_root: Root of the page-space label tree.

    Returns:
        ``(sources, nested)``: the page labels as ``(source id, path)`` in
        sorted path order, and the nested ``.txt`` files (sorted).

    Raises:
        ValueError: If ``pages_root`` is missing.
    """
    if not pages_root.is_dir():
        raise ValueError(f"page labels directory not found: {pages_root}")
    sources: list[tuple[str, Path]] = []
    nested: list[Path] = []
    for path in sorted(pages_root.rglob("*.txt")):
        relative = path.relative_to(pages_root)
        if len(relative.parts) == 1:
            sources.append((path.stem, path))
        else:
            nested.append(path)
    return sources, nested


def dedupe_source(
    path: Path, classes: Sequence[str], min_containment: float, dry_run: bool
) -> SourceSummary:
    """Remove one page label file's cropped duplicate boxes.

    Parses the file strictly, finds the contained same-class copies
    (:func:`duplicate_copies`) and, when there are any and this is not a dry
    run, rewrites the file with the surviving boxes in their original order.
    A file without duplicates is never rewritten, so its bytes are untouched.

    Args:
        path: The page's ``<source-id>.txt`` label file.
        classes: The dataset's class names, indexed by class id (from
            ``--data-yaml``); its length bounds the valid class ids.
        min_containment: Minimum fraction of a smaller box's area that must
            lie inside a larger same-class box for it to count as a copy.
        dry_run: Report the removals without writing the label file.

    Returns:
        The source's ``SourceSummary``.

    Raises:
        ValueError: If the label file is malformed or holds an out-of-range
            class id, or ``min_containment`` is invalid.
        OSError: If the file cannot be read or written.
    """
    parsed = parse_label_file(path, len(classes))
    for item in parsed:
        if not has_usable_area(item.box):
            logger.warning(
                "%s line %d: non-positive or non-finite area, left untouched",
                path.name,
                item.line,
            )

    boxes = [item.box for item in parsed]
    copies = duplicate_copies(boxes, min_containment)
    for index, container, fraction in copies:
        copy = parsed[index]
        original = parsed[container]
        logger.debug(
            "%s line %d: removed class %d (%s) box %.6f %.6f %.6f %.6f; "
            "%.1f%% inside line %d class %d (%s)",
            path.name,
            copy.line,
            copy.box["cls"],
            classes[copy.box["cls"]],
            copy.box["cx"],
            copy.box["cy"],
            copy.box["w"],
            copy.box["h"],
            fraction * 100.0,
            original.line,
            original.box["cls"],
            classes[original.box["cls"]],
        )

    removed = {index for index, _, _ in copies}
    kept = [item.box for index, item in enumerate(parsed) if index not in removed]
    if copies and not dry_run:
        write_page_labels(path, kept)
        logger.debug("%s: wrote %s", path.stem, repo_relative(path))
    return SourceSummary(
        source_id=path.stem,
        n_before=len(parsed),
        n_removed=len(copies),
        n_after=len(kept),
        changed=bool(copies),
    )


# ---------------------------------------------------------------------------
# CLI


def parse_containment(value: str) -> float:
    """Parse and validate a ``--min-containment`` fraction.

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
    """Parse command-line arguments for the page-label dedupe script."""
    parser = argparse.ArgumentParser(
        description="Remove the cropped duplicate boxes from the v2 per-page YOLO labels.",
    )
    parser.add_argument(
        "--pages",
        type=Path,
        default=DEFAULT_PAGES_DIR,
        help="root of the page-space label tree (default: %(default)s)",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="dataset data.yaml naming the classes (default: %(default)s)",
    )
    parser.add_argument(
        "--min-containment",
        type=parse_containment,
        default=DEFAULT_MIN_CONTAINMENT,
        help=(
            "fraction of a smaller box's area that must lie inside a larger "
            "same-class box for it to be removed as a cropped copy "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the removals without writing any file (no label rewrites, no log file)",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="logging verbosity; DEBUG (the default) logs every pipeline step and removed box",
    )
    return parser.parse_args()


def main() -> None:
    """Dedupe every page label under ``--pages`` and print the run summary.

    Logging is configured first from ``--log-level`` (stderr plus, on a normal
    run, the appended ``<pages>/dedupe.log``). The page-label tree is checked
    before the log file is opened, so a missing ``--pages`` never creates it.
    Sources that fail with an expected error are logged and skipped and their
    files are left untouched; the run summary lists every failure and the run
    exits 1 if there was any. Unexpected errors abort the run immediately so
    bugs surface.

    Raises:
        SystemExit: If the arguments are invalid, the page-label tree is
            missing, the class set cannot be loaded, or any source failed.
    """
    args = parse_args()
    pages_root: Path = args.pages.resolve()
    dry_run: bool = args.dry_run
    if not pages_root.is_dir():
        raise SystemExit(f"page labels directory not found: {pages_root}")
    setup_logging(args.log_level, None if dry_run else pages_root / LOG_FILENAME)

    try:
        classes = load_classes(args.data_yaml)
    except ValueError as exc:
        logger.error("cannot load classes: %s", exc)
        raise SystemExit(f"cannot load classes: {exc}") from exc
    logger.info("%d classes from %s", len(classes), repo_relative(args.data_yaml))
    logger.info(
        "minimum containment %.3f (smaller box must lie this much inside its original)",
        args.min_containment,
    )
    if dry_run:
        logger.info("dry run: no files will be written")

    try:
        sources, nested = scan_label_files(pages_root)
    except ValueError as exc:
        logger.error("cannot scan %s: %s", repo_relative(pages_root), exc)
        raise SystemExit(f"cannot scan {pages_root}: {exc}") from exc
    logger.info("%d sources under %s", len(sources), repo_relative(pages_root))

    failed: list[tuple[str, str]] = []
    for path in nested:
        logger.warning(
            "label %s skipped: not a <source-id>.txt page label",
            repo_relative(path),
        )

    summaries: list[SourceSummary] = []
    for source_id, label_path in sources:
        try:
            summary = dedupe_source(
                label_path, classes, args.min_containment, dry_run
            )
        except (ValueError, OSError) as exc:
            logger.warning("source %s failed: %s", source_id, exc)
            failed.append((source_id, str(exc)))
            continue
        summaries.append(summary)
        logger.info(
            "%s: %d boxes (%d cropped duplicates removed, %d remain)",
            summary.source_id,
            summary.n_before,
            summary.n_removed,
            summary.n_after,
        )

    total_before = sum(s.n_before for s in summaries)
    total_removed = sum(s.n_removed for s in summaries)
    total_after = sum(s.n_after for s in summaries)
    n_changed = sum(1 for s in summaries if s.changed)
    verb = "would change" if dry_run else "changed"
    suffix = " (dry run, no files written)" if dry_run else ""
    run_summary = (
        f"{len(sources)} sources / {n_changed} label files {verb} / "
        f"{total_before} boxes ({total_removed} cropped duplicates removed) -> "
        f"{total_after} boxes -> {pages_root} ({len(failed)} failed){suffix}"
    )
    logger.info(run_summary)
    print(run_summary)
    for label, reason in failed:
        print(f"  {label}: {reason}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
