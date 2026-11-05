#!/usr/bin/env python3
"""Bootstrap annotations: run a YOLO model over the non-annotated digits crops.

Usage:

    pixi run python scripts/bootstrap_annotations.py --model runs/detect/train/weights/best.pt

Scans ``datasets/digits/images`` for crops without a mirrored label file under
``datasets/digits/labels`` (the annotation tool's "not yet annotated"
convention) and runs Ultralytics detection on them with the given weights.
Crops flagged in the annotation tool as needing manual adjustment (a marker
under ``datasets/digits/manual``) belong to the manual queue and are skipped:
they are labeled by hand first. One prediction file per crop is written under
``datasets/digits/predictions``,
mirroring the images tree: each line is ``cls cx cy w h conf`` in normalised
0-1 coordinates (the YOLO label format plus a 6th confidence column), and a
crop with no detections above ``--conf`` gets an empty file (a predicted
background crop).

Prediction files are *not* valid YOLO label files (labels take exactly 5
fields); drop the last column before promoting them into ``labels/``. The
``predictions`` tree is owned by this script: prediction files whose crop has
since been annotated, flagged for manual adjustment, or removed are pruned on
every run. The ``manual`` tree is written by the annotation tool; this script
only prunes stale markers (whose crop is annotated or removed), so a run
leaves every marker mirroring an unlabeled crop.

Inference is lazy and batched: crops are processed in batches of
``--batch-size`` and each batch's prediction files are written before the next
batch is loaded, so memory use stays flat no matter how many crops there are.
A run interrupted mid-way leaves the completed batches' prediction files on
disk — harmless, since those crops are still unlabeled and are simply
re-predicted (and overwritten) by the next run.

Run activity is logged to stderr and appended to ``<dataset>/bootstrap.log``:
initialisation (model loading, dataset scan), one line per completed batch,
every written prediction file at DEBUG, and the run totals. ``--log-level``
narrows the verbosity (``DEBUG``, the default, shows everything); the run
summary on stdout is not affected.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "digits"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# A predicted box: (cls, cx, cy, w, h, conf) in normalised 0-1 coordinates.
Prediction = tuple[int, float, float, float, float, float]

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# logging


def setup_logging(level_name: str, log_file: Path) -> None:
    """Configure run logging: stderr plus an appended ``log_file``.

    Both handlers receive identical records with UTC timestamps; the log file
    is appended to, so reruns extend the same trail. The ``PIL`` and
    ``ultralytics`` loggers are quieted to WARNING so Pillow's per-chunk
    image-decoder traces and model-stack chatter do not drown the run's own
    records.

    Args:
        level_name: Logging level name (``DEBUG``..``CRITICAL``). ``DEBUG``,
            the ``--log-level`` default, logs everything including every
            written prediction file.
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
    for noisy in ("PIL", "ultralytics"):
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
    back to defaults. Same contract as ``tools/labeler.dataset.load_classes``.

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


def scan_dataset(dataset: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Return ``(annotated, manual, unlabeled)`` crops under ``dataset/images``.

    A crop is annotated iff the mirrored label file (the image path's ``images``
    tree mapped onto the sibling ``labels`` tree with the suffix changed to
    ``.txt``) exists; an empty label file counts as annotated (an annotated
    background crop). A crop is manual — flagged in the annotation tool as
    needing manual adjustment — iff it is unlabeled and the mirrored ``manual``
    marker exists; such crops belong to the manual queue and are never
    predicted. This is the same convention as
    ``scripts/prepare_dataset.find_labeled_images`` and the annotation tool's
    ``done`` flag and manual queue. All lists are sorted by path for
    deterministic output.

    Args:
        dataset: Dataset root containing the ``images``, ``labels`` and
            ``manual`` trees (``manual`` may be absent — then no crop is
            flagged).

    Returns:
        ``(annotated, manual, unlabeled)`` lists of image paths.

    Raises:
        ValueError: If the ``images`` or ``labels`` tree is missing.
    """
    images_root = dataset / "images"
    labels_root = dataset / "labels"
    manual_root = dataset / "manual"
    if not images_root.is_dir():
        raise ValueError(f"images directory not found: {images_root}")
    if not labels_root.is_dir():
        raise ValueError(f"labels directory not found: {labels_root}")
    annotated: list[Path] = []
    manual: list[Path] = []
    unlabeled: list[Path] = []
    for image in sorted(p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        label = labels_root / image.relative_to(images_root).with_suffix(".txt")
        if label.is_file():
            annotated.append(image)
        elif (manual_root / image.relative_to(images_root).with_suffix(".txt")).is_file():
            manual.append(image)
        else:
            unlabeled.append(image)
    return annotated, manual, unlabeled


# ---------------------------------------------------------------------------
# inference


def check_model_names(model: YOLO, classes: list[str]) -> None:
    """Check that the model's class names match the dataset's ``data.yaml`` names.

    Prediction files store bare class ids, so a mismatched model would silently
    write ids that mean something else in this dataset.

    Args:
        model: The loaded Ultralytics model; ``model.names`` maps id -> name.
        classes: Class names from ``data.yaml``, indexed by class id.

    Raises:
        ValueError: If the model's class names differ from ``classes``.
    """
    names = {int(k): str(v) for k, v in model.names.items()}
    expected = dict(enumerate(classes))
    if names != expected:
        raise ValueError(
            "model class names do not match data.yaml names:\n"
            f"  model    : {[names[i] for i in sorted(names)]}\n"
            f"  data.yaml: {classes}"
        )


def run_inference(model: YOLO, images: list[Path], conf: float) -> dict[Path, list[Prediction]]:
    """Run detection on `images`, returning predicted boxes per image.

    Args:
        model: The loaded Ultralytics detection model.
        images: Images to process.
        conf: Minimum confidence for a returned box.

    Returns:
        Mapping from each image path (resolved) to its predicted boxes in the
        model's output order (highest confidence first).

    Raises:
        ValueError: If the model returns no result for an image.
    """
    if not images:
        return {}
    predicted: dict[Path, list[Prediction]] = {}
    results = model.predict(source=[str(p) for p in images], conf=conf, stream=True, verbose=False)
    for result in results:
        boxes: list[Prediction] = []
        if result.boxes is not None:
            for cls, xywhn, score in zip(
                result.boxes.cls.tolist(), result.boxes.xywhn.tolist(), result.boxes.conf.tolist()
            ):
                cx, cy, w, h = xywhn
                boxes.append((int(cls), float(cx), float(cy), float(w), float(h), float(score)))
        predicted[Path(result.path).resolve()] = boxes
    missing = [str(p) for p in images if p.resolve() not in predicted]
    if missing:
        raise ValueError(f"model returned no result for {len(missing)} images, e.g. {missing[0]}")
    return predicted


# ---------------------------------------------------------------------------
# prediction files


def prediction_path_for(rel: Path, predictions_root: Path) -> Path:
    """``predictions`` root mirror of an image path: <stem>.txt in the same subdirs.

    Args:
        rel: Image path relative to the ``images`` tree.
        predictions_root: The ``predictions`` tree to mirror into.

    Returns:
        The prediction file path for ``rel``.
    """
    return predictions_root / rel.parent / (rel.stem + ".txt")


def write_prediction(path: Path, boxes: list[Prediction]) -> None:
    """Write `boxes` to `path` as YOLO label lines plus a confidence column.

    Each line is ``cls cx cy w h conf`` with normalised coordinates and
    six-decimal formatting: the five label columns of
    ``tools/labeler.dataset.Dataset.write_labels`` plus a 6th confidence column
    in 0-1. An
    empty `boxes` list writes an empty file, a predicted background crop.

    Args:
        path: Destination file; parent directories are created as needed.
        boxes: Predicted boxes ``(cls, cx, cy, w, h, conf)`` in model order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}" for cls, cx, cy, w, h, conf in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prune_stale(predictions_root: Path, targets: list[Path], images_root: Path) -> int:
    """Delete prediction files whose crop is annotated, manual, or gone; return the count.

    Keeps ``predictions/`` owned by this script: after a run every prediction
    file mirrors a crop predicted in this run, so ``predictions/`` never
    overlaps ``labels/`` or the manual queue (crops excluded from `targets`,
    e.g. flagged for manual adjustment). Only ``.txt`` files are touched.

    Args:
        predictions_root: The ``predictions`` tree to prune.
        targets: The images that received predictions in this run.
        images_root: The ``images`` tree that ``predictions_root`` mirrors.

    Returns:
        Number of deleted files.
    """
    keep = {p.relative_to(images_root).with_suffix("").as_posix() for p in targets}
    removed = 0
    for pred in sorted(predictions_root.rglob("*.txt")):
        if pred.relative_to(predictions_root).with_suffix("").as_posix() not in keep:
            pred.unlink()
            removed += 1
    return removed


def prune_manual(manual_root: Path, images_root: Path, labels_root: Path) -> int:
    """Delete manual-queue markers whose crop is annotated or gone; return the count.

    The ``manual`` tree is owned by the annotation tool (it writes and clears
    markers as crops are flagged and labeled); this is only hygiene so a run
    leaves every marker mirroring a currently unlabeled crop — a marker whose
    crop was removed or labeled out of band is stale. Only ``.txt`` files are
    touched.

    Args:
        manual_root: The ``manual`` marker tree to prune.
        images_root: The ``images`` tree that ``manual_root`` mirrors.
        labels_root: The ``labels`` tree that ``manual_root`` mirrors.

    Returns:
        Number of deleted files.
    """
    if not manual_root.is_dir():
        return 0
    image_stems = {
        p.relative_to(images_root).with_suffix("").as_posix()
        for p in images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    removed = 0
    for marker in sorted(manual_root.rglob("*.txt")):
        rel = marker.relative_to(manual_root)
        if rel.with_suffix("").as_posix() not in image_stems or (labels_root / rel).is_file():
            marker.unlink()
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the bootstrap script."""
    parser = argparse.ArgumentParser(
        description="Run a trained YOLO model over the non-annotated crops and write predictions.",
    )
    parser.add_argument("--model", type=Path, required=True, help="trained YOLO weights (.pt)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR, help="dataset root (default: %(default)s)")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum confidence for a written box (default: %(default)s)")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="images per inference batch; bounds peak memory (default: %(default)s)"
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="logging verbosity; DEBUG (the default) logs every written prediction file",
    )
    return parser.parse_args()


def main() -> None:
    """Predict boxes for every non-annotated crop and write the prediction files.

    Logging is configured first from ``--log-level`` (stderr plus the appended
    ``<dataset>/bootstrap.log``); the run summary still goes to stdout. Crops
    are inferred in lazily loaded batches of ``--batch-size`` and each batch's
    prediction files are written before the next batch is loaded, so memory
    use stays flat regardless of dataset size.
    """
    args = parse_args()
    dataset = args.dataset.resolve()
    setup_logging(args.log_level, dataset / "bootstrap.log")
    if not args.model.is_file():
        logger.error("model weights not found: %s", args.model)
        raise SystemExit(f"model weights not found: {args.model}")
    if not 0.0 <= args.conf <= 1.0:
        logger.error("--conf must be between 0 and 1, got %s", args.conf)
        raise SystemExit(f"--conf must be between 0 and 1, got {args.conf}")
    if args.batch_size < 1:
        logger.error("--batch-size must be at least 1, got %s", args.batch_size)
        raise SystemExit(f"--batch-size must be at least 1, got {args.batch_size}")
    logger.info("model=%s dataset=%s conf=%g batch_size=%d", args.model, dataset, args.conf, args.batch_size)

    annotated, manual, targets = scan_dataset(dataset)
    logger.info(
        "scan: %d annotated / %d manual / %d unlabeled crops",
        len(annotated),
        len(manual),
        len(targets),
    )

    predictions_root = dataset / "predictions"
    if targets:
        classes = load_classes(dataset / "data.yaml")
        model = YOLO(str(args.model))
        check_model_names(model, classes)
        logger.info("loaded %s; model classes match data.yaml (%d classes)", args.model, len(classes))
        n_boxes = 0
        for i, batch in enumerate(itertools.batched(targets, args.batch_size), start=1):
            predicted = run_inference(model, list(batch), args.conf)
            for image in batch:
                rel = image.relative_to(dataset / "images")
                boxes = predicted[image.resolve()]
                pred_path = prediction_path_for(rel, predictions_root)
                write_prediction(pred_path, boxes)
                logger.debug("%s: %d boxes -> %s", image, len(boxes), pred_path)
                n_boxes += len(boxes)
            logger.info("batch %d: %d images, %d boxes so far", i, len(batch), n_boxes)
        logger.info("done: %d images / %d boxes -> %s", len(targets), n_boxes, predictions_root)
        print(f"{len(targets)} images / {n_boxes} boxes -> {predictions_root}")
    else:
        logger.info(
            "nothing to do: %d annotated / %d manual, no crops to predict",
            len(annotated),
            len(manual),
        )
        print(
            f"nothing to do: {len(annotated)} annotated / {len(manual)} manual, "
            "no crops to predict"
        )
    removed = prune_stale(predictions_root, targets, dataset / "images")
    if removed:
        logger.info("pruned %d stale prediction files", removed)
        print(f"pruned {removed} stale prediction files")
    pruned_manual = prune_manual(dataset / "manual", dataset / "images", dataset / "labels")
    if pruned_manual:
        logger.info("pruned %d stale manual-queue markers", pruned_manual)
        print(f"pruned {pruned_manual} stale manual-queue markers")


if __name__ == "__main__":
    main()
