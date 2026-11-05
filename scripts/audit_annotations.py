#!/usr/bin/env python3
"""Audit annotations: run a YOLO model over every digits crop for label review.

Usage:

    pixi run python scripts/audit_annotations.py --model runs/detect/train/weights/best.pt

Runs Ultralytics detection with the given weights over **every** crop under
``datasets/digits/images`` and writes one prediction file per crop under
``datasets/digits/audit_predictions``, mirroring the images tree: each line
is ``cls cx cy w h conf`` in normalised 0-1 coordinates (the YOLO label
format plus a 6th confidence column), and a crop with no detections above
``--conf`` gets an empty file (a predicted background crop). The labeler's
audit stage (``tools/labeler``) compares these predictions against the
dataset's labels and surfaces every disagreement for review.

One run is one review round: the audit stage deletes the prediction file of
every crop it decides, and a fresh run re-flags whatever still disagrees.
Prediction files whose crop is gone are pruned on every run. Prediction
files are *not* valid YOLO label files (labels take exactly 5 fields).

Inference is lazy and batched: crops are processed in batches of
``--batch-size`` and each batch's prediction files are written before the next
batch is loaded, so memory use stays flat no matter how many crops there are.
A run interrupted mid-way leaves the completed batches' prediction files on
disk — harmless to the audit stage (``tools/labeler``), which only reviews
crops still holding an audit prediction — but the round is fully refreshed
only once a run completes: un-reached crops keep whatever prediction state
they had (nothing, or a previous run's files). A fresh full run overwrites
every crop's prediction file.

Run activity is logged to stderr and appended to ``<dataset>/audit.log``:
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
from ultralytics.engine.results import Results

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


def scan_images(dataset: Path) -> list[Path]:
    """Return every crop image under ``dataset/images``, sorted by path.

    The audit round covers the whole dataset — annotated or not — because
    the comparison against labels happens later, in the labeler's audit
    stage. Same crop convention as ``scripts/bootstrap_annotations.py``.

    Args:
        dataset: Dataset root containing the ``images`` tree.

    Returns:
        All crop image paths, sorted by path for deterministic output.

    Raises:
        ValueError: If the ``images`` tree is missing.
    """
    images_root = dataset / "images"
    if not images_root.is_dir():
        raise ValueError(f"images directory not found: {images_root}")
    return sorted(p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


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
        # the predict() stub is typed Results | Tensor; detection always yields Results
        if not isinstance(result, Results):
            raise ValueError(f"unexpected prediction result type: {type(result).__name__}")
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
    """``audit_predictions`` root mirror of an image path: <stem>.txt in the same subdirs.

    Args:
        rel: Image path relative to the ``images`` tree.
        predictions_root: The ``audit_predictions`` tree to mirror into.

    Returns:
        The prediction file path for ``rel``.
    """
    return predictions_root / rel.parent / (rel.stem + ".txt")


def write_prediction(path: Path, boxes: list[Prediction]) -> None:
    """Write `boxes` to `path` as YOLO label lines plus a confidence column.

    Each line is ``cls cx cy w h conf`` with normalised coordinates and
    six-decimal formatting: the five label columns of
    ``tools/labeler.dataset.Dataset.write_labels`` plus a 6th confidence column
    in 0-1. An empty `boxes` list writes an empty file, a predicted background
    crop.

    Args:
        path: Destination file; parent directories are created as needed.
        boxes: Predicted boxes ``(cls, cx, cy, w, h, conf)`` in model order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}" for cls, cx, cy, w, h, conf in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prune_stale(predictions_root: Path, targets: list[Path], images_root: Path) -> int:
    """Delete audit prediction files whose crop is gone; return the count.

    Keeps ``audit_predictions/`` owned by this script: every file mirrors a
    current crop. Files for crops the user decided are already deleted by the
    audit stage and stay gone until the next run. Only ``.txt`` files are
    touched.

    Args:
        predictions_root: The ``audit_predictions`` tree to prune.
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


# ---------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the audit script."""
    parser = argparse.ArgumentParser(
        description="Run a trained YOLO model over every crop and write audit predictions for label review.",
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
    """Predict boxes for every crop and write the audit prediction files.

    Logging is configured first from ``--log-level`` (stderr plus the appended
    ``<dataset>/audit.log``); the run summary still goes to stdout. Crops are
    inferred in lazily loaded batches of ``--batch-size`` and each batch's
    prediction files are written before the next batch is loaded, so memory
    use stays flat regardless of dataset size.
    """
    args = parse_args()
    dataset = args.dataset.resolve()
    setup_logging(args.log_level, dataset / "audit.log")
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

    targets = scan_images(dataset)
    logger.info("scan: %d crops", len(targets))

    predictions_root = dataset / "audit_predictions"
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
        logger.info("nothing to do: no crops under %s", dataset / "images")
        print(f"nothing to do: no crops under {dataset / 'images'}")
    removed = prune_stale(predictions_root, targets, dataset / "images")
    if removed:
        logger.info("pruned %d stale audit prediction files", removed)
        print(f"pruned {removed} stale audit prediction files")


if __name__ == "__main__":
    main()
