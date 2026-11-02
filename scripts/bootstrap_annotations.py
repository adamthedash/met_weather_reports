#!/usr/bin/env python3
"""Bootstrap annotations: run a YOLO model over the non-annotated digits crops.

Usage:

    pixi run python scripts/bootstrap_annotations.py --model runs/detect/train/weights/best.pt

Scans ``datasets/digits/images`` for crops without a mirrored label file under
``datasets/digits/labels`` (the annotation tool's "not yet annotated"
convention) and runs Ultralytics detection on them with the given weights. One
prediction file per crop is written under ``datasets/digits/predictions``,
mirroring the images tree: each line is ``cls cx cy w h conf`` in normalised
0-1 coordinates (the YOLO label format plus a 6th confidence column), and a
crop with no detections above ``--conf`` gets an empty file (a predicted
background crop).

Prediction files are *not* valid YOLO label files (labels take exactly 5
fields); drop the last column before promoting them into ``labels/``. The
``predictions`` tree is owned by this script: prediction files whose crop has
since been annotated (or removed) are pruned on every run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "digits"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# A predicted box: (cls, cx, cy, w, h, conf) in normalised 0-1 coordinates.
Prediction = tuple[int, float, float, float, float, float]


# ---------------------------------------------------------------------------
# dataset helpers


def load_classes(yaml_path: Path) -> list[str]:
    """Read class names from the int-keyed ``names:`` mapping of a YOLO data.yaml.

    ``data.yaml`` is the single source of truth for classes; nothing falls
    back to defaults. Same contract as ``tools/annotate.py:load_classes``.

    Args:
        yaml_path: Path to the dataset's ``data.yaml``.

    Returns:
        Class names indexed by class id.

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
            raise ValueError(f"{yaml_path}: names key {key!r} is not an integer") from exc
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"{yaml_path}: names[{key!r}] must be a string, got {type(value).__name__}")
        if idx in names:
            raise ValueError(f"{yaml_path}: duplicate names index {idx}")
        names[idx] = str(value)
    return [names[i] for i in sorted(names)]


def scan_dataset(dataset: Path) -> tuple[list[Path], list[Path]]:
    """Return ``(annotated, unlabeled)`` crops under ``dataset/images``.

    A crop is annotated iff the mirrored label file (the image path's ``images``
    tree mapped onto the sibling ``labels`` tree with the suffix changed to
    ``.txt``) exists; an empty label file counts as annotated (an annotated
    background crop). This is the same convention as
    ``scripts/prepare_dataset.find_labeled_images`` and the annotation tool's
    ``done`` flag. Both lists are sorted by path for deterministic output.

    Args:
        dataset: Dataset root containing the ``images`` and ``labels`` trees.

    Returns:
        ``(annotated, unlabeled)`` lists of image paths.

    Raises:
        ValueError: If the ``images`` or ``labels`` tree is missing.
    """
    images_root = dataset / "images"
    labels_root = dataset / "labels"
    if not images_root.is_dir():
        raise ValueError(f"images directory not found: {images_root}")
    if not labels_root.is_dir():
        raise ValueError(f"labels directory not found: {labels_root}")
    annotated: list[Path] = []
    unlabeled: list[Path] = []
    for image in sorted(p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        label = labels_root / image.relative_to(images_root).with_suffix(".txt")
        (annotated if label.is_file() else unlabeled).append(image)
    return annotated, unlabeled


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
    ``tools/annotate.py:write_labels`` plus a 6th confidence column in 0-1. An
    empty `boxes` list writes an empty file, a predicted background crop.

    Args:
        path: Destination file; parent directories are created as needed.
        boxes: Predicted boxes ``(cls, cx, cy, w, h, conf)`` in model order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}" for cls, cx, cy, w, h, conf in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prune_stale(predictions_root: Path, targets: list[Path], images_root: Path) -> int:
    """Delete prediction files whose crop is annotated or gone; return the count.

    Keeps ``predictions/`` owned by this script: after a run every prediction
    file mirrors a currently non-annotated crop, so ``predictions/`` and
    ``labels/`` never overlap. Only ``.txt`` files are touched.

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
    return parser.parse_args()


def main() -> None:
    """Predict boxes for every non-annotated crop and write the prediction files."""
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"model weights not found: {args.model}")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit(f"--conf must be between 0 and 1, got {args.conf}")
    dataset = args.dataset.resolve()
    annotated, targets = scan_dataset(dataset)

    predictions_root = dataset / "predictions"
    if targets:
        classes = load_classes(dataset / "data.yaml")
        model = YOLO(str(args.model))
        check_model_names(model, classes)
        predicted = run_inference(model, targets, args.conf)
        n_boxes = 0
        for image in targets:
            rel = image.relative_to(dataset / "images")
            boxes = predicted[image.resolve()]
            write_prediction(prediction_path_for(rel, predictions_root), boxes)
            n_boxes += len(boxes)
        print(f"{len(targets)} images / {n_boxes} boxes -> {predictions_root}")
    else:
        print(f"nothing to do: all {len(annotated)} crops are annotated")
    removed = prune_stale(predictions_root, targets, dataset / "images")
    if removed:
        print(f"pruned {removed} stale prediction files")


if __name__ == "__main__":
    main()
