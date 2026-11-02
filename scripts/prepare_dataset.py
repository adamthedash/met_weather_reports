#!/usr/bin/env python3
"""Generate the YOLO train/val split files for the handwritten digits dataset.

Usage:

    pixi run python scripts/prepare_dataset.py

Scans ``datasets/digits/images`` for crops with a mirrored label file under
``datasets/digits/labels`` and splits them deterministically into
``datasets/digits/train.txt`` and ``datasets/digits/val.txt``, one image path
per line. These two txt files are the only files written; everything else in
the dataset is left untouched.

Images without a label file are skipped: per the annotation tool's convention
they are not yet labeled and must not be trained as background crops.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "digits"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def find_labeled_images(dataset: Path) -> list[Path]:
    """Return the images under ``dataset/images`` that have a mirrored label file.

    Mirrors the Ultralytics label lookup: the image path's ``images`` tree is
    mapped onto the sibling ``labels`` tree with the suffix changed to ``.txt``.
    An empty label file counts as labeled (an annotated background crop).

    Args:
        dataset: Dataset root containing ``images`` and ``labels`` trees.

    Returns:
        Image paths sorted by their path for deterministic downstream splitting.

    Raises:
        ValueError: If the ``images`` or ``labels`` tree is missing, if no
            labeled images are found, or if a label file has no matching image.
    """
    images_root = dataset / "images"
    labels_root = dataset / "labels"
    if not images_root.is_dir():
        raise ValueError(f"images directory not found: {images_root}")
    if not labels_root.is_dir():
        raise ValueError(f"labels directory not found: {labels_root}")

    images = sorted(p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    image_stems = {(p.parent.relative_to(images_root), p.stem) for p in images}
    orphans = [
        p
        for p in sorted(labels_root.rglob("*.txt"))
        if (p.parent.relative_to(labels_root), p.stem) not in image_stems
    ]
    if orphans:
        raise ValueError(f"label files without a matching image: {', '.join(str(p) for p in orphans)}")

    labeled = [p for p in images if (labels_root / p.relative_to(images_root).with_suffix(".txt")).is_file()]
    if not labeled:
        raise ValueError(f"no labeled images found under {images_root}")
    return labeled


def train_val_split(images: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Split images deterministically into train and val lists.

    Args:
        images: Labeled image paths to split.
        val_fraction: Fraction of images to place in the validation split.
        seed: Seed for the shuffle; the same seed always yields the same split.

    Returns:
        ``(train, val)`` lists where train has ``int((1 - val_fraction) * n)``
        images and val takes the remainder.

    Raises:
        ValueError: If ``val_fraction`` is not strictly between 0 and 1, or the
            dataset is too small to give both splits at least one image.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be strictly between 0 and 1, got {val_fraction}")
    shuffled = sorted(images, key=lambda p: p.as_posix())
    random.Random(seed).shuffle(shuffled)
    n_train = int((1 - val_fraction) * len(shuffled))
    if n_train == 0 or n_train == len(shuffled):
        raise ValueError(f"dataset of {len(shuffled)} images is too small to split into train and val")
    return shuffled[:n_train], shuffled[n_train:]


def write_split(txt: Path, images: list[Path], dataset: Path) -> None:
    """Write image paths to ``txt``, one per line, as ``./``-relative POSIX paths.

    The ``./`` prefix makes Ultralytics resolve each entry relative to the txt
    file's own directory, so the dataset loads regardless of the working
    directory used for training.

    Args:
        txt: Destination txt file, overwritten if it already exists.
        images: Image paths to list; every path must live under ``dataset``.
        dataset: Dataset root that the txt file lives in.

    Raises:
        ValueError: If an image is not inside ``dataset``.
    """
    lines = []
    for img in images:
        try:
            rel = img.relative_to(dataset)
        except ValueError as exc:
            raise ValueError(f"{img} is not inside {dataset}") from exc
        lines.append("./" + rel.as_posix())
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the split generator."""
    parser = argparse.ArgumentParser(
        description="Generate train.txt and val.txt listing labeled images of the digits dataset.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR, help="dataset root (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed for a reproducible split (default: %(default)s)")
    parser.add_argument(
        "--val-fraction", type=float, default=0.2, help="fraction of labeled images for val (default: %(default)s)"
    )
    return parser.parse_args()


def main() -> None:
    """Build and write train.txt and val.txt, then print a summary."""
    args = parse_args()
    dataset = args.dataset.resolve()
    images = find_labeled_images(dataset)
    train, val = train_val_split(images, args.val_fraction, args.seed)
    write_split(dataset / "train.txt", train, dataset)
    write_split(dataset / "val.txt", val, dataset)
    print(f"{len(train)} train / {len(val)} val -> {dataset / 'train.txt'}, {dataset / 'val.txt'}")


if __name__ == "__main__":
    main()
