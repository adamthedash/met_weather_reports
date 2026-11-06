"""Runtime configuration for the labeler server.

A *dataset* is one directory holding the standard layout — ``images/`` plus
the mirrored ``labels/``, ``manual/``, ``predictions/`` and
``audit_predictions/`` trees, and ``data.yaml`` — so pointing the tool at
another dataset is one flag (or ``LABELER_DATASET`` for entry points that
cannot pass flags). The individual roots stay overridable for non-standard
layouts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from tools.labeler.dataset import Dataset, load_class_map, load_classes

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
FRONTEND_DIST = PACKAGE_ROOT / "frontend" / "dist"
# environment variable selecting the dataset for entry points that cannot
# pass CLI flags (e.g. `uvicorn tools.labeler.app:app`, the dev server)
DATASET_ENV = "LABELER_DATASET"
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "digits"


def default_dataset() -> Path:
    """The dataset root when no flag says otherwise.

    ``$LABELER_DATASET`` when set (relative paths resolve against the working
    directory), else ``datasets/digits``.
    """
    env = os.environ.get(DATASET_ENV)
    return Path(env) if env else DEFAULT_DATASET_DIR


def dataset_paths(dataset: Path) -> dict[str, Path]:
    """The standard layout under dataset root `dataset`.

    Returns the sidecar roots and ``data.yaml`` keyed by the
    :class:`Config` field names they feed (``images_root``, ``labels_root``,
    ``manual_root``, ``predictions_root``, ``audit_predictions_root``,
    ``data_yaml``).
    """
    return {
        "images_root": dataset / "images",
        "labels_root": dataset / "labels",
        "manual_root": dataset / "manual",
        "predictions_root": dataset / "predictions",
        "audit_predictions_root": dataset / "audit_predictions",
        "data_yaml": dataset / "data.yaml",
    }


@dataclass(frozen=True)
class Config:
    """Server settings: bind address, built frontend, and dataset roots.

    The root defaults are the standard layout under :func:`default_dataset`;
    each root is individually overridable for non-standard layouts.
    """

    host: str = "127.0.0.1"
    port: int = 8125
    dist: Path = FRONTEND_DIST
    images_root: Path = field(default_factory=lambda: default_dataset() / "images")
    labels_root: Path = field(default_factory=lambda: default_dataset() / "labels")
    manual_root: Path = field(default_factory=lambda: default_dataset() / "manual")
    predictions_root: Path = field(
        default_factory=lambda: default_dataset() / "predictions"
    )
    audit_predictions_root: Path = field(
        default_factory=lambda: default_dataset() / "audit_predictions"
    )
    data_yaml: Path = field(default_factory=lambda: default_dataset() / "data.yaml")
    class_map: Path | None = None

    def dataset(self) -> Dataset:
        """Load the dataset: resolve the roots, parse the class names and presentation.

        Strict by design: the dataset fully defines
        the tool, so a missing images root or unusable data.yaml/classes.yaml
        aborts startup instead of falling back to defaults. The class
        presentation map defaults to ``classes.yaml`` alongside `data_yaml`
        (``class_map`` overrides it).

        Raises:
            ValueError: If the images root is missing, `data.yaml` is
                missing/invalid, or ``classes.yaml`` is missing/invalid or
                does not match the class names.
        """
        images_root = self.images_root.resolve()
        if not images_root.is_dir():
            raise ValueError(f"images root not found: {images_root}")
        classes = tuple(load_classes(self.data_yaml))
        class_map = (
            self.class_map
            if self.class_map is not None
            else self.data_yaml.parent / "classes.yaml"
        )
        return Dataset(
            id=self.data_yaml.parent.name or "dataset",
            images_root=images_root,
            labels_root=self.labels_root.resolve(),
            manual_root=self.manual_root.resolve(),
            predictions_root=self.predictions_root.resolve(),
            audit_root=self.audit_predictions_root.resolve(),
            classes=classes,
            display=tuple(load_class_map(class_map, classes)),
        )
