"""Runtime configuration for the labeler server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.labeler.dataset import Dataset, load_class_map, load_classes

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
FRONTEND_DIST = PACKAGE_ROOT / "frontend" / "dist"
DEFAULT_IMAGES_ROOT = REPO_ROOT / "datasets" / "digits" / "images"
DEFAULT_LABELS_ROOT = REPO_ROOT / "datasets" / "digits" / "labels"
DEFAULT_MANUAL_ROOT = REPO_ROOT / "datasets" / "digits" / "manual"
DEFAULT_PREDICTIONS_ROOT = REPO_ROOT / "datasets" / "digits" / "predictions"
DEFAULT_AUDIT_PREDICTIONS_ROOT = REPO_ROOT / "datasets" / "digits" / "audit_predictions"
DEFAULT_DATA_YAML = REPO_ROOT / "datasets" / "digits" / "data.yaml"


@dataclass(frozen=True)
class Config:
    """Server settings: bind address, built frontend, and dataset roots."""

    host: str = "127.0.0.1"
    port: int = 8125
    dist: Path = FRONTEND_DIST
    images_root: Path = DEFAULT_IMAGES_ROOT
    labels_root: Path = DEFAULT_LABELS_ROOT
    manual_root: Path = DEFAULT_MANUAL_ROOT
    predictions_root: Path = DEFAULT_PREDICTIONS_ROOT
    audit_predictions_root: Path = DEFAULT_AUDIT_PREDICTIONS_ROOT
    data_yaml: Path = DEFAULT_DATA_YAML
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
            images_root=images_root,
            labels_root=self.labels_root.resolve(),
            manual_root=self.manual_root.resolve(),
            predictions_root=self.predictions_root.resolve(),
            audit_root=self.audit_predictions_root.resolve(),
            classes=classes,
            display=tuple(load_class_map(class_map, classes)),
        )
