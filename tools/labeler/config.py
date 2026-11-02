"""Runtime configuration for the labeler server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.labeler.dataset import Dataset, load_classes

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
FRONTEND_DIST = PACKAGE_ROOT / "frontend" / "dist"
DEFAULT_IMAGES_ROOT = REPO_ROOT / "datasets" / "digits" / "images"
DEFAULT_LABELS_ROOT = REPO_ROOT / "datasets" / "digits" / "labels"
DEFAULT_DATA_YAML = REPO_ROOT / "datasets" / "digits" / "data.yaml"


@dataclass(frozen=True)
class Config:
    """Server settings: bind address, built frontend, and dataset roots."""

    host: str = "127.0.0.1"
    port: int = 8125
    dist: Path = FRONTEND_DIST
    images_root: Path = DEFAULT_IMAGES_ROOT
    labels_root: Path = DEFAULT_LABELS_ROOT
    data_yaml: Path = DEFAULT_DATA_YAML

    def dataset(self) -> Dataset:
        """Load the dataset: resolve the roots and parse the class names.

        Strict by design (as in tools/annotate.py): the dataset fully defines
        the tool, so a missing images root or unusable data.yaml aborts
        startup instead of falling back to defaults.

        Raises:
            ValueError: If the images root is missing or `data.yaml` is
                missing/invalid.
        """
        images_root = self.images_root.resolve()
        if not images_root.is_dir():
            raise ValueError(f"images root not found: {images_root}")
        return Dataset(
            images_root=images_root,
            labels_root=self.labels_root.resolve(),
            classes=tuple(load_classes(self.data_yaml)),
        )
