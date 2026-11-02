"""Runtime configuration for the labeler server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = PACKAGE_ROOT / "frontend" / "dist"


@dataclass(frozen=True)
class Config:
    """Server settings: bind address and the built-frontend directory."""

    host: str = "127.0.0.1"
    port: int = 8125
    dist: Path = FRONTEND_DIST
