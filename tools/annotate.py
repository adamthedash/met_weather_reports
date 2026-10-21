#!/usr/bin/env python3
"""Browser-based bounding-box labeler for the handwritten digits dataset.

Usage:

    pixi run python tools/annotate.py

then open the printed URL (http://127.0.0.1:8123 by default; forward the
port first when working over SSH).

The server exposes a small JSON API plus tools/annotate.html. It lists the
cell crops under ``datasets/digits/images``, serves them, and reads/writes
YOLO label files under ``datasets/digits/labels`` mirroring the images tree
(one ``<source-id>/<row>_<col>.txt`` per crop, lines of
``<class> <cx> <cy> <w> <h>`` normalised to 0-1). An empty label file marks
a crop as "done but nothing to label" (ignored/background).

Standard library only: no dependencies beyond the Python in the pixi env.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
FALLBACK_CLASSES = [str(d) for d in range(10)] + ["dash", "dot"]
MAX_BOXES = 64
MAX_BODY = 1_000_000


class ApiError(Exception):
    """An HTTP error with a JSON-safe message."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Config:
    images_root: Path
    labels_root: Path
    classes: list[str]


CFG: Config  # populated in main()


# ---------------------------------------------------------------------------
# dataset helpers


def load_classes(yaml_path: Path) -> list[str]:
    """Parse the ``names:`` mapping of a YOLO data.yaml (int key -> label)."""
    if not yaml_path.is_file():
        return list(FALLBACK_CLASSES)
    names: dict[int, str] = {}
    in_names = False
    for raw in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line[0].isspace():
            in_names = line.strip() == "names:"
            continue
        if not in_names:
            continue
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        try:
            idx = int(key.strip())
        except ValueError:
            continue
        names[idx] = value.strip().strip("\"'")
    if not names:
        return list(FALLBACK_CLASSES)
    return [names[i] for i in sorted(names)]


_NUM_RE = re.compile(r"(\d+)")


def natural_key(rel: str) -> list:
    """Sort key so 10_2.png sorts before 10_10.png."""
    return [int(p) if p.isdigit() else p.lower() for p in _NUM_RE.split(rel)]


def list_images(root: Path) -> list[str]:
    """All image files under `root`, relative posix paths, natural order."""
    rels = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(rels, key=natural_key)


def safe_path(root: Path, rel: str) -> Path:
    """Resolve `rel` under `root`, rejecting traversal and absolute paths."""
    if not rel or "\\" in rel:
        raise ApiError(400, f"bad path: {rel!r}")
    parts = Path(rel)
    if parts.is_absolute() or ".." in parts.parts:
        raise ApiError(400, f"bad path: {rel!r}")
    resolved = (root / parts).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ApiError(400, f"bad path: {rel!r}")
    return resolved


def label_path_for(rel: str) -> Path:
    """labels root mirror of an image path: <stem>.txt in the same subdirs."""
    p = Path(rel)
    return safe_path(CFG.labels_root, (p.parent / (p.stem + ".txt")).as_posix())


def read_labels(rel: str) -> list[dict]:
    boxes: list[dict] = []
    path = label_path_for(rel)
    if not path.is_file():
        return boxes
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ApiError(
                400, f"{path.name} line {lineno}: expected 5 fields, got {len(parts)}"
            )
        cls = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
        if not 0 <= cls < len(CFG.classes):
            raise ApiError(400, f"{path.name} line {lineno}: class {cls} out of range")
        boxes.append({"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def validate_boxes(raw: object) -> list[dict]:
    if not isinstance(raw, list):
        raise ApiError(400, "'boxes' must be a list")
    if len(raw) > MAX_BOXES:
        raise ApiError(400, f"too many boxes (max {MAX_BOXES})")
    boxes: list[dict] = []
    for i, b in enumerate(raw):
        try:
            cls = int(b["cls"])
            cx, cy, w, h = (float(b[k]) for k in ("cx", "cy", "w", "h"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(400, f"box {i}: expected cls, cx, cy, w, h") from exc
        if not 0 <= cls < len(CFG.classes):
            raise ApiError(400, f"box {i}: class {cls} out of range 0..{len(CFG.classes) - 1}")
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            raise ApiError(400, f"box {i}: non-finite coordinates")
        if w <= 0 or h <= 0:
            raise ApiError(400, f"box {i}: width and height must be positive")
        # clamp to the image, keep the box valid
        x1 = min(max(cx - w / 2, 0.0), 1.0)
        y1 = min(max(cy - h / 2, 0.0), 1.0)
        x2 = min(max(cx + w / 2, 0.0), 1.0)
        y2 = min(max(cy + h / 2, 0.0), 1.0)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            raise ApiError(400, f"box {i}: lies outside the image")
        boxes.append(
            {"cls": cls, "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "w": x2 - x1, "h": y2 - y1}
        )
    return boxes


def write_labels(rel: str, boxes: list[dict]) -> Path:
    path = label_path_for(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}" for b in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# HTTP


def one_arg(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key)
    if not values or not values[0]:
        raise ApiError(400, f"missing query parameter {key!r}")
    return values[0]


class Handler(BaseHTTPRequestHandler):
    server_version = "Annotate/1.0"

    # -- helpers ---------------------------------------------------------
    def _bytes(self, status: int, ctype: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, obj: object) -> None:
        self._bytes(status, "application/json; charset=utf-8", json.dumps(obj).encode())

    def _guard(self, fn) -> None:
        try:
            fn()
        except ApiError as exc:
            self._json(exc.status, {"error": exc.message})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001 - last-resort error reporting
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:
        self._guard(self._get)

    def do_POST(self) -> None:
        self._guard(self._post)

    def _get(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        route = parsed.path
        if route in ("/", "/index.html"):
            html = (Path(__file__).resolve().parent / "annotate.html").read_bytes()
            self._bytes(200, "text/html; charset=utf-8", html)
        elif route == "/api/classes":
            self._json(200, {"classes": CFG.classes})
        elif route == "/api/list":
            self._list()
        elif route == "/api/image":
            self._image(one_arg(query, "path"))
        elif route == "/api/labels":
            rel = one_arg(query, "path")
            self._json(200, {"path": rel, "boxes": read_labels(rel)})
        else:
            raise ApiError(404, f"not found: {route}")

    def _post(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/labels":
            raise ApiError(404, f"not found: {parsed.path}")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ApiError(413, "payload too large")
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            raise ApiError(400, "invalid JSON body") from exc
        rel = payload.get("path")
        if not isinstance(rel, str):
            raise ApiError(400, "missing 'path'")
        image = safe_path(CFG.images_root, rel)
        if not image.is_file():
            raise ApiError(404, f"no such image: {rel}")
        boxes = validate_boxes(payload.get("boxes", []))
        write_labels(rel, boxes)
        self._json(200, {"ok": True, "saved": len(boxes), "has_labels": bool(boxes)})

    def _list(self) -> None:
        images = []
        done = 0
        for rel in list_images(CFG.images_root):
            lp = label_path_for(rel)
            exists = lp.is_file()
            done += int(exists)
            images.append(
                {"path": rel, "done": exists, "has_labels": exists and lp.stat().st_size > 0}
            )
        self._json(200, {"images": images, "total": len(images), "done": done})

    def _image(self, rel: str) -> None:
        path = safe_path(CFG.images_root, rel)
        if not path.is_file():
            raise ApiError(404, f"no such image: {rel}")
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._bytes(200, ctype, path.read_bytes())

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.address_string()}] {fmt % args}")


# ---------------------------------------------------------------------------


def main() -> None:
    global CFG
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--images",
        type=Path,
        default=REPO_ROOT / "datasets" / "digits" / "images",
        help="root directory of crop images",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        default=REPO_ROOT / "datasets" / "digits" / "labels",
        help="output root for YOLO label files (mirrors the images tree)",
    )
    ap.add_argument(
        "--data-yaml",
        type=Path,
        default=REPO_ROOT / "datasets" / "digits" / "data.yaml",
        help="YOLO data.yaml used for class names",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()

    images_root = args.images.resolve()
    labels_root = args.labels.resolve()
    if not images_root.is_dir():
        raise SystemExit(f"images root not found: {images_root}")

    CFG = Config(
        images_root=images_root,
        labels_root=labels_root,
        classes=load_classes(args.data_yaml),
    )
    rels = list_images(images_root)
    done = sum(1 for rel in rels if label_path_for(rel).is_file())

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"annotate server: http://{args.host}:{args.port}")
    print(f"  images : {len(rels)} crops under {images_root}")
    print(f"  labels : {labels_root}")
    print(f"  classes: {', '.join(CFG.classes)} ({len(CFG.classes)})")
    print(f"  done   : {done} / {len(rels)}")
    print("open the URL in a browser, Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
