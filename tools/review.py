#!/usr/bin/env python3
"""Browser-based accept/reject review for the bootstrap prediction files.

Usage:

    pixi run python tools/review.py

then open the printed URL (http://127.0.0.1:8124 by default; forward the
port first when working over SSH).

The server exposes a small JSON API plus tools/review.html. It lists the
prediction files written by ``scripts/bootstrap_annotations.py`` under
``datasets/digits/predictions`` for crops that are not yet annotated, shows
each crop with its predicted boxes and confidence scores, and turns an
accepted prediction into a YOLO label file under ``datasets/digits/labels``
(the confidence column dropped, byte-compatible with tools/annotate.py).
Rejecting discards the prediction file only: the crop stays unannotated and
returns to the manual queue of tools/annotate.py.

Standard library for the server and file I/O; PyYAML (declared in pixi.toml)
is the only third-party dependency and is used (via tools/annotate.py) to
parse data.yaml.
"""

from __future__ import annotations

import argparse
import json
import math
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TypedDict
from urllib.parse import parse_qs, urlparse

import annotate

REPO_ROOT = annotate.REPO_ROOT

PREDICTIONS_ROOT: Path  # populated in main()


class PredBox(TypedDict):
    """A predicted YOLO bounding box in normalised 0-1 coordinates."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float


# ---------------------------------------------------------------------------
# prediction helpers


def prediction_path_for(rel: str) -> Path:
    """Predictions root mirror of an image path: <stem>.txt in the same subdirs.

    Same contract as ``scripts/bootstrap_annotations.py:prediction_path_for``.

    Raises:
        ApiError: If `rel` escapes the predictions root.
    """
    p = Path(rel)
    return annotate.safe_path(PREDICTIONS_ROOT, (p.parent / (p.stem + ".txt")).as_posix())


def read_prediction(rel: str) -> list[PredBox]:
    """Read the prediction file mirroring `rel`; empty for a predicted background.

    Lines hold six fields (``cls cx cy w h conf``, normalised 0-1) as written
    by ``scripts/bootstrap_annotations.py:write_prediction``. Parsing is
    strict: a malformed line is reported, never skipped, so corruption can not
    slip into the dataset as a label.

    Raises:
        ApiError: If the prediction is missing, a line has the wrong field
            count or non-numeric fields, a class id is out of range, or a
            value is non-finite or outside 0-1.
    """
    path = prediction_path_for(rel)
    if not path.is_file():
        raise annotate.ApiError(404, f"no such prediction: {rel}")
    boxes: list[PredBox] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 6:
            raise annotate.ApiError(
                400, f"{path.name} line {lineno}: expected 6 fields, got {len(parts)}"
            )
        try:
            cls = int(parts[0])
            cx, cy, w, h, conf = (float(v) for v in parts[1:])
        except ValueError as exc:
            raise annotate.ApiError(400, f"{path.name} line {lineno}: non-numeric field") from exc
        if not 0 <= cls < len(annotate.CFG.classes):
            raise annotate.ApiError(400, f"{path.name} line {lineno}: class {cls} out of range")
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (cx, cy, w, h, conf)):
            raise annotate.ApiError(
                400, f"{path.name} line {lineno}: values must be finite and in 0-1"
            )
        boxes.append({"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf})
    return boxes


def queue_and_stale() -> tuple[list[dict[str, object]], int]:
    """Return the review queue and the number of stale prediction files.

    The queue holds every crop that has a prediction file and no label file,
    in natural order — exactly the crops the last bootstrap run predicted.
    Prediction files whose crop is annotated or gone are stale (pruned by
    ``scripts/bootstrap_annotations.py`` on its next run) and never queue.

    Each entry is ``{"path", "boxes", "empty", "error"}``: `boxes` counts the
    predicted boxes (None for a malformed file), `empty` flags a predicted
    background crop, and `error` carries the parse message of a malformed
    file (None otherwise).

    Returns:
        The ``(queue, stale_count)`` pair.
    """
    queue: list[dict[str, object]] = []
    queued: set[str] = set()
    for rel in annotate.list_images(annotate.CFG.images_root):
        if annotate.label_path_for(rel).is_file():
            continue  # already annotated (possibly since the bootstrap run)
        if not prediction_path_for(rel).is_file():
            continue  # no prediction for this crop
        queued.add(rel)
        try:
            boxes = read_prediction(rel)
            entry: dict[str, object] = {
                "path": rel,
                "boxes": len(boxes),
                "empty": not boxes,
                "error": None,
            }
        except annotate.ApiError as exc:
            entry = {"path": rel, "boxes": None, "empty": False, "error": exc.message}
        queue.append(entry)
    on_disk = {
        p.relative_to(PREDICTIONS_ROOT).with_suffix("").as_posix()
        for p in PREDICTIONS_ROOT.rglob("*.txt")
    }
    queued_stems = {Path(rel).with_suffix("").as_posix() for rel in queued}
    return queue, len(on_disk - queued_stems)


# ---------------------------------------------------------------------------
# HTTP


class Handler(annotate.Handler):
    """HTTP routes for the review UI and its JSON API.

    Inherits the response helpers, image serving, and error guard of the
    annotate server; only the routes differ.
    """

    server_version = "Review/1.0"

    def _get(self) -> None:
        """Route a GET request to the UI page or one of the /api endpoints."""
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        route = parsed.path
        if route in ("/", "/index.html"):
            html = (Path(__file__).resolve().parent / "review.html").read_bytes()
            self._bytes(200, "text/html; charset=utf-8", html)
        elif route == "/api/classes":
            self._json(200, {"classes": annotate.CFG.classes})
        elif route == "/api/list":
            self._list()
        elif route == "/api/image":
            self._image(annotate.one_arg(query, "path"))
        elif route == "/api/prediction":
            rel = annotate.one_arg(query, "path")
            self._json(200, {"path": rel, "boxes": read_prediction(rel)})
        else:
            raise annotate.ApiError(404, f"not found: {route}")

    def _post(self) -> None:
        """Handle POST /api/decision: apply an accept/reject for one prediction."""
        parsed = urlparse(self.path)
        if parsed.path != "/api/decision":
            raise annotate.ApiError(404, f"not found: {parsed.path}")
        length = int(self.headers.get("Content-Length") or 0)
        if length > annotate.MAX_BODY:
            raise annotate.ApiError(413, "payload too large")
        # an empty body is treated as {}; the missing-field checks below then
        # turn it into a clear 400 rather than a JSON syntax error
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            raise annotate.ApiError(400, "invalid JSON body") from exc
        rel = payload.get("path")
        if not isinstance(rel, str):
            raise annotate.ApiError(400, "missing 'path'")
        accept = payload.get("accept")
        if not isinstance(accept, bool):
            raise annotate.ApiError(400, "missing 'accept'")
        self._decide(rel, accept)

    def _list(self) -> None:
        """Respond with the review queue and the stale prediction count."""
        queue, stale = queue_and_stale()
        self._json(
            200,
            {
                "predictions": queue,
                "total": len(queue) + stale,
                "remaining": len(queue),
            },
        )

    def _decide(self, rel: str, accept: bool) -> None:
        """Apply an accept/reject decision for the prediction mirroring `rel`.

        Accept writes the label file first and deletes the prediction second,
        so an interrupted accept leaves a stale prediction (pruned by the next
        bootstrap run) instead of losing a label. An existing label file is
        never overwritten: the crop was annotated in the meantime, so only the
        stale prediction is dropped. Reject just discards the prediction; the
        crop stays unannotated.

        Raises:
            ApiError: If the prediction is missing, or accept is refused for a
                malformed prediction or a missing crop image.
        """
        pred = prediction_path_for(rel)
        if not pred.is_file():
            raise annotate.ApiError(404, f"no such prediction: {rel}")
        label = annotate.label_path_for(rel)
        already = label.is_file()
        saved = 0
        if accept and not already:
            image = annotate.safe_path(annotate.CFG.images_root, rel)
            if not image.is_file():
                raise annotate.ApiError(404, f"no such image: {rel}")
            boxes = read_prediction(rel)  # strict: refuses malformed files
            annotate.write_labels(
                rel,
                [{"cls": b["cls"], "cx": b["cx"], "cy": b["cy"], "w": b["w"], "h": b["h"]} for b in boxes],
            )
            saved = len(boxes)
        pred.unlink()
        queue, _stale = queue_and_stale()
        self._json(
            200,
            {
                "ok": True,
                "accepted": accept,
                "saved": saved,
                "already_annotated": already,
                "remaining": len(queue),
            },
        )


# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI flags, configure the dataset, and serve until interrupted."""
    global PREDICTIONS_ROOT
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
        help="output root for accepted YOLO label files (mirrors the images tree)",
    )
    ap.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT / "datasets" / "digits" / "predictions",
        help="root of the prediction files to review (mirrors the images tree)",
    )
    ap.add_argument(
        "--data-yaml",
        type=Path,
        default=REPO_ROOT / "datasets" / "digits" / "data.yaml",
        help="YOLO data.yaml defining the class names (must exist and parse)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8124)
    args = ap.parse_args()

    images_root = args.images.resolve()
    labels_root = args.labels.resolve()
    predictions_root = args.predictions.resolve()
    if not images_root.is_dir():
        raise SystemExit(f"images root not found: {images_root}")
    try:
        classes = annotate.load_classes(args.data_yaml)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # the dataset helpers in tools/annotate.py resolve label paths through
    # annotate.CFG, so point it at this tool's roots and reuse them unchanged
    annotate.CFG = annotate.Config(
        images_root=images_root, labels_root=labels_root, classes=classes
    )
    PREDICTIONS_ROOT = predictions_root

    queue, stale = queue_and_stale()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"review server: http://{args.host}:{args.port}")
    print(f"  queue  : {len(queue)} predictions to review under {predictions_root}")
    print(f"  stale  : {stale} (annotated or gone; pruned by the next bootstrap run)")
    print(f"  labels : {labels_root}")
    print(f"  classes: {', '.join(classes)} ({len(classes)})")
    print("open the URL in a browser, Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
