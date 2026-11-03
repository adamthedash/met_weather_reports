"""CLI entry point: parse flags, load the dataset, print a summary, serve."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from tools.labeler.app import create_app
from tools.labeler.config import (
    DEFAULT_AUDIT_PREDICTIONS_ROOT,
    DEFAULT_DATA_YAML,
    DEFAULT_IMAGES_ROOT,
    DEFAULT_LABELS_ROOT,
    DEFAULT_PREDICTIONS_ROOT,
    FRONTEND_DIST,
    Config,
)


def main() -> None:
    """Parse CLI flags and run the labeler server until interrupted."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES_ROOT,
        help="root directory of crop images",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_ROOT,
        help="output root for YOLO label files (mirrors the images tree)",
    )
    ap.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_ROOT,
        help="root of the prediction files to review (mirrors the images tree)",
    )
    ap.add_argument(
        "--audit-predictions",
        type=Path,
        default=DEFAULT_AUDIT_PREDICTIONS_ROOT,
        help="root of the audit prediction files (mirrors the images tree)",
    )
    ap.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="YOLO data.yaml defining the class names (must exist and parse)",
    )
    ap.add_argument("--host", default="127.0.0.1", help="bind address")
    ap.add_argument("--port", type=int, default=8125, help="port")
    ap.add_argument(
        "--dist",
        type=Path,
        default=FRONTEND_DIST,
        help="built frontend directory (tools/labeler/frontend/dist)",
    )
    args = ap.parse_args()

    cfg = Config(
        host=args.host,
        port=args.port,
        dist=args.dist,
        images_root=args.images,
        labels_root=args.labels,
        predictions_root=args.predictions,
        audit_predictions_root=args.audit_predictions,
        data_yaml=args.data_yaml,
    )
    # strict by design: a missing images root or bad data.yaml aborts startup
    # with a clear error instead of running against a half-configured dataset
    try:
        ds = cfg.dataset()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    app = create_app(cfg, ds)

    rels = ds.list_images()
    done = sum(1 for rel in rels if ds.label_path_for(rel).is_file())
    queue, stale = ds.queue_and_stale()
    disagreements, compared = ds.audit_queue()
    dist_built = (cfg.dist / "index.html").is_file()
    print(f"labeler server: http://{cfg.host}:{cfg.port}")
    print(f"  images : {len(rels)} crops under {ds.images_root}")
    print(f"  labels : {ds.labels_root}")
    print(f"  classes: {', '.join(ds.classes)} ({len(ds.classes)})")
    print(f"  done   : {done} / {len(rels)}")
    print(
        f"  queue  : {len(queue)} predictions to review"
        + (f" ({stale} stale)" if stale else "")
    )
    print(f"  audit  : {len(disagreements)} disagreements of {compared} compared")
    print(f"  stages : {', '.join(s.id for s in app.state.stages)}")
    print(
        f"  dist   : {cfg.dist} "
        f"({'built' if dist_built else 'MISSING (run just labeler-build)'})"
    )
    print("open the URL in a browser, Ctrl+C to stop")
    try:
        uvicorn.run(app, host=cfg.host, port=cfg.port)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
