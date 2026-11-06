"""CLI entry point: parse flags, load the dataset, print a summary, serve."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from tools.labeler.app import create_app
from tools.labeler.config import (
    FRONTEND_DIST,
    Config,
    default_dataset,
    dataset_paths,
)


def main() -> None:
    """Parse CLI flags and run the labeler server until interrupted."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="dataset root directory: derives the images/labels/manual/"
        "predictions/audit_predictions roots and data.yaml from its standard "
        "layout (default: $LABELER_DATASET or datasets/digits)",
    )
    ap.add_argument(
        "--images",
        type=Path,
        default=None,
        help="root directory of crop images (default: <dataset>/images)",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="output root for YOLO label files (mirrors the images tree; "
        "default: <dataset>/labels)",
    )
    ap.add_argument(
        "--manual",
        type=Path,
        default=None,
        help="root of the manual-queue marker files (mirrors the images tree; "
        "default: <dataset>/manual)",
    )
    ap.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="root of the prediction files to review (mirrors the images tree; "
        "default: <dataset>/predictions)",
    )
    ap.add_argument(
        "--audit-predictions",
        type=Path,
        default=None,
        help="root of the audit prediction files (mirrors the images tree; "
        "default: <dataset>/audit_predictions)",
    )
    ap.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="YOLO data.yaml defining the class names (must exist and parse; "
        "default: <dataset>/data.yaml)",
    )
    ap.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="classes.yaml defining the display glyph and keyboard shortcut "
        "of every class (default: classes.yaml next to --data-yaml)",
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

    dataset = args.dataset if args.dataset is not None else default_dataset()
    roots = dataset_paths(dataset)
    cfg = Config(
        host=args.host,
        port=args.port,
        dist=args.dist,
        images_root=args.images or roots["images_root"],
        labels_root=args.labels or roots["labels_root"],
        manual_root=args.manual or roots["manual_root"],
        predictions_root=args.predictions or roots["predictions_root"],
        audit_predictions_root=args.audit_predictions
        or roots["audit_predictions_root"],
        data_yaml=args.data_yaml or roots["data_yaml"],
        class_map=args.class_map,
    )
    # strict by design: a missing images root or bad data.yaml/classes.yaml
    # aborts startup with a clear error instead of running against a
    # half-configured dataset
    try:
        ds = cfg.dataset()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    app = create_app(cfg, ds)

    rels = ds.list_images()
    done = sum(1 for rel in rels if ds.label_path_for(rel).is_file())
    manual = sum(1 for rel in rels if ds.is_manual(rel))
    queue, stale = ds.queue_and_stale()
    disagreements, compared = ds.audit_queue()
    dist_built = (cfg.dist / "index.html").is_file()
    print(f"labeler server: http://{cfg.host}:{cfg.port}")
    print(f"  dataset: {dataset}")
    print(f"  images : {len(rels)} crops under {ds.images_root}")
    print(f"  labels : {ds.labels_root}")
    print(f"  classes: {', '.join(ds.classes)} ({len(ds.classes)})")
    print(
        "  keys   : "
        + ", ".join(f"{d.name}={d.key}" for d in ds.display if d.key is not None)
    )
    print(f"  done   : {done} / {len(rels)}")
    print(f"  manual : {manual} flagged for manual adjustment")
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
