"""CLI entry point: parse flags, print a startup summary, serve until Ctrl+C."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from tools.labeler.app import create_app
from tools.labeler.config import FRONTEND_DIST, Config
from tools.labeler.stages import summaries


def main() -> None:
    """Parse CLI flags and run the labeler server until interrupted."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
    cfg = Config(host=args.host, port=args.port, dist=args.dist.resolve())

    app = create_app(cfg.dist)
    print(f"labeler server: http://{cfg.host}:{cfg.port}")
    built = "built" if (cfg.dist / "index.html").is_file() else "MISSING (run 'just labeler-build')"
    print(f"  dist  : {cfg.dist} ({built})")
    names = ", ".join(s["id"] for s in summaries()) or "(none registered yet)"
    print(f"  stages: {names}")
    print("open the URL in a browser, Ctrl+C to stop")
    try:
        uvicorn.run(app, host=cfg.host, port=cfg.port)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
