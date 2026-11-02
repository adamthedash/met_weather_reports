"""FastAPI application: JSON API under ``/api``, built frontend at ``/``.

Response contract (carried over from the annotate/review tools): every error
is a JSON ``{"error": ...}`` body, and every response sets
``Cache-Control: no-store`` so labeling state is never stale.
"""

from __future__ import annotations

import mimetypes

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from tools.labeler import __version__
from tools.labeler.config import FRONTEND_DIST, Config
from tools.labeler.dataset import Dataset
from tools.labeler.errors import ApiError
from tools.labeler.stages import build_stages, summaries
from tools.labeler.web import require_arg

ERROR_FALLBACK = "internal server error"


def create_app(cfg: Config, ds: Dataset | None = None) -> FastAPI:
    """Build the ASGI app: core routes, stage routers, and the built frontend."""
    ds = ds or cfg.dataset()
    stages = build_stages(ds)

    app = FastAPI(
        title="labeler",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.dataset = ds
    app.state.stages = stages

    # -- error contract --------------------------------------------------
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        """Turn tool errors into the ``{"error": ...}`` JSON shape."""
        return JSONResponse({"error": exc.message}, status_code=exc.status)

    @app.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError) -> JSONResponse:
        """Bad input raises ValueError in a few parsers; report it as a 400."""
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        """Turn framework errors into the ``{"error": ...}`` JSON shape."""
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        """Last-resort handler: never drop the connection on a crash."""
        return JSONResponse(
            {"error": f"{type(exc).__name__}: {str(exc) or ERROR_FALLBACK}"},
            status_code=500,
        )

    # -- no stale state --------------------------------------------------
    @app.middleware("http")
    async def no_store(request: Request, call_next):
        """Labeling state must never be served from a cache."""
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    # -- core API --------------------------------------------------------
    @app.get("/api/health")
    async def health() -> dict[str, object]:
        """Liveness probe for the UI and smoke checks."""
        return {"ok": True, "version": __version__}

    @app.get("/api/stages")
    async def stage_list() -> dict[str, object]:
        """Every registered workflow stage."""
        return {"stages": summaries(stages)}

    @app.get("/api/classes")
    async def classes() -> dict[str, object]:
        """The dataset's class names (data.yaml is the single source of truth)."""
        return {"classes": list(ds.classes)}

    @app.get("/api/image")
    async def image(path: str | None = Query(default=None)) -> FileResponse:
        """Stream the crop image at `path` with its guessed media type."""
        rel = require_arg(path, "path")
        file = ds.image_path(rel)
        if not file.is_file():
            raise ApiError(404, f"no such image: {rel}")
        ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        return FileResponse(file, media_type=ctype)

    # -- stage routers (the extension seam) ------------------------------
    for stage in stages:
        app.include_router(stage.router, prefix=f"/api/{stage.id}")

    # -- frontend --------------------------------------------------------
    root = cfg.dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> FileResponse:
        """Serve a built asset, falling back to index.html for SPA routes.

        Raises:
            HTTPException: 404 for unknown API routes and missing files
                (never an HTML shell with 200), 503 when the frontend has not
                been built yet.
        """
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, f"not found: /{path}")
        if path:
            candidate = (root / path).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                return FileResponse(candidate)
            # a request for a missing file is a 404; extension-less paths are
            # SPA routes and fall through to index.html
            if "." in path.rsplit("/", 1)[-1]:
                raise HTTPException(404, f"not found: /{path}")
        index = root / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(
            503, f"frontend not built: {index} missing (run 'just labeler-build')"
        )

    return app


# ASGI target for `uvicorn tools.labeler.app:app` (dev server, --reload).
app = create_app(Config())
