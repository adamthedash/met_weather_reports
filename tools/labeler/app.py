"""FastAPI application: JSON API under ``/api``, built frontend at ``/``.

Response contract (carried over from the annotate/review tools): every error
is a JSON ``{"error": ...}`` body, and every response sets
``Cache-Control: no-store`` so labeling state is never stale.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from tools.labeler import __version__
from tools.labeler.config import FRONTEND_DIST
from tools.labeler.stages import summaries

ERROR_FALLBACK = "internal server error"


def create_app(dist: Path) -> FastAPI:
    """Build the ASGI app serving the API and the built frontend in `dist`."""
    app = FastAPI(
        title="labeler",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # -- error contract --------------------------------------------------
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        """Turn HTTP errors into the ``{"error": ...}`` JSON shape."""
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

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

    # -- API -------------------------------------------------------------
    @app.get("/api/health")
    async def health() -> dict[str, object]:
        """Liveness probe for the UI and smoke tests."""
        return {"ok": True, "version": __version__}

    @app.get("/api/stages")
    async def stages() -> dict[str, object]:
        """Every registered workflow stage."""
        return {"stages": summaries()}

    # -- frontend --------------------------------------------------------
    root = dist.resolve()

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
            if Path(path).suffix:
                raise HTTPException(404, f"not found: /{path}")
        index = root / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(
            503, f"frontend not built: {index} missing (run 'just labeler-build')"
        )

    return app


# ASGI target for `uvicorn tools.labeler.app:app` (dev server, --reload).
app = create_app(FRONTEND_DIST)
