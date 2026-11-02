# labeler — combined, extensible annotation tool

The successor to `tools/annotate.py` and `tools/review.py`: one application
covering manual bounding-box labeling and accept/reject review of bootstrap
predictions, structured as a registry of pluggable workflow **stages** so new
pipeline stages (triage, spot-check, merge, correction rounds) can be added
later without touching the shared engine.

Stack: FastAPI + uvicorn backend, Vite + Svelte + TypeScript frontend with an
imperative `<canvas>` box editor (coming with the shared engine). The on-disk
dataset layout stays byte-compatible with the existing pipeline
(`scripts/bootstrap_annotations.py`, `scripts/prepare_dataset.py`,
ultralytics training). The old tools remain runnable during the transition.

## Layout

| Path | Role |
|---|---|
| `tools/labeler/` | Python package `tools.labeler` (FastAPI backend) |
| `tools/labeler/app.py` | ASGI app: `/api/*` JSON API + built frontend at `/` |
| `tools/labeler/main.py` | CLI flags, startup summary, uvicorn bootstrap |
| `tools/labeler/config.py` | `Config` (host/port/dist) |
| `tools/labeler/stages/base.py` | stage seam: `Stage` registry contract |
| `tools/labeler/frontend/` | Vite + Svelte + TypeScript UI, built to `frontend/dist` |

## Toolchains

- **Python**: the pixi environment at the repo root (adds `fastapi`, `uvicorn`
  to `pixi.toml`; PyYAML stays for `data.yaml` parsing).
- **Node.js**: ≥ 20 on `PATH` with pnpm (tested with Node 24 / pnpm 11) for
  the Vite build. Not managed by pixi; `pnpm-lock.yaml` is tracked.

## Quick start

```sh
just labeler        # pnpm install + vite build, then serve on http://127.0.0.1:8125
```

Open the URL in a browser (forward the port first when working over SSH).
`Ctrl+C` stops the server. CLI flags: `--host`, `--port` (default 8125),
`--dist` (built frontend directory).

Development, in two terminals:

```sh
just labeler-api    # uvicorn --reload on 127.0.0.1:8125
just labeler-ui     # vite dev server on http://127.0.0.1:5173 (proxies /api)
```

## API contract

Errors are always JSON `{"error": ...}` (400/404/413/500); every response sets
`Cache-Control: no-store` so labeling state is never stale — same contract as
the annotate/review tools. Routes so far:

| Route | Returns |
|---|---|
| `GET /api/health` | `{"ok": true, "version": …}` |
| `GET /api/stages` | `{"stages": [{id, title, description}]}` |
| `GET /api/docs` | OpenAPI explorer |
| `GET /` | built frontend (503 with a hint when `frontend/dist` is missing) |

## Adding a stage

A stage (pipeline stage) is a queue of crops + user actions + artifacts
consumed/produced. One stage = one backend module registering a `Stage` in
`tools/labeler/stages/` (API router mounted at `/api/<id>`) + one entry in
`frontend/src/stages/registry.ts` (workspace component + keymap). The shared
engine (canvas box editor, queue navigation, autosave) and core routes stay
untouched. The annotate and review stages land next and serve as the two
worked examples.
