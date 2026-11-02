# labeler — combined, extensible annotation tool

The successor to `tools/annotate.py` and `tools/review.py`: one application
covering manual bounding-box labeling and accept/reject review of bootstrap
predictions, structured as a registry of pluggable workflow **stages** so new
pipeline stages (triage, spot-check, merge, correction rounds) can be added
later without touching the shared engine.

Stack: FastAPI + uvicorn backend, Vite + Svelte + TypeScript frontend with an
imperative `<canvas>` box editor. The on-disk dataset layout stays
byte-compatible with the existing pipeline
(`scripts/bootstrap_annotations.py`, `scripts/prepare_dataset.py`,
ultralytics training). The old tools remain runnable during the transition.

Status: the **annotate** stage is complete; the **review** stage (accept/reject
over bootstrap predictions) lands next.

## Layout

| Path | Role |
|---|---|
| `tools/labeler/app.py` | ASGI app: core `/api` routes, stage routers, built frontend at `/` |
| `tools/labeler/main.py` | CLI flags, startup summary, uvicorn bootstrap |
| `tools/labeler/config.py` | `Config`: bind address, dist, dataset roots |
| `tools/labeler/dataset.py` | `Dataset` + helpers: classes, image listing, YOLO label I/O |
| `tools/labeler/errors.py` | `ApiError` (HTTP error with a JSON-safe message) |
| `tools/labeler/web.py` | request helpers shared by the stage routers |
| `tools/labeler/stages/base.py` | stage seam: `Stage` contract + `summaries` |
| `tools/labeler/stages/annotate.py` | annotate stage: queue listing + label I/O |
| `tools/labeler/frontend/` | Vite + Svelte + TypeScript UI, built to `frontend/dist` |
| `tools/labeler/frontend/src/lib/BoxEditor.svelte` | shared canvas engine (draw/move/resize/delete, read-only mode) |
| `tools/labeler/frontend/src/stages/` | workspace components + `registry.ts` (stage id → workspace) |

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
`Ctrl+C` stops the server. CLI flags: `--images`, `--labels`, `--data-yaml`
(dataset paths, defaults under `datasets/digits`), `--host`, `--port`
(default 8125), `--dist` (built frontend directory).

Development, in two terminals:

```sh
just labeler-api    # uvicorn --reload on 127.0.0.1:8125
just labeler-ui     # vite dev server on http://127.0.0.1:5173 (proxies /api)
```

## API contract

Errors are always JSON `{"error": ...}` (400/404/413/500); every response sets
`Cache-Control: no-store` so labeling state is never stale — same contract as
the annotate/review tools.

Core routes:

| Route | Returns |
|---|---|
| `GET /api/health` | `{"ok": true, "version": …}` |
| `GET /api/stages` | `{"stages": [{id, title, description}]}` |
| `GET /api/classes` | `{"classes": [...]}` (from `data.yaml`, strict) |
| `GET /api/image?path=` | raw crop image bytes |
| `GET /api/docs` | OpenAPI explorer |
| `GET /` | built frontend (503 with a hint when `frontend/dist` is missing) |

Annotate stage (mounted at `/api/annotate`):

| Route | Returns |
|---|---|
| `GET /api/annotate/list` | `{"images": [{path, done, has_labels}], "total", "done"}` |
| `GET /api/annotate/labels?path=` | `{"path", "boxes": [{cls, cx, cy, w, h}]}` (normalised) |
| `POST /api/annotate/labels` | `{"path", "boxes"}` → `{"ok", "saved", "has_labels"}`; empty `boxes` marks the crop empty |

Dataset semantics are unchanged: label file exists ⇒ crop done, empty file ⇒
done-but-background, malformed label files are reported (400) never skipped,
saves are clamped to the image and written as 5-column YOLO at 6 decimals.

## Annotate workflow

Same keymap as `tools/annotate.html`: drag draws a box (current class needed),
shift-click/drag selects and moves, handles resize, right-click deletes,
`0-9 . -` assign the class (sticky), `← →`/`a d` navigate, `⏎` save & next,
`x` mark empty, `z` undo, `⌫` delete, `Esc` cancel/deselect, `f` cycles the
view mode (all → annotated → non-annotated), `[ ]` zoom, `Ctrl+S` save.
Autosave 600 ms after the last edit; last crop/zoom/view mode resume from
`localStorage` (`labeler.annotate.*`).

## Adding a stage

A stage (pipeline stage) is a queue of crops + user actions + artifacts
consumed/produced. One stage = one backend module with a `create_stage(ds)`
factory (API router mounted at `/api/<id>`) registered in
`tools/labeler/stages/__init__.py`, plus one entry in
`frontend/src/stages/registry.ts` (workspace component, usually built from the
shared `BoxEditor`). `stages/annotate.py` is the worked example. The shared
engine and core routes stay untouched.
