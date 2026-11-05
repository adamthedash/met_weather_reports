# labeler — combined, extensible annotation tool

One application for the dataset's annotation workflows: manual bounding-box
labeling (**annotate**), accept/reject review of bootstrap predictions
(**review**), and model-vs-label disagreement resolution (**audit**). It is
structured as a registry of pluggable workflow **stages**
so new pipeline stages (triage, spot-check, merge, correction rounds) can be
added later without touching the shared engine.

Stack: FastAPI + uvicorn backend, Vite + Svelte + TypeScript frontend with an
imperative `<canvas>` box editor. The on-disk dataset layout is byte-compatible
with the surrounding pipeline (`scripts/bootstrap_annotations.py`,
`scripts/prepare_dataset.py`, ultralytics training).

## Quick start

```sh
just labeler        # pnpm install + vite build, then serve on http://127.0.0.1:8125
```

Open the URL in a browser (forward the port first when working over SSH).
The top bar switches between the stages (`/annotate`, `/review`, `/audit`
deep links).
`Ctrl+C` stops the server.

Development, in two terminals:

```sh
just labeler-api    # uvicorn --reload on 127.0.0.1:8125
just labeler-ui     # vite dev server on http://127.0.0.1:5173 (proxies /api)
```

CLI flags (all optional):

| Flag | Default | Meaning |
|---|---|---|
| `--images` | `datasets/digits/images` | root directory of crop images |
| `--labels` | `datasets/digits/labels` | output root for YOLO label files (mirrors the images tree) |
| `--predictions` | `datasets/digits/predictions` | prediction files to review (mirrors the images tree) |
| `--audit-predictions` | `datasets/digits/audit_predictions` | audit model output to compare against labels (mirrors the images tree) |
| `--data-yaml` | `datasets/digits/data.yaml` | class names (file must exist and parse) |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8125` | port |
| `--dist` | `tools/labeler/frontend/dist` | built frontend directory |

## Data layout

```
datasets/digits/
├── data.yaml                              # classes: 0-9, dash, dot
├── images/<source-id>/<r>_<c>.png         # one crop per table cell
├── labels/<source-id>/<r>_<c>.txt         # YOLO labels (5 columns)
├── predictions/<source-id>/<r>_<c>.txt    # bootstrap model output (6 columns)
└── audit_predictions/<source-id>/<r>_<c>.txt  # audit model output (6 columns)
```

A label file holds one line per character box, normalised to 0-1 at 6 decimals:

```
<class> <cx> <cy> <w> <h>
```

A prediction file adds a confidence column (written by
`scripts/bootstrap_annotations.py` or `scripts/audit_annotations.py`,
consumed and deleted by the review and audit stages):

```
<class> <cx> <cy> <w> <h> <conf>
```

Prediction files are intentionally not loadable as YOLO labels; only accepted
files under `labels/` are.

Semantics shared with the training pipeline (`scripts/prepare_dataset.py`
treats them identically):

- **label file exists (any content)** → crop is *done/annotated*
- **empty label file** → done but nothing to label (background/ignored):
  written by annotate's *mark empty* action, or by accepting an empty
  prediction in review
- **non-empty** → crop has labeled characters
- **prediction file + no label file** → in the review queue
- **prediction file for an annotated or missing crop** → stale: ignored by the
  queue, pruned by the next bootstrap run
- **audit prediction file + label file** → compared by the audit stage and
  queued when the two box sets disagree; the file is deleted when the crop is
  decided

New source pages need no setup: any directory tree of images placed under the
images root is picked up on the next load, with labels and predictions mirrored
into the same subpath.

## Workflows

### annotate (`/annotate`)

Draw character boxes on the crops. The current class is sticky: pick a class
once (key or palette chip) and every new box inherits it. Coordinates are
image pixels on screen, normalised on save; edits autosave 600 ms after the
last change, are flushed before navigating, and are guarded by `beforeunload`.
Crops, zoom and view mode resume where you left off (`localStorage`, keys
`labeler.annotate.*`).

| Key | Action |
|---|---|
| drag | draw a box (needs a current class) |
| shift-click / shift-drag | select / select & move a box |
| drag selected box | move |
| drag handle | resize selected box |
| right-click | delete box under cursor |
| `0`–`9`, `.`, `-` | assign class to selected box + set current class |
| `←` `→` / `a` `d` | previous / next crop |
| `⏎` | save & next |
| `x` | mark crop empty/ignored & next |
| `z` | undo (100 steps) |
| `⌫` / `Del` | delete selected box |
| `Esc` | cancel drag / deselect |
| `f` | cycle view mode: all → annotated → non-annotated |
| `[` / `]` | zoom out / in |
| `Ctrl+S` | save now |

The view mode restricts prev/next only — the displayed crop stays put — so the
same queue supports both fast batch runs (non-annotated) and review passes over
already-labeled crops (annotated). Selection is anchored on the crop's path,
never on list position: a save can never detach the displayed annotations from
its file. A done crop with zero boxes shows an `EMPTY` badge (hidden as soon as
a box is drawn).

### review (`/review`)

One keystroke per crop over the bootstrap predictions. Predicted boxes are
display-only here: every box is tagged with its class on a large chip in the
box's class colour (text colour picked for contrast against it) and the
confidence on a separate smaller chip right-aligned at the box's top-right;
wrong predictions are rejected and fixed in the annotate stage. Beside the
image the sidebar shows the crop's whole-image predicted reading — the boxes'
class glyphs concatenated in left-to-right box-centre order (e.g. "0.32") in
large text — so a crop can be judged without reading each box. The text sizes
are tweakable constants in `frontend/src/lib/labels.ts`.

| Key | Action |
|---|---|
| `y` | accept: prediction becomes a label file, prediction deleted |
| `n` | reject: prediction discarded, crop stays non-annotated |
| `←` `→` / `a` `d` | previous / next (no decision) |

- **accept** drops the confidence column and writes the label byte-compatible
  with annotate's output. An empty prediction accepts to an empty label — the
  crop is confirmed as background/ignored. The label is written **before** the
  prediction is deleted, so an interrupted accept leaves a stale prediction
  (cleaned up later) rather than losing work. Existing labels are never
  overwritten: a crop annotated in the meantime gets its stale prediction
  dropped instead and the outcome is reported.
- **reject** deletes only the prediction file; the crop returns to the
  annotate queue (a future bootstrap run may predict it again).
- Decisions are destructive and immediate — there is no undo. Wrong
  predictions are cheap to regenerate, wrong labels are cheap to fix.
- Empty predictions show a `PREDICTED EMPTY` badge; malformed prediction files
  are flagged with the parse error and can only be rejected or skipped, never
  accepted.
- Decided crops never return to the queue; stale prediction files are counted
  in the header and ignored.

The queue can be re-ordered by a **ranking** (selector in the header,
persisted as `localStorage` key `labeler.review.ranking`): a ranking only
changes which crop comes next — every queued prediction stays reachable and
the decisions are identical in every ranking. Available rankings:

- **natural order** (default) — the queue's path order (source page, row,
  column).
- **rare labels first** — predictions containing labels that are rare in the
  accepted annotations come first. Each crop is scored by its rarest predicted
  label's count of labeled boxes over `labels/` (a never-labeled class counts
  0 and is therefore rarest); ties keep the natural order. Predicted
  backgrounds and malformed predictions come last, in natural order. Each
  scored crop shows a `why here:` note in the sidebar; malformed label files
  skipped while counting are reported in the list response (`label_errors`)
  and counted in the header, so a corrupt file cannot silently skew rarity.

### audit (`/audit`)

Find incorrect labels by comparing them against a model's predictions: run
`just audit <model>` (`scripts/audit_annotations.py`) over the dataset, then
resolve every disagreement here. The crop renders twice side by side —
annotations on the left, model predictions on the right, each badged with
its side (an empty side shows "(EMPTY)") — with boxes tagged as in review
(class on a large class-coloured chip, prediction confidences on a separate
chip at the box's top-right) and a plain-words issue list in the sidebar.
One keystroke per crop:

| Key | Action |
|---|---|
| `l` | keep the existing label: prediction dropped, label untouched |
| `p` | take the model's prediction: label overwritten (empty prediction → marked empty/ignored) |
| `x` | discard both: label and prediction deleted, the crop returns to the annotate queue |
| `←` `→` / `a` `d` | previous / next (no decision) |

- A crop is queued when its label and its audit prediction disagree: a box
  missing on either side, a matched pair with different classes, or a
  matched pair overlapping less than `MIN_IOU` (0.5, greedy pairing by
  descending IoU). Pure position/size differences above that overlap do not
  flag a crop. Empty label + empty prediction is agreement (background
  confirmed).
- Every decision consumes the crop's audit prediction file and is destructive
  and immediate — there is no undo. The label is written before the
  prediction is deleted, so an interrupted *take* leaves a still-comparing
  snapshot; a wrongly discarded label is re-annotated in the annotate stage.
- One `just audit <model>` run is one review round: re-running the script
  re-flags every crop that still disagrees, including crops whose labels you
  already kept.
- A malformed file queues with its parse error and can only be resolved from
  its sound side: a malformed prediction can never be taken, a malformed
  label can never be kept; *discard both* is always available.

## API contract

Errors are always JSON `{"error": ...}` (400/404/413/500) and every response
sets `Cache-Control: no-store`, so labeling state is never stale. Decision
routes touch only the decided crop's files — constant time per decision,
independent of dataset size; the queue is only scanned by the `/list` routes.

Core routes:

| Route | Returns |
|---|---|
| `GET /api/health` | `{"ok": true, "version": …}` |
| `GET /api/stages` | `{"stages": [{id, title, description}]}` |
| `GET /api/classes` | `{"classes": [...]}` (from `data.yaml`, strict) |
| `GET /api/image?path=` | raw crop image bytes |
| `GET /api/docs` | OpenAPI explorer |
| `GET /` | built frontend (503 with a build hint when `dist/` is missing) |

Annotate stage (mounted at `/api/annotate`):

| Route | Returns |
|---|---|
| `GET /api/annotate/list` | `{"images": [{path, done, has_labels}], "total", "done"}` |
| `GET /api/annotate/labels?path=` | `{"path", "boxes": [{cls, cx, cy, w, h}]}` (normalised) |
| `POST /api/annotate/labels` | `{"path", "boxes"}` → `{"ok", "saved", "has_labels"}`; empty `boxes` marks the crop empty |

Review stage (mounted at `/api/review`):

| Route | Returns |
|---|---|
| `GET /api/review/list?ranking=` | `{"predictions": [{path, boxes, empty, error, rank_note}], "total", "remaining", "ranking", "label_errors"}` — the queue in the ranking's order (`natural` when omitted, see `GET /api/review/rankings`); `total - remaining` predictions are stale and ignored; `rank_note` is the ranking's explanation of the placement (null when it has none); `label_errors` lists label files the ranking could not parse while scoring |
| `GET /api/review/rankings` | `{"rankings": [{id, title, description}]}` — available queue rankings |
| `GET /api/review/prediction?path=` | `{"path", "boxes": [{cls, cx, cy, w, h, conf}]}` (normalised) |
| `POST /api/review/decision` | `{"path", "accept": true\|false}` → `{"ok", "accepted", "saved", "already_annotated"}` |

Audit stage (mounted at `/api/audit`):

| Route | Returns |
|---|---|
| `GET /api/audit/list` | `{"audits": [{path, label_boxes, pred_boxes, issues, error}], "total", "remaining"}` — `total` counts every compared crop; a side's box count is `null` for a malformed file |
| `GET /api/audit/compare?path=` | `{"path", "labels", "predictions", "issues"}` — both box sets (normalised, predictions carry `conf`) and the disagreements `[{kind, label, pred, iou}]` |
| `POST /api/audit/decision` | `{"path", "action": "keep_label"\|"take_prediction"\|"discard_both"}` → `{"ok", "action", "saved"}` |

On-disk parsing is strict: malformed label or prediction lines are reported
(400), never silently skipped or promoted into the dataset.

## Architecture / layout

| Path | Role |
|---|---|
| `tools/labeler/app.py` | ASGI app: core `/api` routes, stage routers, built frontend at `/` |
| `tools/labeler/main.py` | CLI flags, startup summary, uvicorn bootstrap |
| `tools/labeler/config.py` | `Config`: bind address, dist, dataset roots |
| `tools/labeler/dataset.py` | `Dataset` + helpers: classes, image listing, label/prediction I/O |
| `tools/labeler/compare.py` | model-vs-label comparison: greedy IoU pairing + disagreement issues |
| `tools/labeler/rankings.py` | review queue rankings: pluggable presentation order (natural, rare labels) |
| `tools/labeler/errors.py` | `ApiError` (HTTP error with a JSON-safe message) |
| `tools/labeler/web.py` | request helpers shared by the stage routers |
| `tools/labeler/stages/base.py` | stage seam: `Stage` contract + `summaries` |
| `tools/labeler/stages/annotate.py` | annotate stage: queue listing + label I/O |
| `tools/labeler/stages/review.py` | review stage: prediction queue + accept/reject decisions |
| `tools/labeler/stages/audit.py` | audit stage: disagreement queue + keep/take/discard decisions |
| `tools/labeler/frontend/` | Vite + Svelte + TypeScript UI, built to `frontend/dist` |
| `tools/labeler/frontend/src/lib/BoxEditor.svelte` | shared canvas engine (draw/move/resize/delete, read-only mode) |
| `tools/labeler/frontend/src/lib/labels.ts` | label text sizes (the tweakable constants), chip drawing, reading composition |
| `tools/labeler/frontend/src/lib/` | geometry, palette, label chips, typed API client |
| `tools/labeler/frontend/src/stages/` | workspace components + `registry.ts` (stage id → workspace) |

## Toolchains

- **Python**: the pixi environment at the repo root (`fastapi`, `uvicorn`,
  PyYAML for `data.yaml` parsing).
- **Node.js**: ≥ 20 on `PATH` with pnpm (tested with Node 24 / pnpm 11) for
  the Vite build. Not managed by pixi; `pnpm-lock.yaml` is tracked and
  `pnpm-workspace.yaml` pins the build-script policy (`allowBuilds`).

## Adding a stage

A stage (pipeline stage) is a queue of crops + user actions + artifacts
consumed/produced. One stage = one backend module with a `create_stage(ds)`
factory (API router mounted at `/api/<id>`) registered in
`tools/labeler/stages/__init__.py`, plus one entry in
`frontend/src/stages/registry.ts` (workspace component, usually built from the
shared `BoxEditor`). `stages/annotate.py` is the worked example. The shared
engine and core routes stay untouched.

## Adding a ranking

A ranking (review queue presentation order) is one entry in
`tools/labeler/rankings.py`: a `Ranking` (id, title, description) whose
`order` function maps the review queue entries to themselves — optionally
enriched with a per-entry `rank_note` — plus the label files it skipped while
scoring as label errors. `GET /api/review/rankings` and the review
workspace's selector pick it up automatically; no other file changes.
`natural_order` and `rare_labels_order` are the worked examples. Rankings
never filter: queue membership and decisions stay identical across them.

## Notes & limitations

- **Localhost tool, no authentication** — use it through your machine or an
  SSH tunnel; do not expose the port.
- **Axis-aligned boxes only** — YOLO format has no rotation.
- **Single selection** in the annotate editor; crops hold only a few boxes.
- **Review and audit decisions are irreversible** by design (cheap to
  regenerate or re-annotate instead of undo).
- **`predictions/` is owned by `scripts/bootstrap_annotations.py`** — review
  only deletes files it decides; regeneration and pruning stay in the script.
- **`audit_predictions/` is owned by `scripts/audit_annotations.py`** — the
  audit stage only deletes files it decides; regeneration and pruning stay in
  the script, and one run is one review round.
- **Class set is data-driven** — editing `data.yaml` names changes the palette
  and keyboard map on the next load; ids must stay contiguous from 0.
- **Box-label text sizes are constants** — `frontend/src/lib/labels.ts` holds
  `LABEL_TEXT_PX` (class chip), `CONF_TEXT_PX` (confidence chip) and
  `READING_TEXT_PX` (the review sidebar's whole-image reading); tweak and
  rebuild.
