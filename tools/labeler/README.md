# labeler — combined, extensible annotation tool

One application for the dataset's annotation workflows: manual bounding-box
labeling (**annotate**), accept/reject review of bootstrap predictions one
crop at a time (**review**) or in grid-sized batches (**batch review**), and
model-vs-label disagreement resolution (**audit**), plus a read-only overview
of the dataset's labeling state (**summary**). It is
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
The top bar switches between the stages (`/annotate`, `/review`,
`/batch-review`, `/audit`, `/summary` deep links).
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
| `--manual` | `datasets/digits/manual` | manual-queue marker files (mirrors the images tree) |
| `--predictions` | `datasets/digits/predictions` | prediction files to review (mirrors the images tree) |
| `--audit-predictions` | `datasets/digits/audit_predictions` | audit model output to compare against labels (mirrors the images tree) |
| `--data-yaml` | `datasets/digits/data.yaml` | class names (file must exist and parse) |
| `--class-map` | `classes.yaml` next to `--data-yaml` | class presentation: display glyph + keyboard shortcut per class (file must exist and parse) |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8125` | port |
| `--dist` | `tools/labeler/frontend/dist` | built frontend directory |

## Data layout

```
datasets/digits/
├── data.yaml                              # classes: 0-9, dash, dot, tr, x
├── classes.yaml                           # class presentation: glyph + keyboard shortcut per class
├── images/<source-id>/<r>_<c>.png         # one crop per table cell
├── labels/<source-id>/<r>_<c>.txt         # YOLO labels (5 columns)
├── manual/<source-id>/<r>_<c>.txt         # manual-queue marker (empty file)
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

`classes.yaml` (required) is the class presentation map: one `{glyph, key}`
entry per `data.yaml` class, keyed by class name (`key: null` makes the class
click-only). `glyph` is what the class renders as — on box chips, in the
review reading, and in summary chips — and `key` is its annotate keyboard
shortcut (a `KeyboardEvent.key` value, e.g. `"x"`, `"t"`, `"-"`).

Semantics shared with the training pipeline (`scripts/prepare_dataset.py`
treats them identically):

- **label file exists (any content)** → crop is *done/annotated*
- **empty label file** → done but nothing to label (background/ignored):
  written by annotate's *mark empty* action, or by accepting an empty
  prediction in review
- **non-empty** → crop has labeled characters
- **manual marker + no label file** → crop is in the *manual queue*: flagged
  in review as needing manual adjustment (its prediction was discarded);
  excluded from the review queue, served first in annotate
- **label file written** → the manual marker is removed (a labeled crop is
  done), so the manual queue and `labels/` never overlap
- **prediction file + no label file and no manual marker** → in the review
  queue (worked one crop at a time in *review* or in batches in
  *batch review*)
- **prediction file for an annotated, flagged, or missing crop** → stale:
  ignored by the queue, pruned by the next bootstrap run
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
| the class keys (see the palette chips) | assign class to selected box + set current class |
| `←` `→` / `a` `d` | previous / next crop |
| `⏎` | save & next |
| `⌫` | mark crop empty/ignored & next |
| `z` | undo (100 steps) |
| `Del` | delete selected box |
| `Esc` | cancel drag / deselect |
| `f` | cycle view mode: all → annotated → non-annotated → manual |
| `[` / `]` | zoom out / in |
| `Ctrl+S` | save now |

The view mode restricts prev/next only — the displayed crop stays put — so the
same queue supports both fast batch runs (non-annotated) and review passes over
already-labeled crops (annotated). The `manual` mode confines prev/next to the
flagged manual queue, so it can be worked to completion before the backlog.
Selection is anchored on the crop's path, never on list position: a save can
never detach the displayed annotations from its file. A done crop with zero
boxes shows an `EMPTY` badge (hidden as soon as a box is drawn).

Crops flagged in review as needing manual adjustment lead the queue: the
listing serves the whole manual queue first (natural order within it, then
everything else in natural order), so a fresh session opens the first flagged
crop and prev/next reach every flagged crop before the backlog. The `manual`
view mode restricts prev/next to the flagged crops alone. A flagged crop
on display shows a `NEEDS MANUAL` badge and the header counts what is still
flagged; saving labels for it (including *mark empty*) clears the flag.

### review (`/review`)

One keystroke per crop over the bootstrap predictions. Predicted boxes are
display-only here: every box is tagged with its class glyph on a large chip in
the box's class colour (text colour picked for contrast against it) and the
confidence on a separate smaller chip right-aligned at the box's top-right;
wrong predictions are rejected and fixed in the annotate stage. The crop's
whole-image predicted reading — the boxes' class glyphs concatenated in
left-to-right box-centre order (e.g. "0.32") — is drawn in large text on a
dark chip at the image's top-centre (an area normally free of predicted
boxes), so a crop can be judged without reading each box. The text sizes
are tweakable constants in `frontend/src/lib/labels.ts`.

| Key | Action |
|---|---|
| `y` | accept: prediction becomes a label file, prediction deleted |
| `n` | reject: prediction discarded, crop stays non-annotated |
| `m` | needs manual adjustment: prediction discarded, crop added to the manual queue |
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
- **needs manual adjustment** deletes only the prediction file and flags the
  crop for the *manual queue* (an empty marker under `manual/`), which the
  annotate stage serves first so the crop is worked by hand before the
  backlog. The marker is written **before** the prediction is deleted, and the
  prediction is discarded unparsed, so malformed prediction files can be
  flagged too. A crop annotated in the meantime gets its stale prediction
  dropped and no flag. Saving a label for a flagged crop (or marking it empty)
  clears the flag; the next bootstrap run never predicts flagged crops.
- Decisions are destructive and immediate — there is no undo. Wrong
  predictions are cheap to regenerate, wrong labels are cheap to fix.
- Empty predictions show a `PREDICTED EMPTY` badge; malformed prediction files
  are flagged with the parse error and can only be rejected, flagged for
  manual adjustment, or skipped, never accepted.
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
- **fill the gaps first** — predictions that most reduce the annotation gap
  come first. Each predicted box scores its confidence × `1/(1 + n)` where
  `n` is its class's count of labeled boxes over `labels/` (a never-labeled
  class weighs the most), and a crop's score is the sum over its boxes, so
  covering several under-annotated classes — or more boxes of them —
  outranks covering well-annotated ones. The order is built greedily: each
  pick's boxes are counted as filled gap (the class weights shrink as if the
  crop were accepted) before the next crop is scored, so the queue works
  down what is still missing instead of repeating an already-covered gap.
  Ties keep the natural order; predicted backgrounds and malformed
  predictions come last, in natural order. Each scored crop shows a
  `why here:` note naming the classes it fills (with box counts) and its
  score; malformed label files skipped while counting are reported as in
  *rare labels first*.
- **high confidence first** — predictions whose lowest-confidence box is most
  confident first. Each crop is scored by the minimum confidence of its
  predicted boxes, so a crop comes early only when *every* box is confident
  and the bulk of safe predictions is cleared before the risky ones; ties
  keep the natural order, predicted backgrounds and malformed predictions
  come last in natural order, and each scored crop shows a `why here:` note
  naming its lowest box confidence. This is also the order the *batch
  review* stage builds on.

### batch review (`/batch-review`)

The same review queue worked in batches: a grid of crops with their predicted
boxes and whole-image predicted reading — rendered as in review (class glyph
on a class-coloured chip plus the box's confidence on a small chip
right-aligned at its top-right, reading on a dark chip at the image's
top-centre) at tile scale, using the tile text-size constants in
`frontend/src/lib/labels.ts`. Tiles deliberately carry no caption; a
predicted empty crop's top label reads `EMPTY` (its only empty label). The
grid's image size has `−` / `+` controls (persisted as `localStorage` key
`labeler.batch.tile-scale`). The queue is served **high confidence first**
(the ranking above) with **empty predictions sprinkled in randomly at 25% of
the served slots** (a target rate; individual batches vary naturally around
it), so predicted backgrounds keep being confirmed at a steady rate while
the high-confidence mass is cleared; malformed prediction files come last,
in natural order.

The batch size is a slider (1-32, persisted as `localStorage` key
`labeler.batch.size`): the grid is always the first entries of the served
queue, so growing the batch appends tiles and shrinking it truncates from the
tail (pending marks on truncated tiles are discarded — nothing is written
while marking). Mark tiles, then commit the whole batch with one button whose
label counts the actions (e.g. `accept 6 · reject 2 · flag 1`):

| Click | Action |
|---|---|
| plain click | mark rejected (click again to unmark) |
| shift-click | mark needs manual adjustment (click again to unmark) |

- A tile carries at most one mark; clicking with the other modifier switches
  the mark. On commit every unmarked tile is **accept**ed and every marked
  tile gets its marked action (**reject** / **flag**), all with the review
  stage's exact decision semantics: accept writes the label before deleting
  the prediction (an empty prediction accepts to an empty label, confirming
  the crop as background/ignored), reject deletes only the prediction, and
  flag discards the prediction and marks the crop for the manual queue.
  Existing labels are never overwritten: a crop annotated in the meantime
  gets its stale prediction dropped and reported.
- The commit is one request with a per-crop result: a stale or broken crop
  reports its error and never blocks the rest of the batch; the queue is then
  re-synced from disk.
- Malformed prediction tiles cannot be accepted, as in review: unmarked ones
  are skipped and reported, and they can be rejected or flagged by clicking
  them.
- Decisions are destructive and immediate — there is no undo, as in review.
  Wrong predictions are cheap to regenerate, wrong labels are cheap to fix.

### audit (`/audit`)

Find incorrect labels by comparing them against a model's predictions: run
`just audit <model>` (`scripts/audit_annotations.py`) over the annotated crops,
then resolve every disagreement here. The crop renders twice side by side —
annotations on the left, model predictions on the right, each badged with
its side (an empty side shows "(EMPTY)") — with boxes tagged as in review
(class glyph on a large class-coloured chip, prediction confidences on a
separate chip at the box's top-right) and a plain-words issue list in the
sidebar.
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

### summary (`/summary`)

A read-only overview of the dataset as it stands — no queue, no decisions,
nothing written. Three panels:

- **progress** — headline totals: crops, annotated (including the empty
  ones), of which empty, unlabeled, labeled boxes.
- **class distribution** — labeled boxes per class (every class from
  `data.yaml` is shown, never-labeled classes at 0), with the crops marked
  empty/ignored alongside as its own row (a crop count, not a box count).
- **per-subject distribution** — one row per subject (the crop folder's
  `<source-id>`, so a second page's `<subject>_2` folder is a subject of its
  own; crops directly under the images root group as `(root)`) in natural
  order, with the crop counts and the full per-class box breakdown.

Malformed label files and label files mirroring no crop image (what
`scripts/prepare_dataset.py` rejects as inconsistent) are listed under
**problems** and excluded from every count, so corruption cannot silently
skew the numbers. The page fetches on load and has a refresh button; the
data is rescanned from disk on every request.

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
| `GET /api/classes` | `{"classes": [{"name", "glyph", "key"}]}` — one entry per class: `name` from `data.yaml`, `glyph` (display) and `key` (annotate shortcut, `null` for a click-only class) from `classes.yaml`, both strict |
| `GET /api/image?path=` | raw crop image bytes |
| `GET /api/docs` | OpenAPI explorer |
| `GET /` | built frontend (503 with a build hint when `dist/` is missing) |

Annotate stage (mounted at `/api/annotate`):

| Route | Returns |
|---|---|
| `GET /api/annotate/list` | `{"images": [{path, done, has_labels, manual}], "total", "done", "manual"}` — entries serve `manual` crops (flagged for manual adjustment, not yet labeled) first, each group in natural path order; `manual` is the flagged count |
| `GET /api/annotate/labels?path=` | `{"path", "boxes": [{cls, cx, cy, w, h}]}` (normalised) |
| `POST /api/annotate/labels` | `{"path", "boxes"}` → `{"ok", "saved", "has_labels"}`; empty `boxes` marks the crop empty |

Review stage (mounted at `/api/review`):

| Route | Returns |
|---|---|
| `GET /api/review/list?ranking=` | `{"predictions": [{path, boxes, empty, error, rank_note}], "total", "remaining", "ranking", "label_errors"}` — the queue in the ranking's order (`natural` when omitted, see `GET /api/review/rankings`); `total - remaining` predictions are stale and ignored; `rank_note` is the ranking's explanation of the placement (null when it has none); `label_errors` lists label files the ranking could not parse while scoring |
| `GET /api/review/rankings` | `{"rankings": [{id, title, description}]}` — available queue rankings |
| `GET /api/review/prediction?path=` | `{"path", "boxes": [{cls, cx, cy, w, h, conf}]}` (normalised) |
| `POST /api/review/decision` | `{"path", "accept": true\|false}` → `{"ok", "accepted", "saved", "already_annotated"}` |
| `POST /api/review/manual` | `{"path"}` → `{"ok", "marked", "already_annotated"}` — needs manual adjustment: the prediction is discarded and the crop is flagged for the manual queue; `marked` is false only when the crop was annotated in the meantime (stale prediction dropped) |

Batch review stage (mounted at `/api/batch-review`):

| Route | Returns |
|---|---|
| `GET /api/batch-review/list` | `{"predictions": [{path, boxes, empty, error, rank_note}], "total", "remaining"}` — the queue with its prediction boxes inlined (`boxes` is null for a malformed file) in the stage's served order: high-confidence-first with empty predictions sprinkled in at 25% of the slots and malformed files last in natural order; `total - remaining` predictions are stale and ignored |
| `POST /api/batch-review/decision` | `{"decisions": [{"path", "action": "accept"\|"reject"\|"manual"}]}` (1..64 distinct paths) → `{"results": [{"path", "action", "ok", "saved", "already_annotated", "error"}]}` — one result per decision in request order, each applied with the review stage's accept/reject/flag semantics; a failing crop reports `error` in its result and never blocks the rest of the batch |

Audit stage (mounted at `/api/audit`):

| Route | Returns |
|---|---|
| `GET /api/audit/list` | `{"audits": [{path, label_boxes, pred_boxes, issues, error}], "total", "remaining"}` — `total` counts every compared crop; a side's box count is `null` for a malformed file |
| `GET /api/audit/compare?path=` | `{"path", "labels", "predictions", "issues"}` — both box sets (normalised, predictions carry `conf`) and the disagreements `[{kind, label, pred, iou}]` |
| `POST /api/audit/decision` | `{"path", "action": "keep_label"\|"take_prediction"\|"discard_both"}` → `{"ok", "action", "saved"}` |

Summary stage (mounted at `/api/summary`):

| Route | Returns |
|---|---|
| `GET /api/summary/overview` | `{"classes", "totals": {crops, annotated, empty, unlabeled, boxes}, "class_counts", "subjects", "label_errors", "orphan_labels"}` — the dataset's aggregate state (a scan route like `/list`); `annotated` includes `empty`; each `subjects` row is `{id, crops, annotated, empty, unlabeled, boxes, class_counts}` in natural id order and the rows sum exactly to `totals` / `class_counts`; `label_errors` lists malformed label files (`{path, error}`) and `orphan_labels` the label files mirroring no image — both excluded from every count |

On-disk parsing is strict: malformed label or prediction lines are reported
(400), never silently skipped or promoted into the dataset.

## Architecture / layout

| Path | Role |
|---|---|
| `tools/labeler/app.py` | ASGI app: core `/api` routes, stage routers, built frontend at `/` |
| `tools/labeler/main.py` | CLI flags, startup summary, uvicorn bootstrap |
| `tools/labeler/config.py` | `Config`: bind address, dist, dataset roots |
| `tools/labeler/dataset.py` | `Dataset` + helpers: classes and their presentation, image listing, label/manual-marker/prediction I/O |
| `tools/labeler/compare.py` | model-vs-label comparison: greedy IoU pairing + disagreement issues |
| `tools/labeler/rankings.py` | review queue rankings: pluggable presentation order (natural, rare labels, gap filling) |
| `tools/labeler/errors.py` | `ApiError` (HTTP error with a JSON-safe message) |
| `tools/labeler/web.py` | request helpers shared by the stage routers |
| `tools/labeler/stages/base.py` | stage seam: `Stage` contract + `summaries` |
| `tools/labeler/stages/annotate.py` | annotate stage: queue listing + label I/O |
| `tools/labeler/stages/review.py` | review stage: prediction queue + accept/reject/flag decisions |
| `tools/labeler/stages/batch_review.py` | batch review stage: composed batch queue + bulk accept/reject/flag commits |
| `tools/labeler/stages/audit.py` | audit stage: disagreement queue + keep/take/discard decisions |
| `tools/labeler/stages/summary.py` | summary stage: read-only dataset aggregate (class + per-subject distributions) |
| `tools/labeler/frontend/` | Vite + Svelte + TypeScript UI, built to `frontend/dist` |
| `tools/labeler/frontend/src/lib/BoxEditor.svelte` | shared canvas engine (draw/move/resize/delete, read-only mode) |
| `tools/labeler/frontend/src/lib/Tile.svelte` | batch grid tile: read-only crop + predicted boxes + reading, click-to-mark chrome |
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
`natural_order`, `confidence_order`, `rare_labels_order` and
`fill_gaps_order` are the worked examples. Rankings never filter: queue
membership and decisions stay identical across them. The batch review
stage's 25% empty-prediction sprinkle is its own queue composition
(`stages/batch_review.py`), not a ranking behaviour.

## Notes & limitations

- **Localhost tool, no authentication** — use it through your machine or an
  SSH tunnel; do not expose the port.
- **Axis-aligned boxes only** — YOLO format has no rotation.
- **Single selection** in the annotate editor; crops hold only a few boxes.
- **Review, batch review and audit decisions are irreversible** by design
  (cheap to regenerate or re-annotate instead of undo).
- **`predictions/` is owned by `scripts/bootstrap_annotations.py`** — review
  only deletes files it decides; regeneration and pruning stay in the script.
- **`manual/` is owned by the review/annotate decisions** — review writes the
  markers, saving a label clears them; `scripts/bootstrap_annotations.py`
  skips flagged crops and only prunes markers whose crop is annotated or gone.
- **`audit_predictions/` is owned by `scripts/audit_annotations.py`** — the
  audit stage only deletes files it decides; regeneration and pruning stay in
  the script, and one run is one review round.
- **Class set is data-driven** — `data.yaml` names the classes and the
  required sibling `classes.yaml` gives each its display glyph and annotate
  keyboard shortcut (`key: null` = click-only), so the app is not hard-wired
  to any dataset's class set; ids must stay contiguous from 0. Both files are
  validated strictly at startup: `classes.yaml` must name exactly the
  `data.yaml` classes, and duplicate shortcuts or shortcuts colliding with a
  labeler action key (`a` `d` `f` `z` `[` `]` `⏎` `⌫` `Del` `Esc` `←` `→`)
  abort startup.
- **Box-label text sizes are constants** — `frontend/src/lib/labels.ts` holds
  `LABEL_TEXT_PX` (class chip), `CONF_TEXT_PX` (confidence chip) and
  `READING_TEXT_PX` (the review image's top-centre reading chip), plus the
  batch grid's `TILE_LABEL_TEXT_PX`, `TILE_CONF_TEXT_PX` and
  `TILE_READING_TEXT_PX`; tweak and rebuild.
