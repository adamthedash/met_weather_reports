# review — accept/reject review for the bootstrap predictions

A small companion to `scripts/bootstrap_annotations.py` and
`tools/annotate.py`: it shows each predicted crop with its boxes and
confidence scores overlaid, and reduces the review to one keystroke per
crop — accept (`y`) promotes the prediction to a YOLO label file, reject
(`n`) discards it. PyYAML (via `tools/annotate.py`) remains the only
third-party dependency.

Built from two files on top of the annotation tooling:

| File | Role |
|---|---|
| `tools/review.py` | HTTP server (stdlib + PyYAML): queue listing, prediction file I/O, decisions; imports the dataset helpers of `tools/annotate.py` |
| `tools/review.html` | single-page UI: vanilla JS + `<canvas>`, no build step |

## Quick start

```sh
just bootstrap <model>   # write predictions for the non-annotated crops
just review
# review server: http://127.0.0.1:8124
```

Open the URL in a browser (forward the port first when working over SSH) and
press `y` or `n` on each crop. `Ctrl+C` stops the server. The tool binds its
own port by default and can run alongside `tools/annotate.py`.

CLI flags (all optional):

| Flag | Default | Meaning |
|---|---|---|
| `--images` | `datasets/digits/images` | root directory of crop images |
| `--labels` | `datasets/digits/labels` | output root for accepted YOLO label files |
| `--predictions` | `datasets/digits/predictions` | prediction files to review |
| `--data-yaml` | `datasets/digits/data.yaml` | class names (file must exist and parse) |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8124` | port |

## The review loop

The **queue** is every crop that has a prediction file and no label file, in
natural order — exactly the crops the last bootstrap run predicted.
Prediction files whose crop was annotated in the meantime are ignored as
stale and removed by the next bootstrap run.

Per crop:

- **`y` — accept:** the prediction becomes
  `labels/<source-id>/<r>_<c>.txt` with the confidence column removed
  (byte-compatible with the annotate tool's labels), the prediction file is
  deleted, and the next crop is shown. An empty prediction file accepts to an
  empty label file — the crop is confirmed as background/ignored.
- **`n` — reject:** only the prediction file is deleted. The crop stays
  non-annotated: fix it manually in `tools/annotate.py` (a future bootstrap
  run may predict it again).
- **`←` / `→` (`a` / `d`)** browse without deciding; decided crops never
  return to the queue.

Decisions are destructive and immediate — there is no undo. Wrong
predictions are cheap to regenerate, and wrong labels can be fixed in the
annotate tool.

Empty predictions show a `PREDICTED EMPTY` badge; malformed prediction files
(strict 6-field parse) are flagged with the parse error and can only be
rejected or skipped, never accepted. If a crop was annotated while it sat in
the queue, accepting does **not** overwrite the label: the stale prediction
is dropped instead and the outcome is reported.

## Data layout

```
datasets/digits/
├── images/<source-id>/<r>_<c>.png       # one crop per table cell
├── predictions/<source-id>/<r>_<c>.txt  # model output, 6 columns (input)
└── labels/<source-id>/<r>_<c>.txt       # accepted output, 5 columns
```

A prediction file holds one line per box, confidence descending (as written
by `scripts/bootstrap_annotations.py`):

```
<class> <cx> <cy> <w> <h> <conf>     # normalised to 0-1, 6 decimals
```

Accepting drops the 6th column, producing the standard label format of
`tools/annotate.py`:

```
<class> <cx> <cy> <w> <h>            # normalised to 0-1, 6 decimals
```

Prediction files are therefore intentionally not loadable as YOLO labels;
only accepted files under `labels/` are.

## API

| Route | Returns |
|---|---|
| `GET /` | `review.html` (read from disk per request, so UI edits need no restart) |
| `GET /api/classes` | `{"classes": ["0", …, "dash", "dot"]}` |
| `GET /api/list` | `{"predictions": [{path, boxes, empty, error}], "total", "remaining"}` — the queue; `total - remaining` predictions are stale and ignored |
| `GET /api/image?path=` | raw image bytes |
| `GET /api/prediction?path=` | `{"path", "boxes": [{cls, cx, cy, w, h, conf}]}` (normalized) |
| `POST /api/decision` | body `{"path", "accept": true\|false}` → `{"ok", "accepted", "saved", "already_annotated", "remaining"}` |

The server is a plain localhost tool with no authentication — don't expose
the port beyond your machine or an SSH tunnel.

## Design notes & limitations

- **Builds on `tools/annotate.py`** — dataset helpers (`load_classes`,
  `safe_path`, `label_path_for`, `write_labels`, …) and the HTTP plumbing
  are imported, not copied; `review.py` points `annotate.CFG` at its own
  roots so the shared label writer produces the exact on-disk format of the
  annotate tool.
- **Labels are written before predictions are deleted**, so an interrupted
  accept leaves a stale prediction (cleaned up later) rather than losing
  work. Existing labels are never overwritten.
- **Display-only** — predicted boxes cannot be adjusted here (deliberate:
  fix wrong predictions in `tools/annotate.py` after rejecting them).
- **Strict on-disk format** — malformed prediction lines are reported (400)
  rather than silently skipped or promoted, so corruption never reaches
  `labels/`.
- **`predictions/` is owned by `scripts/bootstrap_annotations.py`** —
  review only removes files it has decided (plus the stale ones it drops
  during accept); regeneration and pruning stay in the bootstrap script.
