# annotate — bounding-box labeler for the handwritten digits dataset

A small tool for drawing character-level bounding boxes on the
table-cell crops in `datasets/digits/images`, with PyYAML as its only
third-party dependency. Boxes are saved as YOLO label
files under `datasets/digits/labels`, ready for `ultralytics` training against
`datasets/digits/data.yaml`.

Built from two files:

| File | Role |
|---|---|
| `tools/annotate.py` | HTTP server (stdlib + PyYAML): image listing, label file I/O, JSON API |
| `tools/annotate.html` | single-page UI: vanilla JS + `<canvas>`, no build step |

## Quick start

```sh
pixi run python tools/annotate.py
# annotate server: http://127.0.0.1:8123
```

Open the URL in a browser (forward the port first when working over SSH) and
label. Progress saves automatically; re-running the tool resumes where you
left off. `Ctrl+C` stops the server.

CLI flags (all optional):

| Flag | Default | Meaning |
|---|---|---|
| `--images` | `datasets/digits/images` | root directory of crop images |
| `--labels` | `datasets/digits/labels` | output root for YOLO label files |
| `--data-yaml` | `datasets/digits/data.yaml` | class names (file must exist and parse) |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8123` | port |

## Data layout

```
datasets/digits/
├── data.yaml                       # classes: 0-9, dash, dot
├── images/<source-id>/<r>_<c>.png  # one crop per table cell
└── labels/<source-id>/<r>_<c>.txt  # written by the tool (mirrors images/)
```

Each label file holds one line per character box:

```
<class> <cx> <cy> <w> <h>     # normalised to 0-1, 6 decimals
```

Example — a cell containing `.14`:

```
11 0.100000 0.500000 0.080000 0.600000
1  0.300000 0.500000 0.100000 0.700000
4  0.500000 0.500000 0.100000 0.700000
```

Semantics used by the tool and by `train.txt`/`val.txt`:

- **label file exists (any content)** → crop is *done* (not offered by
  prev/next while the "hide done" filter is on)
- **empty label file** → crop is done but contains nothing to label
  (ignored/background). The UI's *mark empty* action (`x`) writes one, and the
  UI overlays an `EMPTY` badge on such crops (hidden again while any box is
  drawn).
- **non-empty** → crop has labeled characters

New source pages need no setup: any directory tree of images placed under the
images root is picked up on the next `/api/list`, with labels mirrored into the
same subpath.

## Server architecture (`annotate.py`)

Standard library (`http.server`, `json`, `pathlib`, …) plus PyYAML for
`data.yaml`. Structured in
three layers:

1. **Dataset helpers** — pure functions, no HTTP concerns:
   - `load_classes()` parses the int-keyed `names:` mapping of `data.yaml`
     with PyYAML. Strict by design: the dataset fully defines the classes, so
     a missing file, invalid YAML, or a missing/empty `names:` mapping aborts
     startup with a clear error instead of falling back to defaults.
   - `natural_key()` / `list_images()` — natural sort so `10_2.png` precedes
     `10_10.png`; recursive scan of `.png .jpg .jpeg .bmp .webp`.
   - `safe_path()` — rejects absolute paths, `..` segments, and backslashes,
     then resolves and checks containment under the root (both image and label
     paths go through it).
   - `read_labels()` / `write_labels()` / `validate_boxes()` — YOLO round-trip.
     Reads are strict (5 fields per line, class ids in range, blank lines and
     `#` comments allowed); writes format to 6 decimals. `validate_boxes()`
     caps at 64 boxes, requires finite positive extents, and clamps boxes into
     the image instead of rejecting boundary overshoot.
2. **HTTP layer** — a `BaseHTTPRequestHandler` on a `ThreadingHTTPServer`.
   Every response sets `Cache-Control: no-store` (labeling state must never be
   stale). All errors are JSON `{"error": ...}` with 400/404/413/500 status;
   a catch-all guard turns unexpected exceptions into 500s rather than dropped
   connections. Requests are logged to stdout.
3. **`main()`** — argument parsing, startup summary (crop count, classes,
   progress), graceful `Ctrl+C` shutdown.

### API

| Route | Returns |
|---|---|
| `GET /` | `annotate.html` (read from disk per request, so UI edits need no restart) |
| `GET /api/classes` | `{"classes": ["0", …, "dash", "dot"]}` |
| `GET /api/list` | `{"images": [{path, done, has_labels}], "total", "done"}` |
| `GET /api/image?path=` | raw image bytes |
| `GET /api/labels?path=` | `{"path", "boxes": [{cls, cx, cy, w, h}]}` (normalized) |
| `POST /api/labels` | body `{"path", "boxes": […]}` → `{"ok", "saved", "has_labels"}`; empty `boxes` marks the crop empty |

The server is a plain localhost tool with no authentication — don't expose the
port beyond your machine or an SSH tunnel.

## UI architecture (`annotate.html`)

One page, one `<canvas>`, one module-scoped `state` object. Coordinates are
kept in **image pixels** everywhere and normalized only at save time.

- **Rendering** — the crop is drawn magnified with
  `imageSmoothingEnabled = false` (and CSS `image-rendering: pixelated`) so
  pixels stay crisp; boxes are semi-transparent fills + colored strokes, each
  tagged with its class name. Selected boxes show 8 resize handles. A crop that
  is done but holds no boxes (marked empty) gets a small fixed-size `EMPTY`
  badge in the image's top-left corner, which disappears as soon as a box is
  drawn. The canvas backing store is scaled by `devicePixelRatio` for sharp
  text.
- **Mouse model** — a plain mousedown hit-tests in priority order: selected-box
  handle (resize) → selected-box interior (move) → anything else, *including
  unselected box interiors* (draw a new box; requires a current class). Boxes
  are only ever moved or resized while selected; `Shift`+press is the
  select-and-move gesture — it selects the topmost box under the cursor and
  starts moving it (a click without dragging just selects; clicking the
  selected box keeps it selected, and handles never resize while `Shift` is
  held). Drags clamp to the image bounds; boxes under 3 screen-px are discarded
  as accidental clicks. Right-click deletes the box under the cursor. `Esc`
  cancels an in-progress drag (restoring the pre-drag state) or deselects.
- **Class assignment** — the class palette is generated from `data.yaml`
  names: single-character names map to that key, `dash` → `-`, `dot` → `.`.
  Pressing a key assigns the class to the selected box *and* sets the "current"
  class, which new boxes inherit automatically (sticky — draw four `1`s in a
  row without touching the keyboard). Drawing is refused until a class has been
  chosen (via key or palette chip) with a visible hint, so every box always
  carries class info; new boxes are never auto-selected (shift-click to edit
  one).
- **Persistence** — autosave POSTs 600 ms after the last edit; navigation
  flushes before switching crops; a `beforeunload` guard covers unsaved edits.
  Undo keeps a 100-deep stack of pre-mutation snapshots (covers draw, move,
  resize, delete, reclass).
- **Resume** — `localStorage` remembers the last crop, zoom level, and the
  "hide done" filter; existing label files are loaded and drawn on open.
- **Progress** — header shows `crop N / M · done X / Y` (N/M counts every
  crop), and the *hide done* filter (`f`) restricts prev/next to unlabeled
  crops for fast batch runs; toggling the filter keeps the current crop on
  screen.

### Keyboard reference

| Key | Action |
|---|---|
| drag | draw a box (needs a current class) |
| shift-click / shift-drag | select / select & move a box |
| drag selected box | move |
| drag handle | resize selected box |
| right-click | delete box under cursor |
| `0`–`9`, `.`, `-` | assign class (digit, dot, dash) + set current class |
| `←` `→` / `a` `d` | previous / next crop |
| `⏎` | save & next |
| `x` | mark crop empty/ignored & next |
| `z` | undo |
| `⌫` / `Del` | delete selected box |
| `Esc` | cancel drag / deselect |
| `f` | toggle "hide done" |
| `[` / `]` | zoom out / in |
| `Ctrl+S` | save now |

## Design notes & limitations

- **Selection is path-anchored** — the crop on display is tracked by its path,
  never by position in the (filtered) list, so saving or toggling the filter
  can never detach the displayed image/annotations from its file. The filter
  is purely a navigation predicate over prev/next.
- **Axis-aligned boxes only** — YOLO format has no rotation; the crop grid is
  close to axis-aligned anyway.
- **Single selection** — `Shift`-click selects exactly one box;
  shift-multi-select is intentionally omitted; cells hold only 1–3 characters.
- **Class set is data-driven** — editing `data.yaml` names changes the palette
  and keyboard map on next load; ids must stay contiguous from 0.
- **Strict on-disk format** — hand-edited label files with malformed lines are
  reported (400) rather than silently skipped, so training data never hides
  corruption.
