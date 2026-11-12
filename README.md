# met_weather_reports

Digitising historical Irish weather observations from scanned rainfall register
pages.

## Purpose

This project works toward digitising the handwritten weather observations held
in Met Éireann's National Climate Archive, whose scanned register pages are
published as subjects of the Zooniverse
[Irish Weather Rescue](https://www.zooniverse.org/projects/met-rhonda/irish-weather-rescue)
citizen-science project.

It is a local pipeline plus tooling that turns raw scanned register pages into
an annotated character-detection dataset:

1. **Scrape** the page scans and per-subject provenance metadata from the
   Zooniverse project (`scripts/scrape_zooniverse.py`).
2. **Preprocess** each page into one image per table cell, saving the grid
   geometry so the patches can be re-assembled after inference. The pipeline
   cuts the daily table's guaranteed 12 × 31 equal-cell grid from the
   hand-drawn `daily` boxes in `datasets/tables`, writes one patch per cell
   and saves the grid as `table_grid.json` (`scripts/preprocess_pages.py`).
3. **Label** the handwritten characters in each cell crop — or any other
   region of a page; the class set and its presentation are defined per
   dataset in its `data.yaml`/`classes.yaml` — with the in-repo labeler app
   (`tools/labeler`).
4. **Train** a YOLO character detector over the labeled crops and use it to
   bootstrap and audit annotations at scale
   (`scripts/bootstrap_annotations.py`, `scripts/audit_annotations.py`,
   `notebooks/yolo.py`).
5. **Recombine** the accepted per-patch labels back onto the pages with the
   grid geometry saved in step 2, one YOLO label file per page plus a
   per-page sidecar recording the confirmed-empty cell regions
   (`scripts/recombine_labels.py`; it reads the preprocessing pipeline's
   `table_grid.json`). A character labeled on two overlapping crops maps back
   onto two near-identical boxes; same-class pairs from different patches
   overlapping at least 80% are merged into their enclosing bounding box.
6. **Re-crop** the per-page labels back into per-patch labels when the patch
   cropping changes, so the annotations follow the new crops instead of being
   re-done (`scripts/crop_page_labels.py`; it reads the same `table_grid.json`
   the patches were cut with, re-derives the confirmed-empty patches from the
   sidecars, and omits patch copies that retain less than half a box's area by
   default — `--min-retained-area`; a patch holding only such fragments stays
   unlabeled).

Exploratory work behind the pipeline stages lives in marimo notebooks.

## External resources

- [Irish Weather Rescue](https://www.zooniverse.org/projects/met-rhonda/irish-weather-rescue)
  — the Zooniverse project publishing the scanned register pages
- [Irish Weather Rescue at UCD](https://citizenscience.ucd.ie/irish-weather-rescue-met-eireanns-data-rescue-citizen-science-project)
  — project background from University College Dublin
- [Zooniverse Panoptes API](https://panoptes.zooniverse.org/api) — API used by
  the scraper

Tools and technologies the pipeline is built on:

- [pixi](https://pixi.sh) — Python environment and task runner (`pixi.toml`,
  `justfile`)
- [Ultralytics YOLO](https://docs.ultralytics.com) — detector training and
  inference
- [marimo](https://marimo.io) — notebooks
- [FastAPI](https://fastapi.tiangolo.com) and [Vite](https://vite.dev) /
  [Svelte](https://svelte.dev) — the labeler backend and frontend

## Repository structure

| Entry | Role |
|---|---|
| `pixi.toml` / `pixi.lock` | pixi workspace `met_weather`: Python environment and dependencies |
| `justfile` | task entry points for every pipeline stage (`scrape`, `preprocess`, `prep_dataset`, `bootstrap`, `audit`, `recombine`, `crop-labels`, `dedupe-labels`, `labeler`) and dev servers — the dataset-taking tasks accept a `dataset=` argument (default `datasets/digits`; `preprocess` defaults to `datasets/tables` and writes to `data/v2_patches`) |
| `scripts/` | the pipeline stages, one script each (see below) |
| `tools/labeler/` | combined annotation web app (annotate / review / batch review / audit stages, plus a read-only dataset summary page) — details in [`tools/labeler/README.md`](tools/labeler/README.md) |
| `tools/labeler/frontend/` | Vite + Svelte + TypeScript UI, built into `frontend/dist` |
| `notebooks/` | marimo notebooks: YOLO training (`yolo.py`) |
| `data/`, `datasets/`, `models/`, `runs/` | **gitignored, generated** working data: scraped pages/metadata (`data/images`, `data/metadata`), the page patch set with its QC renders and `table_grid.json` geometry, page labels and re-cropped per-patch labels under `data/v2_patches/`, the datasets (`datasets/digits/` character crops, `datasets/tables/` full-page table regions), model weights (`models/`), Ultralytics training runs (`runs/`) |

Pipeline scripts under `scripts/`:

| Script | Role |
|---|---|
| `scrape_zooniverse.py` | download page scans and metadata from the Zooniverse project |
| `preprocess_pages.py` | cut the daily table's guaranteed 12 × 31 equal-cell grid from the hand-drawn `daily` boxes in `datasets/tables`, crop one image per cell, save `table_grid.json` for reassembly, and render the grid on the page as `qc/<page-id>/grid_overlay.jpg` for visual QC |
| `prepare_dataset.py` | write the YOLO train/val split files |
| `bootstrap_annotations.py` | model predictions for not-yet-annotated crops outside the manual queue (review queue) |
| `audit_annotations.py` | model predictions over every annotated crop (one label-audit round) |
| `recombine_labels.py` | map the per-patch labels back onto the page with the saved `table_grid.json` geometry, one YOLO label file per page (same-class boxes from different patches overlapping at least 80% are merged into their enclosing box), plus one `<source-id>.empty.json` sidecar per page recording the cell regions annotated as empty |
| `crop_page_labels.py` | map the per-page labels back onto a patch set's cell crops (the inverse of `recombine_labels.py`), one YOLO label file per patch under `data/v2_patches/labels`, with confirmed-empty patches re-derived from the `<source-id>.empty.json` sidecars; patch copies retaining less than `--min-retained-area` (default 0.5) of the box's original area are omitted, and a patch holding only such fragments stays unlabeled |
| `dedupe_page_labels.py` | remove cropped duplicate boxes from the per-page labels under `data/v2_patches/page_labels` in place: a same-class box lying at least `--min-containment` (default 0.95) inside a larger same-class box is removed as a clipped copy, leaving only the outermost box; files without duplicates are untouched and `--dry-run` reports without writing (re-run after `recombine`) |

For the annotation data layout and labeler details, see
[`tools/labeler/README.md`](tools/labeler/README.md).
