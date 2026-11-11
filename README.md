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
   geometry so the patches can be re-assembled after inference. Two paths
   exist: the detector pipeline contrast-enhances the scan with CLAHE,
   detects the cells and reconstructs the day × month grid
   (`scripts/preprocess_pages_v1.py`); the ground-truth path cuts the daily
   table's guaranteed 12 × 31 equal-cell grid from the hand-drawn `daily`
   boxes in `datasets/tables` (`scripts/preprocess_pages_v2.py`).
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
   (`scripts/recombine_labels.py`; it reads the ground-truth pipeline's
   `table_grid.json`).
6. **Re-crop** the per-page labels back into per-patch labels when the patch
   cropping changes, so the annotations follow the new crops instead of being
   re-done (`scripts/crop_page_labels.py`; it reads the same `table_grid.json`
   the patches were cut with and re-derives the confirmed-empty patches from
   the sidecars).

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
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — table-cell detection
- [marimo](https://marimo.io) — notebooks
- [FastAPI](https://fastapi.tiangolo.com) and [Vite](https://vite.dev) /
  [Svelte](https://svelte.dev) — the labeler backend and frontend

## Repository structure

| Entry | Role |
|---|---|
| `pixi.toml` / `pixi.lock` | pixi workspace `met_weather`: Python environment and dependencies |
| `justfile` | task entry points for every pipeline stage (`scrape`, `preprocess`, `preprocess-tables`, `prep_dataset`, `bootstrap`, `audit`, `recombine`, `crop-labels`, `labeler`) and dev servers — the dataset-taking tasks accept a `dataset=` argument (default `datasets/digits`; `preprocess-tables` defaults to `datasets/tables`) |
| `scripts/` | the pipeline stages, one script each (see below) |
| `tools/labeler/` | combined annotation web app (annotate / review / batch review / audit stages, plus a read-only dataset summary page) — details in [`tools/labeler/README.md`](tools/labeler/README.md) |
| `tools/labeler/frontend/` | Vite + Svelte + TypeScript UI, built into `frontend/dist` |
| `notebooks/` | marimo notebooks: preprocessing prototypes (`preprocess.py`, `test.py`) and YOLO training (`yolo.py`) |
| `data/`, `datasets/`, `models/`, `runs/` | **gitignored, generated** working data: scraped pages/metadata/cell patches/QC renders/per-page labels (`data/`, including the ground-truth patch set under `data/table_patches/` and the re-cropped per-patch labels under `data/v2_patches/labels/`), the datasets (`datasets/digits/` character crops, `datasets/tables/` full-page table regions), model weights (`models/`), Ultralytics training runs (`runs/`) |

Pipeline scripts under `scripts/`:

| Script | Role |
|---|---|
| `scrape_zooniverse.py` | download page scans and metadata from the Zooniverse project |
| `preprocess_pages_v1.py` | contrast-enhance each page with CLAHE, detect table cells, reconstruct the grid, crop one image per cell, save the fitted grid geometry for reassembly |
| `preprocess_pages_v2.py` | cut the daily table's guaranteed 12 × 31 equal-cell grid from the hand-drawn `daily` boxes in `datasets/tables`, crop one image per cell, save `table_grid.json` for reassembly, and render the grid on the page as `qc/<page-id>/grid_overlay.jpg` for visual QC |
| `prepare_dataset.py` | write the YOLO train/val split files |
| `bootstrap_annotations.py` | model predictions for not-yet-annotated crops outside the manual queue (review queue) |
| `audit_annotations.py` | model predictions over every annotated crop (one label-audit round) |
| `recombine_labels.py` | map the per-patch labels back onto the page with the saved `table_grid.json` geometry, one YOLO label file per page, plus one `<source-id>.empty.json` sidecar per page recording the cell regions annotated as empty |
| `crop_page_labels.py` | map the per-page labels back onto a patch set's cell crops (the inverse of `recombine_labels.py`), one YOLO label file per patch under `data/v2_patches/labels`, with confirmed-empty patches re-derived from the `<source-id>.empty.json` sidecars |

For the annotation data layout and labeler details, see
[`tools/labeler/README.md`](tools/labeler/README.md).
