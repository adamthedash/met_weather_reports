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
2. **Preprocess** each page: detect the table cells, reconstruct the
   day × month grid, and crop one image per cell (`scripts/preprocess_pages.py`).
3. **Label** the handwritten characters in each cell crop (classes `0-9`,
   `dash`, `dot`) with the in-repo labeler app (`tools/labeler`).
4. **Train** a YOLO character detector over the labeled crops and use it to
   bootstrap and audit annotations at scale
   (`scripts/bootstrap_annotations.py`, `scripts/audit_annotations.py`,
   `notebooks/yolo.py`).

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
| `justfile` | task entry points for every pipeline stage (`scrape`, `preprocess`, `prep_dataset`, `bootstrap`, `audit`, `labeler`) and dev servers |
| `scripts/` | the pipeline stages, one script each (see below) |
| `tools/labeler/` | combined annotation web app (annotate / review / audit stages) — details in [`tools/labeler/README.md`](tools/labeler/README.md) |
| `tools/labeler/frontend/` | Vite + Svelte + TypeScript UI, built into `frontend/dist` |
| `notebooks/` | marimo notebooks: preprocessing prototypes (`preprocess.py`, `test.py`) and YOLO training (`yolo.py`) |
| `data/`, `datasets/`, `models/`, `runs/` | **gitignored, generated** working data: scraped pages/metadata/cell patches/QC renders (`data/`), the digits dataset (`datasets/digits/`), model weights (`models/`), Ultralytics training runs (`runs/`) |

Pipeline scripts under `scripts/`:

| Script | Role |
|---|---|
| `scrape_zooniverse.py` | download page scans and metadata from the Zooniverse project |
| `preprocess_pages.py` | detect table cells, reconstruct the grid, crop one image per cell |
| `prepare_dataset.py` | write the YOLO train/val split files |
| `bootstrap_annotations.py` | model predictions for not-yet-annotated crops (review queue) |
| `audit_annotations.py` | model predictions over every crop (one label-audit round) |

For the annotation data layout and labeler details, see
[`tools/labeler/README.md`](tools/labeler/README.md).
