marimo:
    pixi run marimo edit

# scrape subject images + metadata from a public Zooniverse project
scrape:
    pixi run python scripts/scrape_zooniverse.py --project met-rhonda/irish-weather-rescue --out data --limit 100

preprocess_v1:
    pixi run python scripts/preprocess_pages_v1.py

# cut daily-table cell patches from the hand-labelled table regions
preprocess_v2 dataset="datasets/tables" out="data/v2_patches":
    pixi run python scripts/preprocess_pages_v2.py --dataset {{dataset}} --out {{out}}

prep_dataset dataset="datasets/digits":
    pixi run python scripts/prepare_dataset.py --dataset {{dataset}}

bootstrap model dataset="datasets/digits" conf="0.25" batch="32":
    pixi run python scripts/bootstrap_annotations.py --model {{model}} --dataset {{dataset}} --conf {{conf}} --batch-size {{batch}}

audit model dataset="datasets/digits" conf="0.25" batch="32":
    pixi run python scripts/audit_annotations.py --model {{model}} --dataset {{dataset}} --conf {{conf}} --batch-size {{batch}}

recombine:
    pixi run python scripts/recombine_labels.py

# re-crop the per-page labels onto a patch set's exported cell crops
crop-labels:
    pixi run python scripts/crop_page_labels.py

# combined labeler tool (tools/labeler): build the frontend and serve
labeler dataset="datasets/digits": labeler-build
    pixi run python -m tools.labeler --dataset {{dataset}}

labeler-build:
    cd tools/labeler/frontend && pnpm install && pnpm build
