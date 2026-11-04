marimo:
    pixi run marimo edit

# scrape subject images + metadata from a public Zooniverse project
scrape:
    pixi run python scripts/scrape_zooniverse.py --project met-rhonda/irish-weather-rescue --out download --limit 100

preprocess:
    pixi run python scripts/preprocess_pages.py

prep_dataset:
    pixi run python scripts/prepare_dataset.py --dataset datasets/digits

bootstrap model conf="0.25":
    pixi run python scripts/bootstrap_annotations.py --model {{model}} --conf {{conf}}

audit model conf="0.25":
    pixi run python scripts/audit_annotations.py --model {{model}} --conf {{conf}}

# combined labeler tool (tools/labeler): build the frontend and serve
labeler: labeler-build
    pixi run python -m tools.labeler

labeler-build:
    cd tools/labeler/frontend && pnpm install && pnpm build

# dev: API with autoreload (run 'just labeler-ui' in a second terminal)
labeler-api:
    pixi run uvicorn tools.labeler.app:app --host 127.0.0.1 --port 8125 --reload

labeler-ui:
    cd tools/labeler/frontend && pnpm dev
