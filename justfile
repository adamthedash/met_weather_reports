marimo:
    pixi run marimo edit

annotate:
    pixi run python tools/annotate.py

prep_dataset:
    pixi run python scripts/prepare_dataset.py --dataset datasets/digits

bootstrap model conf="0.25":
    pixi run python scripts/bootstrap_annotations.py --model {{model}} --conf {{conf}}
