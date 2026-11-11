import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    return (YOLO,)


@app.cell
def _(YOLO):
    model = YOLO("models/yolo26n.pt")
    model
    return (model,)


@app.cell
def _(model):
    def disable_all_augmentation() -> dict:
        """Return a dictionary of training kwargs that disable all image augmentation."""
        return {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
            "augmentations": []
        }

    def train(dataset_name: str):
        # https://docs.ultralytics.com/usage/cfg#train-settings
        results = model.train(
            data=f"datasets/{dataset_name}/data.yaml",
            project=dataset_name,
            epochs=100,
            patience=20,
            imgsz=1024, # Bigger size better for whatever reason
            cache=True,
            **disable_all_augmentation(),
        )

    return (train,)


@app.cell
def _(train):
    train("tables")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
