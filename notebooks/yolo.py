import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from ultralytics import YOLO

    return (YOLO,)


@app.cell
def _(YOLO):
    model = YOLO("models/yolo26n.pt")
    model
    return (model,)


@app.cell
def _(model):
    res = model.predict("patches/8_5.png", nms=False)
    res[0]
    return (res,)


@app.cell
def _(res):
    res[0].boxes
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
