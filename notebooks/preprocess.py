import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from paddleocr import TableCellsDetection

    model = TableCellsDetection(
        model_name="RT-DETR-L_wireless_table_cell_det",
        engine="onnxruntime",
    )
    output = model.predict(
        "data/210736fe-7e78-45ad-80cf-217e3acc0df3.jpeg",
        threshold=0.3,
        batch_size=1,
    )
    for res in output:
        res.print(json_format=False)
        res.save_to_img("./output2/")
        res.save_to_json("./output2/res.json")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
