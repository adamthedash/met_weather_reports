import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from PIL import Image, ImageFilter
    import numpy as np
    import matplotlib.pyplot as plt

    return Image, np, plt


@app.cell
def _(Image):
    img = Image.open("data/images/103801210.jpeg").convert("L")
    img
    return (img,)


@app.cell
def _(img, np):
    img_np = np.asarray(img)
    img_np.shape
    return (img_np,)


@app.cell
def _():
    import cv2

    return (cv2,)


@app.cell
def _(Image, cv2, img_np):
    clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=[16] * 2)
    img_clahed_np = clahe.apply(img_np)
    Image.fromarray(img_clahed_np)
    return (img_clahed_np,)


@app.cell
def _(Image, cv2, img_clahed_np):
    img_blurred_np = cv2.blur(img_clahed_np, (3, 9))
    Image.fromarray(img_blurred_np)
    return (img_blurred_np,)


@app.cell
def _():
    from scipy.ndimage import convolve

    return


@app.cell
def _(img_blurred_np, plt):
    x = img_blurred_np.mean(axis=1)
    plt.plot(x)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
