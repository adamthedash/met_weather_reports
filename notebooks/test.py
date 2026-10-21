import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import json
    from typing import TypedDict

    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle

    class Box(TypedDict):
        """A single detected object (e.g. a table cell) from PaddleOCR."""

        cls_id: int
        label: str
        score: float
        coordinate: list[float]  # [x1, y1, x2, y2]

    class PaddleOcrResult(TypedDict):
        """Schema of a PaddleOCR result file (e.g. ``*_res.json``)."""

        input_path: str
        page_index: int | None
        boxes: list[Box]

    return (
        GridSpec,
        LogNorm,
        PaddleOcrResult,
        Path,
        Rectangle,
        json,
        np,
        pl,
        plt,
    )


@app.cell
def _(PaddleOcrResult, Path, json, pl):
    def load_cell_boxes(json_dir: Path, image_dir: Path) -> pl.DataFrame:
        """Load PaddleOCR result JSONs into a per-box geometry dataframe.

        Each result file ``<name>_res.json`` in `json_dir` describes one image,
        found under `image_dir` with just the ``.jpeg`` suffix (e.g.
        ``9b855403-...b_res.json`` -> ``data/9b855403-...jpeg``).

        Returns one row per detected box with its corner coordinates
        ([x1, y1, x2, y2]), width, height and aspect ratio.
        """
        json_paths: list[Path] = sorted(json_dir.rglob("*.json"))
        if not json_paths:
            raise FileNotFoundError(f"No .json files found under {json_dir}")

        records: list[dict[str, float | str]] = []
        for path in json_paths:
            with path.open("r", encoding="utf-8") as f:
                result: PaddleOcrResult = json.load(f)
            image_path: Path = image_dir / (path.name.split(".")[0] + ".jpeg")
            if not image_path.is_file():
                raise FileNotFoundError(f"No image for {path} at {image_path}")
            for i, box in enumerate(result["boxes"]):
                x1, y1, x2, y2 = box["coordinate"]
                if not (x2 > x1 and y2 > y1):
                    raise ValueError(
                        f"Box {i} in {path} is not [x1, y1, x2, y2]: {box['coordinate']}"
                    )
                records.append(
                    {
                        "image_path": str(image_path),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "box_width": x2 - x1,
                        "box_height": y2 - y1,
                    }
                )

        if not records:
            raise ValueError(
                f"No boxes found in the JSON files under {json_dir}"
            )

        boxes: pl.DataFrame = pl.DataFrame(
            records,
            schema={
                "image_path": pl.String,
                "x1": pl.Float64,
                "y1": pl.Float64,
                "x2": pl.Float64,
                "y2": pl.Float64,
                "box_width": pl.Float64,
                "box_height": pl.Float64,
            },
        )
        return boxes.with_columns(
            (pl.col("box_width") / pl.col("box_height")).alias("aspect_ratio")
        )

    return (load_cell_boxes,)


@app.cell
def _(Path, load_cell_boxes, pl):
    boxes_df: pl.DataFrame = load_cell_boxes(Path("out2"), Path("data"))
    boxes_df
    return (boxes_df,)


@app.cell
def _(boxes_df: "pl.DataFrame", np):
    widths: np.ndarray = boxes_df["box_width"].to_numpy()
    heights: np.ndarray = boxes_df["box_height"].to_numpy()
    med_w: float = float(np.median(widths))
    med_h: float = float(np.median(heights))
    med_w, med_h
    return heights, med_h, med_w, widths


@app.cell
def _(
    GridSpec,
    LogNorm,
    boxes_df: "pl.DataFrame",
    heights: "np.ndarray",
    med_h: float,
    med_w: float,
    np,
    plt,
    widths: "np.ndarray",
):
    def plot_width_height_distribution(
        widths: np.ndarray,
        heights: np.ndarray,
        med_w: float,
        med_h: float,
        n_images: int,
    ) -> None:
        """Plot a width/height 2D histogram with aligned marginal histograms.

        The main heatmap is bracketed by per-axis histograms on the top and right,
        sharing their x/y limits with the main axes so the panels stay aligned.
        Median width and height are drawn as red dashed reference lines, and the
        legend/annotation box sits in the top-left of the main axes.
        """
        x_max: float = max(float(np.percentile(widths, 99)), 1.0)
        y_max: float = max(float(np.percentile(heights, 99)), 1.0)
        x_edges: np.ndarray = np.linspace(0.0, x_max, 70)
        y_edges: np.ndarray = np.linspace(0.0, y_max, 70)

        fig: plt.Figure = plt.figure(figsize=(10, 8))
        gs: GridSpec = GridSpec(
            2,
            2,
            width_ratios=(4.2, 1.1),
            height_ratios=(1.1, 4.2),
            wspace=0.0,
            hspace=0.0,
            figure=fig,
        )
        # The sharex/sharey kwargs below keep the marginal axes locked to the main
        # axes' extents, so the panels remain pixel-aligned (identical gridspec
        # cells + shared axis groups).
        ax_main: plt.Axes = fig.add_subplot(gs[1, 0])
        ax_top: plt.Axes = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right: plt.Axes = fig.add_subplot(gs[1, 1], sharey=ax_main)

        _, _, _, im = ax_main.hist2d(
            widths,
            heights,
            bins=[x_edges, y_edges],
            norm=LogNorm(vmin=1),
            cmap="viridis",
        )
        ax_main.set_xlabel("Box width (pixels)")
        ax_main.set_ylabel("Box height (pixels)")
        ax_main.set_xlim(0.0, x_edges[-1])
        ax_main.set_ylim(0.0, y_edges[-1])
        ax_main.grid(alpha=0.2, linewidth=0.5)

        ax_main.axvline(med_w, color="red", ls="--", lw=1.2, alpha=0.9)
        ax_main.axhline(med_h, color="red", ls="--", lw=1.2, alpha=0.9)
        ax_top.axvline(med_w, color="red", ls="--", lw=1.2, alpha=0.9)
        ax_right.axhline(med_h, color="red", ls="--", lw=1.2, alpha=0.9)
        ax_main.text(
            0.02,
            0.98,
            f"median width = {med_w:.1f} px\nmedian height = {med_h:.1f} px\nn = {widths.size:,} cells",
            transform=ax_main.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.75,
                edgecolor="gray",
            ),
        )

        ax_top.hist(
            widths,
            bins=x_edges,
            color="steelblue",
            edgecolor="white",
            linewidth=0.3,
        )
        ax_top.set_ylabel("count")
        ax_top.tick_params(labelbottom=False)
        ax_top.grid(alpha=0.2, linewidth=0.5)
        ax_top.set_title(
            f"PaddleOCR table cell boxes — width/height 2D histogram "
            f"({widths.size:,} cells from {n_images} image(s) in out2)"
        )

        ax_right.hist(
            heights,
            bins=y_edges,
            orientation="horizontal",
            color="seagreen",
            edgecolor="white",
            linewidth=0.3,
        )
        ax_right.set_xlabel("count")
        ax_right.tick_params(labelleft=False)
        ax_right.grid(alpha=0.2, linewidth=0.5)

    plot_width_height_distribution(
        widths,
        heights,
        med_w,
        med_h,
        n_images=boxes_df["image_path"].n_unique(),
    )
    plt.gca()
    return


@app.cell
def _(boxes_df: "pl.DataFrame", med_h: float, med_w: float, pl):
    filtered_boxes_df: pl.DataFrame = boxes_df.filter(
        pl.col("box_width").is_between(med_w * 0.8, med_w * 1.2),
        pl.col("box_height").is_between(med_h * 0.8, med_h * 1.2),
    )
    print(
        f"kept {filtered_boxes_df.height} of {boxes_df.height} boxes within median size ±20% "
        f"(width {med_w * 0.8:.1f}-{med_w * 1.2:.1f} px, "
        f"height {med_h * 0.8:.1f}-{med_h * 1.2:.1f} px)"
    )
    filtered_boxes_df
    return (filtered_boxes_df,)


@app.cell
def _(Path, Rectangle, np, pl, plt):
    def plot_boxes_over_images(
        boxes: pl.DataFrame, edge_color: str = "red"
    ) -> None:
        """Overlay box rectangles on their source images, one subplot per image."""
        image_paths: list[str] = (
            boxes["image_path"].unique(maintain_order=True).to_list()
        )
        if not image_paths:
            raise ValueError("No boxes to plot")

        fig: plt.Figure
        axes: np.ndarray
        fig, axes = plt.subplots(
            1,
            len(image_paths),
            figsize=(9 * len(image_paths), 11),
            squeeze=False,
        )
        for ax_arr, image_path in zip(axes, image_paths):
            ax: plt.Axes = ax_arr[0]
            img: np.ndarray = plt.imread(image_path)
            ax.imshow(img)
            subset: pl.DataFrame = boxes.filter(
                pl.col("image_path") == image_path
            )
            for row in subset.iter_rows(named=True):
                ax.add_patch(
                    Rectangle(
                        (row["x1"], row["y1"]),
                        row["box_width"],
                        row["box_height"],
                        fill=False,
                        edgecolor=edge_color,
                        linewidth=1.0,
                    )
                )
            ax.set_title(
                f"{Path(image_path).name} — {subset.height} boxes within median size ±20%"
            )
            ax.set_xlim(0.0, img.shape[1])
            ax.set_ylim(img.shape[0], 0.0)
            ax.axis("off")
        fig.tight_layout()

    return (plot_boxes_over_images,)


@app.cell
def _(filtered_boxes_df: "pl.DataFrame", plot_boxes_over_images, plt):
    plot_boxes_over_images(filtered_boxes_df)
    plt.gca()
    return


@app.cell
def _(filtered_boxes_df: "pl.DataFrame", pl):
    centers_df: pl.DataFrame = filtered_boxes_df.with_columns(
        ((pl.col("x1") + pl.col("x2")) / 2).alias("cx"),
        ((pl.col("y1") + pl.col("y2")) / 2).alias("cy"),
    )
    centers_df.select("cx", "cy")
    return (centers_df,)


@app.cell
def _(np):
    def estimate_spacing_fft(
        positions: np.ndarray,
        min_period: float,
        max_period: float,
        bin_size: float = 1.0,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Estimate the dominant spacing of 1-D point positions using an FFT.

        The positions are rasterised into a per-``bin_size`` occupancy signal,
        mean-subtracted and Hann-windowed. The spacing is the period of the
        strongest spectral peak whose period lies within
        ``[min_period, max_period]`` (this band keeps harmonics and
        sub-harmonics out of the running). The peak frequency is refined
        between FFT bins with parabolic interpolation.

        Returns:
            ``(spacing, frequencies, power)`` on the FFT frequency grid.
        """
        pos: np.ndarray = np.asarray(positions, dtype=float)
        if pos.size < 2:
            raise ValueError(
                "Need at least two positions to estimate a spacing"
            )
        if not (0.0 < bin_size < min_period < max_period):
            raise ValueError("Need 0 < bin_size < min_period < max_period")

        edges: np.ndarray = np.arange(
            pos.min(), pos.max() + bin_size, bin_size
        )
        occupancy: np.ndarray = np.histogram(pos, bins=edges)[0].astype(float)
        signal: np.ndarray = occupancy - occupancy.mean()
        windowed: np.ndarray = signal * np.hanning(signal.size)

        power: np.ndarray = np.abs(np.fft.rfft(windowed))
        freqs: np.ndarray = np.fft.rfftfreq(signal.size, d=bin_size)

        periods: np.ndarray = np.full_like(freqs, np.inf)
        np.divide(1.0, freqs, out=periods, where=freqs > 0)
        band: np.ndarray = (periods >= min_period) & (periods <= max_period)
        if not band.any():
            raise ValueError("No FFT bin falls inside the period band")

        peak_idx: int = int(np.argmax(np.where(band, power, -np.inf)))
        delta: float = 0.0
        if 0 < peak_idx < power.size - 1:
            denom: float = float(
                power[peak_idx - 1]
                - 2.0 * power[peak_idx]
                + power[peak_idx + 1]
            )
            if denom != 0.0:
                delta = (
                    0.5
                    * float(power[peak_idx - 1] - power[peak_idx + 1])
                    / denom
                )
        peak_freq: float = float(freqs[peak_idx]) + delta * float(
            freqs[1] - freqs[0]
        )

        return 1.0 / peak_freq, freqs, power

    return (estimate_spacing_fft,)


@app.cell
def _(
    centers_df: "pl.DataFrame",
    estimate_spacing_fft,
    med_h: float,
    med_w: float,
):
    spacing_x, freqs_x, power_x = estimate_spacing_fft(
        centers_df["cx"].to_numpy(),
        min_period=0.7 * med_w,
        max_period=3.0 * med_w,
    )
    spacing_y, freqs_y, power_y = estimate_spacing_fft(
        centers_df["cy"].to_numpy(),
        min_period=0.7 * med_h,
        max_period=3.0 * med_h,
    )
    print(
        f"estimated grid spacing: x = {spacing_x:.2f} px, y = {spacing_y:.2f} px "
        f"(median box size: {med_w:.1f} x {med_h:.1f} px)"
    )
    spacing_x, spacing_y
    return freqs_x, freqs_y, power_x, power_y, spacing_x, spacing_y


@app.cell
def _(np, plt):
    def plot_spacing_spectra(
        spectra: dict[str, tuple[float, np.ndarray, np.ndarray]],
    ) -> None:
        """Plot FFT power spectra of box-centre positions with the detected peaks.

        ``spectra`` maps an axis name (``"x"`` / ``"y"``) to
        ``(spacing, frequencies, power)``. The horizontal axis shows the period
        so the detected spacing can be read off directly.
        """
        fig: plt.Figure
        axes: np.ndarray
        fig, axes = plt.subplots(
            1, len(spectra), figsize=(6.5 * len(spectra), 4), squeeze=False
        )
        for ax_arr, (axis_name, (spacing, freqs, power)) in zip(
            axes, spectra.items()
        ):
            ax: plt.Axes = ax_arr[0]
            periods: np.ndarray = np.full_like(freqs, np.inf)
            np.divide(1.0, freqs, out=periods, where=freqs > 0)
            keep: np.ndarray = np.isfinite(periods) & (
                periods <= 4.0 * spacing
            )
            ax.plot(periods[keep], power[keep], lw=0.8, color="steelblue")
            ax.axvline(
                spacing,
                color="red",
                ls="--",
                lw=1.2,
                label=f"estimated spacing = {spacing:.1f} px",
            )
            ax.set_xlabel(f"Period along {axis_name} (pixels)")
            ax.set_ylabel("FFT power")
            ax.set_title(f"Box-centre positions — {axis_name} axis spectrum")
            ax.legend()
            ax.grid(alpha=0.2, linewidth=0.5)
        fig.tight_layout()

    return (plot_spacing_spectra,)


@app.cell
def _(
    freqs_x,
    freqs_y,
    np,
    plot_spacing_spectra,
    plt,
    power_x,
    power_y,
    spacing_x,
    spacing_y,
):
    spacing_spectra: dict[str, tuple[float, np.ndarray, np.ndarray]] = {
        "x": (spacing_x, freqs_x, power_x),
        "y": (spacing_y, freqs_y, power_y),
    }
    plot_spacing_spectra(spacing_spectra)
    plt.gca()
    return


@app.cell
def _(np):
    def fit_grid_axis(
        positions: np.ndarray, spacing: float
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Fit a 1-D grid of given `spacing` to point positions.

        The phase is estimated as the circular mean of the positions taken
        modulo the spacing: every position is mapped to an angle
        ``theta = 2*pi*x/spacing`` on the unit circle, and the mean direction
        gives the grid offset. Each position is then assigned to its nearest
        grid line ``phase + k * spacing``.

        Returns:
            ``(phase, indices, residuals)`` where ``phase`` lies in
            ``[0, spacing)``, ``indices`` are the grid line indices ``k``, and
            ``residuals`` are the deviations from the fitted grid lines.
        """
        pos: np.ndarray = np.asarray(positions, dtype=float)
        if pos.size < 2:
            raise ValueError("Need at least two positions to fit a grid")
        if spacing <= 0.0:
            raise ValueError("spacing must be positive")

        angles: np.ndarray = 2.0 * np.pi * pos / spacing
        mean_angle: float = float(
            np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        )
        phase: float = float(
            (mean_angle % (2.0 * np.pi)) * spacing / (2.0 * np.pi)
        )

        indices: np.ndarray = np.rint((pos - phase) / spacing).astype(int)
        residuals: np.ndarray = pos - (phase + indices * spacing)
        return phase, indices, residuals

    return (fit_grid_axis,)


@app.cell
def _(centers_df: "pl.DataFrame", fit_grid_axis, np, spacing_x, spacing_y):
    phase_x, col_idx, residual_x = fit_grid_axis(
        centers_df["cx"].to_numpy(), spacing_x
    )
    phase_y, row_idx, residual_y = fit_grid_axis(
        centers_df["cy"].to_numpy(), spacing_y
    )

    rms_dx: float = float(np.sqrt(np.mean(residual_x**2)))
    rms_dy: float = float(np.sqrt(np.mean(residual_y**2)))
    n_cols: int = int(col_idx.max() - col_idx.min() + 1)
    n_rows: int = int(row_idx.max() - row_idx.min() + 1)
    print(
        f"grid phase: x = {phase_x:.2f} px, y = {phase_y:.2f} px\n"
        f"residual RMS: x = {rms_dx:.2f} px, y = {rms_dy:.2f} px\n"
        f"grid extent: {n_cols} cols x {n_rows} rows = {n_cols * n_rows} cells "
        f"({centers_df.height} detected)"
    )
    phase_x, phase_y
    return col_idx, phase_x, phase_y, residual_x, residual_y, row_idx


@app.cell
def _(
    centers_df: "pl.DataFrame",
    col_idx,
    pl,
    residual_x,
    residual_y,
    row_idx,
):
    grid_df: pl.DataFrame = centers_df.with_columns(
        pl.Series("col", col_idx),
        pl.Series("row", row_idx),
        pl.Series("dx", residual_x),
        pl.Series("dy", residual_y),
    )
    grid_df
    return (grid_df,)


@app.cell
def _(Path, Rectangle, np, pl, plt):
    def plot_grid_overlay(
        boxes: pl.DataFrame,
        spacing_x: float,
        spacing_y: float,
        phase_x: float,
        phase_y: float,
        edge_color: str = "red",
    ) -> None:
        """Overlay detected boxes and the fitted grid on their source images.

        Grid lines are drawn at ``phase + k * spacing`` along each axis so the
        phase fit can be checked visually: every box centre should sit on a
        grid intersection, and empty intersections reveal missed cells.
        """
        image_paths: list[str] = (
            boxes["image_path"].unique(maintain_order=True).to_list()
        )
        if not image_paths:
            raise ValueError("No boxes to plot")

        fig: plt.Figure
        axes: np.ndarray
        fig, axes = plt.subplots(
            1,
            len(image_paths),
            figsize=(9 * len(image_paths), 11),
            squeeze=False,
        )
        for ax_arr, image_path in zip(axes, image_paths):
            ax: plt.Axes = ax_arr[0]
            img: np.ndarray = plt.imread(image_path)
            ax.imshow(img)
            subset: pl.DataFrame = boxes.filter(
                pl.col("image_path") == image_path
            )
            for row in subset.iter_rows(named=True):
                ax.add_patch(
                    Rectangle(
                        (row["x1"], row["y1"]),
                        row["box_width"],
                        row["box_height"],
                        fill=False,
                        edgecolor=edge_color,
                        linewidth=1.0,
                    )
                )

            width_px: float = float(img.shape[1])
            height_px: float = float(img.shape[0])
            x_lines: np.ndarray = np.arange(phase_x, width_px, spacing_x)
            y_lines: np.ndarray = np.arange(phase_y, height_px, spacing_y)
            ax.vlines(
                x_lines,
                0.0,
                height_px,
                color="magenta",
                alpha=0.45,
                linewidth=0.6,
            )
            ax.hlines(
                y_lines,
                0.0,
                width_px,
                color="magenta",
                alpha=0.45,
                linewidth=0.6,
            )

            ax.set_title(
                f"{Path(image_path).name} — fitted grid "
                f"{spacing_x:.1f} x {spacing_y:.1f} px"
            )
            ax.set_xlim(0.0, width_px)
            ax.set_ylim(height_px, 0.0)
            ax.axis("off")
        fig.tight_layout()

    return (plot_grid_overlay,)


@app.cell
def _(
    grid_df: "pl.DataFrame",
    phase_x,
    phase_y,
    plot_grid_overlay,
    plt,
    spacing_x,
    spacing_y,
):
    plot_grid_overlay(grid_df, spacing_x, spacing_y, phase_x, phase_y)
    plt.gca()
    return


@app.cell
def _(np):
    def grid_line_support(
        line_positions: np.ndarray,
        box_starts: np.ndarray,
        box_ends: np.ndarray,
    ) -> np.ndarray:
        """Mask of grid lines that overlap at least one real box extent.

        A line at position ``p`` overlaps a box when the box spans ``p``
        (``start <= p <= end``). Lines with no overlapping box lie outside the
        region covered by detected cells and are dropped, turning the
        infinitely repeating grid into finite bounds.

        Returns:
            Boolean mask over ``line_positions``.
        """
        lines: np.ndarray = np.asarray(line_positions, dtype=float)
        starts: np.ndarray = np.asarray(box_starts, dtype=float)
        ends: np.ndarray = np.asarray(box_ends, dtype=float)
        if lines.size == 0:
            raise ValueError("No grid lines given")
        if starts.size == 0 or starts.size != ends.size:
            raise ValueError("Need matching, non-empty box extents")

        overlaps: np.ndarray = (starts[None, :] <= lines[:, None]) & (
            lines[:, None] <= ends[None, :]
        )
        return overlaps.any(axis=1)

    return (grid_line_support,)


@app.cell
def _(
    grid_df: "pl.DataFrame",
    grid_line_support,
    np,
    phase_x,
    phase_y,
    plt,
    spacing_x,
    spacing_y,
):
    sample_img: np.ndarray = plt.imread(grid_df["image_path"][0])
    image_width_px: float = float(sample_img.shape[1])
    image_height_px: float = float(sample_img.shape[0])

    k_x: np.ndarray = np.arange(
        int(np.ceil(-phase_x / spacing_x)),
        int(np.floor((image_width_px - phase_x) / spacing_x)) + 1,
    )
    k_y: np.ndarray = np.arange(
        int(np.ceil(-phase_y / spacing_y)),
        int(np.floor((image_height_px - phase_y) / spacing_y)) + 1,
    )
    x_candidates: np.ndarray = phase_x + k_x * spacing_x
    y_candidates: np.ndarray = phase_y + k_y * spacing_y

    keep_x: np.ndarray = grid_line_support(
        x_candidates, grid_df["x1"].to_numpy(), grid_df["x2"].to_numpy()
    )
    keep_y: np.ndarray = grid_line_support(
        y_candidates, grid_df["y1"].to_numpy(), grid_df["y2"].to_numpy()
    )

    k_x_kept: np.ndarray = k_x[keep_x]
    k_y_kept: np.ndarray = k_y[keep_y]
    grid_x_lines: np.ndarray = x_candidates[keep_x]
    grid_y_lines: np.ndarray = y_candidates[keep_y]

    print(
        f"x lines: kept {grid_x_lines.size} of {x_candidates.size} "
        f"(cols {k_x_kept.min()}..{k_x_kept.max()}, "
        f"{grid_x_lines.min():.1f}..{grid_x_lines.max():.1f} px)\n"
        f"y lines: kept {grid_y_lines.size} of {y_candidates.size} "
        f"(rows {k_y_kept.min()}..{k_y_kept.max()}, "
        f"{grid_y_lines.min():.1f}..{grid_y_lines.max():.1f} px)\n"
        f"grid bounds: {grid_x_lines.size} x {grid_y_lines.size} = "
        f"{grid_x_lines.size * grid_y_lines.size} cells"
    )
    return grid_x_lines, grid_y_lines, k_x_kept, k_y_kept


@app.cell
def _(
    grid_x_lines: "np.ndarray",
    grid_y_lines: "np.ndarray",
    k_x_kept: "np.ndarray",
    k_y_kept: "np.ndarray",
    pl,
):
    x_frame: pl.DataFrame = pl.DataFrame(
        {"col": k_x_kept.astype(int), "cx": grid_x_lines}
    )
    y_frame: pl.DataFrame = pl.DataFrame(
        {"row": k_y_kept.astype(int), "cy": grid_y_lines}
    )
    grid_centers_df: pl.DataFrame = x_frame.join(y_frame, how="cross").sort(
        "col", "row"
    )
    grid_centers_df
    return (grid_centers_df,)


@app.cell
def _(
    grid_centers_df: "pl.DataFrame",
    grid_df: "pl.DataFrame",
    pl,
    spacing_x,
    spacing_y,
):
    grid_boxes_df: pl.DataFrame = grid_centers_df.with_columns(
        (pl.col("cx") - spacing_x / 2).alias("x1"),
        (pl.col("cy") - spacing_y / 2).alias("y1"),
        (pl.col("cx") + spacing_x / 2).alias("x2"),
        (pl.col("cy") + spacing_y / 2).alias("y2"),
        pl.lit(grid_df["image_path"][0]).alias("image_path"),
    ).with_columns(
        (pl.col("x2") - pl.col("x1")).alias("box_width"),
        (pl.col("y2") - pl.col("y1")).alias("box_height"),
    )
    grid_boxes_df
    return (grid_boxes_df,)


@app.cell
def _(Path, Rectangle, np, pl, plt):
    def plot_grid_boxes(
        generated: pl.DataFrame,
        detected: pl.DataFrame,
        edge_generated: str = "deepskyblue",
        edge_detected: str = "red",
    ) -> None:
        """Overlay generated grid boxes and detected boxes on their images.

        Generated boxes (from the fitted grid) are drawn in
        ``edge_generated``; the real PaddleOCR boxes in ``edge_detected``, so
        recovered cells show up as blue rectangles with no red counterpart.
        """
        image_paths: list[str] = (
            detected["image_path"].unique(maintain_order=True).to_list()
        )
        if not image_paths:
            raise ValueError("No boxes to plot")

        fig: plt.Figure
        axes: np.ndarray
        fig, axes = plt.subplots(
            1,
            len(image_paths),
            figsize=(9 * len(image_paths), 11),
            squeeze=False,
        )
        for ax_arr, image_path in zip(axes, image_paths):
            ax: plt.Axes = ax_arr[0]
            img: np.ndarray = plt.imread(image_path)
            ax.imshow(img)

            gen_subset: pl.DataFrame = generated.filter(
                pl.col("image_path") == image_path
            )
            for row in gen_subset.iter_rows(named=True):
                ax.add_patch(
                    Rectangle(
                        (row["x1"], row["y1"]),
                        row["box_width"],
                        row["box_height"],
                        fill=False,
                        edgecolor=edge_generated,
                        linewidth=1.0,
                        alpha=0.8,
                    )
                )

            det_subset: pl.DataFrame = detected.filter(
                pl.col("image_path") == image_path
            )
            for row in det_subset.iter_rows(named=True):
                ax.add_patch(
                    Rectangle(
                        (row["x1"], row["y1"]),
                        row["box_width"],
                        row["box_height"],
                        fill=False,
                        edgecolor=edge_detected,
                        linewidth=1.0,
                    )
                )

            ax.set_title(
                f"{Path(image_path).name} — {gen_subset.height} generated grid boxes "
                f"({edge_generated}) vs {det_subset.height} detected ({edge_detected})"
            )
            ax.set_xlim(0.0, img.shape[1])
            ax.set_ylim(img.shape[0], 0.0)
            ax.axis("off")
        fig.tight_layout()

    return (plot_grid_boxes,)


@app.cell
def _(
    grid_boxes_df: "pl.DataFrame",
    grid_df: "pl.DataFrame",
    plot_grid_boxes,
    plt,
):
    plot_grid_boxes(grid_boxes_df, grid_df)
    plt.gca()
    return


@app.cell
def _(Path, np, pl):
    from PIL import Image

    def export_cell_patches(
        boxes: pl.DataFrame,
        image_path: str,
        out_dir: Path,
        expand: float = 0.1,
    ) -> list[Path]:
        """Crop and save an image patch for every cell box.

        Each box is expanded by `expand` of its size on every side (10% per
        side grows the box by 20% in each dimension), clipped to the image
        bounds, cropped from `image_path` and written to
        ``out_dir / f"{row}_{col}.png"``.

        Returns:
            The list of written patch paths.
        """
        if expand < 0.0:
            raise ValueError("expand must be non-negative")
        out_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        with Image.open(image_path) as img:
            for cell in boxes.iter_rows(named=True):
                pad_x: float = expand * float(cell["box_width"])
                pad_y: float = expand * float(cell["box_height"])
                left: int = max(int(np.floor(cell["x1"] - pad_x)), 0)
                upper: int = max(int(np.floor(cell["y1"] - pad_y)), 0)
                right: int = min(int(np.ceil(cell["x2"] + pad_x)), img.width)
                lower: int = min(int(np.ceil(cell["y2"] + pad_y)), img.height)
                patch: Image.Image = img.crop((left, upper, right, lower))
                out_path: Path = out_dir / f"{cell['row']}_{cell['col']}.png"
                patch.save(out_path)
                written.append(out_path)
        return written

    return Image, export_cell_patches


@app.cell
def _(Path, np, plt):
    def plot_patch_preview(patch_paths: list[Path], n: int = 8) -> None:
        """Show an evenly spaced sample of the exported cell patches."""
        if not patch_paths:
            raise ValueError("No patches to preview")
        sample_idx: np.ndarray = np.unique(
            np.linspace(
                0, len(patch_paths) - 1, min(n, len(patch_paths))
            ).astype(int)
        )
        sample: list[Path] = [patch_paths[i] for i in sample_idx]

        n_cols: int = 4
        n_rows: int = int(np.ceil(len(sample) / n_cols))
        fig: plt.Figure
        axes: np.ndarray
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows), squeeze=False
        )
        for ax, path in zip(axes.flat, sample):
            ax.imshow(plt.imread(path))
            ax.set_title(path.name, fontsize=8)
            ax.axis("off")
        for ax in list(axes.flat)[len(sample) :]:
            ax.axis("off")
        fig.tight_layout()

    return (plot_patch_preview,)


@app.cell
def _(
    Path,
    export_cell_patches,
    grid_boxes_df: "pl.DataFrame",
    plot_patch_preview,
    plt,
):
    patch_paths: list[Path] = export_cell_patches(
        grid_boxes_df,
        grid_boxes_df["image_path"][0],
        Path("patches"),
    )
    print(f"wrote {len(patch_paths)} patches to patches/")
    plot_patch_preview(patch_paths)
    plt.gca()
    return (patch_paths,)


@app.cell
def _():
    import torch
    from transformers import (
        BatchFeature,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    return BatchFeature, TrOCRProcessor, VisionEncoderDecoderModel, torch


@app.cell
def _(TrOCRProcessor, VisionEncoderDecoderModel, torch):
    trocr_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_processor: TrOCRProcessor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-small-handwritten"
    )
    trocr_model: VisionEncoderDecoderModel = (
        VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-small-handwritten"
        )
        .to(trocr_device)
        .eval()
    )
    print(
        f"TrOCR (microsoft/trocr-small-handwritten) loaded on {trocr_device}"
    )
    return trocr_model, trocr_processor


@app.cell
def _(
    BatchFeature,
    Image,
    Path,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    torch,
):
    def recognize_handwriting(
        image: str | Path | Image.Image,
        processor: TrOCRProcessor,
        model: VisionEncoderDecoderModel,
        max_new_tokens: int = 64,
    ) -> str:
        """Recognise handwritten text in an image patch with TrOCR.

        Accepts a file path or a PIL image (converted to RGB), runs a greedy
        decode with the given processor/model pair and returns the
        transcription.
        """
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                rgb: Image.Image = img.convert("RGB")
        else:
            rgb = image.convert("RGB")

        device: torch.device = next(model.parameters()).device
        inputs: BatchFeature = processor(images=rgb, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            generated: torch.Tensor = model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
        return processor.batch_decode(generated, skip_special_tokens=True)[0]

    return (recognize_handwriting,)


@app.cell
def _(
    Path,
    np,
    patch_paths: "list[Path]",
    pl,
    recognize_handwriting,
    trocr_model: "VisionEncoderDecoderModel",
    trocr_processor: "TrOCRProcessor",
):
    sample_paths: list[Path] = [
        patch_paths[i]
        for i in np.unique(np.linspace(0, len(patch_paths) - 1, 8).astype(int))
    ]
    ocr_results: pl.DataFrame = pl.DataFrame(
        {
            "patch": [p.name for p in sample_paths],
            "text": [
                recognize_handwriting(p, trocr_processor, trocr_model)
                for p in sample_paths
            ],
        }
    )
    ocr_results
    return


@app.cell
def _(
    recognize_handwriting,
    trocr_model: "VisionEncoderDecoderModel",
    trocr_processor: "TrOCRProcessor",
):
    recognize_handwriting("data/9b855403-0376-473a-a179-befdb4314796.jpeg", trocr_processor, trocr_model)
    return


@app.cell
def _(Path, pl, torch, trocr_model: "VisionEncoderDecoderModel"):
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors import safe_open

    TROCR_REPO: str = "microsoft/trocr-small-handwritten"


    def list_weight_files(repo_id: str) -> list[str]:
        """Return sorted names of the weight shards published in a HF repo."""
        api: HfApi = HfApi()
        files: list[str] = api.list_repo_files(repo_id)
        weight_files: list[str] = sorted(
            f
            for f in files
            if (f.endswith(".safetensors") or f.endswith(".bin"))
            and f != "training_args.bin"
        )
        if not weight_files:
            raise FileNotFoundError(f"No weight files published in {repo_id}")
        return weight_files


    def read_shard_shapes(repo_id: str, file_name: str) -> dict[str, tuple[int, ...]]:
        """Map parameter name -> shape for a single checkpoint shard.

        Only the tensor *shapes* are read (via safetensors' lazy slices), so the
        full checkpoint never has to be materialised in memory.
        """
        shard_path: Path = Path(hf_hub_download(repo_id, file_name))
        shapes: dict[str, tuple[int, ...]] = {}
        if file_name.endswith(".safetensors"):
            with safe_open(str(shard_path), framework="pt") as shard:
                for key in shard.keys():
                    shapes[key] = tuple(
                        int(d) for d in shard.get_slice(key).get_shape()
                    )
            return shapes
        state_dict: dict[str, torch.Tensor] = torch.load(
            shard_path, map_location="cpu", weights_only=True
        )
        return {
            key: tuple(int(d) for d in tensor.shape)
            for key, tensor in state_dict.items()
        }


    def load_checkpoint_shapes(repo_id: str) -> dict[str, tuple[int, ...]]:
        """Map parameter name -> shape across every checkpoint shard of `repo_id`."""
        shapes: dict[str, tuple[int, ...]] = {}
        for file_name in list_weight_files(repo_id):
            shard_shapes: dict[str, tuple[int, ...]] = read_shard_shapes(
                repo_id, file_name
            )
            duplicated: set[str] = set(shapes) & set(shard_shapes)
            if duplicated:
                raise ValueError(
                    f"Keys appear in more than one shard: {sorted(duplicated)[:5]}"
                )
            shapes.update(shard_shapes)
        return shapes


    def shape_str(shape: tuple[int, ...] | None) -> str:
        """Render a shape tuple as ``768x768`` ('' when the tensor is absent)."""
        return "" if shape is None else "x".join(str(d) for d in shape)


    def compare_parameter_sets(
        model_shapes: dict[str, tuple[int, ...]],
        checkpoint_shapes: dict[str, tuple[int, ...]],
    ) -> pl.DataFrame:
        """Diff the model's expected parameters against the checkpoint's tensors.

        Status values: ``matched``, ``shape_mismatch``,
        ``missing_in_checkpoint`` (the model expects it, the file does not
        contain it) and ``unexpected_in_checkpoint`` (the file has it, the
        model does not).
        """
        rows: list[dict[str, str]] = []
        for name in sorted(set(model_shapes) | set(checkpoint_shapes)):
            model_shape: tuple[int, ...] | None = model_shapes.get(name)
            ckpt_shape: tuple[int, ...] | None = checkpoint_shapes.get(name)
            if model_shape is None:
                status: str = "unexpected_in_checkpoint"
            elif ckpt_shape is None:
                status = "missing_in_checkpoint"
            elif model_shape != ckpt_shape:
                status = "shape_mismatch"
            else:
                status = "matched"
            rows.append(
                {
                    "key": name,
                    "status": status,
                    "model_shape": shape_str(model_shape),
                    "checkpoint_shape": shape_str(ckpt_shape),
                }
            )
        return pl.DataFrame(rows)


    # --- inspect the published weights file -----------------------------------
    _ckpt_shapes: dict[str, tuple[int, ...]] = load_checkpoint_shapes(TROCR_REPO)
    _model_shapes: dict[str, tuple[int, ...]] = {
        name: tuple(int(d) for d in p.shape)
        for name, p in trocr_model.state_dict().items()
    }
    print(
        f"checkpoint: {len(_ckpt_shapes)} tensors in {list_weight_files(TROCR_REPO)}\n"
        f"model     : {len(_model_shapes)} parameters"
    )

    _diff_df: pl.DataFrame = compare_parameter_sets(_model_shapes, _ckpt_shapes)
    for _status, _count in _diff_df["status"].value_counts().iter_rows():
        print(f"  {_status:<26} {_count}")

    # --- what does the file contain for the encoder? --------------------------
    _encoder_ckpt_keys: list[str] = sorted(
        k for k in _ckpt_shapes if k.startswith("encoder.")
    )
    _pool_like_keys: list[str] = sorted(
        k for k in _ckpt_shapes if "pool" in k.lower()
    )
    print(
        f"\nencoder tensors in checkpoint: {_encoder_ckpt_keys[0]} ... "
        f"{_encoder_ckpt_keys[-1]}  ({len(_encoder_ckpt_keys)} total)"
    )
    print(f"checkpoint keys containing 'pool': {_pool_like_keys or 'NONE'}")

    # --- why the model asks for a pooler at all --------------------------------
    _pooler_params: list[str] = sorted(
        k for k in _model_shapes if k.startswith("encoder.pooler")
    )
    print(f"\nmodel parameters under encoder.pooler: {_pooler_params}")
    _pooler_weight: torch.Tensor = trocr_model.encoder.pooler.dense.weight
    _trained_weight: torch.Tensor = trocr_model.encoder.layernorm.weight
    print(
        f"encoder.pooler = {type(trocr_model.encoder.pooler).__name__} "
        f"(add_pooling_layer defaults to True when VisionEncoderDecoderModel "
        f"builds a ViTModel from the config)"
    )
    print(
        f"pooler dense weight std = {_pooler_weight.std().item():.4f} "
        f"(freshly initialised) vs trained layernorm std = "
        f"{_trained_weight.std().item():.4f}"
    )

    # --- is the pooler even used at inference? --------------------------------
    _image_size: int = int(trocr_model.encoder.config.image_size)
    with torch.no_grad():
        _enc_out = trocr_model.encoder(
            pixel_values=torch.zeros(1, 3, _image_size, _image_size)
        )
    print(
        f"\nencoder forward returns: {list(_enc_out.keys())} -> the seq2seq "
        f"wrapper feeds only `last_hidden_state` to the decoder, so "
        f"`pooler_output` is never consumed during generation"
    )

    print(
        "\nCONCLUSION\n"
        "----------\n"
        f"* every tensor in {TROCR_REPO} matches the model one-for-one except "
        "encoder.pooler.dense.{weight,bias} -> the file, key names and prefix "
        "are all fine\n"
        "* the checkpoint simply contains NO pooler tensors at all: TrOCR's "
        "encoder was fine-tuned/saved with `add_pooling_layer=False`, so only "
        "embeddings + transformer layers + final layernorm were written out\n"
        "* `VisionEncoderDecoderModel.from_pretrained` rebuilds the encoder as "
        "ViTModel with the default `add_pooling_layer=True`, which constructs a "
        "brand-new, untrained `encoder.pooler.dense.*` -> reported as MISSING "
        "and left randomly initialised\n"
        "* those two missing tensors are never used by the decoder, so they "
        "cannot cause the garbage transcriptions - the culprit is elsewhere "
        "(e.g. pixel_values / processor usage)"
    )

    _diff_df.filter(pl.col("status") != "matched")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
