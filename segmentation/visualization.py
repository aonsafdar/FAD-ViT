"""
Publication-standard qualitative visualization for segmentation results.

Standard layout: Input | Ground truth (contour) | Prediction (contour) | Error map.
- Contour overlay: GT and prediction as outlines on grayscale image (green=GT, red=pred).
- Error map: True positive=green, False positive=red, False negative=blue (TN=dark).
Output: high DPI (300), optional PDF, no axes for clean figures.
"""
from pathlib import Path

import numpy as np


def _ensure_2d_grayscale(img: np.ndarray) -> np.ndarray:
    """Return (H, W) float in [0, 1] for display."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = img[0] if img.shape[0] in (1, 3) else img.mean(axis=0)
    if img.max() > 1.0:
        img = img / 255.0 if img.max() > 1 else img
    return np.clip(img, 0, 1)


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return (H, W) 0/1 uint8."""
    mask = np.asarray(mask, dtype=np.int64)
    if mask.ndim > 2:
        mask = mask.squeeze()
    return (mask > 0).astype(np.uint8)


def draw_contour_on_axis(
    ax,
    img: np.ndarray,
    mask: np.ndarray,
    color: str = "lime",
    linewidth: float = 1.5,
    level: float = 0.5,
) -> None:
    """Draw contour of mask on axis over grayscale img. mask: binary 0/1."""
    img = _ensure_2d_grayscale(img)
    mask = _ensure_binary_mask(mask).astype(np.float64)
    ax.imshow(img, cmap="gray", aspect="equal")
    try:
        from skimage import measure
        contours = measure.find_contours(mask, level=level)
        for contour in contours:
            # contour is (N, 2) with (row, col); matplotlib uses (col, row) for plot
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)
    except ImportError:
        ax.contour(mask, levels=[level], colors=[color], linewidths=[linewidth])
    ax.set_axis_off()
    ax.set_aspect("equal")


def build_error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    RGB error map: TP=green, FP=red, FN=blue, TN=black.
    Returns (H, W, 3) uint8 in [0, 255].
    """
    pred = _ensure_binary_mask(pred)
    target = _ensure_binary_mask(target)
    if pred.shape != target.shape:
        raise ValueError("pred and target shape mismatch")
    tp = (pred == 1) & (target == 1)
    fp = (pred == 1) & (target == 0)
    fn = (pred == 0) & (target == 1)
    # RGB: R=FP, G=TP, B=FN
    r = np.where(fp, 255, 0).astype(np.uint8)
    g = np.where(tp, 255, 0).astype(np.uint8)
    b = np.where(fn, 255, 0).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def save_qualitative_figure(
    img: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    path: Path,
    *,
    dpi: int = 300,
    fig_format: str = "png",
    layout: str = "four_panel",
    num_classes: int = 2,
) -> None:
    """
    Save publication-style qualitative figure.

    layout:
      - "four_panel": Input | GT contour | Pred contour | Error map (default)
      - "overlay_only": Input with GT (green) and Pred (red) contours on same image
      - "side_by_side": Input | GT overlay | Pred overlay (semi-transparent)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = _ensure_2d_grayscale(img)
    # For multi-class, use foreground (class 1) for binary-style contour; or all classes
    if target.max() > 1 or pred.max() > 1:
        target_bin = (target > 0).astype(np.uint8)
        pred_bin = (pred > 0).astype(np.uint8)
    else:
        target_bin = _ensure_binary_mask(target)
        pred_bin = _ensure_binary_mask(pred)

    if layout == "overlay_only":
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(img, cmap="gray", aspect="equal")
        try:
            from skimage import measure
            for mask, color in [(target_bin, "lime"), (pred_bin, "red")]:
                m = mask.astype(np.float64)
                contours = measure.find_contours(m, level=0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.5)
        except ImportError:
            ax.contour(target_bin.astype(np.float64), levels=[0.5], colors=["lime"], linewidths=[1.5])
            ax.contour(pred_bin.astype(np.float64), levels=[0.5], colors=["red"], linewidths=[1.5])
        ax.set_axis_off()
        ax.set_aspect("equal")
        plt.tight_layout(pad=0)
        plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, format=fig_format)
        plt.close()
        return

    if layout == "side_by_side":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, label, mask in zip(axes, ["Input", "Ground Truth", "Prediction"], [None, target_bin, pred_bin]):
            ax.imshow(img, cmap="gray", aspect="equal")
            if mask is not None:
                ax.imshow(mask, alpha=0.4, cmap="Greens", vmin=0, vmax=1)
            ax.set_title(label, fontsize=11)
            ax.set_axis_off()
            ax.set_aspect("equal")
        plt.tight_layout(pad=0.5)
        plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, format=fig_format)
        plt.close()
        return

    # four_panel (default)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img, cmap="gray", aspect="equal")
    axes[0].set_title("Input", fontsize=11)
    axes[0].set_axis_off()
    axes[0].set_aspect("equal")

    draw_contour_on_axis(axes[1], img, target_bin, color="lime", linewidth=1.5)
    axes[1].set_title("Ground Truth", fontsize=11)

    draw_contour_on_axis(axes[2], img, pred_bin, color="red", linewidth=1.5)
    axes[2].set_title("Prediction", fontsize=11)

    err_rgb = build_error_map(pred_bin, target_bin)
    axes[3].imshow(img, cmap="gray", aspect="equal")
    axes[3].imshow(err_rgb, alpha=0.6, aspect="equal")
    axes[3].set_title("Error (TP=G, FP=R, FN=B)", fontsize=10)
    axes[3].set_axis_off()
    axes[3].set_aspect("equal")

    plt.tight_layout(pad=0.5)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, format=fig_format)
    plt.close()


def save_qualitative_grid(
    imgs: list,
    targets: list,
    preds: list,
    path: Path,
    *,
    nrows: int = 2,
    ncols: int = 4,
    dpi: int = 300,
    fig_format: str = "png",
    layout: str = "overlay_only",
) -> None:
    """Save a grid of qualitative figures (e.g. 2x4 samples) for paper figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(nrows * ncols, len(imgs), len(targets), len(preds))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for idx in range(n):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        img = _ensure_2d_grayscale(imgs[idx])
        tb = _ensure_binary_mask(targets[idx]).astype(np.float64)
        pb = _ensure_binary_mask(preds[idx]).astype(np.float64)
        ax.imshow(img, cmap="gray", aspect="equal")
        try:
            from skimage import measure
            for m, color in [(tb, "lime"), (pb, "red")]:
                contours = measure.find_contours(m, level=0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.2)
        except ImportError:
            ax.contour(tb, levels=[0.5], colors=["lime"], linewidths=[1.2])
            ax.contour(pb, levels=[0.5], colors=["red"], linewidths=[1.2])
        ax.set_axis_off()
        ax.set_aspect("equal")
    for idx in range(n, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r, c].set_visible(False)
    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, format=fig_format)
    plt.close()


def save_comparison_figure(
    img: np.ndarray,
    target: np.ndarray,
    preds_by_model: dict,
    path: Path,
    *,
    dpi: int = 300,
    fig_format: str = "png",
    model_labels: list = None,
) -> None:
    """
    Publication-quality comparison of multiple models on one sample.

    Layout: Input | Ground truth (contour) | Model1 pred (contour) | Model2 pred | Model3 pred ...
    preds_by_model: dict mapping model name -> prediction array (e.g. {"FADViT": pred, "ViT": pred}).
    model_labels: optional list of display names in order (default: keys of preds_by_model).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = _ensure_2d_grayscale(img)
    target_bin = _ensure_binary_mask(target).astype(np.float64)
    labels = model_labels or list(preds_by_model.keys())
    n_models = len(labels)
    n_cols = 2 + n_models  # Input | GT | M1 | M2 | M3
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.2))
    if n_cols == 1:
        axes = [axes]
    # Input
    axes[0].imshow(img, cmap="gray", aspect="equal")
    axes[0].set_title("Input", fontsize=11)
    axes[0].set_axis_off()
    axes[0].set_aspect("equal")
    # GT
    draw_contour_on_axis(axes[1], img, target_bin, color="lime", linewidth=1.5)
    axes[1].set_title("Ground truth", fontsize=11)
    # Per-model prediction
    for i, name in enumerate(labels):
        pred = preds_by_model[name]
        pred_bin = _ensure_binary_mask(pred).astype(np.float64)
        draw_contour_on_axis(axes[2 + i], img, pred_bin, color="red", linewidth=1.5)
        axes[2 + i].set_title(name, fontsize=10)
    plt.tight_layout(pad=0.4)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, format=fig_format)
    plt.close()


def save_comparison_grid(
    imgs: list,
    targets: list,
    preds_by_model: list,
    path: Path,
    *,
    model_labels: list = None,
    nrows: int = 2,
    ncols: int = 3,
    dpi: int = 300,
    fig_format: str = "png",
) -> None:
    """
    Grid of comparison figures: each cell = one sample, (Input|GT|M1|M2|M3) per sample.
    preds_by_model: list of dicts, one per sample; each dict maps model name -> pred array.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(nrows * ncols, len(imgs), len(targets), len(preds_by_model))
    labels = model_labels or (list(preds_by_model[0].keys()) if preds_by_model else [])
    n_models = len(labels)
    n_panels = 2 + n_models
    fig, axes = plt.subplots(nrows, ncols * n_panels, figsize=(2.5 * n_panels * ncols, 2.5 * nrows))
    axes = np.atleast_2d(axes)
    for idx in range(n):
        row, col = idx // ncols, idx % ncols
        base = col * n_panels
        img = _ensure_2d_grayscale(imgs[idx])
        target_bin = _ensure_binary_mask(targets[idx]).astype(np.float64)
        axes[row, base + 0].imshow(img, cmap="gray", aspect="equal")
        axes[row, base + 0].set_axis_off()
        axes[row, base + 0].set_aspect("equal")
        draw_contour_on_axis(axes[row, base + 1], img, target_bin, color="lime", linewidth=1.2)
        axes[row, base + 1].set_axis_off()
        for i, name in enumerate(labels):
            pred_bin = _ensure_binary_mask(preds_by_model[idx][name]).astype(np.float64)
            draw_contour_on_axis(axes[row, base + 2 + i], img, pred_bin, color="red", linewidth=1.2)
            axes[row, base + 2 + i].set_axis_off()
    for idx in range(n, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        base = col * n_panels
        for j in range(n_panels):
            axes[row, base + j].set_visible(False)
    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, format=fig_format)
    plt.close()


def save_datasets_comparison(
    cols_data: list,
    path: Path,
    *,
    dpi: int = 300,
    fig_format: str = "png",
    model_labels: list = None,
    model_display: str = "prediction",
) -> None:
    """
    One figure: Rows = Ground truth | model1 | model2 | ..., Columns = datasets.

    No Original or Error row. Row labels vertical on the left.
    cols_data: list of (dataset_label, img, target, preds_by_model) — one per dataset (column).
    model_display: "prediction" = contour overlay per model; "error" = error map (TP=G, FP=R, FN=B) per model vs GT.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = model_labels or (list(cols_data[0][3].keys()) if cols_data and cols_data[0][3] else [])
    n_models = len(labels)
    n_rows = 1 + n_models  # Ground truth | models
    n_cols = len(cols_data)
    row_titles = ["Ground truth"] + list(labels)
    if model_display == "error":
        row_titles = ["Ground truth"] + [f"{l} (error)" for l in labels]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    for col, (dataset_label, img, target, preds_by_model) in enumerate(cols_data):
        img = _ensure_2d_grayscale(img)
        target_bin = _ensure_binary_mask(target).astype(np.float64)
        # Row 0: Ground truth (column title = dataset name)
        draw_contour_on_axis(axes[0, col], img, target_bin, color="lime", linewidth=1.5)
        axes[0, col].set_axis_off()
        axes[0, col].set_title(dataset_label, fontsize=14, fontweight="bold")
        # Rows 1..n_models: each model — prediction (contour) or error map
        for i, name in enumerate(labels):
            pred_bin = _ensure_binary_mask(preds_by_model[name]).astype(np.float64)
            if model_display == "prediction":
                draw_contour_on_axis(axes[1 + i, col], img, pred_bin, color="red", linewidth=1.5)
                axes[1 + i, col].set_axis_off()
            else:
                err_rgb = build_error_map(pred_bin, target_bin)
                axes[1 + i, col].imshow(img, cmap="gray", aspect="equal")
                axes[1 + i, col].imshow(err_rgb, alpha=0.6, aspect="equal")
                axes[1 + i, col].set_axis_off()
    plt.tight_layout(pad=0.4)
    # Vertical row labels on the left
    for row, title in enumerate(row_titles):
        bbox = axes[row, 0].get_position()
        fig.text(
            bbox.x0 - 0.008, (bbox.y0 + bbox.y1) / 2, title,
            rotation=90, fontsize=14, fontweight="bold",
            va="center", ha="right", transform=fig.transFigure,
        )
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, format=fig_format)
    plt.close()


def save_one_model_across_datasets(
    rows_data: list,
    path: Path,
    *,
    row_titles: list = None,
    dpi: int = 300,
    fig_format: str = "png",
) -> None:
    """
    One figure: Rows = Original | GT | Prediction | Error, Columns = datasets.

    rows_data: list of (dataset_label, img, target, pred) — one per dataset (column).
    row_titles: optional list of 4 row titles (default: Original, Ground truth, Prediction, Error).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 4  # Original, GT, Prediction, Error
    n_cols = len(rows_data)
    if row_titles is None:
        row_titles = ["Original", "Ground truth", "Prediction", "Error (TP=G, FP=R, FN=B)"]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    for col, (dataset_label, img, target, pred) in enumerate(rows_data):
        img = _ensure_2d_grayscale(img)
        target_bin = _ensure_binary_mask(target).astype(np.float64)
        pred_bin = _ensure_binary_mask(pred).astype(np.float64)
        # Row 0: Original (with dataset name in panel)
        axes[0, col].imshow(img, cmap="gray", aspect="equal")
        axes[0, col].set_axis_off()
        axes[0, col].set_aspect("equal")
        axes[0, col].set_title(dataset_label, fontsize=14, fontweight="bold")
        # Row 1: GT contour
        draw_contour_on_axis(axes[1, col], img, target_bin, color="lime", linewidth=1.5)
        axes[1, col].set_axis_off()
        # Row 2: Prediction contour
        draw_contour_on_axis(axes[2, col], img, pred_bin, color="red", linewidth=1.5)
        axes[2, col].set_axis_off()
        # Row 3: Error map
        err_rgb = build_error_map(pred_bin, target_bin)
        axes[3, col].imshow(img, cmap="gray", aspect="equal")
        axes[3, col].imshow(err_rgb, alpha=0.6, aspect="equal")
        axes[3, col].set_axis_off()
    plt.tight_layout(pad=0.4)
    # Vertical row labels (same style as column titles, placed just left of first column)
    for row, title in enumerate(row_titles):
        bbox = axes[row, 0].get_position()
        fig.text(
            bbox.x0 - 0.008, (bbox.y0 + bbox.y1) / 2, title,
            rotation=90, fontsize=14, fontweight="bold",
            va="center", ha="right", transform=fig.transFigure,
        )
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, format=fig_format)
    plt.close()
