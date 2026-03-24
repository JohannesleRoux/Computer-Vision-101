"""
utils/visualization.py
----------------------
Shared visualization helpers for the Computer Vision 101 course.
Handles Tensor / ndarray / PIL inputs uniformly.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# PIL and torch are imported lazily so the module stays importable even when
# only a subset of dependencies is installed.
try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # type: ignore

try:
    import torch
except ImportError:
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(img) -> np.ndarray:
    """Convert Tensor / PIL Image / ndarray to uint8 HxWxC numpy array."""
    if torch is not None and isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dtype == torch.float32 or img.dtype == torch.float64:
            img = img.clamp(0, 1).mul(255).byte()
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):  # CHW → HWC
            img = img.permute(1, 2, 0)
        img = img.numpy()
    elif PILImage is not None and isinstance(img, PILImage.Image):
        img = np.array(img)
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(img)}")
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    # Squeeze single-channel trailing dim
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    return img


def _default_colors(n: int) -> List[Tuple[float, float, float]]:
    """Return n visually distinct colors using the HSV color wheel."""
    return [plt.cm.hsv(i / max(n, 1))[:3] for i in range(n)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def show_image(
    img,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    cmap: Optional[str] = None,
) -> plt.Axes:
    """Display a single image.

    Parameters
    ----------
    img:
        Image as a numpy array (H×W or H×W×C), PIL Image, or torch Tensor
        (C×H×W or H×W×C).
    title:
        Optional axes title.
    ax:
        Existing ``matplotlib.axes.Axes`` to draw into.  Creates a new figure
        if *None*.
    cmap:
        Colormap passed to ``imshow`` (default: ``"gray"`` for single-channel).
    """
    arr = _to_numpy(img)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    if arr.ndim == 2 and cmap is None:
        cmap = "gray"
    ax.imshow(arr, cmap=cmap)
    if title:
        ax.set_title(title)
    ax.axis("off")
    return ax


def show_images_grid(
    images,
    titles: Optional[List[str]] = None,
    cols: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Display a list of images in a grid layout.

    Parameters
    ----------
    images:
        Iterable of images (numpy arrays, PIL Images, or Tensors).
    titles:
        Optional list of per-image titles.
    cols:
        Number of columns in the grid.
    figsize:
        Figure size override.
    """
    images = list(images)
    n = len(images)
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (cols * 3, rows * 3))
    axes_flat = np.array(axes).flatten()
    for i, (ax, img) in enumerate(zip(axes_flat, images)):
        arr = _to_numpy(img)
        cmap = "gray" if arr.ndim == 2 else None
        ax.imshow(arr, cmap=cmap)
        if titles and i < len(titles):
            ax.set_title(str(titles[i]), fontsize=9)
        ax.axis("off")
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    return fig


def draw_boxes(
    image,
    boxes,
    labels: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    threshold: float = 0.5,
    colors: Optional[List] = None,
    line_width: float = 2.0,
) -> plt.Figure:
    """Overlay bounding boxes on an image.

    Parameters
    ----------
    image:
        Background image.
    boxes:
        Array-like of shape (N, 4) in *xyxy* format.
    labels:
        Optional list of N string labels.
    scores:
        Optional list of N confidence scores.  Boxes below *threshold* are
        skipped when scores are provided.
    threshold:
        Score threshold for filtering.
    colors:
        Per-class colour list.  Defaults to a rainbow palette.
    line_width:
        Rectangle edge width in points.
    """
    arr = _to_numpy(image)
    boxes = np.asarray(boxes)
    if boxes.ndim == 1:
        boxes = boxes[np.newaxis]

    n = len(boxes)
    if colors is None:
        colors = _default_colors(max(n, 1))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)

    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < threshold:
            continue
        x1, y1, x2, y2 = box[:4]
        color = colors[i % len(colors)]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=line_width, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        # Build annotation text
        parts = []
        if labels:
            parts.append(str(labels[i]))
        if scores is not None:
            parts.append(f"{scores[i]:.2f}")
        if parts:
            ax.text(
                x1, y1 - 4, " ".join(parts),
                color="white", fontsize=8,
                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"),
            )

    ax.axis("off")
    plt.tight_layout()
    return fig


def draw_masks(
    image,
    masks,
    alpha: float = 0.5,
    colors: Optional[List] = None,
) -> plt.Figure:
    """Overlay binary instance masks on an image.

    Parameters
    ----------
    image:
        Background image (H×W×3 or similar).
    masks:
        Array-like of shape (N, H, W) with boolean or 0/1 values.
    alpha:
        Mask opacity.
    colors:
        Per-mask RGB colors (values in [0, 1]).
    """
    arr = _to_numpy(image).copy()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    masks = np.asarray(masks, dtype=bool)
    if masks.ndim == 2:
        masks = masks[np.newaxis]

    n = len(masks)
    if colors is None:
        colors = _default_colors(max(n, 1))

    overlay = arr.astype(np.float32)
    for i, mask in enumerate(masks):
        color = np.array(colors[i % len(colors)]) * 255
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(overlay.clip(0, 255).astype(np.uint8))
    ax.axis("off")
    plt.tight_layout()
    return fig


def draw_panoptic(
    image,
    panoptic_seg: np.ndarray,
    segments_info: List[dict],
) -> plt.Figure:
    """Visualise a panoptic segmentation map.

    Parameters
    ----------
    image:
        Background image.
    panoptic_seg:
        Integer array of shape (H, W).  Each pixel holds a *segment id*.
    segments_info:
        List of dicts with at minimum keys ``id``, ``label_id``,
        ``is_thing`` (bool) and optionally ``category_name``.
    """
    arr = _to_numpy(image).copy()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    n = len(segments_info)
    palette = _default_colors(max(n, 1))
    color_map = {seg["id"]: palette[i] for i, seg in enumerate(segments_info)}

    colored = np.zeros_like(arr, dtype=np.float32)
    for seg_id, color in color_map.items():
        mask = panoptic_seg == seg_id
        colored[mask] = np.array(color) * 255

    blended = (0.5 * arr.astype(np.float32) + 0.5 * colored).clip(0, 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.imshow(blended)
    # Legend
    patches = []
    for i, seg in enumerate(segments_info):
        name = seg.get("category_name", f"id={seg['id']}")
        kind = "thing" if seg.get("is_thing") else "stuff"
        patches.append(mpatches.Patch(color=palette[i], label=f"{name} ({kind})"))
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    ax.axis("off")
    plt.tight_layout()
    return fig


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Convert an optical flow field to an HSV colour-wheel visualisation.

    Parameters
    ----------
    flow:
        Float array of shape (H, W, 2) with (u, v) displacements.

    Returns
    -------
    np.ndarray
        uint8 RGB image of shape (H, W, 3).
    """
    import cv2  # local import — opencv may not be installed in all envs

    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255  # full saturation

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2           # hue encodes direction
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    metrics: Optional[dict] = None,
    figsize: Tuple[float, float] = (12, 4),
) -> plt.Figure:
    """Plot training and optional validation loss curves plus extra metrics.

    Parameters
    ----------
    train_losses:
        Per-epoch training losses.
    val_losses:
        Optional per-epoch validation losses.
    metrics:
        Optional dict of ``{metric_name: [values]}`` plotted in a third panel.
    figsize:
        Figure dimensions.
    """
    n_panels = 1 + (val_losses is not None) + (metrics is not None and len(metrics) > 0)
    # Reuse single loss panel when both train and val are given
    has_val = val_losses is not None
    has_metrics = metrics is not None and len(metrics) > 0
    n_panels = 1 + int(has_metrics)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="train loss", linewidth=2)
    if has_val:
        axes[0].plot(epochs, val_losses, label="val loss", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if has_metrics:
        ax = axes[1]
        for name, values in metrics.items():
            ax.plot(range(1, len(values) + 1), values, label=name, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_title("Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
