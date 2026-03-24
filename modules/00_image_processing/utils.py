"""
modules/00_image_processing/utils.py
-------------------------------------
Local helpers for Module 00 — Image Processing Fundamentals.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_image(path: str, as_array: bool = True):
    """Load an image from *path* and optionally convert to a NumPy array.

    Parameters
    ----------
    path:
        File path to the image.
    as_array:
        If *True* (default) return an ``(H, W, 3)`` uint8 ndarray; otherwise
        return the PIL ``Image`` object.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img) if as_array else img


def show_images_grid(images, titles=None, cols=4, figsize=None):
    """Display a list of images in a grid.

    Parameters
    ----------
    images:
        List of numpy arrays (H×W) or (H×W×C).
    titles:
        Optional list of strings, one per image.
    cols:
        Number of columns in the grid.
    figsize:
        Figure size override (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        if titles:
            ax.set_title(titles[i])
        ax.axis("off")
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    return fig


def frequency_spectrum(image_gray: np.ndarray):
    """Compute the 2-D FFT of a greyscale image and return the shifted
    complex spectrum and its log-magnitude.

    Parameters
    ----------
    image_gray:
        2-D greyscale image as a float or uint8 array.

    Returns
    -------
    fshift : np.ndarray
        Shifted complex DFT coefficients (DC at centre).
    magnitude : np.ndarray
        ``log(1 + |fshift|)`` — suitable for visualisation.
    """
    f = np.fft.fft2(image_gray.astype(float))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    return fshift, magnitude


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Manually apply a 2-D convolution (cross-correlation) to *image*.

    Uses zero-padding so the output has the same shape as the input.
    Operates on single-channel images only.

    Parameters
    ----------
    image:
        2-D greyscale float image.
    kernel:
        2-D convolution kernel.

    Returns
    -------
    np.ndarray
        Filtered image of the same shape as *image*.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float64)
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            out[i, j] = (padded[i: i + kh, j: j + kw] * kernel).sum()
    return out


def apply_clahe(image_gray: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalisation (CLAHE).

    Divides *image_gray* into a grid of tiles, equalises each tile's
    histogram with clipping, then bilinearly interpolates the results.
    This is a simplified (non-optimised) reference implementation.

    Parameters
    ----------
    image_gray:
        uint8 greyscale image.
    clip_limit:
        Normalised clip limit (fraction of tile pixel count that triggers
        redistribution).
    grid:
        Number of tiles along each axis.

    Returns
    -------
    np.ndarray
        uint8 CLAHE-equalised image.
    """
    h, w = image_gray.shape
    out = np.zeros_like(image_gray, dtype=np.float32)
    tile_h = h // grid
    tile_w = w // grid

    # Build a lookup table (LUT) for each tile.
    luts = np.zeros((grid, grid, 256), dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            tile = image_gray[r * tile_h:(r + 1) * tile_h,
                              c * tile_w:(c + 1) * tile_w]
            hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256))
            # Clip redistribution
            clip_val = int(clip_limit * tile.size / 256)
            excess = np.maximum(0, hist - clip_val).sum()
            hist = np.minimum(hist, clip_val)
            hist += excess // 256
            cdf = hist.cumsum().astype(np.float32)
            cdf_min = cdf[cdf > 0].min()
            luts[r, c] = np.clip(
                (cdf - cdf_min) / (tile.size - cdf_min + 1e-6) * 255,
                0, 255,
            )

    # Apply LUT with bilinear interpolation between tile centres.
    for y in range(h):
        for x in range(w):
            # Tile indices
            r = min(grid - 1, y // tile_h)
            c = min(grid - 1, x // tile_w)
            out[y, x] = luts[r, c, image_gray[y, x]]

    return out.clip(0, 255).astype(np.uint8)


def background_subtraction(frames: list, method: str = "mean") -> list:
    """Simple background subtraction on a list of greyscale frames.

    Parameters
    ----------
    frames:
        List of (H, W) uint8 greyscale numpy arrays.
    method:
        ``"mean"`` uses the temporal mean as background model;
        ``"median"`` uses the temporal median (more robust).

    Returns
    -------
    list
        List of foreground masks (uint8, 0 or 255).
    """
    stack = np.stack(frames, axis=0).astype(np.float32)
    if method == "median":
        bg = np.median(stack, axis=0)
    else:
        bg = stack.mean(axis=0)

    masks = []
    for frame in frames:
        diff = np.abs(frame.astype(np.float32) - bg)
        threshold = diff.mean() + diff.std()
        mask = (diff > threshold).astype(np.uint8) * 255
        masks.append(mask)
    return masks
