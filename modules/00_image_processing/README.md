# Module 00 — Image Processing Fundamentals

## Learning Objectives

By the end of this module you will be able to:

1. Represent and manipulate images as multi-dimensional NumPy arrays.
2. Convert between colour spaces (RGB, HSV, LAB, YCbCr) and understand when each is useful.
3. Implement 2-D convolution from scratch and apply common spatial filters.
4. Detect edges using gradient-based operators (Sobel, Prewitt) and the Canny pipeline.
5. Apply morphological operations (erosion, dilation, opening, closing) for shape processing.
6. Compute image histograms and apply histogram equalisation / CLAHE.
7. Analyse images in the frequency domain using the 2-D FFT and design simple frequency filters.

---

## Key Concepts

### Pixels and colour channels

A digital image is a 2-D grid of *pixels*.  A greyscale image is a 2-D array of shape
`(H, W)`.  A colour image is a 3-D array of shape `(H, W, C)` where `C` is the number
of channels (typically 3 for RGB or BGR).  Pixel values are usually stored as unsigned
8-bit integers in [0, 255] or normalised floats in [0, 1].

### Colour spaces

| Space | Channels | Typical use |
|-------|----------|-------------|
| RGB | Red, Green, Blue | Display, storage |
| BGR | Blue, Green, Red | OpenCV default |
| HSV | Hue, Saturation, Value | Colour-based segmentation |
| LAB | Lightness, a*, b* | Perceptually uniform, colour distance |
| YCbCr | Luma, Cb, Cr | JPEG compression, skin detection |

### Convolution

Spatial filtering is implemented as a 2-D *convolution* (or cross-correlation) between
the image `I` and a *kernel* (filter) `K`:

```
(I * K)[i,j] = Σ_m Σ_n I[i+m, j+n] · K[m,n]
```

Common kernels include:
- **Box blur**: uniform averaging kernel.
- **Gaussian blur**: weighted average, reduces noise without sharp edges.
- **Sobel / Prewitt**: approximate first-order derivatives — edge detection.
- **Laplacian**: second-order derivative — sharpness / blob detection.

### Morphological operations

Morphological operators process the *shape* of objects in a binary image:

- **Erosion**: shrinks foreground regions (removes thin protrusions).
- **Dilation**: expands foreground regions (fills small holes).
- **Opening** = erosion then dilation (removes small objects).
- **Closing** = dilation then erosion (fills small gaps).

### Fourier domain

The 2-D Discrete Fourier Transform (DFT) decomposes an image into sinusoidal
components.  Low frequencies correspond to smooth large-scale variations; high
frequencies correspond to fine detail and edges.  Frequency-domain filtering
(e.g. low-pass blurring, high-pass sharpening, notch filters) can be faster than
spatial convolution for large kernels.

---

## Exercises

| # | Title | Skill |
|---|-------|-------|
| 1 | Unsharp masking | Convolution, blurring |
| 2 | Simple background subtraction | Morphology, thresholding |
| 3 | CLAHE from first principles | Histograms, local equalisation |

Solutions are in `notebook_solutions.ipynb`.

---

## Key References

- Gonzalez & Woods, *Digital Image Processing*, 4th ed. (Pearson, 2018) — Chapters 2–5.
- Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. — Chapter 3. [Free PDF](https://szeliski.org/Book/)
- OpenCV documentation: <https://docs.opencv.org/>
- NumPy FFT docs: <https://numpy.org/doc/stable/reference/routines.fft.html>

> **Note:** This module intentionally avoids neural networks.  We build geometric intuition
> about images before introducing learned representations in later modules.
