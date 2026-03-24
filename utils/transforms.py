"""
utils/transforms.py
--------------------
Data augmentation transforms for the Computer Vision 101 course.
Covers standard ImageNet augmentations, SSL (SimCLR-style) augmentations,
and detection-specific transforms that keep boxes in sync with the image.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

try:
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
except ImportError:
    torch = None  # type: ignore
    T = None  # type: ignore
    TF = None  # type: ignore

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # type: ignore

# ImageNet statistics
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Standard supervised transforms
# ---------------------------------------------------------------------------

def get_train_transforms(size: int = 224):
    """Standard ImageNet training augmentation pipeline.

    Parameters
    ----------
    size:
        Target spatial size (square crop).

    Returns
    -------
    torchvision.transforms.Compose
    """
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_val_transforms(size: int = 224):
    """Standard ImageNet validation transform (resize → centre crop → normalise).

    Parameters
    ----------
    size:
        Target spatial size (square crop).

    Returns
    -------
    torchvision.transforms.Compose
    """
    return T.Compose([
        T.Resize(int(size * 256 / 224)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Self-supervised (SimCLR) augmentations
# ---------------------------------------------------------------------------

class SimCLRAugmentation:
    """Paired augmentation for contrastive learning (SimCLR / MoCo style).

    Calling an instance of this class on a PIL Image returns two randomly
    augmented views ``(view_1, view_2)`` suitable for contrastive objectives.

    Parameters
    ----------
    size:
        Output crop size.
    gaussian_blur:
        Whether to include Gaussian blur (recommended for ImageNet-scale data;
        less critical for small datasets like CIFAR).
    """

    def __init__(self, size: int = 224, gaussian_blur: bool = True):
        blur_transforms = [T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))] if gaussian_blur else []
        self.transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            *blur_transforms,
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def __call__(self, img) -> Tuple:
        """Return two independently augmented views of *img*."""
        return self.transform(img), self.transform(img)


# ---------------------------------------------------------------------------
# MAE masking
# ---------------------------------------------------------------------------

class MAEMasking:
    """Random patch masking for Masked Autoencoders (MAE).

    Splits the image into a grid of patches and randomly masks a fraction of
    them by returning a boolean mask tensor.

    Parameters
    ----------
    img_size:
        Spatial size of the (square) input image.
    patch_size:
        Spatial size of each (square) patch.
    mask_ratio:
        Fraction of patches to mask (default: 0.75 as in the MAE paper).
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, mask_ratio: float = 0.75):
        self.num_patches = (img_size // patch_size) ** 2
        self.num_masked = int(mask_ratio * self.num_patches)

    def __call__(self, _img=None) -> "torch.Tensor":
        """Return a boolean mask of shape (num_patches,).

        ``True`` indicates a patch that is masked (hidden from the encoder).
        """
        mask = torch.zeros(self.num_patches, dtype=torch.bool)
        masked_indices = torch.randperm(self.num_patches)[: self.num_masked]
        mask[masked_indices] = True
        return mask


# ---------------------------------------------------------------------------
# Multi-crop transform (DINO / SwAV style)
# ---------------------------------------------------------------------------

class MultiCropTransform:
    """Multi-crop augmentation as used in DINO and SwAV.

    Produces *n_global* large-scale views and *n_local* small-scale views.

    Parameters
    ----------
    global_size:
        Spatial size of global crops.
    local_size:
        Spatial size of local crops.
    global_scale:
        Scale range for global crops.
    local_scale:
        Scale range for local crops.
    n_global:
        Number of global crops (default: 2).
    n_local:
        Number of local crops (default: 6).
    """

    def __init__(
        self,
        global_size: int = 224,
        local_size: int = 96,
        global_scale: Tuple[float, float] = (0.4, 1.0),
        local_scale: Tuple[float, float] = (0.05, 0.4),
        n_global: int = 2,
        n_local: int = 6,
    ):
        self.n_global = n_global
        self.n_local = n_local

        flip_and_color = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        self.global_transform = T.Compose([
            T.RandomResizedCrop(global_size, scale=global_scale),
            flip_and_color,
        ])
        self.local_transform = T.Compose([
            T.RandomResizedCrop(local_size, scale=local_scale),
            flip_and_color,
        ])

    def __call__(self, img) -> List:
        """Return a list of augmented views: global first, then local."""
        views = [self.global_transform(img) for _ in range(self.n_global)]
        views += [self.local_transform(img) for _ in range(self.n_local)]
        return views


# ---------------------------------------------------------------------------
# Detection transform — keeps bounding boxes in sync
# ---------------------------------------------------------------------------

class DetectionTransform:
    """Synchronized image + bounding box augmentation for detection models.

    Applies random horizontal flip and optional resizing, updating xyxy boxes
    accordingly.

    Parameters
    ----------
    min_size:
        Minimum size of the resized short edge.
    max_size:
        Maximum size of the resized long edge.
    flip_prob:
        Probability of applying horizontal flip.
    """

    def __init__(
        self,
        min_size: int = 600,
        max_size: int = 1000,
        flip_prob: float = 0.5,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.flip_prob = flip_prob

    def __call__(self, image, target: dict) -> Tuple:
        """Apply transform to *image* and update *target* boxes.

        Parameters
        ----------
        image:
            PIL Image.
        target:
            Dict with at minimum ``boxes`` key (xyxy format, shape N×4).

        Returns
        -------
        Tuple of (transformed image tensor, updated target dict).
        """
        w, h = image.size

        # Random horizontal flip
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            if "boxes" in target and target["boxes"] is not None:
                boxes = target["boxes"].clone() if hasattr(target["boxes"], "clone") \
                    else target["boxes"].copy()
                # Flip x-coordinates: x_new = W - x_old
                if hasattr(boxes, "clone"):  # torch tensor
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                else:
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes

        # Resize: scale so that short side = min_size, long side ≤ max_size
        scale = self.min_size / min(h, w)
        if scale * max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        image = TF.resize(image, [new_h, new_w])

        if "boxes" in target and target["boxes"] is not None:
            target["boxes"] = target["boxes"] * scale

        image = TF.to_tensor(image)
        return image, target
