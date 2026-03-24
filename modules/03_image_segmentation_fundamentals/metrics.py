"""
modules/03_image_segmentation_fundamentals/metrics.py
------------------------------------------------------
Segmentation evaluation metrics.
"""

from __future__ import annotations
import numpy as np
import torch


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6,
              threshold: float = 0.5) -> float:
    """Binary IoU (Jaccard Index)."""
    pred_bin = (pred > threshold).float()
    target = target.float()
    inter = (pred_bin * target).sum(dim=(-2, -1))
    union = pred_bin.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6,
               threshold: float = 0.5) -> float:
    """Binary Dice coefficient."""
    pred_bin = (pred > threshold).float()
    target = target.float()
    inter = (pred_bin * target).sum(dim=(-2, -1))
    return ((2 * inter + smooth) / (pred_bin.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + smooth)).mean().item()


def pixel_accuracy(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    """Pixel accuracy for multi-class segmentation.

    Parameters
    ----------
    pred_logits:
        Raw logits of shape (N, C, H, W).
    target:
        Class indices of shape (N, H, W).
    """
    pred_classes = pred_logits.argmax(dim=1)
    return (pred_classes == target).float().mean().item()


def mean_iou(pred_logits: torch.Tensor, target: torch.Tensor,
             num_classes: int, ignore_index: int = 255) -> float:
    """Mean IoU over all classes.

    Parameters
    ----------
    pred_logits:
        (N, C, H, W) logit tensor.
    target:
        (N, H, W) ground-truth class tensor.
    num_classes:
        Total number of classes.
    ignore_index:
        Class index to ignore (e.g. void label).
    """
    pred = pred_logits.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_mask   = (pred == cls)
        target_mask = (target == cls) & (target != ignore_index)
        inter = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return float(np.mean(ious)) if ious else 0.0


def confusion_matrix_seg(pred_logits: torch.Tensor, target: torch.Tensor,
                          num_classes: int) -> np.ndarray:
    """Compute the (num_classes × num_classes) confusion matrix.

    Parameters
    ----------
    pred_logits:
        (N, C, H, W) tensor.
    target:
        (N, H, W) ground-truth class tensor.
    num_classes:
        Total number of classes.

    Returns
    -------
    np.ndarray of shape (num_classes, num_classes).
    """
    pred = pred_logits.argmax(dim=1).cpu().numpy().flatten()
    gt   = target.cpu().numpy().flatten()
    cm   = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (gt >= 0) & (gt < num_classes)
    np.add.at(cm, (gt[valid], pred[valid]), 1)
    return cm
