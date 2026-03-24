"""
modules/09_yolo_family/yolo_utils.py
--------------------------------------
YOLO utility functions: anchor k-means, NMS, decode helpers.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np


def anchor_kmeans(
    gt_boxes_wh: np.ndarray,
    k: int = 9,
    n_iter: int = 300,
    random_state: int = 42,
) -> np.ndarray:
    """Compute optimal anchor box sizes using k-means clustering on GT box WH.

    Uses IoU-based distance (1 - IoU) instead of Euclidean distance.

    Parameters
    ----------
    gt_boxes_wh:
        (N, 2) array of ground-truth box widths and heights (normalised).
    k:
        Number of anchor clusters.
    n_iter:
        Maximum k-means iterations.
    random_state:
        Random seed.

    Returns
    -------
    np.ndarray of shape (k, 2): anchor (w, h) pairs, sorted by area.
    """
    np.random.seed(random_state)
    n = len(gt_boxes_wh)
    idx = np.random.choice(n, k, replace=False)
    centroids = gt_boxes_wh[idx].copy()

    for _ in range(n_iter):
        # IoU-based distance: centre boxes at origin
        iw = np.minimum(gt_boxes_wh[:, 0:1], centroids[:, 0])  # (N, k)
        ih = np.minimum(gt_boxes_wh[:, 1:2], centroids[:, 1])
        inter = iw * ih
        area_gt  = gt_boxes_wh[:, 0] * gt_boxes_wh[:, 1]
        area_anc = centroids[:, 0] * centroids[:, 1]
        iou = inter / (area_gt[:, None] + area_anc[None] - inter + 1e-9)
        distances = 1 - iou  # (N, k)

        assign = distances.argmin(axis=1)
        new_centroids = np.array([
            gt_boxes_wh[assign == j].mean(axis=0) if (assign == j).any() else centroids[j]
            for j in range(k)
        ])
        if np.abs(new_centroids - centroids).max() < 1e-6:
            break
        centroids = new_centroids

    # Sort by area
    areas = centroids[:, 0] * centroids[:, 1]
    return centroids[areas.argsort()]


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.25,
) -> List[int]:
    """Class-aware NMS.

    Parameters
    ----------
    boxes:
        (N, 4) in xyxy format.
    scores:
        (N,) confidence scores.
    class_ids:
        (N,) integer class labels.
    iou_threshold:
        Suppress if IoU above this.
    score_threshold:
        Ignore boxes below this score.

    Returns
    -------
    List of kept indices.
    """
    keep_all = []
    for cls in np.unique(class_ids):
        cls_mask = (class_ids == cls) & (scores >= score_threshold)
        idx = np.where(cls_mask)[0]
        if len(idx) == 0:
            continue
        order = idx[scores[idx].argsort()[::-1]]
        while len(order) > 0:
            best = order[0]
            keep_all.append(int(best))
            rest = order[1:]
            x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
            inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
            area_b = (boxes[best,2]-boxes[best,0]) * (boxes[best,3]-boxes[best,1])
            area_r = (boxes[rest,2]-boxes[rest,0]) * (boxes[rest,3]-boxes[rest,1])
            iou = inter / (area_b + area_r - inter + 1e-9)
            order = rest[iou < iou_threshold]
    return keep_all


def decode_yolo_output(
    pred: np.ndarray,
    anchors: np.ndarray,
    input_size: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw YOLOv3 output into boxes, objectness scores, and class probs.

    Parameters
    ----------
    pred:
        (batch, H, W, num_anchors*(5+num_classes)) raw output.
    anchors:
        (num_anchors, 2) anchor (w, h) in absolute pixels.
    input_size:
        Network input resolution.
    num_classes:
        Number of classes.

    Returns
    -------
    boxes : (batch, H*W*num_anchors, 4) xyxy normalised to [0,1].
    obj   : (batch, H*W*num_anchors) objectness probability.
    cls   : (batch, H*W*num_anchors, num_classes) class probabilities.
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x.clip(-500, 500)))

    B, H, W, _ = pred.shape
    A = len(anchors)
    pred = pred.reshape(B, H, W, A, 5 + num_classes)

    # Grid offsets
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    grid_xy = np.stack([grid_x, grid_y], axis=-1)[None, :, :, None]  # (1,H,W,1,2)

    # Decode
    bxy = (sigmoid(pred[..., :2]) + grid_xy) / np.array([[W, H]])
    bwh = np.exp(pred[..., 2:4]) * anchors[None, None, None] / input_size
    obj = sigmoid(pred[..., 4:5])
    cls = sigmoid(pred[..., 5:])

    # Convert to xyxy
    x1 = bxy[..., 0] - bwh[..., 0] / 2
    y1 = bxy[..., 1] - bwh[..., 1] / 2
    x2 = bxy[..., 0] + bwh[..., 0] / 2
    y2 = bxy[..., 1] + bwh[..., 1] / 2
    boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(B, -1, 4)
    obj   = obj.reshape(B, -1)
    cls   = cls.reshape(B, -1, num_classes)
    return boxes, obj, cls


def fcos_decode(
    cls_logits: np.ndarray,
    box_regression: np.ndarray,
    centerness: np.ndarray,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode FCOS predictions (l, r, t, b distances) into xyxy boxes.

    Parameters
    ----------
    cls_logits:
        (H, W, num_classes) class logits.
    box_regression:
        (H, W, 4) predicted (l, r, t, b) distances.
    centerness:
        (H, W) centerness scores.
    stride:
        Feature map stride (pixels per cell).

    Returns
    -------
    boxes : (H*W, 4) xyxy boxes.
    scores: (H*W, num_classes) scores (sigmoid(cls) * sqrt(centerness)).
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x.clip(-500, 500)))

    H, W = cls_logits.shape[:2]
    gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    cx = (gx + 0.5) * stride
    cy = (gy + 0.5) * stride

    l, r, t, b = [box_regression[:, :, i] for i in range(4)]
    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b
    boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)

    c = np.sqrt(sigmoid(centerness)).reshape(-1, 1)
    scores = sigmoid(cls_logits).reshape(-1, cls_logits.shape[-1]) * c
    return boxes, scores
