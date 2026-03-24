"""
modules/04_region_proposals/selective_search.py
-------------------------------------------------
Selective search wrappers and proposal evaluation utilities.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None  # type: ignore


def selective_search(image: np.ndarray, mode: str = "fast", max_proposals: int = 2000) -> np.ndarray:
    """Run selective search on *image* using OpenCV's implementation.

    Parameters
    ----------
    image:
        BGR uint8 image.
    mode:
        ``"fast"`` or ``"quality"``.
    max_proposals:
        Return at most this many proposals.

    Returns
    -------
    np.ndarray of shape (N, 4) with (x, y, w, h) boxes.
    """
    assert cv2 is not None, "Install opencv-contrib-python for selective search."
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    return rects[:max_proposals]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """Greedy non-maximum suppression.

    Parameters
    ----------
    boxes:
        (N, 4) array in **xyxy** format.
    scores:
        (N,) confidence scores.
    iou_threshold:
        IoU threshold above which boxes are suppressed.

    Returns
    -------
    List[int]
        Indices of kept boxes.
    """
    order = scores.argsort()[::-1]
    keep  = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        # Compute IoU of box i against all remaining
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_i    = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        order = rest[iou < iou_threshold]
    return keep


def soft_nms(boxes: np.ndarray, scores: np.ndarray,
             sigma: float = 0.5, score_threshold: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Soft-NMS: decay scores of overlapping boxes instead of discarding.

    Parameters
    ----------
    boxes:
        (N, 4) array in xyxy format.
    scores:
        (N,) float scores.
    sigma:
        Gaussian decay parameter.
    score_threshold:
        Boxes with decayed score below this are removed.

    Returns
    -------
    Tuple of (kept_boxes, kept_scores).
    """
    boxes  = boxes.copy().astype(np.float32)
    scores = scores.copy().astype(np.float32)
    indices = list(range(len(boxes)))
    result  = []

    while indices:
        # Pick highest score
        best = max(indices, key=lambda i: scores[i])
        result.append(best)
        indices.remove(best)
        remaining = list(indices)
        for j in remaining:
            x1 = max(boxes[best, 0], boxes[j, 0])
            y1 = max(boxes[best, 1], boxes[j, 1])
            x2 = min(boxes[best, 2], boxes[j, 2])
            y2 = min(boxes[best, 3], boxes[j, 3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            area_b = (boxes[best,2]-boxes[best,0]) * (boxes[best,3]-boxes[best,1])
            area_j = (boxes[j,2]-boxes[j,0]) * (boxes[j,3]-boxes[j,1])
            iou = inter / (area_b + area_j - inter + 1e-6)
            scores[j] *= np.exp(-(iou**2) / sigma)
        indices = [i for i in indices if scores[i] > score_threshold]

    kept_boxes  = boxes[result]
    kept_scores = scores[result]
    return kept_boxes, kept_scores


def recall_vs_proposals(proposals_xyxy: np.ndarray,
                         gt_boxes_xyxy: np.ndarray,
                         iou_threshold: float = 0.5,
                         max_proposals: int = 2000,
                         step: int = 50) -> Tuple[List[int], List[float]]:
    """Compute recall as a function of the number of proposals used.

    Parameters
    ----------
    proposals_xyxy:
        (N, 4) proposal boxes sorted by score (best first).
    gt_boxes_xyxy:
        (M, 4) ground-truth boxes.
    iou_threshold:
        IoU threshold to count a proposal as covering a gt box.
    max_proposals, step:
        Range and increment for the recall curve.

    Returns
    -------
    Tuple of (proposal_counts, recall_values).
    """
    counts, recalls = [], []
    for n in range(step, min(max_proposals, len(proposals_xyxy)) + 1, step):
        props = proposals_xyxy[:n]
        covered = 0
        for gt in gt_boxes_xyxy:
            # Check if any proposal covers this gt box
            x1 = np.maximum(props[:, 0], gt[0])
            y1 = np.maximum(props[:, 1], gt[1])
            x2 = np.minimum(props[:, 2], gt[2])
            y2 = np.minimum(props[:, 3], gt[3])
            inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
            area_p  = (props[:, 2]-props[:, 0]) * (props[:, 3]-props[:, 1])
            area_gt = (gt[2]-gt[0]) * (gt[3]-gt[1])
            iou     = inter / (area_p + area_gt - inter + 1e-6)
            if iou.max() >= iou_threshold:
                covered += 1
        counts.append(n)
        recalls.append(covered / max(len(gt_boxes_xyxy), 1))
    return counts, recalls


def visualize_proposals(image: np.ndarray, proposals_xywh: np.ndarray,
                         max_show: int = 100, figsize=(8, 8)) -> "plt.Figure":
    """Draw region proposals on the image.

    Parameters
    ----------
    image:
        RGB uint8 image.
    proposals_xywh:
        (N, 4) proposals in (x, y, w, h) format.
    max_show:
        Maximum number of proposals to draw.
    """
    assert plt is not None
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    colors = plt.cm.rainbow(np.linspace(0, 1, min(max_show, len(proposals_xywh))))
    for (x, y, w, h), c in zip(proposals_xywh[:max_show], colors):
        rect = mpatches.Rectangle((x, y), w, h, linewidth=0.8,
                                   edgecolor=c, facecolor='none', alpha=0.6)
        ax.add_patch(rect)
    ax.set_title(f'Top {min(max_show, len(proposals_xywh))} proposals')
    ax.axis('off')
    plt.tight_layout()
    return fig
