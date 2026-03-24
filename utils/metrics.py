"""
utils/metrics.py
----------------
Evaluation metrics for segmentation, detection, and depth estimation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def iou_score(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    smooth: float = 1e-6,
    threshold: float = 0.5,
) -> float:
    """Compute Intersection-over-Union (Jaccard Index) for binary masks.

    Parameters
    ----------
    pred:
        Predicted probability map or binary mask, shape (N, H, W) or (H, W).
    target:
        Ground-truth binary mask, same shape as *pred*.
    smooth:
        Laplace smoothing term to avoid division by zero.
    threshold:
        Binarisation threshold applied to *pred* when it contains probabilities.

    Returns
    -------
    float
        Mean IoU across the batch (or scalar IoU for a single pair).
    """
    if torch is not None and isinstance(pred, torch.Tensor):
        pred = (pred > threshold).float()
        target = target.float()
        intersection = (pred * target).sum(dim=(-2, -1))
        union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean().item()
    else:
        pred_bin = (np.asarray(pred) > threshold).astype(float)
        tgt = np.asarray(target).astype(float)
        intersection = (pred_bin * tgt).sum()
        union = pred_bin.sum() + tgt.sum() - intersection
        return float((intersection + smooth) / (union + smooth))


def dice_score(
    pred,
    target,
    smooth: float = 1e-6,
    threshold: float = 0.5,
) -> float:
    """Compute the Dice (F1) coefficient for binary masks.

    Parameters
    ----------
    pred:
        Predicted probability map or binary mask.
    target:
        Ground-truth binary mask.
    smooth:
        Laplace smoothing term.
    threshold:
        Binarisation threshold.

    Returns
    -------
    float
        Dice score in [0, 1].
    """
    if torch is not None and isinstance(pred, torch.Tensor):
        pred = (pred > threshold).float()
        target = target.float()
        intersection = (pred * target).sum(dim=(-2, -1))
        dice = (2.0 * intersection + smooth) / (
            pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + smooth
        )
        return dice.mean().item()
    else:
        pred_bin = (np.asarray(pred) > threshold).astype(float)
        tgt = np.asarray(target).astype(float)
        intersection = (pred_bin * tgt).sum()
        return float(
            (2.0 * intersection + smooth) / (pred_bin.sum() + tgt.sum() + smooth)
        )


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

def _box_iou_numpy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of xyxy boxes."""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def mean_average_precision(
    detections: List[dict],
    ground_truths: List[dict],
    iou_threshold: float = 0.5,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Compute mean Average Precision (mAP) for object detection.

    Parameters
    ----------
    detections:
        List of detection dicts, each with keys:
        ``image_id``, ``class_id``, ``score``, ``box`` (xyxy, length-4 array).
    ground_truths:
        List of ground-truth dicts, each with keys:
        ``image_id``, ``class_id``, ``box`` (xyxy, length-4 array).
    iou_threshold:
        IoU threshold for a true positive.
    num_classes:
        Total number of classes.  Inferred from data if *None*.

    Returns
    -------
    dict
        ``{"mAP": float, "AP_per_class": {class_id: ap}}``
    """
    if num_classes is None:
        classes = set(d["class_id"] for d in detections) | set(
            g["class_id"] for g in ground_truths
        )
        num_classes = max(classes) + 1 if classes else 0

    ap_per_class: dict = {}

    for cls in range(num_classes):
        cls_dets = sorted(
            [d for d in detections if d["class_id"] == cls],
            key=lambda x: -x["score"],
        )
        cls_gts_by_image: dict = {}
        for g in ground_truths:
            if g["class_id"] == cls:
                cls_gts_by_image.setdefault(g["image_id"], []).append(g)

        tp = np.zeros(len(cls_dets))
        fp = np.zeros(len(cls_dets))
        n_gt = sum(len(v) for v in cls_gts_by_image.values())

        if n_gt == 0:
            ap_per_class[cls] = 0.0
            continue

        matched: dict = {}  # image_id → set of matched gt indices

        for i, det in enumerate(cls_dets):
            img_id = det["image_id"]
            gts = cls_gts_by_image.get(img_id, [])
            if not gts:
                fp[i] = 1
                continue

            det_box = np.array(det["box"])[np.newaxis]
            gt_boxes = np.array([g["box"] for g in gts])
            ious = _box_iou_numpy(det_box, gt_boxes)[0]

            best_idx = int(np.argmax(ious))
            if ious[best_idx] >= iou_threshold:
                seen = matched.setdefault(img_id, set())
                if best_idx not in seen:
                    tp[i] = 1
                    seen.add(best_idx)
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / (n_gt + 1e-6)
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)

        # VOC-style 11-point interpolation
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            prec_at_thr = precision[recall >= thr]
            ap += (prec_at_thr.max() if len(prec_at_thr) else 0) / 11
        ap_per_class[cls] = float(ap)

    mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return {"mAP": mAP, "AP_per_class": ap_per_class}


# ---------------------------------------------------------------------------
# Panoptic quality
# ---------------------------------------------------------------------------

def panoptic_quality(
    pred_panoptic: np.ndarray,
    gt_panoptic: np.ndarray,
    num_things: int,
    num_stuff: int,
) -> Dict[str, float]:
    """Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and
    Recognition Quality (RQ) following Kirillov et al. (2019).

    Parameters
    ----------
    pred_panoptic:
        Integer array of shape (H, W).  Each value encodes
        ``class_id * max_inst + instance_id``.
    gt_panoptic:
        Ground-truth panoptic map with the same encoding.
    num_things:
        Number of *thing* categories.
    num_stuff:
        Number of *stuff* categories.

    Returns
    -------
    dict
        Keys: ``PQ``, ``SQ``, ``RQ``, ``PQ_things``, ``PQ_stuff``.
    """
    pred = pred_panoptic.flatten()
    gt = gt_panoptic.flatten()

    pred_ids = np.unique(pred)
    gt_ids = np.unique(gt)

    # For each gt segment, find the best-matching pred segment.
    tp = sq_sum = 0
    fp = fn = 0

    matched_pred: set = set()

    for gt_id in gt_ids:
        if gt_id == 0:  # ignore background / void
            continue
        gt_mask = gt == gt_id
        overlapping_preds = np.unique(pred[gt_mask])

        best_iou = 0.0
        best_pred_id = -1
        for pred_id in overlapping_preds:
            if pred_id == 0:
                continue
            pred_mask = pred == pred_id
            intersection = int((gt_mask & pred_mask).sum())
            union = int((gt_mask | pred_mask).sum())
            iou = intersection / (union + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_pred_id = pred_id

        if best_iou > 0.5:
            tp += 1
            sq_sum += best_iou
            matched_pred.add(best_pred_id)
        else:
            fn += 1

    fp = len([p for p in pred_ids if p != 0 and p not in matched_pred])

    pq = sq_sum / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    sq = sq_sum / (tp + 1e-6)
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)

    return {
        "PQ": float(pq),
        "SQ": float(sq),
        "RQ": float(rq),
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

def depth_metrics(
    pred_depth,
    gt_depth,
    min_depth: float = 1e-3,
    max_depth: float = 80.0,
) -> Dict[str, float]:
    """Compute standard monocular depth evaluation metrics.

    Parameters
    ----------
    pred_depth:
        Predicted depth map, shape (H, W) or (N, H, W).
    gt_depth:
        Ground-truth depth map, same shape.
    min_depth:
        Minimum valid depth value (depths below this are masked out).
    max_depth:
        Maximum valid depth value.

    Returns
    -------
    dict
        Keys: ``AbsRel``, ``SqRel``, ``RMSE``, ``RMSElog``,
              ``delta1``, ``delta2``, ``delta3``.
    """
    pred = np.asarray(pred_depth, dtype=np.float64)
    gt = np.asarray(gt_depth, dtype=np.float64)

    valid = (gt > min_depth) & (gt < max_depth) & (pred > min_depth)
    pred = pred[valid]
    gt = gt[valid]

    if len(gt) == 0:
        return {k: float("nan") for k in
                ["AbsRel", "SqRel", "RMSE", "RMSElog", "delta1", "delta2", "delta3"]}

    thresh = np.maximum(gt / pred, pred / gt)
    delta1 = float((thresh < 1.25).mean())
    delta2 = float((thresh < 1.25 ** 2).mean())
    delta3 = float((thresh < 1.25 ** 3).mean())

    abs_rel = float(np.mean(np.abs(gt - pred) / gt))
    sq_rel = float(np.mean(((gt - pred) ** 2) / gt))
    rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2)))

    return {
        "AbsRel": abs_rel,
        "SqRel": sq_rel,
        "RMSE": rmse,
        "RMSElog": rmse_log,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }
