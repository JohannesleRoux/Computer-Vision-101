"""
modules/08_panoptic_segmentation/panoptic_utils.py
----------------------------------------------------
Panoptic segmentation utilities: PQ metric, ID encoding, and merging.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def encode_panoptic_id(category_id: int, instance_id: int, max_instances: int = 1000) -> int:
    """Encode (category, instance) into a single integer panoptic ID.

    Parameters
    ----------
    category_id:
        Semantic class label.
    instance_id:
        Instance index (0 for stuff categories).
    max_instances:
        Maximum number of instances per category.

    Returns
    -------
    int
        Unique panoptic ID.
    """
    return category_id * max_instances + instance_id


def decode_panoptic_id(panoptic_id: int, max_instances: int = 1000) -> Tuple[int, int]:
    """Decode a panoptic ID back to (category_id, instance_id)."""
    return divmod(panoptic_id, max_instances)


def panoptic_quality(
    pred_panoptic: np.ndarray,
    gt_panoptic: np.ndarray,
    void_ids: tuple = (0,),
) -> Dict[str, float]:
    """Compute PQ, SQ, RQ for a single image.

    Parameters
    ----------
    pred_panoptic:
        (H, W) integer array with predicted panoptic IDs.
    gt_panoptic:
        (H, W) integer array with ground-truth panoptic IDs.
    void_ids:
        IDs to ignore (e.g. void / ignore class).

    Returns
    -------
    dict with keys: PQ, SQ, RQ, TP, FP, FN.
    """
    pred_ids = set(np.unique(pred_panoptic)) - set(void_ids)
    gt_ids   = set(np.unique(gt_panoptic))  - set(void_ids)

    tp = 0
    sq_sum = 0.0
    matched_pred: set = set()

    for gt_id in gt_ids:
        gt_mask   = gt_panoptic == gt_id
        best_iou  = 0.0
        best_pred = -1
        for pred_id in np.unique(pred_panoptic[gt_mask]):
            if pred_id in void_ids:
                continue
            pred_mask  = pred_panoptic == pred_id
            intersection = int((gt_mask & pred_mask).sum())
            union        = int((gt_mask | pred_mask).sum())
            iou = intersection / (union + 1e-6)
            if iou > best_iou:
                best_iou, best_pred = iou, pred_id

        if best_iou > 0.5:
            tp      += 1
            sq_sum  += best_iou
            matched_pred.add(best_pred)

    fp = len(pred_ids - matched_pred)
    fn = len(gt_ids) - tp

    denominator = tp + 0.5 * fp + 0.5 * fn + 1e-9
    pq = sq_sum / denominator
    sq = sq_sum / (tp + 1e-9)
    rq = tp / denominator

    return {"PQ": float(pq), "SQ": float(sq), "RQ": float(rq),
            "TP": tp, "FP": fp, "FN": fn}


def merge_semantic_instance(
    semantic_map: np.ndarray,
    instance_map: np.ndarray,
    thing_ids: List[int],
    max_instances: int = 1000,
) -> Tuple[np.ndarray, List[Dict]]:
    """Merge semantic and instance maps into a panoptic representation.

    Parameters
    ----------
    semantic_map:
        (H, W) integer class labels.
    instance_map:
        (H, W) integer instance IDs (0 = background/stuff).
    thing_ids:
        List of category IDs considered as things (countable objects).
    max_instances:
        Encoding factor.

    Returns
    -------
    panoptic_map : (H, W) integer array of panoptic IDs.
    segments_info : list of dicts with id, category_id, is_thing.
    """
    H, W = semantic_map.shape
    panoptic = np.zeros((H, W), dtype=np.int64)
    segments_info = []

    # Handle thing classes (use instance IDs)
    for inst_id in np.unique(instance_map):
        if inst_id == 0:
            continue
        mask      = instance_map == inst_id
        cat_id    = int(np.median(semantic_map[mask]))
        if cat_id not in thing_ids:
            continue
        pan_id    = encode_panoptic_id(cat_id, inst_id, max_instances)
        panoptic[mask] = pan_id
        segments_info.append({"id": pan_id, "category_id": cat_id, "is_thing": True})

    # Handle stuff classes (use semantic label, one segment per class)
    stuff_mask = panoptic == 0
    for cat_id in np.unique(semantic_map[stuff_mask]):
        if cat_id == 0 or cat_id in thing_ids:
            continue
        mask   = (semantic_map == cat_id) & stuff_mask
        pan_id = encode_panoptic_id(cat_id, 0, max_instances)
        panoptic[mask] = pan_id
        segments_info.append({"id": pan_id, "category_id": cat_id, "is_thing": False})

    return panoptic, segments_info
