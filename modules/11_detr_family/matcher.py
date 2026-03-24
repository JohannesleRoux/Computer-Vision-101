"""
modules/11_detr_family/matcher.py
-----------------------------------
Hungarian matcher and box utilities for DETR-family detectors.
"""

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalised IoU (GIoU) between two sets of boxes.

    Both inputs in xyxy format, normalised to [0, 1].

    Parameters
    ----------
    boxes1:
        (N, 4) tensor.
    boxes2:
        (M, 4) tensor.

    Returns
    -------
    torch.Tensor of shape (N, M) with GIoU values in [-1, 1].
    """
    # Pairwise intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    iou   = inter / (union + 1e-6)

    # Enclosing box
    ex1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    ey1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    ex2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    ey2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enclose_area = (ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0)

    giou = iou - (enclose_area - union) / (enclose_area + 1e-6)
    return giou


class HungarianMatcher(torch.nn.Module):
    """Hungarian matcher for DETR bipartite set assignment.

    Computes the optimal one-to-one matching between predictions and
    ground-truth objects by minimising a combined cost:

    ``cost = λ_cls * cost_class + λ_bbox * cost_L1 + λ_giou * cost_GIoU``

    Parameters
    ----------
    cost_class:
        Weight for classification cost.
    cost_bbox:
        Weight for L1 box cost.
    cost_giou:
        Weight for GIoU cost.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        tgt_classes: List[torch.Tensor],
        tgt_boxes: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Match predictions to ground-truth for a batch.

        Parameters
        ----------
        pred_logits:
            (B, N_q, num_classes) class logits.
        pred_boxes:
            (B, N_q, 4) predicted boxes in cxcywh normalised format.
        tgt_classes:
            List of B tensors of shape (N_gt_i,) with class labels.
        tgt_boxes:
            List of B tensors of shape (N_gt_i, 4) in cxcywh normalised format.

        Returns
        -------
        List of (pred_idx, tgt_idx) tuples (one per image).
        """
        from scipy.optimize import linear_sum_assignment

        B, N_q = pred_logits.shape[:2]
        indices = []

        for b in range(B):
            n_gt = len(tgt_classes[b])
            if n_gt == 0:
                indices.append((torch.tensor([], dtype=torch.int64),
                                 torch.tensor([], dtype=torch.int64)))
                continue

            # Class cost: negative log-probability for GT class
            prob = pred_logits[b].softmax(-1)  # (N_q, C)
            cost_cls = -prob[:, tgt_classes[b]]  # (N_q, n_gt)

            # L1 box cost
            cost_l1 = torch.cdist(pred_boxes[b], tgt_boxes[b], p=1)  # (N_q, n_gt)

            # GIoU cost
            pb_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])
            gb_xyxy = box_cxcywh_to_xyxy(tgt_boxes[b])
            cost_gi = -generalized_box_iou(pb_xyxy, gb_xyxy)  # (N_q, n_gt)

            # Combined cost
            cost = (
                self.cost_class * cost_cls
                + self.cost_bbox  * cost_l1
                + self.cost_giou  * cost_gi
            ).cpu().numpy()

            row_idx, col_idx = linear_sum_assignment(cost)
            indices.append((
                torch.as_tensor(row_idx, dtype=torch.int64),
                torch.as_tensor(col_idx, dtype=torch.int64),
            ))

        return indices
