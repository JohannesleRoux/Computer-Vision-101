"""
modules/07_mask_rcnn_instance_seg/mask_head.py
-----------------------------------------------
Mask R-CNN mask head and paste-masks utility.
"""

from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskHead(nn.Module):
    """FCN-based mask head as in Mask R-CNN (He et al., 2017).

    Input: RoI-Aligned features of shape (N, in_channels, roi_size, roi_size).
    Output: Per-class binary mask logits of shape (N, num_classes, 2*roi_size, 2*roi_size).

    Parameters
    ----------
    in_channels:
        Number of input feature channels.
    roi_size:
        Spatial size of RoI-Aligned input (default: 14 → 28×28 output).
    num_classes:
        Number of object classes (one mask per class).
    hidden_channels:
        Number of channels in intermediate conv layers.
    """

    def __init__(
        self,
        in_channels: int = 256,
        roi_size: int = 14,
        num_classes: int = 80,
        hidden_channels: int = 256,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(4):
            inc = in_channels if i == 0 else hidden_channels
            layers += [
                nn.Conv2d(inc, hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ]
        # Transposed conv: 14 → 28
        layers.append(nn.ConvTranspose2d(hidden_channels, hidden_channels, 2, stride=2))
        layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*layers)
        self.predictor   = nn.Conv2d(hidden_channels, num_classes, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        roi_features:
            (N, C, roi_size, roi_size) RoI-Aligned features.

        Returns
        -------
        torch.Tensor of shape (N, num_classes, 2*roi_size, 2*roi_size).
        """
        x = self.conv_layers(roi_features)
        return self.predictor(x)


def paste_masks_in_image(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    image_shape: tuple,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Paste predicted masks back into the full image coordinate space.

    Parameters
    ----------
    masks:
        (N, H_mask, W_mask) float mask predictions (probabilities after sigmoid).
    boxes:
        (N, 4) bounding boxes in xyxy image coordinates.
    image_shape:
        (H_img, W_img) target image dimensions.
    threshold:
        Binarisation threshold.

    Returns
    -------
    torch.Tensor of shape (N, H_img, W_img) with dtype torch.bool.
    """
    N = masks.shape[0]
    H_img, W_img = image_shape
    full_masks = torch.zeros(N, H_img, W_img, dtype=torch.bool, device=masks.device)

    for i in range(N):
        x1, y1, x2, y2 = [int(v.item()) for v in boxes[i]]
        x1, x2 = max(0, x1), min(W_img, x2)
        y1, y2 = max(0, y1), min(H_img, y2)
        w, h = max(x2 - x1, 1), max(y2 - y1, 1)

        # Resize mask to box size
        mask_resized = F.interpolate(
            masks[i:i+1].unsqueeze(0).float(),
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        )[0, 0]

        full_masks[i, y1:y2, x1:x2] = mask_resized > threshold

    return full_masks
