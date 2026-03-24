"""
modules/05_rcnn_family/models.py
----------------------------------
RoI Pooling, RoI Align, AnchorGenerator, and RPNHead implementations.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoIPool(nn.Module):
    """RoI Pooling (Fast R-CNN style).

    Extracts a fixed-size (output_size × output_size) feature map for each proposal.
    Uses quantised (rounded) coordinates — may lose sub-pixel precision.

    Parameters
    ----------
    output_size:
        Spatial size of the output feature map.
    spatial_scale:
        Ratio of feature map to input image spatial dimensions.
    """

    def __init__(self, output_size: int = 7, spatial_scale: float = 1.0 / 16):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(
        self,
        features: torch.Tensor,
        rois: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            (N, C, H, W) feature map.
        rois:
            (K, 5) tensor where each row is (batch_idx, x1, y1, x2, y2)
            in image coordinates.

        Returns
        -------
        torch.Tensor of shape (K, C, output_size, output_size).
        """
        K = rois.shape[0]
        C, H, W = features.shape[1:]
        out = torch.zeros(K, C, self.output_size, self.output_size,
                          device=features.device, dtype=features.dtype)

        for k in range(K):
            batch_idx = int(rois[k, 0])
            x1 = round(float(rois[k, 1]) * self.spatial_scale)
            y1 = round(float(rois[k, 2]) * self.spatial_scale)
            x2 = round(float(rois[k, 3]) * self.spatial_scale)
            y2 = round(float(rois[k, 4]) * self.spatial_scale)
            x1, x2 = max(0, x1), min(W, max(x2, x1 + 1))
            y1, y2 = max(0, y1), min(H, max(y2, y1 + 1))
            roi_feat = features[batch_idx:batch_idx+1, :, y1:y2, x1:x2]
            out[k] = F.adaptive_max_pool2d(roi_feat, self.output_size)

        return out


class RoIAlign(nn.Module):
    """RoI Align (Mask R-CNN style) with bilinear interpolation.

    Eliminates quantisation error by sampling at exact (fractional) positions.

    Parameters
    ----------
    output_size:
        Spatial size (H_out, W_out) of the output.
    spatial_scale:
        Ratio of feature map to input image.
    sampling_ratio:
        Number of sampling points per output cell (default: 2).
    """

    def __init__(
        self,
        output_size: int = 7,
        spatial_scale: float = 1.0 / 16,
        sampling_ratio: int = 2,
    ):
        super().__init__()
        self.output_size   = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            (N, C, H, W).
        rois:
            (K, 5) with (batch_idx, x1, y1, x2, y2).

        Returns
        -------
        torch.Tensor of shape (K, C, output_size, output_size).
        """
        # Use PyTorch's built-in torchvision RoI Align when available.
        try:
            from torchvision.ops import roi_align
            return roi_align(
                features, rois,
                output_size=self.output_size,
                spatial_scale=self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
                aligned=True,
            )
        except ImportError:
            # Fallback: use bilinear interpolation manually (simplified)
            return self._manual_roi_align(features, rois)

    def _manual_roi_align(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        K = rois.shape[0]
        C, H, W = features.shape[1:]
        out = torch.zeros(K, C, self.output_size, self.output_size,
                          device=features.device, dtype=features.dtype)
        for k in range(K):
            b = int(rois[k, 0])
            x1 = float(rois[k, 1]) * self.spatial_scale
            y1 = float(rois[k, 2]) * self.spatial_scale
            x2 = float(rois[k, 3]) * self.spatial_scale
            y2 = float(rois[k, 4]) * self.spatial_scale
            roi_w = max(x2 - x1, 1e-3)
            roi_h = max(y2 - y1, 1e-3)
            bin_w = roi_w / self.output_size
            bin_h = roi_h / self.output_size
            for i in range(self.output_size):
                for j in range(self.output_size):
                    cy = y1 + (i + 0.5) * bin_h
                    cx = x1 + (j + 0.5) * bin_w
                    # Normalise to [-1, 1] for grid_sample
                    ny = 2.0 * cy / H - 1.0
                    nx = 2.0 * cx / W - 1.0
                    grid = torch.tensor([[[[nx, ny]]]], device=features.device, dtype=features.dtype)
                    val  = F.grid_sample(features[b:b+1], grid, align_corners=True, mode='bilinear', padding_mode='zeros')
                    out[k, :, i, j] = val[0, :, 0, 0]
        return out


class AnchorGenerator(nn.Module):
    """Generate anchor boxes for each feature map location.

    Parameters
    ----------
    sizes:
        Anchor areas (sqrt) per feature map level.
    aspect_ratios:
        Aspect ratios (h/w) to use at each scale.
    """

    def __init__(
        self,
        sizes: Tuple[int, ...] = (32, 64, 128, 256, 512),
        aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
    ):
        super().__init__()
        self.sizes        = sizes
        self.aspect_ratios = aspect_ratios
        self._base_anchors = self._generate_base_anchors()

    def _generate_base_anchors(self) -> torch.Tensor:
        """Return (A, 4) base anchors centred at (0, 0) in xyxy format."""
        anchors = []
        for size in self.sizes:
            for ar in self.aspect_ratios:
                w = size / math.sqrt(ar)
                h = size * math.sqrt(ar)
                anchors.append([-w/2, -h/2, w/2, h/2])
        return torch.tensor(anchors, dtype=torch.float32)

    def forward(self, feature_map: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Place base anchors at every feature map cell.

        Parameters
        ----------
        feature_map:
            (N, C, H, W) feature map.
        image_size:
            (img_H, img_W) original image dimensions.

        Returns
        -------
        torch.Tensor of shape (H*W*A, 4) in image coordinates.
        """
        H, W = feature_map.shape[-2:]
        img_H, img_W = image_size
        stride_h = img_H / H
        stride_w = img_W / W

        # Grid of centre points
        gy = (torch.arange(H, dtype=torch.float32) + 0.5) * stride_h
        gx = (torch.arange(W, dtype=torch.float32) + 0.5) * stride_w
        cy, cx = torch.meshgrid(gy, gx, indexing='ij')
        shifts = torch.stack([cx.flatten(), cy.flatten(),
                               cx.flatten(), cy.flatten()], dim=1)

        base = self._base_anchors.to(feature_map.device)  # (A, 4)
        # Broadcasting: (H*W, 1, 4) + (1, A, 4) → (H*W, A, 4)
        all_anchors = (shifts.unsqueeze(1) + base.unsqueeze(0)).reshape(-1, 4)
        return all_anchors


class RPNHead(nn.Module):
    """Region Proposal Network head.

    Parameters
    ----------
    in_channels:
        Number of input feature channels.
    num_anchors:
        Number of anchors per location.
    """

    def __init__(self, in_channels: int = 256, num_anchors: int = 9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred  = nn.Conv2d(in_channels, num_anchors * 4, 1)

        for m in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        cls_logits: (N, A, H, W) — objectness logits
        bbox_deltas: (N, A*4, H, W) — box regression deltas
        """
        t = F.relu(self.conv(x))
        return self.cls_logits(t), self.bbox_pred(t)
