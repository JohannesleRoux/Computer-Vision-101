"""
modules/02_cnns_deep_dive/layers.py
-------------------------------------
Custom conv layer implementations for the CNNs Deep Dive module.
"""

from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def receptive_field(kernel_sizes: List[int], strides: List[int], dilations: Optional[List[int]] = None) -> int:
    """Compute the theoretical receptive field of a stacked conv sequence.

    Parameters
    ----------
    kernel_sizes:
        Kernel size for each layer.
    strides:
        Stride for each layer.
    dilations:
        Dilation rate for each layer (default: all 1).

    Returns
    -------
    int
        Receptive field size in pixels.
    """
    if dilations is None:
        dilations = [1] * len(kernel_sizes)
    rf = 1
    stride_prod = 1
    for k, s, d in zip(kernel_sizes, strides, dilations):
        eff_k = k + (k - 1) * (d - 1)  # effective kernel size with dilation
        rf = rf + (eff_k - 1) * stride_prod
        stride_prod *= s
    return rf


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (MobileNet-style).

    Parameters
    ----------
    in_channels, out_channels:
        Input/output channel counts.
    kernel_size:
        Spatial kernel size for the depthwise stage.
    stride, padding:
        Applied to the depthwise conv.
    bias:
        Whether to use bias in the pointwise conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

    @staticmethod
    def param_ratio(in_ch: int, out_ch: int, k: int = 3) -> float:
        """Return ratio of depthwise-separable params vs. standard conv params."""
        standard = in_ch * out_ch * k * k
        dw_sep   = in_ch * k * k + in_ch * out_ch
        return dw_sep / standard


class DilatedConv(nn.Module):
    """Dilated (atrous) convolution block with BN and ReLU.

    Parameters
    ----------
    in_channels, out_channels:
        Channel dimensions.
    dilation:
        Dilation rate.  Effective kernel size = k + (k-1)*(d-1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 2,
        bias: bool = False,
    ):
        super().__init__()
        padding = dilation  # 'same' padding for 3×3 dilated conv
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class TransposedConvBlock(nn.Module):
    """Transposed convolution for upsampling (decoder block).

    Parameters
    ----------
    in_channels, out_channels:
        Channel dimensions.
    scale_factor:
        Upsampling factor (stride of the transposed conv).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        bias: bool = False,
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=scale_factor * 2,
            stride=scale_factor,
            padding=scale_factor // 2,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.deconv(x)))


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module (DeepLab v3).

    Applies parallel atrous convolutions with multiple dilation rates
    plus global average pooling, then concatenates and fuses.

    Parameters
    ----------
    in_channels:
        Number of input channels.
    out_channels:
        Number of channels per branch (and output).
    dilations:
        Dilation rates for the parallel branches.
    """

    def __init__(
        self,
        in_channels: int = 2048,
        out_channels: int = 256,
        dilations: List[int] = (1, 6, 12, 18),
    ):
        super().__init__()
        # 1×1 conv for rate=1
        self.branches = nn.ModuleList()
        for d in dilations:
            if d == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ))

        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Projection after concatenation
        total_in = out_channels * (len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_in, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        # Global pooling branch: upsample to match spatial size
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=(H, W), mode='bilinear', align_corners=False)
        features.append(gp)
        return self.project(torch.cat(features, dim=1))
