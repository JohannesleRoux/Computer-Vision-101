"""
modules/06_feature_pyramids/fpn.py
------------------------------------
FPN, PANet, and BiFPN implementations as nn.Module classes.
"""

from __future__ import annotations
from typing import Dict, List, OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Feature Pyramid Network (Lin et al., 2017).

    Adds a top-down pathway with lateral connections to a ResNet backbone.

    Parameters
    ----------
    in_channels_list:
        Number of channels for each backbone level (C2, C3, C4, C5).
    out_channels:
        Number of output channels for each pyramid level (P2-P6).
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs  = nn.ModuleList()
        for in_c in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_c, out_channels, 1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        # Extra level P6 via stride-2 conv on P5
        self.p6_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        features:
            OrderedDict of backbone feature maps in ascending spatial resolution order,
            e.g., {'0': C2, '1': C3, '2': C4, '3': C5}.

        Returns
        -------
        Dict ``{'p2': P2, 'p3': P3, 'p4': P4, 'p5': P5, 'p6': P6}``.
        """
        names  = list(features.keys())
        values = [features[k] for k in names]

        # Lateral connections
        lat_features = [self.lateral_convs[i](values[i]) for i in range(len(values))]

        # Top-down pathway (from highest semantic to lowest)
        top_down = [lat_features[-1]]
        for i in range(len(lat_features) - 2, -1, -1):
            upsampled = F.interpolate(top_down[-1], size=lat_features[i].shape[-2:],
                                      mode='nearest')
            top_down.append(lat_features[i] + upsampled)
        top_down = top_down[::-1]  # reverse so index 0 = P2

        # Apply output convs
        out = {f'p{i+2}': self.output_convs[i](td) for i, td in enumerate(top_down)}
        out['p6'] = self.p6_conv(out[f'p{len(top_down)+1}'])
        return out


class PANet(nn.Module):
    """Path Aggregation Network (Liu et al., 2018).

    Extends FPN with a bottom-up path augmentation for richer feature fusion.

    Parameters
    ----------
    in_channels_list:
        Backbone channel counts (C2-C5).
    out_channels:
        Output channels for all pyramid levels.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.fpn = FPN(in_channels_list, out_channels)

        # Bottom-up path augmentation convolutions
        self.bu_convs = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            self.bu_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
        self.out_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(len(in_channels_list))
        ])

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        fpn_out = self.fpn(features)  # p2, p3, p4, p5, p6
        keys = ['p2', 'p3', 'p4', 'p5']

        # Bottom-up augmentation
        bu = [fpn_out['p2']]
        for i, conv in enumerate(self.bu_convs):
            fused = conv(bu[-1]) + fpn_out[keys[i + 1]]
            bu.append(fused)

        result = {k: self.out_convs[i](bu[i]) for i, k in enumerate(keys)}
        result['p6'] = fpn_out['p6']
        return result


class BiFPNLayer(nn.Module):
    """Single BiFPN layer with weighted bidirectional fusion.

    Parameters
    ----------
    num_levels:
        Number of feature pyramid levels.
    out_channels:
        Channel count (same for all levels).
    """

    def __init__(self, num_levels: int = 5, out_channels: int = 256, eps: float = 1e-4):
        super().__init__()
        self.num_levels = num_levels
        self.eps = eps

        # Learnable fusion weights (fast normalised fusion)
        self.w_td = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(num_levels - 1)])
        self.w_bu = nn.ParameterList([nn.Parameter(torch.ones(3)) for _ in range(num_levels - 2)])
        self.w_bu.append(nn.Parameter(torch.ones(2)))  # for level 0

        # Depthwise separable convs
        self.td_convs = nn.ModuleList([self._dw_conv(out_channels) for _ in range(num_levels - 1)])
        self.bu_convs = nn.ModuleList([self._dw_conv(out_channels) for _ in range(num_levels - 1)])

    @staticmethod
    def _dw_conv(c):
        return nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

    def _wfuse(self, feats, weights):
        w = F.relu(weights)
        w = w / (w.sum() + self.eps)
        return sum(w[i] * feats[i] for i in range(len(feats)))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        features:
            List of (N, C, H_i, W_i) tensors from coarsest to finest.

        Returns
        -------
        List of fused feature tensors in the same order.
        """
        n = self.num_levels
        td = [None] * n
        td[0] = features[0]

        # Top-down
        for i in range(1, n):
            up = F.interpolate(td[i-1], size=features[i].shape[-2:], mode='nearest')
            td[i] = self.td_convs[i-1](self._wfuse([features[i], up], self.w_td[i-1]))

        # Bottom-up
        bu = [None] * n
        bu[-1] = td[-1]
        for i in range(n - 2, -1, -1):
            dn = F.interpolate(bu[i+1], size=td[i].shape[-2:], mode='nearest')
            if i == 0:
                bu[i] = self.bu_convs[n-2-i](self._wfuse([features[i], dn], self.w_bu[-1]))
            else:
                bu[i] = self.bu_convs[n-2-i](self._wfuse([features[i], td[i], dn], self.w_bu[n-2-i]))
        return bu


class BiFPN(nn.Module):
    """Stacked BiFPN (EfficientDet-style).

    Parameters
    ----------
    in_channels_list:
        Backbone channel counts (C3-C7 or similar).
    out_channels:
        Output channels.
    num_layers:
        Number of BiFPN layers stacked.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 64, num_layers: int = 3):
        super().__init__()
        # Project backbone features to out_channels
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
            for c in in_channels_list
        ])
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(num_levels=len(in_channels_list), out_channels=out_channels)
            for _ in range(num_layers)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        x = [self.proj[i](features[i]) for i in range(len(features))]
        for layer in self.bifpn_layers:
            x = layer(x)
        return x
