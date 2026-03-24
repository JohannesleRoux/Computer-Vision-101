"""
modules/10_vision_transformers/attention.py
--------------------------------------------
Multi-head attention, patch embedding, and Transformer/Swin blocks.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention.

    Parameters
    ----------
    d_model:
        Token embedding dimension.
    n_heads:
        Number of attention heads.
    dropout:
        Attention dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.scale   = math.sqrt(self.d_k)
        self.to_qkv  = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj= nn.Linear(d_model, d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x:
            (B, N, D) input tokens.
        mask:
            Optional (B, N, N) boolean attention mask.

        Returns
        -------
        out  : (B, N, D) attended output.
        attn : (B, h, N, N) attention weights.
        """
        B, N, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.n_heads, self.d_k).transpose(1, 2) for t in qkv]

        attn = (q @ k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = self.drop(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out), attn


class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and linearly project each.

    Parameters
    ----------
    img_size:
        Input image resolution (square).
    patch_size:
        Patch resolution (square).
    in_channels:
        Input image channels.
    embed_dim:
        Output embedding dimension.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size  = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, D)."""
        x = self.proj(x)           # (B, D, H/P, W/P)
        x = x.flatten(2)           # (B, D, N)
        x = x.transpose(1, 2)      # (B, N, D)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block (Pre-LN variant).

    Parameters
    ----------
    d_model:
        Embedding dimension.
    n_heads:
        Number of attention heads.
    mlp_ratio:
        Ratio of MLP hidden dim to d_model.
    dropout:
        Dropout applied in attention and MLP.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim    = int(d_model * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.

    Parameters
    ----------
    x:
        (B, H, W, C) feature map.
    window_size:
        Window size W_s.

    Returns
    -------
    (num_windows*B, W_s, W_s, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window_partition back to (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinBlock(nn.Module):
    """Swin Transformer block with (possibly shifted) window attention.

    Parameters
    ----------
    d_model:
        Embedding dimension.
    n_heads:
        Number of attention heads.
    window_size:
        Local window size W_s.
    shift_size:
        Cyclic shift offset (0 = regular, window_size//2 = shifted).
    mlp_ratio:
        MLP expansion ratio.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, n_heads, dropout)
        mlp_dim    = int(d_model * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, mlp_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            (B, H*W, C) flattened spatial tokens.

        Returns
        -------
        (B, H*W, C)
        """
        # NOTE: This is a simplified implementation.
        # Full Swin uses relative position bias and masking for shifted windows.
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size)          # (nW*B, Ws, Ws, C)
        windows = windows.view(-1, self.window_size**2, C)
        attn_out, _ = self.attn(windows)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_out, self.window_size, H, W)     # (B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
