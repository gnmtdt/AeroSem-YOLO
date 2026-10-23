# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Frequency-spatial modules adapted from FSCFNet-style designs."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv

__all__ = ("HaarDWT2d", "FSConv", "ACA", "MRCB")


class HaarDWT2d(nn.Module):
    """Fixed 2D Haar wavelet decomposition."""

    def __init__(self) -> None:
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        filt = torch.stack((ll, lh, hl, hh), dim=0).unsqueeze(1)
        self.register_buffer("filt", filt, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return LL, LH, HL, HH frequency bands."""
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        c = x.shape[1]
        weight = self.filt.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, padding=0, groups=c)
        y = y.view(x.shape[0], c, 4, y.shape[-2], y.shape[-1])
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


class FSConv(nn.Module):
    """
    Frequency-spatial convolution with a lightweight Haar branch.

    The module keeps a standard spatial path while extracting low- and high-frequency
    representations through a fixed Haar decomposition. Frequency features are resized
    to match the spatial path, then fused by a final 1x1 projection.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1) -> None:
        super().__init__()
        if c2 < 4:
            raise ValueError(f"FSConv requires c2 >= 4, got {c2}")

        spatial_c = max(1, c2 // 2)
        low_c = max(1, (c2 - spatial_c) // 2)
        high_c = c2 - spatial_c - low_c

        self.pre = DWConv(c1, c1, 3, 1)
        self.dwt = HaarDWT2d()
        self.spatial = Conv(c1, spatial_c, k, s)
        self.low = Conv(c1, low_c, 3, 1)
        self.high = Conv(c1 * 3, high_c, 3, 1, g=math.gcd(c1 * 3, high_c))
        self.mix = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse spatial, low-frequency and high-frequency features."""
        spatial = self.spatial(x)

        ll, lh, hl, hh = self.dwt(self.pre(x))
        low = self.low(ll)
        high = self.high(torch.cat((lh, hl, hh), dim=1))

        target_hw = spatial.shape[-2:]
        if low.shape[-2:] != target_hw:
            low = F.interpolate(low, size=target_hw, mode="bilinear", align_corners=False)
        if high.shape[-2:] != target_hw:
            high = F.interpolate(high, size=target_hw, mode="bilinear", align_corners=False)

        return self.mix(torch.cat((spatial, low, high), dim=1))


class ACA(nn.Module):
    """Asymmetric cross-domain attention with directional convolutions."""

    def __init__(self, c1: int, c2: int, ratio: float = 0.5) -> None:
        super().__init__()
        if ratio <= 0:
            raise ValueError(f"ACA ratio must be positive, got {ratio}")

        c_mid = max(8, int(min(c1, c2) * ratio))
        self.q = nn.Sequential(
            nn.Conv2d(c1, c_mid, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),
        )
        self.k = nn.Sequential(
            nn.Conv2d(c1, c_mid, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),
        )
        self.v = nn.Sequential(
            nn.Conv2d(c1, c_mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(c_mid * 3, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.short = Conv(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply center-aware reweighting while preserving residual information."""
        base = self.short(x)
        attn = torch.sigmoid(self.proj(torch.cat((self.q(x), self.k(x), self.v(x)), dim=1)))
        return base + base * attn


class MRCB(nn.Module):
    """Multiscale receptive contextual block with dilated branches."""

    def __init__(
        self,
        c1: int,
        c2: int,
        dilation: Sequence[int] = (1, 3, 5),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        if len(dilation) != 3:
            raise ValueError(f"MRCB dilation expects 3 values, got {dilation}")
        if e <= 0:
            raise ValueError(f"MRCB expansion must be positive, got {e}")

        c_mid = max(8, int(c2 * e))
        self.stem = Conv(c1, c_mid, 1, 1)
        self.branch1 = Conv(c_mid, c_mid, 3, 1, d=int(dilation[0]))
        self.branch2 = nn.Sequential(
            Conv(c_mid, c_mid, 3, 1),
            Conv(c_mid, c_mid, 3, 1, d=int(dilation[1])),
        )
        self.branch3 = nn.Sequential(
            Conv(c_mid, c_mid, 3, 1),
            Conv(c_mid, c_mid, 5, 1, d=int(dilation[2])),
        )
        self.fuse = Conv(c_mid * 3, c2, 1, 1, act=False)
        self.short = Conv(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity()
        self.act = nn.SiLU()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate local and long-range context with residual scaling."""
        y = self.stem(x)
        y = torch.cat((self.branch1(y), self.branch2(y), self.branch3(y)), dim=1)
        y = self.fuse(y)
        return self.act(self.short(x) + self.scale * y)
