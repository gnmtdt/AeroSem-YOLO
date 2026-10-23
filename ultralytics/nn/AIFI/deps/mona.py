"""Mona block used by AIFI_Mona (ported from RTDETR-main)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


class MonaOp(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = (self.conv1(x) + self.conv2(x) + self.conv3(x)) / 3.0 + identity
        identity = x
        x = self.projector(x)
        return identity + x


class Mona(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.project1 = nn.Conv2d(in_dim, 64, 1)
        self.nonlinear = F.gelu
        self.project2 = nn.Conv2d(64, in_dim, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = MonaOp(64)
        self.norm = LayerNorm2d(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax
        x = self.project1(x)
        x = self.adapter_conv(x)
        x = self.dropout(self.nonlinear(x))
        x = self.project2(x)
        return identity + x

