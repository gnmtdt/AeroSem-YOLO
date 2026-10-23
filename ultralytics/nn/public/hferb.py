"""HFERB (High-Frequency Enhancement Residual Block) used by C2f_HFERB."""

from __future__ import annotations

import torch
import torch.nn as nn


class HFERB(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mid_dim = dim // 2
        self.act = nn.GELU()
        self.last_fc = nn.Conv2d(self.dim, self.dim, 1)

        self.fc = nn.Conv2d(self.mid_dim, self.mid_dim, 1)
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.conv = nn.Conv2d(self.mid_dim, self.mid_dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = x
        lfe = self.act(self.conv(x[:, : self.mid_dim, :, :]))
        hfe = self.act(self.fc(self.max_pool(x[:, self.mid_dim :, :, :])))
        x = torch.cat([lfe, hfe], dim=1)
        x = short + self.last_fc(x)
        return x

