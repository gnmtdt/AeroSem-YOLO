"""Re-parameterized batch norm utilities used by AIFI variants."""

from __future__ import annotations

import torch
import torch.nn as nn


class RepBN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] -> BN on C with [B, C, L]
        x = x.transpose(1, 2)
        x = self.bn(x) + self.alpha * x
        return x.transpose(1, 2)


class LinearNorm(nn.Module):
    def __init__(self, dim: int, norm1: type[nn.Module], norm2: type[nn.Module], warm: int = 0, step: int = 300000, r0: float = 1.0):
        super().__init__()
        self.register_buffer("warm", torch.tensor(int(warm)))
        self.register_buffer("iter", torch.tensor(int(step)))
        self.register_buffer("total_step", torch.tensor(int(step)))
        self.r0 = float(r0)
        self.norm1 = norm1(dim)
        self.norm2 = norm2(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if int(self.warm) > 0:
                self.warm.copy_(self.warm - 1)
                return self.norm1(x)
            lamda = self.r0 * self.iter / self.total_step
            if int(self.iter) > 0:
                self.iter.copy_(self.iter - 1)
            x1 = self.norm1(x)
            x2 = self.norm2(x)
            return lamda * x1 + (1 - lamda) * x2
        return self.norm2(x)

