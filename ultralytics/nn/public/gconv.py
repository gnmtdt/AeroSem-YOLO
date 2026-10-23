"""gConv 系列公共模块（迁移自 RTDETR-main `nn/extra_modules/block.py`）。"""

from __future__ import annotations

import math

import torch.nn as nn
from timm.models.layers import trunc_normal_

__all__ = ["gConvBlock"]


class gConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, gate_act=nn.Sigmoid, net_depth: int = 8):
        super().__init__()
        self.dim = dim
        self.net_depth = net_depth
        self.kernel_size = kernel_size

        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                padding_mode="reflect",
            ),
        )

        self.Wg = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True),
        )

        self.proj = nn.Conv2d(dim, dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.net_depth) ** (-1 / 4)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.Wv(x) * self.Wg(x)
        return self.proj(out)

