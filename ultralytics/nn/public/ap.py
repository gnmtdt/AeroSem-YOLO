"""AP/PSConv 系列公共模块（迁移自 RTDETR-main `nn/extra_modules/block.py`）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["PSConv", "APBottleneck"]


class PSConv(nn.Module):
    """Pinwheel-shaped Convolution using the Asymmetric Padding method."""

    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=p[g]) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class APBottleneck(nn.Module):
    """Asymmetric Padding bottleneck."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        p = [(2, 0, 2, 0), (0, 2, 0, 2), (0, 2, 2, 0), (2, 0, 0, 2)]
        self.pad = [nn.ZeroPad2d(padding=p[g_]) for g_ in range(4)]
        self.cv1 = Conv(c1, c_ // 4, k[0], 1, p=0)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = bool(shortcut and c1 == c2)

    def forward(self, x):
        y = self.cv2(torch.cat([self.cv1(self.pad[g_](x)) for g_ in range(4)], 1))
        return x + y if self.add else y

