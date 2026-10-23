"""Partial-FDConv bottleneck blocks used by C2f_PFDConv."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.public.faster_fdconv import Partial_FDConv


class Bottleneck_PFDConv(Bottleneck):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = Partial_FDConv(c1)
        self.cv2 = Partial_FDConv(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

