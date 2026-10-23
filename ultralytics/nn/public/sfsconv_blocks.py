"""SFSConv-related bottleneck blocks used by C2f_SFSConv / C2f_PSFSConv."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.public.sfsconv import SFS_Conv
from ultralytics.nn.public.faster_sfsconv import Partial_SFSConv


class Bottleneck_SFSConv(Bottleneck):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = SFS_Conv(c1, c_)
        self.cv2 = SFS_Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_PSFSConv(Bottleneck):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = Partial_SFSConv(c1)
        self.cv2 = Partial_SFSConv(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

