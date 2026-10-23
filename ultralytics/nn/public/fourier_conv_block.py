"""FourierConv Bottleneck 封装（使用本仓库 `nn/modules/conv.py:FourierConv`）。"""

from __future__ import annotations

from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import FourierConv

__all__ = ["Bottleneck_FourierConv"]


class Bottleneck_FourierConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), size=None, e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = FourierConv(c1, c_, out_size=size, stride=1)
        self.cv2 = FourierConv(c_, c2, out_size=size, stride=1)

