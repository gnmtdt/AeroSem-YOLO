"""wConv2d 系列公共模块（迁移自 RTDETR-main `nn/extra_modules/block.py`）。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import autopad

__all__ = ["wConv2d", "Bottleneck_wConv"]


class wConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=1, groups=1, dilation=1, bias=False):
        super().__init__()
        self.stride = _pair(stride)
        self.kernel_size = _pair(kernel_size)
        self.padding = autopad(self.kernel_size, d=dilation)
        self.groups = groups
        self.dilation = _pair(dilation)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device("cpu")
        alfa = torch.tensor(den, device=device)
        self.register_buffer("alfa", torch.cat([alfa, torch.tensor([1.0], device=device), torch.flip(alfa, dims=[0])]))
        self.register_buffer("Phi", torch.outer(self.alfa, self.alfa))

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv2d(
            x,
            weight_Phi,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            dilation=self.dilation,
        )


class Bottleneck_wConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), den=None, e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = wConv2d(c1, c_, k[0], den, padding=k[0] // 2)
        self.cv2 = wConv2d(c_, c2, k[1], den, padding=k[0] // 2)

