"""SMPConv + CGLU block used by C2f_SMPCGLU."""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.public.common_glu import ConvolutionalGLU
from ultralytics.nn.public.smpconv import SMPConv


class SMPCGLU(nn.Module):
    def __init__(self, inc: int, kernel_size: int, drop_path: float = 0.1, n_points: int = 4):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = ConvolutionalGLU(inc)
        self.smpconv = nn.Sequential(
            SMPConv(inc, kernel_size, n_points, 1, padding=kernel_size // 2, groups=1),
            Conv.default_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.smpconv(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

