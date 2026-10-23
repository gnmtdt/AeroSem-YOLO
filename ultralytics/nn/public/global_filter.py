"""GlobalFilter 系列公共模块（迁移自 RTDETR-main `nn/extra_modules/block.py`）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from timm.models.layers import DropPath

from .common_glu import ConvolutionalGLU
from .tsdn import LayerNorm

__all__ = ["GlobalFilterBlock"]


class GlobalFilter(nn.Module):
    def __init__(self, dim: int, size: int):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(dim, size, size // 2 + 1, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, a, b = x.size()
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        return torch.fft.irfft2(x, s=(a, b), dim=(2, 3), norm="ortho")


class GlobalFilterBlock(nn.Module):
    def __init__(self, dim: int, size: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.filter = GlobalFilter(dim, size=size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))

