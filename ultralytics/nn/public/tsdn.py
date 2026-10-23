"""
TSDN 相关公共依赖（迁移自 RTDETR-main `nn/extra_modules/tsdn.py` 的 LayerNorm 部分）。

说明：
- 本文件仅提供 LayerNorm 与必要的张量形状转换工具；
- 不包含 DTAB 等其它模块（保持公共依赖最小且可复用）。
"""

from __future__ import annotations

import numbers

import torch
import torch.nn as nn

__all__ = ["LayerNorm"]


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    # [B, C, H, W] -> [B, H*W, C]
    b, c, h, w = x.shape
    return x.view(b, c, h * w).transpose(1, 2).contiguous()


def _to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # [B, H*W, C] -> [B, C, H, W]
    b, n, c = x.shape
    if n != h * w:
        raise ValueError(f"to_4d expects n==h*w, got n={n}, h*w={h*w}")
    return x.transpose(1, 2).contiguous().view(b, c, h, w)


class _BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int, ...]):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError(f"BiasFreeLayerNorm expects 1D normalized_shape, got {normalized_shape}")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class _WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int, ...]):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError(f"WithBiasLayerNorm expects 1D normalized_shape, got {normalized_shape}")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """2D LayerNorm：对 NCHW 特征做 flatten(HW) 的 LayerNorm。"""

    def __init__(self, dim: int, layernorm_type: str = "BiasFree"):
        super().__init__()
        self.body = _BiasFreeLayerNorm(dim) if layernorm_type == "BiasFree" else _WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)

