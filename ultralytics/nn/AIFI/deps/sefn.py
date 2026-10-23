"""SEFN used by AIFI_SEFN (ported from RTDETR-main, no optional deps)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class _WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layernorm_type: str):
        super().__init__()
        self.body = _BiasFreeLayerNorm(dim) if layernorm_type == "BiasFree" else _WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


class SEFN(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.0, bias: bool = False):
        super().__init__()
        hidden_features = int(dim * float(ffn_expansion_factor))
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, "WithBias"),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, "WithBias"),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)

        # Spatial branch
        y = self.avg_pool(spatial)
        y = self.conv(y)
        y = self.upsample(y)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.fusion(torch.cat((x1, y), dim=1))
        x1 = self.dwconv_afterfusion(x1)

        x = F.gelu(x1) * x2
        return self.project_out(x)

