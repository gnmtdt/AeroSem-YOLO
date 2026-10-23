"""SpectralEnhancedFFN used by AIFI_SEFFN (ported from RTDETR-main, minimal)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEnhancedFFN(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.0, bias: bool = False):
        super().__init__()
        hidden_features = int(dim * float(ffn_expansion_factor))
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=2,
            groups=hidden_features * 2,
            bias=bias,
            dilation=2,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.fft_channel_weight = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))
        self.fft_channel_bias = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))

    @staticmethod
    def _pad_to_factor(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, tuple[int, int]]:
        hw = x.shape[-1]
        pad = (0, 0) if hw % factor == 0 else (0, (hw // factor + 1) * factor - hw)
        return F.pad(x, pad, "constant", 0), pad

    @staticmethod
    def _unpad(x: torch.Tensor, pad: tuple[int, int]) -> torch.Tensor:
        hw = x.shape[-1]
        return x[..., pad[0] : hw - pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = self.dwconv(self.project_in(x))
        x, pad_w = self._pad_to_factor(x, 2)
        x_fft = torch.fft.rfft2(x.float())
        x_fft = self.fft_channel_weight * x_fft + self.fft_channel_bias
        x = torch.fft.irfft2(x_fft)
        x = self._unpad(x, pad_w)
        x1, x2 = x.chunk(2, dim=1)
        x = F.silu(x1) * x2
        return self.project_out(x.to(x_dtype))

