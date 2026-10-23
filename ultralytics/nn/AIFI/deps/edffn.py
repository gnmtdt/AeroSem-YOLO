"""EDFFN used by AIFI_EDFFN (ported from RTDETR-main, minimal)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDFFN(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.0, bias: bool = False, patch_size: int = 8):
        super().__init__()
        hidden_features = int(dim * float(ffn_expansion_factor))
        self.patch_size = int(patch_size)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias
        )
        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        p = self.patch_size
        hn = (p - h % p) % p
        wn = (p - w % p) % p
        x = F.pad(x, (0, wn, 0, hn), mode="reflect")

        hp, wp = (h + hn) // p, (w + wn) // p
        x_patch = x.view(b, c, hp, p, wp, p).permute(0, 1, 2, 4, 3, 5).contiguous()
        x_patch_fft = torch.fft.rfft2(x_patch.float(), dim=(-2, -1))
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(p, p), dim=(-2, -1))
        x = x_patch.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, h + hn, w + wn)
        x = x[:, :, :h, :w]
        return x.to(x_dtype)

