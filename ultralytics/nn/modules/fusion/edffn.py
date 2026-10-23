# 论文: Efficient Visual State Space Model for Image Deblurring (CVPR 2025)
# 链接: https://arxiv.org/pdf/2405.14343
# 模块作用: 在融合后单路特征上执行频域选择性增强，抑制跨模态冗余并突出判别性纹理/结构。

import torch
import torch.nn as nn


class EDFFN(nn.Module):
    def __init__(self, dim: int | None = None, patch_size: int = 8, ffn_expansion: float = 4.0, bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.patch = int(patch_size)
        self.ffn_expansion = float(ffn_expansion)
        self.bias = bool(bias)
        self._built = False
        self._c = None
        self._hidden = None
        self.project_in: nn.Module | None = None
        self.dwconv: nn.Module | None = None
        self.project_out: nn.Module | None = None
        self.fft_weight: torch.nn.Parameter | None = None

    def _build_if_needed(self, c: int) -> None:
        if self._built and self._c == c:
            return
        hidden = max(int(c * self.ffn_expansion), 1)
        self.project_in = nn.Conv2d(c, hidden * 2, kernel_size=1, bias=self.bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1, groups=hidden * 2,
                                 bias=self.bias)
        self.project_out = nn.Conv2d(hidden, c, kernel_size=1, bias=self.bias)
        self.fft_weight = None
        self._built = True
        self._c = c
        self._hidden = hidden

    def _ensure_fft_weight(self, c: int) -> None:
        if self.fft_weight is None:
            P = self.patch
            w = torch.ones(c, 1, 1, P, P // 2 + 1)
            self.fft_weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise TypeError("EDFFN 期望输入 [B, C, H, W]")
        B, C, H, W = x.shape
        if H % self.patch != 0 or W % self.patch != 0:
            raise ValueError(f"EDFFN 要求 H、W 能被 patch_size={self.patch} 整除，got {(H, W)}")
        self._build_if_needed(C)
        z = self.project_in(x)
        z1, z2 = self.dwconv(z).chunk(2, dim=1)
        z = torch.nn.functional.gelu(z1) * z2
        z = self.project_out(z)
        P = self.patch
        Hb, Wb = H // P, W // P
        z_patch = z.view(B, C, Hb, P, Wb, P).permute(0, 1, 2, 4, 3, 5)
        z_fft = torch.fft.rfft2(z_patch.float(), dim=(-2, -1))
        self._ensure_fft_weight(C)
        z_fft = z_fft * self.fft_weight
        z_patch = torch.fft.irfft2(z_fft, s=(P, P))
        z_rec = z_patch.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
        return z_rec
