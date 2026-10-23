# 论文: HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection (AAAI 2025)
# 链接: https://arxiv.org/abs/2412.10116
# 模块作用: 双路 patch 级依赖建模，通过块内 QK 相关将高相关空间依赖从一侧引导到另一侧，提升跨模态对齐后单路表达。

import torch
import torch.nn as nn


class SpatialDependencyPerception(nn.Module):
    def __init__(self, dim: int | None = None, patch: int = 8, inter_dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.patch = int(patch)
        self.inter_dim = inter_dim
        self._built = False
        self._c = None
        self._ci = None
        self.conv_q: nn.Module | None = None
        self.conv_k: nn.Module | None = None
        self.softmax = nn.Softmax(dim=-1)
        self.conv1x1: nn.Module | None = None

    def _build_if_needed(self, c: int) -> None:
        if self._built and self._c == c:
            return
        ci = c if self.inter_dim is None else int(self.inter_dim)
        self.conv_q = nn.Sequential(nn.Conv2d(c, ci, 1, bias=False), nn.GroupNorm(min(32, ci), ci))
        self.conv_k = nn.Sequential(nn.Conv2d(c, ci, 1, bias=False), nn.GroupNorm(min(32, ci), ci))
        self.conv1x1 = nn.Conv2d(c, ci, 1) if ci != c else nn.Identity()
        self._built = True
        self._c = c
        self._ci = ci

    def forward(self, x_low, x_high=None):
        if x_high is None and isinstance(x_low, (list, tuple)):
            x_low, x_high = x_low
        if not isinstance(x_low, torch.Tensor) or not isinstance(x_high, torch.Tensor):
            raise TypeError("SpatialDependencyPerception 需要两路输入张量")
        if x_low.shape != x_high.shape:
            raise ValueError(f"SDFM 要求两路输入形状一致，got {x_low.shape} vs {x_high.shape}")
        B, C, H, W = x_low.shape
        if H % self.patch != 0 or W % self.patch != 0:
            raise ValueError(f"SDFM 要求 H/W 能被 patch={self.patch} 整除，got {(H, W)}")
        self._build_if_needed(C)
        ci = self._ci
        q = self.conv_q(x_low)
        k = self.conv_k(x_high)
        p = self.patch
        q_unf = torch.nn.functional.unfold(q, kernel_size=p, stride=p)
        k_unf = torch.nn.functional.unfold(k, kernel_size=p, stride=p)
        L = q_unf.shape[-1]
        pa = p * p
        q_blk = q_unf.transpose(1, 2).contiguous().view(B * L, ci, pa).transpose(1, 2)
        k_blk = k_unf.transpose(1, 2).contiguous().view(B * L, ci, pa)
        attn = torch.bmm(q_blk, k_blk)
        attn = attn / (ci ** 0.5)
        attn = self.softmax(attn)
        v = k_blk.transpose(1, 2)
        out_blk = torch.bmm(attn, v)
        out_unf = out_blk.transpose(1, 2).contiguous().view(B, L, ci, pa).transpose(1, 2).contiguous().view(B, ci * pa, L)
        out = torch.nn.functional.fold(out_unf, output_size=(H, W), kernel_size=p, stride=p)
        if C != ci:
            x_low = self.conv1x1(x_low)
        return out + x_low
