# 论文: A Lightweight Semantic Segmentation Network Based on Self-Attention Mechanism and State Space Model for Efficient Urban Scene Segmentation (2025)
# 链接: https://ieeexplore.ieee.org/abstract/document/10969832
# 模块作用: 在单路融合特征上沿 H/W 方向进行信息整合与加权，强化方向性结构一致性并减弱跨模态错配。

import torch
import torch.nn as nn


class IIA(nn.Module):
    def __init__(self, channel: int | None = None, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel = channel
        self.kernel_size = int(kernel_size)
        self._built = False
        self._c = None
        self.conv_h: nn.Module | None = None
        self.conv_w: nn.Module | None = None

    def _build_if_needed(self, c: int) -> None:
        if self._built and self._c == c:
            return
        k = self.kernel_size
        p = k // 2
        self.conv_h = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, k), padding=(0, p), bias=False),
            nn.Sigmoid(),
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(k, 1), padding=(p, 0), bias=False),
            nn.Sigmoid(),
        )
        self._built = True
        self._c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise TypeError("IIA 期望输入 [B, C, H, W]")
        B, C, H, W = x.shape
        self._build_if_needed(C)
        # 方向性空间注意力：先在通道维做池化，得到 2 通道的空间描述，再用(1,k)/(k,1)卷积注入方向先验
        avg = torch.mean(x, dim=1, keepdim=True)          # [B, 1, H, W]
        maxv, _ = torch.max(x, dim=1, keepdim=True)       # [B, 1, H, W]
        pooled = torch.cat([avg, maxv], dim=1)            # [B, 2, H, W]
        attn_h = self.conv_h(pooled)                      # [B, 1, H, W] (沿 W 方向增强)
        x_h = x * attn_h
        attn_w = self.conv_w(pooled)                      # [B, 1, H, W] (沿 H 方向增强)
        x_w = x * attn_w
        return x + x_h + x_w
