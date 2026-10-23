# 论文: MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration (CVPR 2025)
# 链接: https://arxiv.org/pdf/2412.20066
# 模块作用: 在单路融合特征上进行分组通道洗牌与门控，挖掘不同展开序列间的互补性并抑制冗余。

import torch
import torch.nn as nn


class SequenceShuffleAttention(nn.Module):
    def __init__(self, group: int = 4) -> None:
        super().__init__()
        self.group = int(group)
        self.gating: nn.Module | None = None
        self._c: int | None = None

    def _build_if_needed(self, c: int) -> None:
        if self.gating is not None and self._c == c:
            return
        if c % self.group != 0:
            raise ValueError(f"SequenceShuffleAttention: 通道数 {c} 不能被分组数 {self.group} 整除")
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, groups=self.group, bias=True),
            nn.Sigmoid(),
        )
        self._c = c

    def channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.group
        if c % g != 0:
            raise ValueError(f"SequenceShuffleAttention.channel_shuffle: 通道数 {c} 不能被分组数 {g} 整除")
        gc = c // g
        x = x.reshape(b, gc, g, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, h, w)
        return x

    def channel_rearrange(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.group
        if c % g != 0:
            raise ValueError(f"SequenceShuffleAttention.channel_rearrange: 通道数 {c} 不能被分组数 {g} 整除")
        gc = c // g
        x = x.reshape(b, g, gc, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, h, w)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise TypeError("SequenceShuffleAttention 期望输入 [B, C, H, W]")
        _, c, _, _ = x.shape
        self._build_if_needed(c)
        residual = x
        x = self.channel_shuffle(x)
        g = self.gating(x)
        g = self.channel_rearrange(g)
        return residual * g
