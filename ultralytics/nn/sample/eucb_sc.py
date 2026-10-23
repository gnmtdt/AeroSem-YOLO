"""EUCB_SC: 带通道混合的增强型上采样卷积块.

论文: Efficient Up-sampling Convolution Block (CVPR 2024)
论文链接: https://arxiv.org/abs/2405.06880
扩展版论文链接: https://arxiv.org/abs/2503.02394
迁移自参考库 Ultralytics_674595707/nn/extra_modules/upsample/eucb_sc.py

EUCB_SC 在 EUCB 基础上增加了 Shift_channel_mix 通道混合操作，
通过通道分割与空间循环移位增强跨通道信息交互。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["EUCB_SC"]


class _ShiftChannelMix(nn.Module):
    """通道分割循环移位混合模块.

    将输入沿通道维度四等分，分别在高度和宽度方向施加正负循环移位，
    实现跨通道的信息交互，不引入额外参数。

    Args:
        shift_size: 循环移位的像素数。
    """

    def __init__(self, shift_size: int):
        super(_ShiftChannelMix, self).__init__()
        self.shift_size = shift_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: 分块 -> 移位 -> 拼接."""
        x1, x2, x3, x4 = x.chunk(4, dim=1)

        x1 = torch.roll(x1, self.shift_size, dims=2)
        x2 = torch.roll(x2, -self.shift_size, dims=2)
        x3 = torch.roll(x3, self.shift_size, dims=3)
        x4 = torch.roll(x4, -self.shift_size, dims=3)

        return torch.cat([x1, x2, x3, x4], 1)


class EUCB_SC(nn.Module):
    """带通道混合的增强型上采样卷积块 (Efficient Up-sampling Convolution Block with Shift Channel Mix).

    先上采样再深度可分离卷积，然后通道混洗 + 循环移位混合，最后 1x1 卷积。

    Args:
        in_channels: 输入通道数（同时也是输出通道数）。
        kernel_size: 深度可分离卷积的核大小，默认 3。
        stride: 卷积步幅，默认 1。
    """

    def __init__(self, in_channels: int, kernel_size: int = 3, stride: int = 1):
        super(EUCB_SC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels, s=stride, act=nn.ReLU()),
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self._shift_channel_mix = _ShiftChannelMix(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            x: 输入张量 (B, C, H, W)。

        Returns:
            上采样后的张量 (B, C, 2*H, 2*W)。
        """
        x = self.up_dwc(x)
        x = self._channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

    def _channel_shuffle(self, x: torch.Tensor, groups: int) -> torch.Tensor:
        """通道混洗 + 循环移位混合."""
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        x = self._shift_channel_mix(x)
        return x
