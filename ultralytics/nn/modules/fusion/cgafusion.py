"""
Channel-Guided Attention Fusion (CGAFusion).

论文: TIP2024 - CGAFusion
来源: https://arxiv.org/pdf/2301.04805
参考: ultralytics/nn/module_images/TIP2024-CGAFusion.md

CGAFusion 实现通道引导注意力融合，通过三级注意力机制融合两路特征：
1. 空间注意力 (SpatialAttention_CGA): 基于均值/最大值的 7x7 卷积空间权重
2. 通道注意力 (ChannelAttention_CGA): 自适应全局池化 + MLP 通道权重
3. 像素注意力 (PixelAttention_CGA): 像素级融合权重，结合空间和通道注意力
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ['CGAFusion']


class SpatialAttention_CGA(nn.Module):
    """空间注意力：通过均值和最大值池化生成空间权重图.

    使用 7x7 反射填充卷积，输出单通道空间注意力图。
    """

    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        return self.sa(x2)


class ChannelAttention_CGA(nn.Module):
    """通道注意力：通过全局平均池化 + MLP 生成通道权重.

    Args:
        dim (int): 输入通道数
        reduction (int): 通道缩减比例，默认 8
    """

    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.ca(self.gap(x))


class PixelAttention_CGA(nn.Module):
    """像素注意力：像素级融合权重，结合空间和通道注意力先验.

    Args:
        dim (int): 通道维度
    """

    def __init__(self, dim):
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode="reflect", groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        b, c, h, w = x.shape
        x_pair = torch.cat([x.unsqueeze(2), pattn1.unsqueeze(2)], dim=2).reshape(b, 2 * c, h, w)
        return self.sigmoid(self.pa2(x_pair))


class CGAFusion(nn.Module):
    """通道引导注意力融合模块.

    通过空间/通道/像素三级注意力机制自适应融合两路输入特征。

    Args:
        in_dim (list[int]): 两路输入通道数 [c1, c2]
        out_dim (int): 输出通道数
        reduction (int): 通道注意力缩减比例，默认 8

    Inputs:
        data (list[Tensor]): 包含两个特征图的列表，空间尺寸需一致

    Shape:
        - 输入: [(B, c1, H, W), (B, c2, H, W)]
        - 输出: (B, out_dim, H, W)
    """

    def __init__(self, in_dim, out_dim, reduction=8):
        super().__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(out_dim, reduction)
        self.pa = PixelAttention_CGA(out_dim)
        self.conv = nn.Conv2d(out_dim, out_dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.conv_adjust = nn.ModuleList([])
        for i in in_dim:
            if i != out_dim:
                self.conv_adjust.append(Conv(i, out_dim, 1))
            else:
                self.conv_adjust.append(nn.Identity())

    def forward(self, data):
        x, y = data
        x = self.conv_adjust[0](x)
        y = self.conv_adjust[1](y)
        initial = x + y
        pattn1 = self.sa(initial) + self.ca(initial)
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        out = initial + pattn2 * x + (1 - pattn2) * y
        return self.conv(out)
