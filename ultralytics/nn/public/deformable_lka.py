"""
DeformableLKA - 可变形大核注意力 (Deformable Large Kernel Attention)
论文: Deformable Large Kernel Attention for Generic Backbone Architectures
来源: WACV 2024
论文链接: https://arxiv.org/abs/2309.00121

包含模块:
- DeformConv: 可变形卷积辅助模块
- DeformableLKA: 可变形大核注意力模块（即插即用）

接口签名:
    DeformableLKA(dim)
    - dim: 输入通道数

依赖:
    - torchvision (用于 DeformConv2d)
"""

import torch
import torch.nn as nn
import torchvision

__all__ = ["DeformConv", "DeformableLKA"]


class DeformConv(nn.Module):
    """可变形卷积模块。

    通过学习的偏移量实现自适应感受野。

    Args:
        in_channels (int): 输入通道数
        groups (int): 分组卷积的组数
        kernel_size (tuple): 卷积核大小，默认 (3, 3)
        padding (int): 填充大小，默认 1
        stride (int): 步长，默认 1
        dilation (int): 膨胀率，默认 1
        bias (bool): 是否使用偏置，默认 True
    """

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super().__init__()

        self.offset_net = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=True,
        )

        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=False,
        )

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class DeformableLKA(nn.Module):
    """可变形大核注意力模块 (Deformable Large Kernel Attention)。

    【核心功能】
    结合可变形卷积和大核注意力机制，实现自适应的长距离依赖建模。

    【工作机制】
    1. 通过 5x5 可变形深度卷积捕获局部特征
    2. 通过 7x7 膨胀可变形深度卷积扩大感受野（等效大核）
    3. 通过 1x1 卷积生成注意力权重
    4. 注意力权重与原始输入相乘（残差连接）

    Args:
        dim (int): 输入通道数

    依赖:
        torchvision (用于 DeformConv2d)
    """

    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            输出特征图 [B, C, H, W]（通道数不变）
        """
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn
