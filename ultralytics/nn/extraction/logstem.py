# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
LoGStem: Laplacian of Gaussian Stem Module

LEGNet: Learning Gaussian Filtering with Laplacian Operator for Visual Perception
论文来源: 可能在 CVPR2025 或相关期刊发表

模块说明:
    LoGStem 是基于高斯-拉普拉斯算子的网络骨干初始层，用于提取边缘特征并进行下采样。
    结合了 LoGFilter、Gaussian 和 DRFD_LoG 模块实现多尺度特征融合。

主要组件:
    - Conv_Extra: 3层卷积块 (Conv 1x1 -> Conv 3x3 -> Conv 1x1)
    - Gaussian: 高斯滤波器模块
    - LoGFilter: 高斯-拉普拉斯核滤波器
    - DRFD_LoG: 下采样融合模块
    - LoGStem: 主模块，对外导出
"""

import math
from typing import List

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["LoGStem", "LoGStem2x", "DRFD", "Cut"]


class Conv_Extra(nn.Module):
    """
    3层卷积块，用于特征增强

    Args:
        channel (int): 输入输出通道数
    """

    def __init__(self, channel):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(
            Conv(channel, 64, 1),
            Conv(64, 64, 3),
            Conv(64, channel, 1, act=False)
        )

    def forward(self, x):
        out = self.block(x)
        return out


class Gaussian(nn.Module):
    """
    高斯滤波器模块

    Args:
        dim (int): 通道维度
        size (int): 卷积核大小
        sigma (float): 高斯核标准差
        feature_extra (bool): 是否使用额外特征增强
    """

    def __init__(self, dim, size, sigma, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        if feature_extra:
            self.conv_extra = Conv_Extra(dim)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

    @staticmethod
    def gaussian_kernel(size: int, sigma: float):
        """生成高斯核"""
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


class DRFD_LoG(nn.Module):
    """
    DRFD 下采样融合模块 (Downsample Residual Fusion with Gaussian)

    Args:
        dim (int): 输入通道维度
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = nn.SiLU()
        self.norm_c = nn.BatchNorm2d(dim * 2)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = nn.BatchNorm2d(dim * 2)
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)
        # gaussian
        self.gaussian = Gaussian(self.outdim, 5, 0.5, feature_extra=False)
        self.norm_g = nn.BatchNorm2d(self.outdim)

    def forward(self, x):
        """Forward pass

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, 2C, H/2, W/2]
        """
        x = self.conv(x)  # x = [B, 2C, H, W]
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)
        max = self.norm_m(self.max_m(x))  # m = [B, 2C, H/2, W/2]
        conv = self.norm_c(self.act_c(self.conv_c(x)))  # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)  # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]

        return x


class LoGFilter(nn.Module):
    """
    高斯-拉普拉斯核滤波器

    Args:
        in_c (int): 输入通道数
        out_c (int): 输出通道数
        kernel_size (int): 卷积核大小
        sigma (float): 高斯核标准差
    """

    def __init__(self, in_c, out_c, kernel_size, sigma):
        super(LoGFilter, self).__init__()
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        # 创建高斯-拉普拉斯核
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax)
        # 计算高斯-拉普拉斯核
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        # 归一化
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.act = nn.SiLU()
        self.norm1 = nn.BatchNorm2d(out_c)
        self.norm2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        """Forward pass

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C/4, H, W]
        """
        x = self.conv_init(x)  # x = [B, C/4, H, W]
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x


class LoGStem(nn.Module):
    """
    LoG Stem 模块 - 基于高斯-拉普拉斯算子的网络初始层

    用于替代标准 Conv stem，实现边缘感知的特征提取。

    Args:
        in_chans (int): 输入通道数 (RGB=3, Dual=6)
        stem_dim (int): Stem 维度，决定输出通道数 (默认128)

    Example:
        >>> model = LoGStem(in_chans=3, stem_dim=128)
        >>> x = torch.randn(1, 3, 640, 640)
        >>> out = model(x)  # out = [B, 128, 160, 160]
    """

    def __init__(self, in_chans, stem_dim=128):
        super().__init__()
        out_c14 = int(stem_dim / 4)  # stem_dim / 4
        out_c12 = int(stem_dim / 2)  # stem_dim / 2
        # original size to 2x downsampling layer
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            Conv(out_c12, out_c12, 3, 2, g=out_c12)
        )
        # 定义LoG滤波器
        self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0)
        # gaussian
        self.gaussian = Gaussian(out_c12, 9, 0.5)
        self.norm = nn.BatchNorm2d(out_c12)
        self.drfd = DRFD_LoG(out_c12)

    def forward(self, x):
        """Forward pass

        Args:
            x: Input tensor [B, in_chans, H, W]

        Returns:
            Output tensor [B, stem_dim, H/4, W/4]
        """
        x = self.LoG(x)
        # original size to 2x downsampling layer
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))
        x = self.drfd(x)

        return x  # x = [B, C, H/4, W/4]


class LoGStem2x(nn.Module):
    """
    LoG Stem2x 模块 - 基于高斯-拉普拉斯算子的标准输入层 (2x 下采样)

    LoGStem 的变体，仅做 2x 下采样（与标准 Conv 一致），可作为中间层替代 Conv。
    输出: [B, stem_dim, H/2, W/2] 从输入 [B, in_chans, H, W]

    Args:
        in_chans (int): 输入通道数 (RGB=3, Dual=6)
        stem_dim (int): Stem 维度，决定输出通道数 (默认128)

    Example:
        >>> model = LoGStem2x(in_chans=3, stem_dim=64)
        >>> x = torch.randn(1, 3, 640, 640)
        >>> out = model(x)  # out = [B, 64, 320, 320]
    """

    def __init__(self, in_chans, stem_dim=128):
        super().__init__()
        out_c14 = int(stem_dim / 4)  # stem_dim / 4
        out_c12 = int(stem_dim / 2)  # stem_dim / 2
        # LoG 滤波 + 2x 下采样
        self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0)
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            Conv(out_c12, out_c12, 3, 2, g=out_c12)
        )
        # Gaussian 残差
        self.gaussian = Gaussian(out_c12, 9, 0.5)
        self.norm = nn.BatchNorm2d(out_c12)
        # 通道扩展到 stem_dim
        self.expand = Conv(out_c12, stem_dim, k=1)

    def forward(self, x):
        """Forward pass

        Args:
            x: Input tensor [B, in_chans, H, W]

        Returns:
            Output tensor [B, stem_dim, H/2, W/2]
        """
        x = self.LoG(x)
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))
        x = self.expand(x)
        return x  # x = [B, stem_dim, H/2, W/2]


class Cut(nn.Module):
    """Cut downsample module - space-to-depth with fusion.

    Performs space-to-depth transformation followed by 1x1 convolution for downsample.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)  # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)
        return x


class DRFD(nn.Module):
    """Downsample Residual Fusion with Gaussian (DRFD) module.

    A downsample module combining Cut, Conv, and MaxPool with Gaussian filtering.
    Output: [B, 2C, H/2, W/2] from input [B, C, H/2, W/2]

    Args:
        c1: Input channel count (from parse_model)
        c2: Output channel count (specified in YAML, typically 2x input)
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.cut_c = Cut(in_channels=c1, out_channels=c2)
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, groups=c1)
        self.conv_x = nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1, groups=c2)
        self.act_x = nn.GELU()
        self.batch_norm_x = nn.BatchNorm2d(c2)
        self.batch_norm_m = nn.BatchNorm2d(c2)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion = nn.Conv2d(3 * c2, c2, kernel_size=1, stride=1)

    def forward(self, x):  # input: x = [B, C, H, W]
        c = x  # c = [B, C, H, W]
        x = self.conv(x)  # x = [B, C, H, W] --> [B, 2C, H, W]
        m = x  # m = [B, 2C, H, W]

        # CutD
        c = self.cut_c(c)  # c = [B, C, H, W] --> [B, 2C, H/2, W/2]

        # ConvD
        x = self.conv_x(x)  # x = [B, 2C, H, W] --> [B, 2C, H/2, W/2]
        x = self.act_x(x)
        x = self.batch_norm_x(x)

        # MaxD
        m = self.max_m(m)  # m = [B, 2C, H/2, W/2]
        m = self.batch_norm_m(m)

        # Concat + conv
        x = torch.cat([c, x, m], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 6C, H/2, W/2] --> [B, 2C, H/2, W/2]

        return x
