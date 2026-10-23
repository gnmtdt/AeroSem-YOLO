"""
ASF - 注意力尺度融合模块 (arXiv 2312.06458)
论文链接: https://arxiv.org/pdf/2312.06458

通过通道注意力和局部注意力实现多尺度特征的自适应融合。
包含:
  - Zoom_cat: 多尺度特征缩放拼接
  - ScalSeq: 3D 卷积序列特征融合
  - Add: 逐元素相加
  - asf_attention_model: 综合注意力融合模型（通道注意力 + 局部注意力）
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class Zoom_cat(nn.Module):
    """多尺度特征缩放拼接模块。

    将三个尺度的特征对齐到中间尺度后进行拼接。
    大尺度特征使用自适应最大池化和平均池化之和，
    小尺度特征使用最近邻插值上采样。

    Args:
        无特殊参数
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        large, medium, small = x
        target_size = medium.shape[2:]
        large = F.adaptive_max_pool2d(large, target_size) + F.adaptive_avg_pool2d(large, target_size)
        small = F.interpolate(small, target_size, mode="nearest")
        return torch.cat([large, medium, small], dim=1)


class ScalSeq(nn.Module):
    """3D 卷积序列特征融合模块。

    将三个尺度的特征对齐到最小尺度后，通过 3D 卷积在尺度维度进行融合。

    Args:
        inc: 三个尺度的输入通道数列表
        channel: 统一输出通道数
    """

    def __init__(self, inc, channel):
        super().__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        p3, p4, p5 = x
        if hasattr(self, "conv0"):
            p3 = self.conv0(p3)
        p4 = F.interpolate(self.conv1(p4), p3.shape[2:], mode="nearest")
        p5 = F.interpolate(self.conv2(p5), p3.shape[2:], mode="nearest")
        combine = torch.cat([p3.unsqueeze(2), p4.unsqueeze(2), p5.unsqueeze(2)], dim=2)
        fused = self.act(self.bn(self.conv3d(combine)))
        return self.pool_3d(fused).squeeze(2)


class Add(nn.Module):
    """逐元素相加模块，将多个特征图求和。"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(torch.stack(x, dim=0), dim=0)


class asf_channel_att(nn.Module):
    """ASF 通道注意力模块。

    通过自适应平均池化和 1D 卷积生成通道注意力权重。

    Args:
        channel: 输入通道数
        b: 计算核大小的参数 b，默认 1
        gamma: 计算核大小的参数 gamma，默认 2
    """

    def __init__(self, channel, b=1, gamma=2):
        super().__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class asf_local_att(nn.Module):
    """ASF 局部注意力模块。

    通过水平/垂直方向的特征压缩和交叉注意力实现局部空间注意力。

    Args:
        channel: 输入通道数
        reduction: 压缩比例，默认 16
    """

    def __init__(self, channel, reduction=16):
        super().__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class asf_attention_model(nn.Module):
    """综合注意力融合模型，结合通道注意力和局部空间注意力。

    Args:
        ch: 输入通道数，默认 256
    """

    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = asf_channel_att(ch)
        self.local_att = asf_local_att(ch)

    def forward(self, x):
        input1, input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x


__all__ = ("Add", "ScalSeq", "Zoom_cat", "asf_attention_model")
