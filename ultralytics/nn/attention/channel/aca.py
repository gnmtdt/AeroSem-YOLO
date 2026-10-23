"""
ACA - Asymmetric Convolution Attention

论文: Asymmetric Convolution Attention for Remote Sensing Image Change Detection
期刊: IEEE Transactions on Geoscience and Remote Sensing (TGRS 2025)
论文链接: https://ieeexplore.ieee.org/document/11175146

利用非对称卷积(1x3, 3x1, 3x3)分别构建Q/K/V，通过交叉注意力机制生成注意力权重，
对输入特征进行通道-空间联合加权。forward 返回 x * attn_factor。
"""

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


def make_divisible(x, divisor):
    return int(math.ceil(x / divisor) * divisor)


class ACA(nn.Module):
    def __init__(self, in_channels, expansion: float = 0.5):
        super().__init__()

        self.padding_1x3 = nn.ZeroPad2d(padding=(2, 0, 0, 0))
        self.padding_3x1 = nn.ZeroPad2d(padding=(0, 0, 2, 0))
        self.padding_3x3 = nn.ZeroPad2d(padding=(0, 2, 0, 2))

        hidden_channels = make_divisible(int(in_channels * expansion), 8)

        self.conv1x3_q = Conv(in_channels, hidden_channels, k=(1, 3), p=0, g=hidden_channels)  # Query
        self.conv3x1_k = Conv(in_channels, hidden_channels, k=(3, 1), p=0, g=hidden_channels)  # Key
        self.conv3x3_v = Conv(in_channels, hidden_channels, k=(3, 3), p=0, g=hidden_channels)  # Value

        self.cross_attn_conv = Conv(hidden_channels * 3, in_channels, k=1, g=1)

        self.act = nn.Sigmoid()

    def forward(self, x):
        q = self.conv1x3_q(self.padding_1x3(x))
        k = self.conv3x1_k(self.padding_3x1(x))
        v = self.conv3x3_v(self.padding_3x3(x))
        b, c, h, w = q.shape

        d_k = q.size(1)
        attn_map = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn_map = attn_map.flatten(2)

        attn_weights = torch.nn.functional.softmax(attn_map, dim=-1)

        attn_weights = attn_weights.view(b, c, h, h)

        attn_output = torch.matmul(attn_weights, v)

        attn_output_cat = torch.cat([attn_output, q, k], dim=1)

        out = self.cross_attn_conv(attn_output_cat)

        attn_factor = self.act(out)

        return x * attn_factor


__all__ = ['ACA']
