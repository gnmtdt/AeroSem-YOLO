"""
CDFA - Contrast-Driven Feature Aggregation (对比驱动特征聚合)

论文: Contrast-Driven Feature Aggregation for Improved Vision Transformer
期刊/会议: arXiv 2024
论文链接: https://arxiv.org/pdf/2407.19768
         https://arxiv.org/abs/2412.08345
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ['HaarWaveletConv', 'ContrastDrivenFeatureAggregation']


class HaarWaveletConv(nn.Module):
    """Haar 小波卷积，将输入分解为低频(近似)和高频(细节)分量。"""

    def __init__(self, in_channels, grad=False):
        super(HaarWaveletConv, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        # h (水平方向高频)
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        # v (垂直方向高频)
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        # d (对角方向高频)
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x):
        B, _, H, W = x.size()
        x = F.pad(x, [0, 1, 0, 1], value=0)
        out = F.conv2d(x, self.haar_weights, bias=None, stride=1, groups=self.in_channels) / 4.0
        out = out.reshape([B, self.in_channels, 4, H, W])
        out = torch.transpose(out, 1, 2)
        out = out.reshape([B, self.in_channels * 4, H, W])

        # a: 低频(近似), h: 水平高频, v: 垂直高频, d: 对角高频
        a, h, v, d = out.chunk(4, 1)

        # 返回低频和高频
        return a, h + v + d


class ContrastDrivenFeatureAggregation(nn.Module):
    """对比驱动特征聚合注意力模块。

    利用 Haar 小波将特征分解为前景(高频)和背景(低频)，
    通过对比驱动的方式分别进行注意力聚合。

    Args:
        dim (int): 输入通道数。
        num_heads (int): 注意力头数。默认 8。
        kernel_size (int): 局部注意力核大小。默认 3。
        padding (int): 填充大小。默认 1。
        stride (int): 步幅。默认 1。
        attn_drop (float): 注意力 dropout 率。默认 0.。
        proj_drop (float): 投影 dropout 率。默认 0.。
    """

    def __init__(self, dim, num_heads=8, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.wavelet = HaarWaveletConv(dim)

        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            Conv(dim, dim, 3),
            Conv(dim, dim, 3),
        )
        self.output_cbr = nn.Sequential(
            Conv(dim, dim, 3),
            Conv(dim, dim, 3),
        )

    def forward(self, x):
        x = self.input_cbr(x)
        bg, fg = self.wavelet(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)

        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')

        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(
            B, self.num_heads, self.head_dim,
            self.kernel_size * self.kernel_size,
            -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')

        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted_bg)

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted
