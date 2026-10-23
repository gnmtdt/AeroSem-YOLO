"""
Wavelet-Domain Adaptive Fusion (WDAF).

来源: 自研模块 (魔傀面具整理)
参考: ultralytics/nn/module_images/自研模块-WDAF.md

WDAF 通过小波域自适应融合实现双模态特征的频域增强融合：
1. Haar 小波分解：将特征图分解为低频（近似）和高频（细节）分量
2. 空间域门控：通过 sigmoid 门控自适应加权两路特征
3. 频域门控：分别对低频和高频分量进行门控融合
4. 双域互补：空间域融合 + 小波域引导（可学习比例）作为最终输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ['WDAF']


class HaarWaveletDecomposition(nn.Module):
    """Haar 小波分解模块.

    使用固定 Haar 小波核对输入特征进行小波分解，
    输出低频（近似）和高频（细节）两个分量。

    Args:
        channels (int): 输入通道数
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        weights = torch.ones(4, 1, 2, 2)
        weights[1, 0, 0, 1] = -1
        weights[1, 0, 1, 1] = -1
        weights[2, 0, 1, 0] = -1
        weights[2, 0, 1, 1] = -1
        weights[3, 0, 1, 0] = -1
        weights[3, 0, 0, 1] = -1
        weights = torch.cat([weights] * channels, dim=0)
        self.register_buffer("weights", weights)

    def forward(self, x):
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        out = F.conv2d(x, self.weights, bias=None, stride=2, groups=self.channels) / 4.0
        batch_size, _, height, width = out.shape
        out = out.view(batch_size, self.channels, 4, height, width)
        out = out.transpose(1, 2).reshape(batch_size, self.channels * 4, height, width)
        low, lh, hl, hh = out.chunk(4, dim=1)
        high = lh + hl + hh
        return low, high


class WDAF(nn.Module):
    """小波域自适应融合模块.

    通过空间域门控和 Haar 小波频域引导实现双模态特征融合。

    Args:
        inc (list[int]): 两路输入通道数 [c1, c2]
        ouc (int): 输出通道数

    Inputs:
        x (list[Tensor]): 包含两个特征图的列表或元组，空间尺寸需一致

    Shape:
        - 输入: [(B, c1, H, W), (B, c2, H, W)]
        - 输出: (B, ouc, H, W)
    """

    def __init__(self, inc, ouc):
        super().__init__()
        if len(inc) != 2:
            raise ValueError(f"WDAF expects exactly two input channels, got {len(inc)}")

        self.conv_align1 = Conv(inc[0], ouc, 1)
        self.conv_align2 = Conv(inc[1], ouc, 1)

        self.spatial_gate = Conv(ouc * 2, ouc * 2, 3)
        self.low_gate = Conv(ouc * 2, ouc * 2, 3)
        self.high_gate = Conv(ouc * 2, ouc * 2, 3)
        self.sigmoid = nn.Sigmoid()

        self.wavelet = HaarWaveletDecomposition(ouc)
        self.branch_balance = nn.Parameter(torch.zeros(2))
        self.frequency_balance = nn.Parameter(torch.zeros(2))
        self.wavelet_scale = nn.Parameter(torch.tensor(0.1))

        self.wavelet_proj = Conv(ouc, ouc, 3)
        self.conv_final = Conv(ouc, ouc, 1)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("WDAF expects a list or tuple with two feature maps")

        x1, x2 = x
        if x1.shape[-2:] != x2.shape[-2:]:
            raise ValueError("WDAF expects both inputs to have the same spatial shape")

        height, width = x1.shape[-2:]
        x1 = self.conv_align1(x1)
        x2 = self.conv_align2(x2)

        # 空间域门控融合
        spatial_logits = self.sigmoid(self.spatial_gate(torch.cat([x1, x2], dim=1)))
        x1_weight, x2_weight = torch.chunk(spatial_logits, 2, dim=1)
        branch_weight = torch.softmax(self.branch_balance, dim=0)
        spatial_fused = branch_weight[0] * (x1 * x1_weight) + branch_weight[1] * (x2 * x2_weight)

        # 小波频域引导
        low1, high1 = self.wavelet(x1)
        low2, high2 = self.wavelet(x2)

        low_logits = self.sigmoid(self.low_gate(torch.cat([low1, low2], dim=1)))
        high_logits = self.sigmoid(self.high_gate(torch.cat([high1, high2], dim=1)))
        low1_weight, low2_weight = torch.chunk(low_logits, 2, dim=1)
        high1_weight, high2_weight = torch.chunk(high_logits, 2, dim=1)

        freq_weight = torch.softmax(self.frequency_balance, dim=0)
        low_fused = low1 * low1_weight + low2 * low2_weight
        high_fused = high1 * high1_weight + high2 * high2_weight
        wavelet_guidance = freq_weight[0] * low_fused + freq_weight[1] * high_fused
        wavelet_guidance = self.wavelet_proj(wavelet_guidance)
        wavelet_guidance = F.interpolate(wavelet_guidance, size=(height, width), mode='bilinear', align_corners=False)

        return self.conv_final(spatial_fused + self.wavelet_scale * wavelet_guidance)
