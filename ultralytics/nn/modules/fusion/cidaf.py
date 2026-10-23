"""
Cross-modal Interaction and Dual-stream Alignment Fusion (CIDAF).

来源: 自研模块 (魔傀面具整理)
参考: ultralytics/nn/module_images/自研模块-CIDAF.md

CIDAF 通过交互编码器 + 互补残差实现双模态特征融合：
1. 通道对齐：将两路输入统一到相同通道维度
2. 交互编码：拼接 top/bottom/差异/一致性四路特征，通过门控 softmax 自适应加权
3. 互补残差：对差异性和一致性特征分别提取后相加，以可学习残差比例融合
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ['CIDAF']


class _InteractionEncoder(nn.Module):
    """交互编码器：对拼接后的四路特征进行编码并输出门控 logits.

    Args:
        channels (int): 统一后的通道维度
    """

    def __init__(self, channels):
        super().__init__()
        merged_channels = channels * 4
        self.local = Conv(merged_channels, merged_channels, 3, g=merged_channels)
        self.mix = Conv(merged_channels, channels, 1)
        self.out = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)

    def forward(self, x):
        return self.out(self.mix(self.local(x)))


class _ComplementaryResidual(nn.Module):
    """互补残差模块：融合差异性和一致性信息.

    Args:
        channels (int): 通道维度
    """

    def __init__(self, channels):
        super().__init__()
        self.discrepancy = Conv(channels, channels, 3, g=channels)
        self.consistency = Conv(channels, channels, 1)
        self.project = Conv(channels, channels, 1)

    def forward(self, discrepancy, consistency):
        return self.project(self.discrepancy(discrepancy) + self.consistency(consistency))


class CIDAF(nn.Module):
    """跨模态交互对齐融合模块.

    接收两路空间尺寸一致的特征图，通过交互编码器生成门控权重，
    并以互补残差增强融合结果。

    Args:
        inc (list[int]): 两路输入通道数 [c1, c2]
        ouc (int): 输出通道数

    Inputs:
        x (list[Tensor]): 包含两个特征图的列表或元组，空间尺寸必须一致

    Shape:
        - 输入: [(B, c1, H, W), (B, c2, H, W)]
        - 输出: (B, ouc, H, W)
    """

    def __init__(self, inc, ouc):
        super().__init__()
        if len(inc) != 2:
            raise ValueError(f"CIDAF expects exactly two input channels, got {len(inc)}")

        self.align_top = Conv(inc[0], ouc, 1)
        self.align_bottom = Conv(inc[1], ouc, 1)
        self.interaction = _InteractionEncoder(ouc)
        self.complementary = _ComplementaryResidual(ouc)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.output = Conv(ouc, ouc, 1)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("CIDAF expects a list or tuple with two feature maps")

        x_top, x_bottom = x
        if x_top.shape[-2:] != x_bottom.shape[-2:]:
            raise ValueError("CIDAF expects both inputs to have the same spatial shape")

        top = self.align_top(x_top)
        bottom = self.align_bottom(x_bottom)

        discrepancy = torch.abs(top - bottom)
        consistency = top * bottom
        interaction = torch.cat((top, bottom, discrepancy, consistency), dim=1)

        gate_logits = self.interaction(interaction)
        batch_size, doubled_channels, height, width = gate_logits.shape
        gate_logits = gate_logits.view(batch_size, 2, doubled_channels // 2, height, width)
        gate_weights = torch.softmax(gate_logits, dim=1)
        top_weight, bottom_weight = gate_weights.unbind(dim=1)

        residual = self.complementary(discrepancy, consistency)
        fused = top_weight * top + bottom_weight * bottom + self.residual_scale * residual
        return self.output(fused)
