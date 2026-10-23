"""
Dynamic Align Fusion (DAF / DynamicAlignFusion).

来源: 自研模块 (魔傀面具整理)
参考: ultralytics/nn/module_images/自研模块-DAF.md

DAF 通过动态对齐融合实现自适应空间对齐的双模态特征融合：
1. 通道对齐：将两路输入通过 1x1 卷积统一到相同通道维度
2. 动态门控：拼接两路特征通过 sigmoid 门控，分别生成各路的权重
3. 可学习参数：两路各自的缩放参数（带 clamp 约束），自适应平衡两路贡献
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ['DAF']


class DynamicAlignFusion(nn.Module):
    """动态对齐融合模块.

    通过动态门控和可学习缩放参数实现双模态特征的自适应融合。

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

        self.conv_align1 = Conv(inc[0], ouc, 1)
        self.conv_align2 = Conv(inc[1], ouc, 1)

        self.conv_concat = Conv(ouc * 2, ouc * 2, 3)
        self.sigmoid = nn.Sigmoid()

        self.x1_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        self.x2_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)

        self.conv_final = Conv(ouc, ouc, 1)

    def forward(self, x):
        self._clamp_abs(self.x1_param.data, 1.0)
        self._clamp_abs(self.x2_param.data, 1.0)

        x1, x2 = x
        x1, x2 = self.conv_align1(x1), self.conv_align2(x2)
        x_concat = self.sigmoid(self.conv_concat(torch.cat([x1, x2], dim=1)))
        x1_weight, x2_weight = torch.chunk(x_concat, 2, dim=1)
        x1, x2 = x1 * x1_weight, x2 * x2_weight

        return self.conv_final(x1 * self.x1_param + x2 * self.x2_param)

    @staticmethod
    def _clamp_abs(data, value):
        """将参数绝对值约束在 [-value, value] 范围内."""
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign


# 别名导出，保持与参考库的兼容性
DAF = DynamicAlignFusion
