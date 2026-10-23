"""
EMA_Attention - 高效多尺度注意力 (Efficient Multi-Scale Attention)
论文: Efficient Multi-Scale Attention Module with Cross-Spatial Learning
来源: ICASSP 2023
论文链接: https://arxiv.org/pdf/2305.13563v2

包含模块:
- EMA_Attention: 高效多尺度注意力模块（即插即用独立版本）

接口签名:
    EMA_Attention(channels, factor=8)
    - channels: 输入通道数
    - factor: 分组因子，默认 8

注意:
    本模块与 c3k2_base.py 中的 EMA 功能相同但独立导出，
    用于即插即用场景（直接在 YAML 中引用）。
    c3k2_base.py 中的 EMA 专用于 C3k2 变体内部。
"""

import torch
import torch.nn as nn

__all__ = ["EMA_Attention"]


class EMA_Attention(nn.Module):
    """高效多尺度注意力模块 (Efficient Multi-Scale Attention)。

    【核心功能】
    通过分组策略并行处理多尺度特征，利用跨空间学习机制
    建立不同空间位置之间的依赖关系。

    【工作机制】
    1. 将通道维度按 factor 分组
    2. 对每组进行 1D 池化（H/W 方向）获取坐标特征
    3. 通过 1x1 卷积融合方向信息并生成空间注意力
    4. 通过 GroupNorm + 3x3 卷积建立跨尺度特征交互
    5. 通过自适应池化 + 矩阵乘法建立跨空间依赖
    6. 聚合权重并与原始特征相乘

    Args:
        channels (int): 输入通道数
        factor (int): 分组因子，默认 8。channels 必须能被 factor 整除
    """

    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0, f"channels({channels}) 必须能被 factor({factor}) 整除"
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            输出特征图 [B, C, H, W]（通道数不变）
        """
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
