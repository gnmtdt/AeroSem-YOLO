"""
Dynamic Tanh (DyT) Normalization Module.

论文: UMIS-YOLO: Underwater Multimodal Images Instance Segmentation With YOLO
来源: IEEE Transactions on Geoscience and Remote Sensing, Vol.63, 2025
参考: Zhu et al., "Transformers without normalization" (CVPR 2025)

DyT 是一种无需传统归一化层的动态变换模块，通过元素级操作实现与 BatchNorm/LayerNorm 类似的效果。
"""

import torch
import torch.nn as nn

__all__ = ['DyT']


class DyT(nn.Module):
    """Dynamic Tanh (DyT) Normalization.

    实现公式: DyT(X) = tanh(α * X) ⊙ W + β

    其中:
    - α: 可学习标量，控制非线性强度
    - W: 通道级可学习权重向量
    - β: 通道级可学习偏置向量
    - ⊙: 通道逐元素乘法

    Args:
        channels (int): 输入通道数
        alpha_init (float): α 的初始值，默认 0.5
    """

    def __init__(self, channels: int, alpha_init: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: 输入张量，形状 (B, C, H, W) 或 (B, C)

        Returns:
            归一化后的张量，形状与输入相同
        """
        # tanh(α * x)
        out = torch.tanh(self.alpha * x)

        # 通道级乘法和偏置
        if x.dim() == 4:
            # (B, C, H, W) 形状
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        elif x.dim() == 2:
            # (B, C) 形状
            out = out * self.weight.view(1, -1) + self.bias.view(1, -1)
        else:
            # 通用形状处理
            shape = [1] * x.dim()
            shape[1] = -1
            out = out * self.weight.view(*shape) + self.bias.view(*shape)

        return out
