"""
CoordAtt - 坐标注意力机制 (Coordinate Attention)
论文: Coordinate Attention for Efficient Mobile Network Design
来源: CVPR 2021
论文链接: https://arxiv.org/pdf/2103.02907

包含模块:
- h_sigmoid: 硬 sigmoid 激活函数（ReLU6 近似）
- h_swish: 硬 Swish 激活函数
- CoordAtt: 坐标注意力模块（即插即用）

接口签名:
    CoordAtt(inp, reduction=32)
    - inp: 输入通道数
    - reduction: 通道压缩比例，默认 32
"""

import torch
import torch.nn as nn

__all__ = ["h_sigmoid", "h_swish", "CoordAtt"]


class h_sigmoid(nn.Module):
    """硬 Sigmoid 激活函数，使用 ReLU6 近似。"""

    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """硬 Swish 激活函数: x * h_sigmoid(x)。"""

    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """坐标注意力模块 (Coordinate Attention)。

    【核心功能】
    通过沿空间方向聚合特征来捕获远程依赖关系，
    同时保留精确的位置信息。

    【工作机制】
    1. 沿 H/W 方向进行 1D 全局池化，获取方向感知特征
    2. 拼接后通过共享 1x1 卷积压缩通道
    3. 沿空间维度分割，分别生成 H/W 方向的注意力权重
    4. 通过 Sigmoid 激活后与原始特征相乘

    Args:
        inp (int): 输入通道数
        reduction (int): 通道压缩比例，默认 32
    """

    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            输出特征图 [B, C, H, W]（通道数不变）
        """
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
