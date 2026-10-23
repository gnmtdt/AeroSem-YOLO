"""
LSKBlock / LSKBlock_SA - 大选择性核注意力 (Large Selective Kernel Attention)
论文: Large Selective Kernel Network for Remote Sensing Object Detection
来源: ICCV 2023
论文链接: https://arxiv.org/pdf/2403.11735

包含模块:
- LSKBlock_SA: 大选择性核注意力的空间门控单元
- LSKBlock: 大选择性核注意力模块（即插即用）

接口签名:
    LSKBlock_SA(dim)
    - dim: 输入通道数

    LSKBlock(d_model)
    - d_model: 输入通道数
"""

import torch
import torch.nn as nn

__all__ = ["LSKBlock_SA", "LSKBlock"]


class LSKBlock_SA(nn.Module):
    """大选择性核注意力的空间门控单元 (Spatial Gating Unit)。

    【核心功能】
    使用标准卷积和膨胀卷积生成多尺度空间特征，
    通过通道注意力机制自适应选择不同感受野的特征。

    【工作机制】
    1. 5x5 深度卷积捕获局部特征
    2. 7x7 膨胀深度卷积（dilation=3）扩大感受野
    3. 分别通过 1x1 卷积降维
    4. 拼接后通过 avg/max pooling + 7x7 卷积生成选择权重
    5. 加权融合两个分支并通过 1x1 卷积输出注意力

    Args:
        dim (int): 输入通道数
    """

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            注意力加权后的特征图 [B, C, H, W]
        """
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class LSKBlock(nn.Module):
    """大选择性核注意力模块 (Large Selective Kernel Block)。

    【核心功能】
    通过投影层和空间门控单元实现高效的长距离依赖建模。

    【工作机制】
    1. 通过 1x1 卷积投影到特征空间
    2. GELU 激活
    3. 通过 LSKBlock_SA 空间门控单元捕获多尺度特征
    4. 通过 1x1 卷积投影回原始维度
    5. 残差连接

    Args:
        d_model (int): 输入/输出通道数
    """

    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKBlock_SA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            输出特征图 [B, C, H, W]（通道数不变，带残差连接）
        """
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x
