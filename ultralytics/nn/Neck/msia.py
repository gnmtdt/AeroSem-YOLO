"""
MSIA Module - Multiscale Iterative Aggregation
论文: MROD-YOLO: Multimodal Joint Representation for Small Object Detection
      in Remote Sensing Imagery via Multiscale Iterative Aggregation
来源: IEEE TGRS 2025

包含模块:
- MCA (Multiscale Channel Attention): 多尺度通道注意力
- MSIA (Multiscale Iterative Aggregation): 多尺度迭代聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MCA', 'MSIA']


class MCA(nn.Module):
    """Multiscale Channel Attention - 多尺度通道注意力

    【核心功能】
    MCA 同时建模全局通道上下文和局部通道上下文，
    通过两个分支动态调整通道权重，增强特征表达能力。

    【工作机制】
    1. 局部上下文分支：通过逐点卷积捕获局部通道交互
    2. 全局上下文分支：通过全局平均池化 + 逐点卷积捕获全局通道依赖
    3. 两个分支的输出相加后通过 Sigmoid 生成通道注意力权重

    Args:
        in_channels (int): 输入通道数
        reduction (int): 通道压缩比例，默认16
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 8)

        # 局部上下文分支
        self.local_branch = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        # 全局上下文分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            通道注意力权重 [B, C, H, W]（已与输入相乘）
        """
        # 局部上下文
        local_out = self.local_branch(x)  # [B, C, H, W]

        # 全局上下文
        global_out = self.global_branch(x)  # [B, C, 1, 1]

        # 融合并生成注意力权重
        attn = self.sigmoid(local_out + global_out)  # [B, C, H, W]

        # 加权输出
        out = x * attn  # [B, C, H, W]

        return out


class MSIA(nn.Module):
    """Multiscale Iterative Aggregation - 多尺度迭代聚合模块

    【核心功能】
    MSIA 用于连接 backbone 和 neck，通过迭代注意力融合机制
    高效地聚合高层语义特征和低层细节特征，防止小目标特征被稀释。

    【工作机制】
    1. 初始整合：对两个输入特征分别应用 MCA，然后相加得到初始融合特征
    2. 迭代精炼：使用融合后的特征再次通过 MCA 对原始特征进行加权
    3. 最终输出：两个加权特征相加得到最终融合结果

    【输入输出】
    - 输入: (F_X, F_Y)，低维特征图和高维特征图
    - 输出: 聚合后的单一特征图

    Args:
        in_channels (int): 输入通道数（两个输入需相同通道数）
        reduction (int): MCA 的通道压缩比例，默认16
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels

        # 初始整合的 MCA
        self.mca_x = MCA(in_channels, reduction)
        self.mca_y = MCA(in_channels, reduction)

        # 迭代精炼的 MCA
        self.mca_refine_x = MCA(in_channels, reduction)
        self.mca_refine_y = MCA(in_channels, reduction)

    def forward(self, x):
        """
        Args:
            x: 输入特征图列表 [f_x, f_y]，每个形状为 [B, C, H, W]
        Returns:
            聚合后的特征图 [B, C, H, W]
        """
        f_x, f_y = x[0], x[1]

        # 1. 初始整合
        # F'_Z = (F_X * MCA(F_Z)) + (F_Y * MCA(F_Z))
        # 其中 F_Z = F_X + F_Y
        f_z = f_x + f_y  # [B, C, H, W]

        x_attn = self.mca_x(f_z)  # [B, C, H, W]
        y_attn = self.mca_y(f_z)  # [B, C, H, W]

        f_z_prime = (f_x * x_attn) + (f_y * y_attn)  # [B, C, H, W]

        # 2. 迭代精炼
        # F''_Z = (F_X * MCA(F'_Z)) + (F_Y * MCA(F'_Z))
        x_attn_refine = self.mca_refine_x(f_z_prime)  # [B, C, H, W]
        y_attn_refine = self.mca_refine_y(f_z_prime)  # [B, C, H, W]

        f_z_double_prime = (f_x * x_attn_refine) + (f_y * y_attn_refine)  # [B, C, H, W]

        return f_z_double_prime
