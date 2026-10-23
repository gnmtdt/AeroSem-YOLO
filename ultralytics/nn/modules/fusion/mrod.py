"""
MROD-YOLO Fusion Modules
论文: MROD-YOLO: Multimodal Joint Representation for Small Object Detection
      in Remote Sensing Imagery via Multiscale Iterative Aggregation
来源: IEEE TGRS 2025

包含模块:
- GCB (Global Context Block): 全局上下文注意力块
- MJRNet (Multimodal Joint Representation Network): 多模态联合表示网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GCB', 'MJRNet']


class GCB(nn.Module):
    """Global Context Block - 全局上下文注意力块

    【核心功能】
    GCB 通过全局注意力池化和瓶颈变换实现高效的全局上下文建模，
    融合了 NLNet 和 SENet 的优势，同时保持轻量级设计。

    【工作机制】
    1. 全局注意力池化：通过 1x1 卷积和 Softmax 获取注意力权重，进行全局上下文提取
    2. 瓶颈变换：使用两层瓶颈结构（带 LayerNorm）进行通道变换
    3. 广播加法：将全局上下文特征加到每个位置的特征上

    Args:
        in_channels (int): 输入通道数
        reduction (int): 瓶颈层的通道压缩比例，默认16
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        mid_channels = max(in_channels // reduction, 8)

        # 全局注意力池化：1x1 卷积生成注意力权重
        self.conv_attn = nn.Conv2d(in_channels, 1, kernel_size=1)

        # 瓶颈变换模块（带 LayerNorm）
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.LayerNorm([mid_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            增强后的特征图 [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 1. 全局注意力池化
        # 生成注意力权重 [B, 1, H, W] -> [B, 1, H*W]
        attn = self.conv_attn(x).view(B, 1, -1)
        attn = F.softmax(attn, dim=-1)  # [B, 1, H*W]

        # 加权求和获取全局上下文 [B, C, H*W] @ [B, H*W, 1] -> [B, C, 1]
        x_flat = x.view(B, C, -1)  # [B, C, H*W]
        context = torch.bmm(x_flat, attn.transpose(1, 2))  # [B, C, 1]
        context = context.view(B, C, 1, 1)  # [B, C, 1, 1]

        # 2. 瓶颈变换
        context = self.bottleneck(context)  # [B, C, 1, 1]

        # 3. 广播加法
        out = x + context  # [B, C, H, W]

        return out


class MJRNet(nn.Module):
    """Multimodal Joint Representation Network - 多模态联合表示网络

    【核心功能】
    MJRNet 是 MROD-YOLO 的核心融合模块，采用早期融合策略，
    通过 GCB 模块动态加权不同模态的信息，实现高质量的多模态融合。

    【工作机制】
    1. 各模态通过 GCB 捕获长程依赖
    2. 生成掩码提取并增强各模态的显著特征区域
    3. 通过残差连接和卷积处理获取各模态的最终特征
    4. 通道拼接后再次使用 GCB 进行融合，输出单一特征图

    【输入输出】
    - 输入: x = [rgb_feat, ir_feat]，两个模态的特征图列表
    - 输出: 融合后的单一特征图

    Args:
        in_channels (int): 每个模态的输入通道数
        reduction (int): GCB 瓶颈层的压缩比例，默认16
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels

        # 各模态的 GCB 模块
        self.gcb_rgb = GCB(in_channels, reduction)
        self.gcb_ir = GCB(in_channels, reduction)

        # 掩码生成：1x1 卷积
        self.mask_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.mask_ir = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 特征精炼：3x3 卷积
        self.refine_rgb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.refine_ir = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        # 融合后的 GCB
        self.gcb_fusion = GCB(in_channels * 2, reduction)

        # 通道压缩：将 2C 压缩回 C
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图列表 [rgb_feat, ir_feat]，每个形状为 [B, C, H, W]
        Returns:
            融合后的特征图 [B, C, H, W]
        """
        rgb, ir = x[0], x[1]

        # 1. 各模态通过 GCB 捕获全局上下文
        g_rgb = self.gcb_rgb(rgb)  # [B, C, H, W]
        g_ir = self.gcb_ir(ir)     # [B, C, H, W]

        # 2. 生成掩码并加权
        m_rgb = self.mask_rgb(rgb)  # [B, C, H, W]
        m_ir = self.mask_ir(ir)     # [B, C, H, W]

        # 3. 掩码加权
        rgb_weighted = g_rgb * torch.sigmoid(m_rgb)  # [B, C, H, W]
        ir_weighted = g_ir * torch.sigmoid(m_ir)     # [B, C, H, W]

        # 4. 残差连接 + 特征精炼
        rgb_out = self.refine_rgb(rgb + rgb_weighted)  # [B, C, H, W]
        ir_out = self.refine_ir(ir + ir_weighted)      # [B, C, H, W]

        # 5. 通道拼接后融合
        fused = torch.cat([rgb_out, ir_out], dim=1)  # [B, 2C, H, W]
        fused = self.gcb_fusion(fused)               # [B, 2C, H, W]

        # 6. 通道压缩输出
        out = self.compress(fused)  # [B, C, H, W]

        return out
