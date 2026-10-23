"""
Residual Feature Fusion (RFF) Module.

论文: UMIS-YOLO: Underwater Multimodal Images Instance Segmentation With YOLO
来源: IEEE Transactions on Geoscience and Remote Sensing, Vol.63, 2025

RFF 模块用于融合低级（P1级）特征和高级特征，以保留像素级信息:
1. 通道对齐：通过 1×1 卷积对齐通道数
2. 空间对齐：通过双线性插值对齐空间尺寸
3. 双分支融合：主分支(Concat+增强) + 辅助分支(Add+Sigmoid门控)
4. 多尺度融合：使用分组卷积融合不同尺度特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dyt import DyT

__all__ = ['RFF']


class FBlock(nn.Module):
    """Feature Block: Conv1x1 + DyT + ReLU.

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.dyt = DyT(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.dyt(self.conv(x)))


class RFF(nn.Module):
    """Residual Feature Fusion Module.

    融合低级特征（如 P1 层）和高级特征（如检测头前的特征），
    通过残差学习保留像素级信息以提升分割精度。

    Args:
        low_channels (int | None): 低级特征通道数 (用于输出通道对齐)
        high_channels (int | None): 高级特征通道数
        groups (int): 分组卷积的组数，默认 4

    Note:
        - 如果 low_channels 和 high_channels 都为 None，则延迟构建
        - 输出通道数等于 low_channels（对齐到低级特征）
    """

    def __init__(
        self,
        low_channels: int | None = None,
        high_channels: int | None = None,
        groups: int = 4,
    ):
        super().__init__()
        self.low_channels = low_channels
        self.high_channels = high_channels
        self.groups = groups
        self._built = False

        # 延迟初始化的组件
        self.low_align: nn.Module | None = None
        self.high_align: nn.Module | None = None
        self.fblock1: FBlock | None = None
        self.fblock2: FBlock | None = None
        self.fblock3: nn.Module | None = None
        self.branch_conv_low: nn.Module | None = None
        self.branch_conv_high: nn.Module | None = None
        self.residual_block: FBlock | None = None
        self.out_dyt: DyT | None = None

        if low_channels is not None and high_channels is not None:
            self._build(low_channels, high_channels)

    def _build(self, low_channels: int, high_channels: int):
        """延迟构建模块组件."""
        if self._built and self.low_channels == low_channels and self.high_channels == high_channels:
            return

        self.low_channels = low_channels
        self.high_channels = high_channels
        dim = low_channels  # 输出对齐到低级特征通道

        # 通道对齐层
        self.low_align = nn.Conv2d(low_channels, dim, 1, bias=False) if low_channels != dim else nn.Identity()
        self.high_align = nn.Conv2d(high_channels, dim, 1, bias=False) if high_channels != dim else nn.Identity()

        # 主分支: Concat + FBlock1 + FBlock2
        self.fblock1 = FBlock(dim * 2, dim)  # Concat 后通道翻倍
        self.fblock2 = FBlock(dim, dim)

        # 辅助分支: Add + Sigmoid 门控
        self.branch_conv_low = nn.Conv2d(dim, dim, 1, bias=False)
        self.branch_conv_high = nn.Conv2d(dim, dim, 1, bias=False)

        # 多尺度融合: 分组卷积
        groups = min(self.groups, dim)  # 确保 groups 不超过通道数
        if dim % groups != 0:
            groups = 1  # 如果不能整除，退化为普通卷积
        self.fblock3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=groups, bias=False),
            DyT(dim),
            nn.ReLU(inplace=True),
        )

        # 残差分支
        self.residual_block = FBlock(dim * 2, dim)

        # 输出 DyT
        self.out_dyt = DyT(dim)

        self._built = True

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor = None) -> torch.Tensor:
        """RFF 前向传播.

        Args:
            x_low: 低级特征（P1层）或 (x_low, x_high) 元组，形状 (B, C_low, H_low, W_low)
            x_high: 高级特征（检测头前），形状 (B, C_high, H_high, W_high)

        Returns:
            融合后的特征，形状 (B, C_low, H_high, W_high)
        """
        # 处理输入格式
        if x_high is None:
            if isinstance(x_low, (list, tuple)) and len(x_low) == 2:
                x_low, x_high = x_low
            else:
                raise ValueError("RFF 需要两路输入 (低级特征, 高级特征)")

        # 延迟构建
        if not self._built:
            self._build(x_low.shape[1], x_high.shape[1])
            self.to(x_low.device)

        # 1. 通道对齐
        x_low_adj = self.low_align(x_low)
        x_high_adj = self.high_align(x_high)

        # 2. 空间对齐 (将低级特征下采样到高级特征尺寸)
        if x_low_adj.shape[-2:] != x_high_adj.shape[-2:]:
            x_low_aligned = F.interpolate(
                x_low_adj,
                size=x_high_adj.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        else:
            x_low_aligned = x_low_adj

        # 3. 主分支: Concat -> FBlock1 -> FBlock2
        concat_feat = torch.cat([x_low_aligned, x_high_adj], dim=1)
        ce = self.fblock1(concat_feat)  # Channel Enhancement
        f_fused = self.fblock2(ce)

        # 4. 辅助分支: Add + Sigmoid 门控
        branch_low = self.branch_conv_low(x_low_aligned)
        branch_high = self.branch_conv_high(x_high_adj)
        a = torch.sigmoid(branch_low + branch_high)  # Attention gate

        # 5. 多尺度融合
        f_multi = self.fblock3(f_fused * a)

        # 6. 残差分支
        r = self.residual_block(concat_feat)

        # 7. 最终融合
        out = self.out_dyt(f_multi + r)
        return out
