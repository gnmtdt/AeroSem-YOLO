"""
Other Base Modules
存放不属于 C3k2/C2f/C2PSA/SPPF 类别的独立模块。

包含模块:
- RFEM (Receptive Field Expansion Mechanism): 感受野扩展机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['RFEM']


class RFEM(nn.Module):
    """Receptive Field Expansion Mechanism - 感受野扩展机制

    【核心功能】
    RFEM 通过多尺度深度可分离卷积动态调整感受野范围，
    增强对不同尺度背景信息的捕获能力，特别适合小目标检测任务。

    【工作机制】
    1. 使用多个不同膨胀率的深度可分离卷积获取多尺度感受野特征
    2. 对每个尺度的特征进行逐点卷积降维
    3. 通过空间注意力机制为每个尺度生成独立的权重
    4. 加权融合多尺度特征，输出增强后的特征图

    Args:
        in_channels (int): 输入通道数
        num_scales (int): 感受野尺度数量，默认3
        kernel_sizes (list): 各尺度的卷积核大小，默认 [3, 5, 7]
        dilations (list): 各尺度的膨胀率，默认 [1, 2, 3]
    """

    def __init__(self, in_channels, num_scales=3, kernel_sizes=None, dilations=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales

        # 默认配置
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        if dilations is None:
            dilations = [1, 2, 3]

        assert len(kernel_sizes) == num_scales, "kernel_sizes 长度必须等于 num_scales"
        assert len(dilations) == num_scales, "dilations 长度必须等于 num_scales"

        # 多尺度深度可分离卷积
        self.dwconvs = nn.ModuleList()
        self.pwconvs = nn.ModuleList()

        for i in range(num_scales):
            k = kernel_sizes[i]
            d = dilations[i]
            p = (k + (k - 1) * (d - 1) - 1) // 2  # 保持空间尺寸不变的 padding

            # 深度可分离卷积
            self.dwconvs.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=p,
                         dilation=d, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
            ))

            # 逐点卷积降维
            self.pwconvs.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
            ))

        # 空间注意力：为每个尺度生成空间权重
        self.spatial_attns = nn.ModuleList()
        for _ in range(num_scales):
            self.spatial_attns.append(nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid(),
            ))

        # 最终融合：1x1 卷积
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * num_scales, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            增强后的特征图 [B, C, H, W]
        """
        multi_scale_feats = []

        # 1. 多尺度特征提取
        feat = x
        for i in range(self.num_scales):
            # 深度可分离卷积
            feat = self.dwconvs[i](feat)
            # 逐点卷积
            feat_pw = self.pwconvs[i](feat)
            multi_scale_feats.append(feat_pw)

        # 2. 空间注意力加权
        weighted_feats = []
        for i in range(self.num_scales):
            attn = self.spatial_attns[i](multi_scale_feats[i])  # [B, 1, H, W]
            weighted = multi_scale_feats[i] * attn  # [B, C, H, W]
            weighted_feats.append(weighted)

        # 3. 多尺度特征融合
        concat_feat = torch.cat(weighted_feats, dim=1)  # [B, C*num_scales, H, W]
        out = self.fuse(concat_feat)  # [B, C, H, W]

        # 4. 残差连接
        out = out * x  # 与输入相乘（论文公式10）

        return out
