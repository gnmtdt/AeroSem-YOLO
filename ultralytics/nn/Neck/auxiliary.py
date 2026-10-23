# Ultralytics YOLOMM - SOEP Auxiliary Modules
# SOEP辅助增强模块集

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv

__all__ = ['SNI', 'GSConvE', 'MFM']


class SNI(nn.Module):
    """Soft Nearest Neighbor Interpolation - 软最近邻插值上采样.

    相比传统最近邻插值，通过缩放因子alpha实现更平滑的上采样，
    有助于次要特征的对齐，减少上采样伪影。

    设计原理:
        传统最近邻: y = nearest(x)
        软最近邻: y = (1/up_f²) * nearest(x)

    优势:
        - 减少上采样导致的棋盘效应
        - 更好的特征对齐
        - 改善特征金字塔网络的性能

    Args:
        up_f: 上采样倍数（默认为2）

    Reference:
        ECCV2024 - Rethinking Feature Pyramid Network
        https://github.com/AlanLi1997/rethinking-fpn

    Examples:
        >>> sni = SNI(up_f=2)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = sni(x)  # shape: (1, 256, 64, 64)
    """

    def __init__(self, up_f=2):
        """Initialize SNI with upsampling factor."""
        super(SNI, self).__init__()
        self.us = nn.Upsample(None, up_f, 'nearest')
        self.alpha = 1 / (up_f ** 2)

    def forward(self, x):
        """Apply soft nearest neighbor interpolation."""
        return self.alpha * self.us(x)


class GSConvE(nn.Module):
    """GSConv Enhancement - 增强型GSConv模块.

    在单个卷积模块中生成多种感受野和纹理特征，用于表示学习增强。

    技术特点:
        - 双分支结构: 一半通道经过标准卷积，一半经过增强处理
        - 增强路径: 3×3标准卷积 + 3×3深度卷积 + GELU激活
        - Channel Shuffle: 改善通道间的信息交互

    设计思想:
        - 生成多样化的感受野
        - 提取丰富的纹理特征
        - 保持计算效率

    Args:
        c1: 输入通道数
        c2: 输出通道数
        k: 卷积核大小（默认为1）
        s: 步长（默认为1）
        g: 分组数（默认为1）
        d: 膨胀率（默认为1）
        act: 是否使用激活函数（默认为True）

    Reference:
        SlimNeck by GSConv
        https://github.com/AlanLi1997/slim-neck-by-gsconv

    Examples:
        >>> gsconv = GSConvE(256, 128, k=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = gsconv(x)  # shape: (1, 128, 32, 32)
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        """Initialize GSConvE with channel split and enhancement."""
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, bias=False),        # 标准3×3卷积
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),  # 深度卷积
            nn.GELU()
        )

    def forward(self, x):
        """Apply GSConv enhancement with channel shuffle."""
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)

        # Channel shuffle for better information flow
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class MFM(nn.Module):
    """Multi-scale Feature Modulation - 多尺度特征调制模块.

    通过学习自适应权重来融合不同尺度的特征，替代传统的Concat操作。

    核心思想:
        - 全局池化提取特征统计信息
        - MLP生成每个尺度的注意力权重
        - Softmax归一化保证权重和为1
        - 加权融合多尺度特征

    技术优势:
        - 自适应融合: 根据输入内容动态调整权重
        - 参数高效: 使用1×1卷积和全局池化
        - 性能提升: 相比简单Concat更有效

    Args:
        inc: 输入通道数列表（多尺度）
        dim: 统一的通道维度
        reduction: MLP的缩减比例（默认为8）

    Reference:
        CVPR2024 - DCMPNet
        https://github.com/zhoushen1/DCMPNet

    Examples:
        >>> mfm = MFM([256, 512, 1024], dim=256)
        >>> feats = [torch.randn(1, 256, 32, 32),
        ...          torch.randn(1, 512, 32, 32),
        ...          torch.randn(1, 1024, 32, 32)]
        >>> out = mfm(feats)  # shape: (1, 256, 32, 32)
    """

    def __init__(self, inc, dim, reduction=8):
        """Initialize MFM with multi-scale channels and MLP."""
        super(MFM, self).__init__()

        self.height = len(inc)
        d = max(int(dim / reduction), 4)

        # 全局池化和MLP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * self.height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

        # 通道对齐卷积
        self.conv1x1 = nn.ModuleList([])
        for i in inc:
            if i != dim:
                self.conv1x1.append(Conv(i, dim, 1))
            else:
                self.conv1x1.append(nn.Identity())

    def forward(self, in_feats_):
        """Apply multi-scale feature modulation."""
        # 通道对齐
        in_feats = []
        for idx, layer in enumerate(self.conv1x1):
            in_feats.append(layer(in_feats_[idx]))

        B, C, H, W = in_feats[0].shape

        # 拼接所有尺度
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        # 生成注意力权重
        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        # 加权融合
        out = torch.sum(in_feats * attn, dim=1)
        return out
