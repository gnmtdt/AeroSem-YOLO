# Ultralytics YOLOMM - SOEP (Small Object Enhance Pyramid) Module
# SOEP: 小目标增强金字塔模块

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv

__all__ = ['SPDConv', 'FGM', 'OmniKernel', 'CSPOmniKernel']


class SPDConv(nn.Module):
    """Space-to-Depth Convolution - 空间到深度卷积.

    将空间维度的信息压缩到通道维度，避免传统下采样导致的信息丢失。
    特别适用于保留小目标的细节信息。

    设计原理:
        将2x2的空间区域重组为4个通道，使得:
        - 空间分辨率降低2倍
        - 通道数增加4倍
        - 信息完全保留，无损失

    Args:
        inc: 输入通道数
        ouc: 输出通道数
        dimension: 维度参数（默认为1）

    Examples:
        >>> spdconv = SPDConv(256, 128)
        >>> x = torch.randn(1, 256, 64, 64)
        >>> out = spdconv(x)  # shape: (1, 128, 32, 32)
    """

    def __init__(self, inc, ouc, dimension=1):
        """Initialize SPDConv with input/output channels."""
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        """Apply space-to-depth transformation followed by convolution."""
        # 将2x2区域的4个位置分别提取并拼接到通道维度
        x = torch.cat([
            x[..., ::2, ::2],   # 左上
            x[..., 1::2, ::2],  # 右上
            x[..., ::2, 1::2],  # 左下
            x[..., 1::2, 1::2]  # 右下
        ], 1)
        x = self.conv(x)
        return x


class FGM(nn.Module):
    """Frequency Gating Module - 频域门控模块.

    结合空间域和频域信息，通过FFT变换和门控机制增强特征表达。

    技术特点:
        - 频域门控: 使用FFT进行频域特征处理
        - 可学习参数: alpha和beta控制频域和空间域的融合权重
        - 残差连接: 保持原始信息的同时增强频域特征

    Args:
        dim: 特征通道数

    Reference:
        AAAI2024 - OmniKernel论文的核心组件
        https://ojs.aaai.org/index.php/AAAI/article/view/27907
    """

    def __init__(self, dim):
        """Initialize FGM with learnable alpha and beta parameters."""
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1, groups=dim)
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        """Apply frequency gating mechanism."""
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        # 频域变换
        x2_fft = torch.fft.fft2(x2, norm='backward')

        # 频域门控
        out = x1 * x2_fft

        # 逆变换回空间域
        out = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')
        out = torch.abs(out)

        # 加权融合
        return out * self.alpha + x * self.beta


class OmniKernel(nn.Module):
    """OmniKernel - 全方位多尺度核模块.

    集成多尺度感受野、频域增强和空间通道注意力的创新模块。

    核心创新:
        1. 多尺度核融合: 1x31, 31x1, 31x31, 1x1 四种不同尺度的卷积核
        2. 频域通道注意力(FCA): FFT变换 + 通道加权
        3. 空间通道注意力(SCA): 自适应池化 + 通道调制
        4. 频域门控模块(FGM): 进一步增强频域特征

    技术特点:
        - 水平核(1x31): 捕获水平方向的特征
        - 垂直核(31x1): 捕获垂直方向的特征
        - 全局核(31x31): 捕获大范围上下文
        - 局部核(1x1): 保留局部细节
        - 深度可分离: 所有大核都使用groups=dim降低计算量

    Args:
        dim: 输入输出通道数

    Reference:
        AAAI2024 - OmniKernel: Building Omni Kernel for Convolutional Neural Networks
        https://ojs.aaai.org/index.php/AAAI/article/view/27907
    """

    def __init__(self, dim):
        """Initialize OmniKernel with multi-scale kernels and attention mechanisms."""
        super().__init__()

        ker = 31
        pad = ker // 2

        # 输入输出卷积
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)

        # 多尺度深度可分离卷积
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), stride=1, groups=dim)  # 水平
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), stride=1, groups=dim)  # 垂直
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)           # 全局
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)               # 局部

        self.act = nn.ReLU()

        # 空间通道注意力 (SCA)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 频域通道注意力 (FCA)
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgm = FGM(dim)

    def forward(self, x):
        """Apply multi-scale kernels with frequency and spatial attention."""
        out = self.in_conv(x)

        # === 频域通道注意力 (FCA) ===
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca)

        # === 空间通道注意力 (SCA) ===
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        x_sca = self.fgm(x_sca)

        # === 多尺度核融合 ===
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)


class CSPOmniKernel(nn.Module):
    """CSP-OmniKernel - 结合CSP思想的OmniKernel模块.

    采用Cross Stage Partial (CSP) 结构，将特征分为两个分支处理：
    - OmniKernel分支 (25%): 经过OmniKernel处理，获得多尺度和频域增强特征
    - Identity分支 (75%): 直接通过，降低计算量

    设计原则:
        - 平衡性能和效率
        - 默认e=0.25，即25%通道经过复杂处理
        - 保持梯度流动性

    Args:
        dim: 输入输出通道数
        e: OmniKernel分支的通道比例（默认0.25）

    Examples:
        >>> csp_ok = CSPOmniKernel(256, e=0.25)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = csp_ok(x)  # shape: (1, 256, 32, 32)
    """

    def __init__(self, dim, e=0.25):
        """Initialize CSP-OmniKernel with channel split ratio."""
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        """Apply CSP structure with OmniKernel."""
        # 分支分割
        ok_branch, identity = torch.split(
            self.cv1(x),
            [int(self.cv1.conv.out_channels * self.e), int(self.cv1.conv.out_channels * (1 - self.e))],
            dim=1
        )
        # 融合输出
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))
