"""
Frequency-Domain Feature Enhancement Fusion (FDFEF) Module.

论文: UMIS-YOLO: Underwater Multimodal Images Instance Segmentation With YOLO
来源: IEEE Transactions on Geoscience and Remote Sensing, Vol.63, 2025

FDFEF 模块通过傅里叶变换在频域内增强和融合多模态特征：
1. 频域特征增强：FFT -> 可学习权重调制 -> IFFT + 残差
2. 幅度谱和相位谱融合：通过可学习权重融合两个模态的频域信息
"""

import torch
import torch.nn as nn
import torch.fft as fft

from .dyt import DyT

__all__ = ['FDFEF']


class FrequencyEnhancement(nn.Module):
    """单模态频域特征增强分支.

    实现公式:
    - F(X) = FFT(X)
    - F_enhanced = F(X) ⊙ W_l (复数 Hadamard 积)
    - X_enhanced = Re{IFFT(F_enhanced)} + X (残差连接)

    Args:
        channels (int): 输入通道数
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # 可学习的复数权重矩阵 (实部和虚部分开存储)
        # 使用 Xavier 初始化实部，虚部初始化为 0
        self.weight_real = nn.Parameter(torch.empty(channels, 1, 1))
        self.weight_imag = nn.Parameter(torch.zeros(channels, 1, 1))
        nn.init.xavier_uniform_(self.weight_real.view(channels, 1))
        # DyT 用于输出稳定
        self.dyt = DyT(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """频域增强前向传播.

        Args:
            x: 输入特征，形状 (B, C, H, W)

        Returns:
            增强后的特征，形状 (B, C, H, W)
        """
        # 2D FFT
        x_fft = fft.rfft2(x, norm='ortho')

        # 构建复数权重并扩展到匹配 rfft2 输出尺寸
        weight_complex = torch.complex(self.weight_real, self.weight_imag)

        # 复数 Hadamard 积 (广播)
        x_fft_enhanced = x_fft * weight_complex

        # IFFT 回到空间域
        x_enhanced = fft.irfft2(x_fft_enhanced, s=x.shape[-2:], norm='ortho')

        # 残差连接 + DyT 稳定
        out = self.dyt(x_enhanced + x)
        return out


class FDFEF(nn.Module):
    """Frequency-Domain Feature Enhancement Fusion Module.

    FDFEF 模块包含两个阶段:
    1. 频域特征增���: 分别增强两个模态的特征
    2. 幅度-相位融合: 通过可学习权重融合两个模态的幅度谱和相位谱

    Args:
        dim (int | None): 输入通道数，None 时延迟构建
    """

    def __init__(self, dim: int | None = None):
        super().__init__()
        self.dim = dim
        self._built = False

        # 延迟初始化的组件
        self.enhance1: FrequencyEnhancement | None = None
        self.enhance2: FrequencyEnhancement | None = None
        self.alpha1: nn.Parameter | None = None
        self.alpha2: nn.Parameter | None = None
        self.beta1: nn.Parameter | None = None
        self.beta2: nn.Parameter | None = None
        self.out_dyt: DyT | None = None

        if dim is not None:
            self._build(dim)

    def _build(self, dim: int):
        """延迟构建模块组件."""
        if self._built and self.dim == dim:
            return
        self.dim = dim

        # 两个模态的频域增强分支
        self.enhance1 = FrequencyEnhancement(dim)
        self.enhance2 = FrequencyEnhancement(dim)

        # 幅度融合权重 (通道级)
        self.alpha1 = nn.Parameter(torch.ones(dim, 1, 1) * 0.5)
        self.alpha2 = nn.Parameter(torch.ones(dim, 1, 1) * 0.5)

        # 相位融合权重 (通道级)
        self.beta1 = nn.Parameter(torch.ones(dim, 1, 1) * 0.5)
        self.beta2 = nn.Parameter(torch.ones(dim, 1, 1) * 0.5)

        # 输出稳定
        self.out_dyt = DyT(dim)

        self._built = True

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> torch.Tensor:
        """FDFEF 前向传播.

        Args:
            x1: RGB 模态特征或 (x1, x2) 元组/列表，形状 (B, C, H, W)
            x2: X 模态特征，形状 (B, C, H, W)，当 x1 是元组时可省略

        Returns:
            融合后的特征，形状 (B, C, H, W)
        """
        # 处理输入格式
        if x2 is None:
            if isinstance(x1, (list, tuple)) and len(x1) == 2:
                x1, x2 = x1
            else:
                raise ValueError("FDFEF 需要两路输入")

        if x1.shape != x2.shape:
            raise ValueError(f"FDFEF 要求两路输入形状一致，got {x1.shape} vs {x2.shape}")

        # 延迟构建
        if not self._built:
            self._build(x1.shape[1])
            # 确保参数在正确的设备上
            device = x1.device
            self.to(device)

        # 阶段1: 频域特征增强
        x1_enhanced = self.enhance1(x1)
        x2_enhanced = self.enhance2(x2)

        # 阶段2: 频域融合
        # 对增强后的特征进行 FFT
        f1 = fft.rfft2(x1_enhanced, norm='ortho')
        f2 = fft.rfft2(x2_enhanced, norm='ortho')

        # 提取幅度和相位
        amp1 = torch.abs(f1)
        amp2 = torch.abs(f2)
        phase1 = torch.angle(f1)
        phase2 = torch.angle(f2)

        # 幅度融合: |F_fused| = α1|F1| + α2|F2|
        amp_fused = self.alpha1 * amp1 + self.alpha2 * amp2

        # 相位融合: ∠F_fused = β1∠F1 + β2∠F2
        phase_fused = self.beta1 * phase1 + self.beta2 * phase2

        # 通过欧拉公式重建复数: F_recon = |F_fused| * e^(j*∠F_fused)
        f_recon = amp_fused * torch.exp(1j * phase_fused)

        # IFFT 回到空间域
        x_fused = fft.irfft2(f_recon, s=x1.shape[-2:], norm='ortho')

        # 输出稳定
        out = self.out_dyt(x_fused)
        return out
