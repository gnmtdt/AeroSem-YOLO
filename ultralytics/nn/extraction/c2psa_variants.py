"""
C2PSA Extraction - Variant Classes
存放所有C2PSA变体的最终封装类

包含内容：
- Batch 1: C2PSA基础类和第一批变体 (PSABlock, PSA, C2PSA, C2fPSA, C2BRA)

迁移自 ultralytics-yolo11-main/ultralytics/nn/
"""

import torch
import torch.nn as nn

from ..modules.conv import Conv
from ..modules.block import C2f
from .c2psa_base import (
    PSABlock, PSA,
    BiLevelRoutingAttention_nchw,
    LocalWindowAttention,
    DAttention,
    DPB_Attention,
    PolaLinearAttention,
    AttentionTSSA,
    # Batch 3 additions
    DynamicTanh,
    AdaptiveSparseSA,
    MSLA,
    # Batch 4 additions - FFN Enhancement modules
    FMFFN,
    ConvolutionalGLU,
    SEFN,
    SpectralEnhancedFFN,
    EDFFN,
    # Batch 5 additions - Mona modular attention normalization
    Mona,
)

__all__ = [
    # Batch 1 - 基础类和第一批变体
    'C2PSA',
    'C2fPSA',
    'C2BRA',
    'BRABlock',
    # Batch 2 - 注意力机制变体
    'C2CGA',
    'CGABlock',
    'C2DA',
    'DABlock',
    'C2DPB',
    'DPBlock',
    'C2Pola',
    'Polalock',
    'C2TSSA',
    'TSSAlock',
    # Batch 3 - 注意力机制变体 + 归一化增强变体
    'C2ASSA',
    'ASSAlock',
    'C2MSLA',
    'MSLAlock',
    'C2PSA_DYT',
    'PSABlock_DYT',
    'C2TSSA_DYT',
    'TSSAlock_DYT',
    'C2Pola_DYT',
    'Polalock_DYT',
    # Batch 4 - FFN增强变体
    'C2PSA_FMFFN',
    'PSABlock_FMFFN',
    'C2PSA_CGLU',
    'PSABlock_CGLU',
    'C2PSA_SEFN',
    'PSABlock_SEFN',
    'C2PSA_SEFFN',
    'PSABlock_SEFFN',
    'C2PSA_EDFFN',
    'PSABlock_EDFFN',
    # Batch 5 - Mona模块化注意力归一化 + 复合增强变体
    'C2PSA_Mona',
    'PSABlock_Mona',
    'C2TSSA_DYT_Mona',
    'TSSAlock_DYT_Mona',
    'C2TSSA_DYT_Mona_SEFN',
    'TSSAlock_DYT_Mona_SEFN',
    'C2TSSA_DYT_Mona_SEFFN',
    'TSSAlock_DYT_Mona_SEFFN',
    'C2TSSA_DYT_Mona_EDFFN',
    'TSSAlock_DYT_Mona_EDFFN',
]


# ================================ Batch 1: 基础类和第一批变体 ================================

class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5, shortcut=False):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, shortcut=shortcut, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


# ================================ Batch 1: 注意力机制变体 ================================

class BRABlock(PSABlock):
    """
    BiLevel Routing Attention Block
    替换PSABlock中的attention为BiLevelRoutingAttention
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.attn = BiLevelRoutingAttention_nchw(dim=c)


class C2BRA(C2PSA):
    """
    C2PSA with BiLevel Routing Attention
    使用双层路由注意力的C2PSA变体，优化长距离依赖建模

    Examples:
        >>> c2bra = C2BRA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2bra(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(BRABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


# ================================ Batch 2: 注意力机制变体 ================================

class CGABlock(PSABlock):
    """
    Local Window Attention Block (CascadedGroupAttention)
    局部窗口注意力块，减少计算复杂度
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.attn = LocalWindowAttention(dim=c)


class C2CGA(C2PSA):
    """
    C2PSA with Local Window Attention
    使用局部窗口注意力的C2PSA变体，适合小目标检测

    Examples:
        >>> c2cga = C2CGA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2cga(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(CGABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


class DABlock(PSABlock):
    """
    Deformable Attention Block
    可变形注意力块，适应不规则目标形状
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.attn = DAttention(c, q_size=[20, 20])


class C2DA(C2PSA):
    """
    C2PSA with Deformable Attention (CVPR2022)
    使用可变形注意力的C2PSA变体，适应不规则目标

    Examples:
        >>> c2da = C2DA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2da(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(DABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


class DPBlock(PSABlock):
    """
    Dynamic Position Bias Block (CrossFormer ICLR2022)
    动态位置偏置注意力块，通过MLP预测位置偏置
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.attn = DPB_Attention(c, group_size=[20, 20], num_heads=num_heads)

    def forward(self, x):
        """Executes a forward pass through DPBlock, applying DPB attention and feed-forward layers."""
        BS, C, H, W = x.size()
        x = x + self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2DPB(C2PSA):
    """
    C2PSA with Dynamic Position Bias (CrossFormer ICLR2022)
    使用动态位置偏置的C2PSA变体，增强位置编码

    Examples:
        >>> c2dpb = C2DPB(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2dpb(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(DPBlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of DPB blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class Polalock(PSABlock):
    """
    Polarized Linear Attention Block (PolaFormer ICLR2025)
    极化线性注意力块，复杂度为O(n)
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.attn = PolaLinearAttention(c, hw=[20, 20], num_heads=num_heads)

    def forward(self, x):
        """Executes a forward pass through Polalock, applying polarized linear attention."""
        BS, C, H, W = x.size()
        x = x + self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2Pola(C2PSA):
    """
    C2PSA with Polarized Linear Attention (PolaFormer ICLR2025)
    使用极化线性注意力的C2PSA变体，O(n)复杂度，适合实时推理

    Examples:
        >>> c2pola = C2Pola(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2pola(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(Polalock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of Pola blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class TSSAlock(PSABlock):
    """
    Token Statistics Self-Attention Block (ToST ICLR2025)
    Token统计自注意力块
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.attn = AttentionTSSA(c, num_heads=num_heads)

    def forward(self, x):
        """Executes a forward pass through TSSAlock, applying token statistics attention."""
        BS, C, H, W = x.size()
        x = x + self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2TSSA(C2PSA):
    """
    C2PSA with Token Statistics Self-Attention (ToST ICLR2025)
    使用Token统计自注意力的C2PSA变体

    Examples:
        >>> c2tssa = C2TSSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2tssa(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(TSSAlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of TSSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        BS, C, H, W = b.size()
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


# ================================ Batch 3: 注意力机制变体（ASSA、MSLA）================================

class ASSAlock(PSABlock):
    """
    PSABlock with Adaptive Sparse Self-Attention (CVPR2024)
    使用自适应稀疏自注意力的PSA块

    来源: CVPR2024 Adaptive Sparse Transformer
    特点: 自适应稀疏注意力机制，适用于图像恢复任务
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = AdaptiveSparseSA(c, num_heads=num_heads, sparseAtt=True)

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying ASSA and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(x).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2ASSA(C2PSA):
    """
    C2PSA with Adaptive Sparse Self-Attention (CVPR2024)
    使用自适应稀疏自注意力的C2PSA变体

    来源: CVPR2024 - Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration
    特点: 自适应稀疏注意力，结合窗口注意力和稀疏机制
    适用场景: 图像恢复任务、稀疏特征学习

    Examples:
        >>> c2assa = C2ASSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2assa(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(ASSAlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of ASSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class MSLAlock(PSABlock):
    """
    PSABlock with Multi-Scale Linear Attention
    使用多尺度线性注意力的PSA块

    特点: 结合3x3, 5x5, 7x7, 9x9四种尺度的深度卷积
    复杂度: O(n) 线性复杂度
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = MSLA(c, num_heads=num_heads)

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying MSLA and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        z = x.flatten(2).permute(0, 2, 1)
        z = self.attn(z, hw=(H, W))
        z = z.permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + z if self.add else z
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2MSLA(C2PSA):
    """
    C2PSA with Multi-Scale Linear Attention
    使用多尺度线性注意力的C2PSA变体

    特点:
    - 多尺度特征提取（3x3, 5x5, 7x7, 9x9卷积）
    - 线性复杂度 O(n)
    - 适用于多尺度目标检测

    Examples:
        >>> c2msla = C2MSLA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2msla(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(MSLAlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of MSLA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


# ================================ Batch 3: 归一化增强变体（DynamicTanh）================================

class PSABlock_DYT(PSABlock):
    """
    PSABlock with Dynamic Tanh normalization (CVPR2025)
    使用动态Tanh归一化的PSA块

    来源: CVPR2025 DynamicTanh
    特点: 通过可学习的alpha参数实现自适应激活
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)

    def forward(self, x):
        x = x + self.attn(self.dyt1(x)) if self.add else self.attn(self.dyt1(x))
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x


class C2PSA_DYT(C2PSA):
    """
    C2PSA with Dynamic Tanh normalization (CVPR2025)
    使用动态Tanh归一化的C2PSA变体

    来源: CVPR2025
    特点: 动态Tanh激活归一化，增强模型表达能力
    适用场景: 需要自适应归一化的检测任务

    Examples:
        >>> c2psa_dyt = C2PSA_DYT(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa_dyt(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(PSABlock_DYT(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


class TSSAlock_DYT(PSABlock):
    """
    PSABlock with Token Statistics Self-Attention and Dynamic Tanh
    结合TSSA和DynamicTanh的PSA块

    特点:
    - Token统计自注意力 (ICLR2025 ToST)
    - 动态Tanh归一化 (CVPR2025)
    - 双重增强机制
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.attn = AttentionTSSA(c, num_heads=num_heads)

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying TSSA with DYT normalization."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x


class C2TSSA_DYT(C2PSA):
    """
    C2PSA with Token Statistics Self-Attention and Dynamic Tanh
    结合TSSA和DynamicTanh的C2PSA变体

    来源:
    - ICLR2025 ToST (Token Statistics Transformer)
    - CVPR2025 DynamicTanh

    特点:
    - Token统计自注意力，O(n)复杂度
    - 动态Tanh归一化
    - 双重增强，提升性能

    Examples:
        >>> c2tssa_dyt = C2TSSA_DYT(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2tssa_dyt(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(TSSAlock_DYT(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


class Polalock_DYT(PSABlock):
    """
    PSABlock with Polarized Linear Attention and Dynamic Tanh
    结合Polarized Linear Attention和DynamicTanh的PSA块

    特点:
    - 极化线性注意力 (ICLR2025 PolaFormer)
    - 动态Tanh归一化 (CVPR2025)
    - O(n)复杂度 + 自适应归一化
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.attn = PolaLinearAttention(c, hw=[20, 20], num_heads=num_heads)

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying Polarized Linear Attention with DYT normalization."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x


class C2Pola_DYT(C2PSA):
    """
    C2PSA with Polarized Linear Attention and Dynamic Tanh
    结合Polarized Linear Attention和DynamicTanh的C2PSA变体

    来源:
    - ICLR2025 PolaFormer (Polarized Linear Attention)
    - CVPR2025 DynamicTanh

    特点:
    - 极化线性注意力，O(n)复杂度
    - 动态Tanh归一化
    - 速度快 + 性能优秀
    - 适合实时推理场景

    Examples:
        >>> c2pola_dyt = C2Pola_DYT(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2pola_dyt(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(Polalock_DYT(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


# ================================ Batch 4: FFN增强变体 ================================

class PSABlock_FMFFN(PSABlock):
    """
    PSABlock with Frequency-Modulated FFN (ICLR2024)
    使用频率调制FFN的PSA块

    来源: ICLR2024 Frequency-Modulated FFN
    特点: FFT-based频率调制，增强频域特征
    适用: 需要频域信息的检测任务
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)
        # 替换FFN为FMFFN
        self.ffn = FMFFN(in_features=c, hidden_features=int(c * 2.66), window_size=4)

    def forward(self, x):
        """Executes a forward pass with PSA attention and Frequency-Modulated FFN."""
        BS, C, H, W = x.size()
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_FMFFN(C2PSA):
    """
    C2PSA with Frequency-Modulated FFN (ICLR2024)
    使用频率调制FFN的C2PSA变体

    来源: ICLR2024
    特点:
    - FFT-based频率调制
    - 增强频域特征提取
    - 适合需要频域信息的检测任务

    Examples:
        >>> c2psa_fmffn = C2PSA_FMFFN(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa_fmffn(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(PSABlock_FMFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


class PSABlock_CGLU(PSABlock):
    """
    PSABlock with Convolutional GLU (CVPR2024)
    使用卷积门控线性单元的PSA块

    来源: CVPR2024 Convolutional GLU
    特点: 卷积 + 门控机制，增强局部特征
    适用: 需要局部特征增强的检测任务
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)
        # 替换FFN为ConvolutionalGLU
        self.ffn = ConvolutionalGLU(in_features=c, out_features=c)

    def forward(self, x):
        """Executes a forward pass with PSA attention and Convolutional GLU."""
        BS, C, H, W = x.size()
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_CGLU(C2PSA):
    """
    C2PSA with Convolutional GLU (CVPR2024)
    使用卷积门控线性单元的C2PSA变体

    来源: CVPR2024
    特点:
    - 卷积 + 门控机制
    - 增强局部特征提取
    - 适合需要局部特征的检测任务

    Examples:
        >>> c2psa_cglu = C2PSA_CGLU(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa_cglu(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(PSABlock_CGLU(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


class PSABlock_SEFN(PSABlock):
    """
    PSABlock with Spatial-Enhanced FFN (WACV2025)
    使用空间增强FFN的PSA块

    来源: WACV2025 Spatial-Enhanced FFN
    特点: 空间分支 + FFN，双重特征增强
    适用: 需要空间特征增强的检测任务
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)
        # 替换FFN为SEFN
        self.ffn = SEFN(dim=c, ffn_expansion_factor=2.66, bias=False)

    def forward(self, x):
        """Executes a forward pass with PSA attention and Spatial-Enhanced FFN."""
        BS, C, H, W = x.size()
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_SEFN(C2PSA):
    """
    C2PSA with Spatial-Enhanced FFN (WACV2025)
    使用空间增强FFN的C2PSA变体

    来源: WACV2025
    特点:
    - 空间分支 + FFN双重增强
    - 增强空间特征提取
    - 适合需要空间特征的检测任务

    Examples:
        >>> c2psa_sefn = C2PSA_SEFN(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa_sefn(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(PSABlock_SEFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


class PSABlock_SEFFN(PSABlock):
    """
    PSABlock with Spectral-Enhanced FFN (TransMamba)
    使用频谱增强FFN的PSA块

    来源: TransMamba Spectral-Enhanced FFN
    特点: FFT频谱增强 + FFN，提取频域特征
    适用: 需要频域特征的检测任务
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)
        # 替换FFN为SpectralEnhancedFFN
        self.ffn = SpectralEnhancedFFN(dim=c, expansion_factor=2.66, drop=0.0)

    def forward(self, x):
        """Executes a forward pass with PSA attention and Spectral-Enhanced FFN."""
        BS, C, H, W = x.size()
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_SEFFN(C2PSA):
    """
    C2PSA with Spectral-Enhanced FFN (TransMamba)
    使用频谱增强FFN的C2PSA变体

    来源: TransMamba
    特点:
    - FFT频谱增强
    - 提取频域特征
    - 适合需要频域信息的检测任务

    Examples:
        >>> c2psa_seffn = C2PSA_SEFFN(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa_seffn(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(PSABlock_SEFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


class PSABlock_EDFFN(PSABlock):
    """
    PSABlock with Enhanced Dynamic FFN (CVPR2025)
    使用增强动态FFN的PSA块

    来源: CVPR2025 Enhanced Dynamic FFN
    特点: patch-based FFT + 动态权重，全局建模能力
    适用: 需要全局建模的检测任务
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)
        # 替换FFN为EDFFN（采用频域扩展因子2.66，无bias）
        self.ffn = EDFFN(dim=c, ffn_expansion_factor=2.66, bias=False)

    def forward(self, x):
        """Executes a forward pass with PSA attention and Enhanced Dynamic FFN."""
        BS, C, H, W = x.size()
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_EDFFN(C2PSA):
    """
    C2PSA with Enhanced Dynamic FFN (CVPR2025)
    使用增强动态FFN的C2PSA变体

    来源: CVPR2025
    特点:
    - patch-based FFT处理
    - 动态权重调整
    - 全局建模能力强
    - 适合需要全局特征的检测任务

    Examples:
        >>> c2psa_edffn = C2PSA_EDFFN(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa_edffn(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(PSABlock_EDFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


# ================================ Batch 5: Mona模块化注意力归一化 + 复合增强变体 ================================

class PSABlock_Mona(PSABlock):
    """
    PSABlock with Mona (CVPR2025)
    使用Mona模块化注意力归一化的PSA块

    来源: CVPR2025 Mona
    特点: 模块化注意力归一化，增强归一化能力
    适用: 提升模型性能的通用增强模块
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)

        # 添加Mona模块
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)

    def forward(self, x):
        """Executes a forward pass with PSA attention and Mona normalization."""
        BS, C, H, W = x.size()
        # 注意力 + Mona归一化
        x = x + self.attn(x) if self.add else self.attn(x)
        x = self.mona1(x)
        # FFN + Mona归一化
        x = x + self.ffn(x) if self.add else self.ffn(x)
        x = self.mona2(x)
        return x


class C2PSA_Mona(C2PSA):
    """
    C2PSA with Mona (CVPR2025)
    使用Mona模块化注意力归一化的C2PSA变体

    来源: CVPR2025
    特点:
    - 模块化注意力归一化
    - 增强归一化能力
    - 提升模型整体性能

    Examples:
        >>> c2psa_mona = C2PSA_Mona(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa_mona(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(PSABlock_Mona(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


class TSSAlock_DYT_Mona(PSABlock):
    """
    PSABlock with TSSA + DynamicTanh + Mona（三重增强）
    结合Token统计注意力、动态Tanh归一化和Mona的PSA块

    特点:
    - Token统计自注意力 (ICLR2025 ToST)
    - 动态Tanh归一化 (CVPR2025)
    - Mona模块化注意力归一化 (CVPR2025)
    - 三重增强机制
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)

        # 动态Tanh归一化
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)

        # Mona模块
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)

        # Token统计自注意力
        self.attn = AttentionTSSA(c, num_heads=num_heads)

    def forward(self, x):
        """Executes a forward pass with TSSA, DynamicTanh and Mona."""
        BS, C, H, W = x.size()
        # TSSA注意力 + DynamicTanh归一化
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        # Mona归一化
        x = self.mona1(x)
        # FFN + DynamicTanh归一化
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        # Mona归一化
        x = self.mona2(x)
        return x


class C2TSSA_DYT_Mona(C2PSA):
    """
    C2PSA with TSSA + DynamicTanh + Mona（三重增强）
    结合Token统计注意力、动态Tanh归一化和Mona的C2PSA变体

    来源:
    - ICLR2025 ToST (Token Statistics Transformer)
    - CVPR2025 DynamicTanh
    - CVPR2025 Mona

    特点:
    - Token统计自注意力，O(n)复杂度
    - 动态Tanh归一化
    - Mona模块化注意力归一化
    - 三重增强，高性能

    Examples:
        >>> c2tssa_dyt_mona = C2TSSA_DYT_Mona(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2tssa_dyt_mona(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(TSSAlock_DYT_Mona(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


class TSSAlock_DYT_Mona_SEFN(PSABlock):
    """
    PSABlock with TSSA + DYT + Mona + SEFN（四重增强）
    结合Token统计注意力、动态Tanh归一化、Mona和空间增强FFN的PSA块

    特点:
    - Token统计自注意力 (ICLR2025 ToST)
    - 动态Tanh归一化 (CVPR2025)
    - Mona模块化注意力归一化 (CVPR2025)
    - 空间增强FFN (WACV2025)
    - 四重增强机制
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)

        # 空间增强FFN
        self.ffn = SEFN(dim=c, ffn_expansion_factor=2.66, bias=False)

        # 动态Tanh归一化
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)

        # Mona模块
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)

        # Token统计自注意力
        self.attn = AttentionTSSA(c, num_heads=num_heads)

    def forward(self, x):
        """Executes a forward pass with TSSA, DYT, Mona and SEFN."""
        x_spatial = x  # 保存用于SEFN的空间特征
        BS, C, H, W = x.size()
        # TSSA注意力 + DynamicTanh归一化
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        # Mona归一化
        x = self.mona1(x)
        # SEFN + DynamicTanh归一化
        x = x + self.ffn(self.dyt2(x), x_spatial) if self.add else self.ffn(self.dyt2(x), x_spatial)
        # Mona归一化
        x = self.mona2(x)
        return x


class C2TSSA_DYT_Mona_SEFN(C2PSA):
    """
    C2PSA with TSSA + DYT + Mona + SEFN（四重增强）
    结合Token统计注意力、动态Tanh归一化、Mona和空间增强FFN的C2PSA变体

    来源:
    - ICLR2025 ToST (Token Statistics Transformer)
    - CVPR2025 DynamicTanh
    - CVPR2025 Mona
    - WACV2025 SEMNet

    特点:
    - Token统计自注意力，O(n)复杂度
    - 动态Tanh归一化
    - Mona模块化注意力归一化
    - 空间增强FFN
    - 四重增强，极高性能

    Examples:
        >>> c2tssa_dyt_mona_sefn = C2TSSA_DYT_Mona_SEFN(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2tssa_dyt_mona_sefn(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(TSSAlock_DYT_Mona_SEFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


class TSSAlock_DYT_Mona_SEFFN(PSABlock):
    """
    PSABlock with TSSA + DYT + Mona + SEFFN（四重增强）
    结合Token统计注意力、动态Tanh归一化、Mona和频谱增强FFN的PSA块

    特点:
    - Token统计自注意力 (ICLR2025 ToST)
    - 动态Tanh归一化 (CVPR2025)
    - Mona模块化注意力归一化 (CVPR2025)
    - 频谱增强FFN (TransMamba)
    - 四重增强机制
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)

        # 频谱增强FFN
        self.ffn = SpectralEnhancedFFN(dim=c, expansion_factor=2.66, drop=0.0)

        # 动态Tanh归一化
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)

        # Mona模块
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)

        # Token统计自注意力
        self.attn = AttentionTSSA(c, num_heads=num_heads)

    def forward(self, x):
        """Executes a forward pass with TSSA, DYT, Mona and SEFFN."""
        BS, C, H, W = x.size()
        # TSSA注意力 + DynamicTanh归一化
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        # Mona归一化
        x = self.mona1(x)
        # SEFFN + DynamicTanh归一化
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        # Mona归一化
        x = self.mona2(x)
        return x


class C2TSSA_DYT_Mona_SEFFN(C2PSA):
    """
    C2PSA with TSSA + DYT + Mona + SEFFN（四重增强）
    结合Token统计注意力、动态Tanh归一化、Mona和频谱增强FFN的C2PSA变体

    来源:
    - ICLR2025 ToST (Token Statistics Transformer)
    - CVPR2025 DynamicTanh
    - CVPR2025 Mona
    - TransMamba

    特点:
    - Token统计自注意力，O(n)复杂度
    - 动态Tanh归一化
    - Mona模块化注意力归一化
    - 频谱增强FFN
    - 四重增强，极高性能

    Examples:
        >>> c2tssa_dyt_mona_seffn = C2TSSA_DYT_Mona_SEFFN(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2tssa_dyt_mona_seffn(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(TSSAlock_DYT_Mona_SEFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))


class TSSAlock_DYT_Mona_EDFFN(PSABlock):
    """
    PSABlock with TSSA + DYT + Mona + EDFFN（四重增强，最强配置）
    结合Token统计注意力、动态Tanh归一化、Mona和增强动态FFN的PSA块

    特点:
    - Token统计自注意力 (ICLR2025 ToST)
    - 动态Tanh归一化 (CVPR2025)
    - Mona模块化注意力归一化 (CVPR2025)
    - 增强动态FFN (CVPR2025 EVSSM)
    - 四重增强机制，最强配置
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)

        # 增强动态FFN
        self.ffn = EDFFN(dim=c, ffn_expansion_factor=2.66, bias=False)

        # 动态Tanh归一化
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)

        # Mona模块
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)

        # Token统计自注意力
        self.attn = AttentionTSSA(c, num_heads=num_heads)

    def forward(self, x):
        """Executes a forward pass with TSSA, DYT, Mona and EDFFN."""
        BS, C, H, W = x.size()
        # TSSA注意力 + DynamicTanh归一化
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        # Mona归一化
        x = self.mona1(x)
        # EDFFN + DynamicTanh归一化
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        # Mona归一化
        x = self.mona2(x)
        return x


class C2TSSA_DYT_Mona_EDFFN(C2PSA):
    """
    C2PSA with TSSA + DYT + Mona + EDFFN（四重增强，最强配置）
    结合Token统计注意力、动态Tanh归一化、Mona和增强动态FFN的C2PSA变体

    来源:
    - ICLR2025 ToST (Token Statistics Transformer)
    - CVPR2025 DynamicTanh
    - CVPR2025 Mona
    - CVPR2025 EVSSM

    特点:
    - Token统计自注意力，O(n)复杂度
    - 动态Tanh归一化
    - Mona模块化注意力归一化
    - 增强动态FFN（patch-based FFT）
    - 四重增强，最强配置
    - 适合追求极致性能的场景

    Examples:
        >>> c2tssa_dyt_mona_edffn = C2TSSA_DYT_Mona_EDFFN(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2tssa_dyt_mona_edffn(input_tensor)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(TSSAlock_DYT_Mona_EDFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
