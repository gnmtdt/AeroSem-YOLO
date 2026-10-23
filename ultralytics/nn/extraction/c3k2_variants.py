"""
C3k2 Extraction - Variant Modules
存放所有C3k2最终封装类

当前包含批次：
- Batch 1 (第一批 - 7个):
  C3k2_Faster, C3k2_PConv, C3k2_ODConv, C3k2_Faster_EMA,
  C3k2_DBB, C3k2_WDBB, C3k2_DeepDBB

- Batch 2 (第二批 - 5个):
  C3k2_CloAtt, C3k2_SCConv, C3k2_ScConv,
  C3k2_EMSC, C3k2_EMSCP

迁移自 ultralytics-yolo11-main/ultralytics/nn/extra_modules/block.py
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k
from ultralytics.nn.public import RelPos2d, RetBlock, HeatBlock, WTConv2d, FMB

# 从c3k2_base导入所有辅助组件
from .c3k2_base import (
    # Batch 1 组件
    C3k_Faster, C3k_PConv, C3k_ODConv, C3k_Faster_EMA,
    C3k_DBB, C3k_WDBB, C3k_DeepDBB,
    Faster_Block, Faster_Block_EMA,
    Bottleneck_PConv, Bottleneck_ODConv,
    Bottleneck_DBB, Bottleneck_WDBB, Bottleneck_DeepDBB,
    # Batch 2 组件
    C3k_CloAtt, C3k_SCConv, C3k_ScConv,
    C3k_EMSC, C3k_EMSCP,
    Bottleneck_CloAtt, Bottleneck_SCConv, Bottleneck_ScConv,
    Bottleneck_EMSC, Bottleneck_EMSCP,
    # Batch 3 组件
    C3k_ContextGuided, ContextGuidedBlock,
    C3k_MSBlock, MSBlock,
    C3k_EMBC, MBConv,
    C3k_EMA, Bottleneck_EMA,
    # Batch 4 组件
    C3k_DLKA, Bottleneck_DLKA,
    C3k_DAttention, Bottleneck_DAttention,
    C3k_Parc, Bottleneck_ParC,
    C3k_DWR, DWR,
    C3k_RFAConv, Bottleneck_RFAConv,
    # Batch 5 组件 (部分使用Batch 4的辅助类)
    C3k_RFCBAMConv, Bottleneck_RFCBAMConv,
    C3k_RFCAConv, Bottleneck_RFCAConv,
    # Batch 5 - AKConv 模块（补充缺失类）
    C3k_AKConv, Bottleneck_AKConv,
    # Retention / Heat / WTConv / FMB
    C3k_RetBlock,
    C3k_Heat,
    C3k_WTConv,
    C3k_FMB,
    C3k_MSMHSA_CGLU,
    C3k_MogaBlock,
    C3k_SHSA,
    C3k_SHSA_CGLU,
    C3k_MutilScaleEdgeInformationEnhance,
    C3k_MutilScaleEdgeInformationSelect,
    C3k_FFCM,
    C3k_SMAFB,
    C3k_SMAFB_CGLU,
    C3k_MSM,
    C3k_HDRAB,
    C3k_RAB,
    C3k_LFE,
    C3k_IDWC,
    C3k_IDWB,
    C3k_CAMixer,
    Bottleneck_IDWC,
    MetaNeXtBlock,
    CAMixer,
    # Batch 9+ 组件补充
    C3k_PKIModule, PKIModule,
    C3k_PPA, PPA,
    C3k_Faster_CGLU, Faster_Block_CGLU,
    C3k_Star, Star_Block,
    C3k_Star_CAA, Star_Block_CAA,
    C3k_EIEM, EIEM,
    C3k_DEConv, Bottleneck_DEConv,
    C3k_MLCA, Bottleneck_MLCA,
    C3k_UniRepLKNetBlock, UniRepLKNetBlock,
    C3k_DRB, Bottleneck_DRB,
    C3k_DWR_DRB, DWR_DRB,
    C3k_SWC, Bottleneck_SWC,
    C3k_FocusedLinearAttention, Bottleneck_FocusedLinearAttention,
    C3k_AggregatedAtt, Bottleneck_AggregatedAttention,
    C3k_gConv, gConvBlock,
    C3k_AdditiveBlock, C3k_AdditiveBlock_CGLU,
)

# 公共基础：仅导入原始基础实现（不导入已组装的变体类）
from .common_base import AdditiveBlock, AdditiveBlock_CGLU

__all__ = [
    # Batch 1 - 第一批(行639-988)
    'C3k2_Faster', 'C3k2_PConv', 'C3k2_ODConv', 'C3k2_Faster_EMA',
    'C3k2_DBB', 'C3k2_WDBB', 'C3k2_DeepDBB',
    # Batch 2 - 第二批(行1111-1407)
    'C3k2_CloAtt', 'C3k2_SCConv', 'C3k2_ScConv',
    'C3k2_EMSC', 'C3k2_EMSCP',
    # Batch 3 - 第三批(行2396-2787)
    'C3k2_ContextGuided', 'C3k2_MSBlock', 'C3k2_EMBC', 'C3k2_EMA',
    # Batch 4 - 第四批(行2480-2920)
    'C3k2_DLKA', 'C3k2_DAttention', 'C3k2_Parc', 'C3k2_DWR', 'C3k2_RFAConv',
    # Batch 5 - 第五批(行2940-2960)
    'C3k2_RFCBAMConv', 'C3k2_RFCAConv',
    # Batch 6+ 扩展
    'C3k2_RetBlock', 'C3k2_Heat', 'C3k2_WTConv', 'C3k2_FMB',
    'C3k2_MSMHSA_CGLU', 'C3k2_MogaBlock', 'C3k2_SHSA', 'C3k2_SHSA_CGLU',
    'C3k2_MutilScaleEdgeInformationEnhance', 'C3k2_MutilScaleEdgeInformationSelect', 'C3k2_FFCM',
    'C3k2_SMAFB', 'C3k2_SMAFB_CGLU',
    'C3k2_MSM', 'C3k2_HDRAB', 'C3k2_RAB', 'C3k2_LFE',
    'C3k2_IDWC', 'C3k2_IDWB', 'C3k2_CAMixer',
    # LEGNet 系列
    'C3k2_LFEM', 'C3k2_LEGM',
]


# ================================ Batch 1: 第一批C3k2变体 ================================
class C3k2_Faster(nn.Module):
    """C3k2 with Faster Block.

    使用FasterNet Block的C3k2模块，通过部分卷积减少计算量。
    源代码位置: extra_modules/block.py:639
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_Faster(self.c, self.c, 2, shortcut, g) if c3k else Faster_Block(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_PConv(nn.Module):
    """C3k2 with Partial Convolution.

    使用部分卷积的C3k2模块。
    源代码位置: extra_modules/block.py:657
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_PConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_PConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_ODConv(nn.Module):
    """C3k2 with Omni-Dimensional Dynamic Convolution.

    使用全维度动态卷积的C3k2模块。
    源代码位置: extra_modules/block.py:855
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_ODConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_ODConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_Faster_EMA(nn.Module):
    """C3k2 with Faster Block and EMA attention.

    使用FasterNet Block + EMA注意力的C3k2模块。
    源代码位置: extra_modules/block.py:926
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_Faster_EMA(self.c, self.c, 2, shortcut, g) if c3k else Faster_Block_EMA(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_DBB(nn.Module):
    """C3k2 with Diverse Branch Block (Reparameterization).

    使用多样化分支块(重参数化)的C3k2模块。
    源代码位置: extra_modules/block.py:948
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_DBB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DBB(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_WDBB(nn.Module):
    """C3k2 with Wide Diverse Branch Block (Reparameterization).

    使用宽度多样化分支块(重参数化，含水平垂直卷积)的C3k2模块。
    源代码位置: extra_modules/block.py:966
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_WDBB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_WDBB(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_DeepDBB(nn.Module):
    """C3k2 with Deep Diverse Branch Block (Reparameterization).

    使用深度多样化分支块(重参数化)的C3k2模块。
    源代码位置: extra_modules/block.py:984
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_DeepDBB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DeepDBB(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ================================ Batch 2: 第二批C3k2变体 ================================
class C3k2_CloAtt(nn.Module):
    """C3k2 with Efficient Attention (CloAtt).

    使用高效注意力机制(EfficientAttention)的C3k2模块。
    源代码位置: extra_modules/block.py:1111
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_CloAtt(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_CloAtt(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_SCConv(nn.Module):
    """C3k2 with Spatial and Channel Convolution (SCConv CVPR2020).

    使用空间通道卷积的C3k2模块。
    论文: http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf
    源代码位置: extra_modules/block.py:1154
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_SCConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_SCConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_ScConv(nn.Module):
    """C3k2 with Spatial and Channel Reconstruction Convolution (ScConv CVPR2023).

    使用空间和通道重构卷积的C3k2模块。
    论文: https://openaccess.thecvf.com/content/CVPR2023/papers/...
    源代码位置: extra_modules/block.py:1291
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_ScConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_ScConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_EMSC(nn.Module):
    """C3k2 with Efficient Multi-Scale Convolution.

    使用高效多尺度卷积的C3k2模块。
    源代码位置: extra_modules/block.py:1386
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_EMSC(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_EMSC(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_EMSCP(nn.Module):
    """C3k2 with Efficient Multi-Scale Convolution Plus.

    使用高效多尺度卷积Plus版本的C3k2模块。
    源代码位置: extra_modules/block.py:1404
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_EMSCP(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_EMSCP(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ================================ Batch 3: 第三批C3k2变体 ================================
class C3k2_ContextGuided(nn.Module):
    """C3k2 with Context Guided Block.

    使用上下文引导块的C3k2模块，用于精炼局部特征和周围上下文。
    源代码位置: extra_modules/block.py:2396
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_ContextGuided(self.c, self.c, 2, shortcut, g) if c3k else ContextGuidedBlock(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_MSBlock(nn.Module):
    """C3k2 with Multi-Scale Block.

    使用多尺度块的C3k2模块，支持多尺度特征提取。
    源代码位置: extra_modules/block.py:2458
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_MSBlock(self.c, self.c, 2, shortcut=shortcut, g=g) if c3k
            else MSBlock(self.c, self.c, kernel_sizes=[1, 3, 3])
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_EMBC(nn.Module):
    """C3k2 with EfficientNet MBConv Block.

    使用EfficientNet风格MBConv块的C3k2模块。
    源代码位置: extra_modules/block.py:2708
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_EMBC(self.c, self.c, 2, shortcut, g) if c3k else MBConv(self.c, self.c, shortcut)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_EMA(nn.Module):
    """C3k2 with Efficient Multi-scale Attention.

    使用高效多尺度注意力的C3k2模块。
    源代码位置: extra_modules/block.py:2787
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_EMA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_EMA(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ================================ Batch 4: 第四批C3k2变体 ================================
class C3k2_DLKA(nn.Module):
    """C3k2 with Deformable Large Kernel Attention.

    使用可变形大核注意力的C3k2模块。
    源代码位置: extra_modules/block.py:2480
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_DLKA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DLKA(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_DAttention(nn.Module):
    """C3k2 with Deformable Attention (CVPR2022).

    使用可变形注意力的C3k2模块。
    源代码位置: extra_modules/block.py:2756
    """
    def __init__(self, c1, c2, n=1, fmapsize=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_DAttention(self.c, self.c, 2, fmapsize, shortcut, g) if c3k
            else Bottleneck_DAttention(self.c, self.c, fmapsize, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_Parc(nn.Module):
    """C3k2 with Parallel Convolution.

    使用并行卷积的C3k2模块。
    源代码位置: extra_modules/block.py:2864
    """
    def __init__(self, c1, c2, n=1, fmapsize=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_Parc(self.c, self.c, 2, fmapsize, shortcut, g) if c3k
            else Bottleneck_ParC(self.c, self.c, fmapsize, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_DWR(nn.Module):
    """C3k2 with Dilation-wise Residual.

    使用扩张残差模块的C3k2。
    源代码位置: extra_modules/block.py:2896
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_DWR(self.c, self.c, 2, shortcut, g) if c3k else DWR(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_RFAConv(nn.Module):
    """C3k2 with Receptive-Field Attention Convolution.

    使用感受野注意力卷积的C3k2模块。
    源代码位置: extra_modules/block.py:2920
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_RFAConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_RFAConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ================================ Batch 5: 第五批C3k2变体 ================================
class C3k2_RFCBAMConv(nn.Module):
    """C3k2 with RFA + CBAM Convolution.

    使用感受野注意力+CBAM卷积的C3k2模块。
    源代码位置: extra_modules/block.py:2940
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_RFCBAMConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_RFCBAMConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_RFCAConv(nn.Module):
    """C3k2 with RFA + Coordinate Attention Convolution.

    使用感受野注意力+坐标注意力卷积的C3k2模块。
    源代码位置: extra_modules/block.py:2960
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_RFCAConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_RFCAConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_FocusedLinearAttention(nn.Module):
    """C3k2 with Focused Linear Attention.

    使用聚焦线性注意力的C3k2模块。
    源代码位置: extra_modules/block.py:3076
    """
    def __init__(self, c1, c2, n=1, fmapsize=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_FocusedLinearAttention(self.c, self.c, 2, fmapsize, shortcut, g) if c3k else Bottleneck_FocusedLinearAttention(self.c, self.c, fmapsize, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_MLCA(nn.Module):
    """C3k2 with Multi-Level Channel Attention.

    使用多级通道注意力的C3k2模块。
    源代码位置: extra_modules/block.py:3101
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_MLCA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_MLCA(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_AKConv(nn.Module):
    """C3k2 with Alterable Kernel Convolution.

    使用可变核卷积的C3k2模块。
    源代码位置: extra_modules/block.py:3260
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_AKConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_AKConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ================================ Batch 6: 第六批C3k2变体 ================================
class C3k2_UniRepLKNetBlock(nn.Module):
    """C3k2 with UniRepLKNet Block.

    使用UniRepLKNet大核卷积块的C3k2模块。
    源代码位置: extra_modules/block.py:3464
    """
    def __init__(self, c1, c2, n=1, k=7, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_UniRepLKNetBlock(self.c, self.c, 2, k, shortcut, g) if c3k else UniRepLKNetBlock(self.c, k)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_DRB(nn.Module):
    """C3k2 with Dilated Reparam Block.

    使用膨胀重参数化块的C3k2模块。
    源代码位置: extra_modules/block.py:3483
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_DRB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DRB(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_DWR_DRB(nn.Module):
    """C3k2 with Dilation-wise Residual DRB.

    使用膨胀残差DRB的C3k2模块。
    源代码位置: extra_modules/block.py:3517
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_DWR_DRB(self.c, self.c, 2, shortcut, g) if c3k else DWR_DRB(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_AggregatedAtt(nn.Module):
    """C3k2 with Aggregated Attention.

    使用聚合注意力的C3k2模块。
    源代码位置: extra_modules/block.py:3763
    """
    def __init__(self, c1, c2, n=1, input_resolution=None, sr_ratio=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_AggregatedAtt(self.c, self.c, 2, input_resolution, sr_ratio, shortcut, g) if c3k
            else Bottleneck_AggregatedAttention(self.c, self.c, input_resolution, sr_ratio, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_SWC(nn.Module):
    """C3k2 with Shift-wise Convolution.

    使用位移卷积的C3k2模块。
    源代码位置: extra_modules/block.py:4227
    """
    def __init__(self, c1, c2, n=1, kernel_size=13, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_SWC(self.c, self.c, 2, kernel_size, shortcut, g) if c3k
            else Bottleneck_SWC(self.c, self.c, kernel_size, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ================================ Batch 7 - iRMB和DynamicConv变体 ================================

class C3k2_iRMB(nn.Module):
    """使用iRMB模块的C3k2变体

    源代码位置: block.py:4548
    依赖: iRMB, C3k_iRMB
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.extraction.c3k2_base import C3k_iRMB, iRMB
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_iRMB(self.c, self.c, 2, shortcut, g) if c3k
            else iRMB(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_iRMB_Cascaded(nn.Module):
    """使用iRMB_Cascaded模块的C3k2变体

    源代码位置: block.py:4559
    依赖: iRMB_Cascaded, C3k_iRMB_Cascaded
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.extraction.c3k2_base import C3k_iRMB_Cascaded, iRMB_Cascaded
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_iRMB_Cascaded(self.c, self.c, 2, shortcut, g) if c3k
            else iRMB_Cascaded(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_iRMB_DRB(nn.Module):
    """使用iRMB_DRB模块的C3k2变体

    源代码位置: block.py:4570
    依赖: iRMB_DRB, C3k_iRMB_DRB
    """
    def __init__(self, c1, c2, n=1, kernel_size=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.extraction.c3k2_base import C3k_iRMB_DRB, iRMB_DRB
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_iRMB_DRB(self.c, self.c, 2, kernel_size, shortcut, g) if c3k
            else iRMB_DRB(self.c, self.c, dw_ks=kernel_size)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_iRMB_SWC(nn.Module):
    """使用iRMB_SWC模块的C3k2变体

    源代码位置: block.py:4581
    依赖: iRMB_SWC, C3k_iRMB_SWC
    """
    def __init__(self, c1, c2, n=1, kernel_size=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.extraction.c3k2_base import C3k_iRMB_SWC, iRMB_SWC
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_iRMB_SWC(self.c, self.c, 2, kernel_size, shortcut, g) if c3k
            else iRMB_SWC(self.c, self.c, dw_ks=kernel_size)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_DynamicConv(nn.Module):
    """使用DynamicConv模块的C3k2变体

    源代码位置: block.py:4948
    依赖: Bottleneck_DynamicConv, C3k_DynamicConv
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.extraction.c3k2_base import C3k_DynamicConv, Bottleneck_DynamicConv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_DynamicConv(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_DynamicConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ================================ Batch 8 - RepViT和GhostDynamicConv变体 ================================

class C3k2_GhostDynamicConv(nn.Module):
    """使用GhostModule的C3k2变体

    源代码位置: block.py:4959
    依赖: GhostModule, C3k_GhostDynamicConv
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.extraction.c3k2_base import C3k_GhostDynamicConv, GhostModule
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_GhostDynamicConv(self.c, self.c, 2, shortcut, g) if c3k
            else GhostModule(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_RVB(nn.Module):
    """使用RepViTBlock的C3k2变体

    源代码位置: block.py:5005
    依赖: RepViTBlock, C3k_RVB
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.extraction.c3k2_base import C3k_RVB, RepViTBlock
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_RVB(self.c, self.c, 2, shortcut, g) if c3k
            else RepViTBlock(self.c, self.c, False)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_RVB_SE(nn.Module):
    """使用RepViTBlock（带SE）的C3k2变体

    源代码位置: block.py:5016
    依赖: RepViTBlock, C3k_RVB_SE
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.extraction.c3k2_base import C3k_RVB_SE, RepViTBlock
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_RVB_SE(self.c, self.c, 2, shortcut, g) if c3k
            else RepViTBlock(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_RVB_EMA(nn.Module):
    """使用RepViTBlock_EMA的C3k2变体

    源代码位置: block.py:5027
    依赖: RepViTBlock_EMA, C3k_RVB_EMA
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.extraction.c3k2_base import C3k_RVB_EMA, RepViTBlock_EMA
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_RVB_EMA(self.c, self.c, 2, shortcut, g) if c3k
            else RepViTBlock_EMA(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ===================== Batch 9: PKIModule, PPA, Faster_CGLU, Star =====================

class C3k2_PKIModule(nn.Module):
    """使用PKIModule块的C3k2模块

    源代码位置: block.py:5187-5190

    Args:
        c1: 输入通道数
        c2: 输出通道数
        n: Bottleneck重复次数，默认为1
        kernel_sizes: 多尺度卷积核大小序列，默认(3, 5, 7, 9, 11)
        expansion: PKIModule内部扩展比例，默认1.0
        with_caa: 是否使用CAA注意力，默认True
        caa_kernel_size: CAA卷积核大小，默认11
        add_identity: 是否添加恒等映射，默认True
        c3k: 是否使用C3k结构，默认False
        e: 通道扩展比例，默认0.5
        g: 分组卷积数，默认1
        shortcut: 是否使用shortcut连接，默认True
    """
    def __init__(self, c1, c2, n=1, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_PKIModule(self.c, self.c, 2, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) if c3k
            else PKIModule(self.c, self.c, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_PPA(nn.Module):
    """使用PPA块的C3k2模块

    源代码位置: block.py:5279-5282

    Args:
        c1: 输入通道数
        c2: 输出通道数
        n: Bottleneck重复次数，默认为1
        c3k: 是否使用C3k结构，默认False
        e: 通道扩展比例，默认0.5
        g: 分组卷积数，默认1
        shortcut: 是否使用shortcut连接，默认True
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_PPA(self.c, self.c, 2, shortcut, g) if c3k
            else PPA(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_Faster_CGLU(nn.Module):
    """使用Faster_Block_CGLU的C3k2模块

    源代码位置: block.py:5864-5867

    Args:
        c1: 输入通道数
        c2: 输出通道数
        n: Bottleneck重复次数，默认为1
        c3k: 是否使用C3k结构，默认False
        e: 通道扩展比例，默认0.5
        g: 分组卷积数，默认1
        shortcut: 是否使用shortcut连接，默认True
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_Faster_CGLU(self.c, self.c, 2, shortcut, g) if c3k
            else Faster_Block_CGLU(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_Star(nn.Module):
    """使用Star_Block的C3k2模块

    源代码位置: block.py:6046-6049

    Args:
        c1: 输入通道数
        c2: 输出通道数
        n: Bottleneck重复次数，默认为1
        c3k: 是否使用C3k结构，默认False
        e: 通道扩展比例，默认0.5
        g: 分组卷积数，默认1
        shortcut: 是否使用shortcut连接，默认True
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_Star(self.c, self.c, 2, shortcut, g) if c3k
            else Star_Block(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ===================== Batch 10: Star_CAA, EIEM, DEConv =====================

class C3k2_Star_CAA(nn.Module):
    """使用Star_Block_CAA的C3k2模块

    源代码位置: block.py:6057-6060

    Args:
        c1: 输入通道数
        c2: 输出通道数
        n: Bottleneck重复次数，默认为1
        c3k: 是否使用C3k结构，默认False
        e: 通道扩展比例，默认0.5
        g: 分组卷积数，默认1
        shortcut: 是否使用shortcut连接，默认True
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_Star_CAA(self.c, self.c, 2, shortcut, g) if c3k
            else Star_Block_CAA(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_EIEM(nn.Module):
    """使用EIEM块的C3k2模块

    源代码位置: block.py:6164-6167

    Args:
        c1: 输入通道数
        c2: 输出通道数
        n: Bottleneck重复次数，默认为1
        c3k: 是否使用C3k结构，默认False
        e: 通道扩展比例，默认0.5
        g: 分组卷积数，默认1
        shortcut: 是否使用shortcut连接，默认True
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_EIEM(self.c, self.c, 2, shortcut, g) if c3k
            else EIEM(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_DEConv(nn.Module):
    """使用DEConv的C3k2模块

    源代码位置: block.py:6214-6217

    Args:
        c1: 输入通道数
        c2: 输出通道数
        n: Bottleneck重复次数，默认为1
        c3k: 是否使用C3k结构，默认False
        e: 通道扩展比例，默认0.5
        g: 分组卷积数，默认1
        shortcut: 是否使用shortcut连接，默认True
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_DEConv(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_DEConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ===================== Batch 11: gConv和Additive注意力变体 =====================

class C3k2_gConv(nn.Module):
    """
    使用gConvBlock的C3k2模块
    门控卷积块适用于需要自适应特征门控的场景

    源代码位置: extra_modules/block.py:7470-7473
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_gConv(self.c, self.c, 2, shortcut, n) if c3k
            else gConvBlock(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_AdditiveBlock(nn.Module):
    """
    使用AdditiveBlock的C3k2模块
    加性注意力机制，通过Q+K的加性融合实现高效注意力

    源代码位置: extra_modules/block.py:7602-7605
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_AdditiveBlock(self.c, self.c, 2, shortcut, g) if c3k
            else AdditiveBlock(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_AdditiveBlock_CGLU(nn.Module):
    """
    使用AdditiveBlock_CGLU的C3k2模块
    结合加性注意力和卷积门控线性单元的高效变体

    源代码位置: extra_modules/block.py:7613-7616
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_AdditiveBlock_CGLU(self.c, self.c, 2, shortcut, g) if c3k
            else AdditiveBlock_CGLU(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_RetBlock(nn.Module):
    """C3k2 with Retention Block (RMT)."""
    def __init__(self, c1, c2, n=1, retention='chunk', num_heads=8, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.retention = retention
        self.relpos = None if c3k else RelPos2d(self.c, num_heads, 2, 4)
        self.m = nn.ModuleList(
            C3k_RetBlock(self.c, self.c, 2, retention, num_heads, shortcut, g) if c3k
            else RetBlock(retention, self.c, num_heads, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        rel_pos = None if self.relpos is None else self.relpos((h, w), chunkwise_recurrent=self.retention == 'chunk')
        y = list(self.cv1(x).chunk(2, 1))
        for layer in self.m:
            if rel_pos is not None:
                y.append(layer(y[-1].permute(0, 2, 3, 1), None, self.retention == 'chunk', rel_pos).permute(0, 3, 1, 2))
            else:
                y.append(layer(y[-1]))
        return self.cv2(torch.cat(y, 1))


class C3k2_Heat(nn.Module):
    """C3k2 with HeatBlock."""
    def __init__(self, c1, c2, n=1, feat_size=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_Heat(self.c, self.c, 2, feat_size, shortcut, g) if c3k
            else HeatBlock(self.c, res=feat_size or 14)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_WTConv(nn.Module):
    """C3k2 with Wavelet Convolution."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_WTConv(self.c, self.c, 2, shortcut, g) if c3k
            else WTConv2d(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_FMB(nn.Module):
    """C3k2 with FMB module."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_FMB(self.c, self.c, 2, shortcut, g) if c3k
            else FMB(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_MSMHSA_CGLU(nn.Module):
    """C3k2 with Multi-Scale MHSA + CGLU."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_MSMHSA_CGLU(self.c, self.c, 2, shortcut, g) if c3k
            else MSMHSA_CGLU(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_MogaBlock(nn.Module):
    """C3k2 with MogaBlock."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_MogaBlock(self.c, self.c, 2, shortcut, g) if c3k
            else MogaBlock(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_SHSA(nn.Module):
    """C3k2 with SHSA (单头自注意)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_SHSA(self.c, self.c, 2, shortcut, g) if c3k
            else SHSABlock(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_SHSA_CGLU(nn.Module):
    """C3k2 with SHSA + CGLU."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_SHSA_CGLU(self.c, self.c, 2, shortcut, g) if c3k
            else SHSABlock_CGLU(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_MutilScaleEdgeInformationEnhance(nn.Module):
    """C3k2 with Multi-Scale Edge Information Enhance."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_MutilScaleEdgeInformationEnhance(self.c, self.c, 2, shortcut, g) if c3k
            else MutilScaleEdgeInformationEnhance(self.c, [3, 6, 9, 12])
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_MutilScaleEdgeInformationSelect(nn.Module):
    """C3k2 with Multi-Scale Edge Information Select (DSM)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_MutilScaleEdgeInformationSelect(self.c, self.c, 2, shortcut, g) if c3k
            else MutilScaleEdgeInformationSelect(self.c, [3, 6, 9, 12])
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_FFCM(nn.Module):
    """C3k2 with Fused Fourier Conv Mixer."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_FFCM(self.c, self.c, 2, shortcut, g) if c3k
            else Fused_Fourier_Conv_Mixer(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_SMAFB(nn.Module):
    """C3k2 with SMAFormer block."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_SMAFB(self.c, self.c, 2, shortcut, g) if c3k
            else SMAFormerBlock(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_SMAFB_CGLU(nn.Module):
    """C3k2 with SMAFormer + CGLU."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_SMAFB_CGLU(self.c, self.c, 2, shortcut, g) if c3k
            else SMAFormerBlock_CGLU(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_MSM(nn.Module):
    """C3k2 with DeepPoolLayer MSM block."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_MSM(self.c, self.c, 2, shortcut, g) if c3k
            else DeepPoolLayer(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_HDRAB(nn.Module):
    """C3k2 with HDRAB block."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_HDRAB(self.c, self.c, 2, shortcut, g) if c3k
            else HDRAB(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_RAB(nn.Module):
    """C3k2 with RAB block."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_RAB(self.c, self.c, 2, shortcut, g) if c3k
            else RAB(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_LFE(nn.Module):
    """C3k2 with LFE (shift conv) block."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_LFE(self.c, self.c, 2, shortcut, g) if c3k
            else LFE(self.c, self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_IDWC(nn.Module):
    """C3k2 with Inception Depthwise Conv."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_IDWC(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck_IDWC(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_IDWB(nn.Module):
    """C3k2 with MetaNeXtBlock backbone."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_IDWB(self.c, self.c, 2, shortcut, g) if c3k
            else MetaNeXtBlock(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_CAMixer(nn.Module):
    """C3k2 with CAMixer token mixer."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_CAMixer(self.c, self.c, 2, shortcut, g) if c3k
            else CAMixer(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ================================ LEGNet 系列变体 ================================
# C3k_LFEM / C3k2_LFEM / C3k2_LEGM
# 依赖: LFE_Module(lfem.py), LEGM(legm.py), C3k(modules.block)


class C3k_LFEM(C3k):
    """C3k with LFE Module - integrates Laplacian of Gaussian edge detection into C3k.

    Uses LFE_Module which combines Gaussian/Scharr edge detection with attention mechanisms.

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of LFE_Module blocks
        shortcut: Whether to use shortcut connections
        g: Groups for convolutions
        e: Expansion ratio
        k: Kernel size
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        from ultralytics.nn.public.lfem import LFE_Module

        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(LFE_Module(c_) for _ in range(n)))


class C3k2_LFEM(nn.Module):
    """C3k2 with LFE Module - LEGNet variant of C3k2.

    Uses LFE_Module (Laplacian of Gaussian edge detection + attention) inside C3k2 structure.

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of bottleneck blocks
        c3k: Whether to use C3k_LFEM blocks (True) or direct LFE_Module (False)
        e: Expansion ratio
        g: Groups for convolutions
        shortcut: Whether to use shortcut connections
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.public.lfem import LFE_Module

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_LFEM(self.c, self.c, n, shortcut, g) if c3k else LFE_Module(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k_LEGM(C3k):
    """C3k with LEGM - integrates LEGM (Laplacian Edge Gaussian Module) into C3k.

    Uses LEGM which combines window attention with Gaussian filtering.

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of LEGM blocks
        shortcut: Whether to use shortcut connections
        g: Groups for convolutions
        e: Expansion ratio
        k: Kernel size
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        from ultralytics.nn.public.legm import LEGM

        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(LEGM(c_) for _ in range(n)))


class C3k2_LEGM(nn.Module):
    """C3k2 with LEGM - LEGNet variant of C3k2.

    Uses LEGM (Laplacian Edge Gaussian Module) inside C3k2 structure.

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of bottleneck blocks
        c3k: Whether to use C3k_LEGM blocks (True) or direct LEGM (False)
        e: Expansion ratio
        g: Groups for convolutions
        shortcut: Whether to use shortcut connections
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.public.legm import LEGM

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k_LEGM(self.c, self.c, n, shortcut, g) if c3k else LEGM(self.c)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ================================ 后续批次将在这里追加 ================================
# Batch 13, Batch 14, ... 后续迁移时继续添加
