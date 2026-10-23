# Ultralytics YOLOMM - Neck Module
# SOEP (Small Object Enhance Pyramid) 小目标增强金字塔模块集

from .auxiliary import GSConvE, MFM, SNI
from .soep import CSPOmniKernel, FGM, OmniKernel, SPDConv
# Neck 变体（AFPN/HSFPN/CFPT/融合等）
from .neck_variants import (
    AFPN_P345,
    AFPN_P345_Custom,
    AFPN_P2345,
    AFPN_P2345_Custom,
    HFP,
    SDP,
    SDP_Improved,
    ChannelAttention_HSFPN,
    ELA_HSFPN,
    CA_HSFPN,
    CAA_HSFPN,
    CrossLayerSpatialAttention,
    CrossLayerChannelAttention,
    FreqFusion,
    LocalSimGuidedSampler,
    Fusion,
    SDI,
    CSPStage,
    BiFusion,
    OREPANCSPELAN4,
    SBA,
    EUCB,
    MSDC,
    MSCB,
    CSP_MSCB,
)
# MSIA 多尺度迭代聚合模块
from .msia import MCA, MSIA
# CTrans 跨尺度通道 Transformer (AAAI 2022)
from .ctrans import ChannelTransformer
# HyperComputeModule 超图计算模块 (TPAMI 2025)
from .hypercompute import HyperComputeModule, HyPConv
# FDPN 频率动态金字塔网络
from .fdpn import FocusFeature, DynamicFrequencyFocusFeature, AlignmentGuidedFocusFeature
# GoldYOLO Transformer 聚合注入融合
from .goldyolo import (
    AdvPoolFusion,
    IFM,
    InjectionMultiSum_Auto_pool,
    PyramidPoolAgg,
    SimFusion_3in,
    SimFusion_4in,
    TopBasicLayer,
)
# SlimNeck 轻量化 Neck
from .slimneck import GSConv, GSBottleneck, GSBottleneckC, VoVGSCSP, HaarWavelet, WFU
# ASF 注意力尺度融合 (arXiv 2312.06458)
from .asf import Add, ScalSeq, Zoom_cat, asf_attention_model

__all__ = [
    # SOEP核心模块
    'SPDConv',
    'FGM',
    'OmniKernel',
    'CSPOmniKernel',
    # SOEP辅助模块
    'SNI',
    'GSConvE',
    'MFM',
    # 颈部变体
    'AFPN_P345','AFPN_P345_Custom','AFPN_P2345','AFPN_P2345_Custom',
    'HFP','SDP','SDP_Improved',
    'ChannelAttention_HSFPN','ELA_HSFPN','CA_HSFPN','CAA_HSFPN',
    'CrossLayerSpatialAttention','CrossLayerChannelAttention',
    'FreqFusion','LocalSimGuidedSampler',
    'Fusion','SDI','CSPStage','BiFusion','OREPANCSPELAN4',
    'SBA','EUCB','MSDC','MSCB','CSP_MSCB',
    # MSIA 多尺度迭代聚合模块
    'MCA', 'MSIA',
    # CTrans 跨尺度通道 Transformer
    'ChannelTransformer',
    # HyperComputeModule 超图计算模块
    'HyperComputeModule', 'HyPConv',
    # FDPN 频率动态金字塔
    'FocusFeature', 'DynamicFrequencyFocusFeature', 'AlignmentGuidedFocusFeature',
    # GoldYOLO 聚合注入融合
    'AdvPoolFusion', 'IFM', 'InjectionMultiSum_Auto_pool',
    'PyramidPoolAgg', 'SimFusion_3in', 'SimFusion_4in', 'TopBasicLayer',
    # SlimNeck 轻量化
    'GSConv', 'GSBottleneck', 'GSBottleneckC', 'VoVGSCSP', 'HaarWavelet', 'WFU',
    # ASF 注意力尺度融合
    'Add', 'ScalSeq', 'Zoom_cat', 'asf_attention_model',
]
