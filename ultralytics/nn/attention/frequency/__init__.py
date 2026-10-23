"""
frequency - 频域注意力模块集合

包含7种频域注意力模块:
- ContrastDrivenFeatureAggregation (CDFA): 对比驱动特征聚合
- CTA: 通道转置注意力
- SFA: 空间频率注意力
- FSSA: 频域空域双分支自注意力
- CPIA_SA: 复数相位注意力
- KSFA: 大核频率选择注意力
- FSA: 频率空间注意力
"""

from .cdfa import ContrastDrivenFeatureAggregation, HaarWaveletConv
from .cta import CTA, ChannelProjection, SpatialProjection
from .sfa import SFA, FrequencyProjection, SpatialProjection as SpatialProjection_SFA
from .fssa import FSSA
from .cpia_sa import CPIA_SA
from .ksfa import KSFA
from .fsa import FSA, Adaptive_global_filter

__all__ = [
    'ContrastDrivenFeatureAggregation',
    'HaarWaveletConv',
    'CTA',
    'ChannelProjection',
    'SpatialProjection',
    'SpatialProjection_SFA',
    'SFA',
    'FrequencyProjection',
    'FSSA',
    'CPIA_SA',
    'KSFA',
    'FSA',
    'Adaptive_global_filter',
]
