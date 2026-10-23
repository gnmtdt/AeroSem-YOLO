"""公共模块出口（延迟导入以避免循环依赖）。"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    # NOTE: 这里使用延迟导入（PEP 562 __getattr__）以避免与 `ultralytics.nn.modules.*` 的循环依赖。
    "ConvolutionalGLU": (".common_glu", "ConvolutionalGLU"),
    "RetBlock": (".rmt", "RetBlock"),
    "RelPos2d": (".rmt", "RelPos2d"),
    "FeedForwardNetwork": (".rmt", "FeedForwardNetwork"),
    "MaSA": (".rmt", "MaSA"),
    "MaSAd": (".rmt", "MaSAd"),
    "DWConv2d": (".rmt", "DWConv2d"),
    "Heat2D": (".heat", "Heat2D"),
    "HeatBlock": (".heat", "HeatBlock"),
    "WTConv2d": (".wtconv2d", "WTConv2d"),
    "DMlp": (".fmb", "DMlp"),
    "SMFA": (".fmb", "SMFA"),
    "PCFN": (".fmb", "PCFN"),
    "FMB": (".fmb", "FMB"),
    "MutilScal": (".msmhsa_cglu", "MutilScal"),
    "Mutilscal_MHSA": (".msmhsa_cglu", "Mutilscal_MHSA"),
    "MSMHSA_CGLU": (".msmhsa_cglu", "MSMHSA_CGLU"),
    "ElementScale": (".mogablock", "ElementScale"),
    "ChannelAggregationFFN": (".mogablock", "ChannelAggregationFFN"),
    "MultiOrderDWConv": (".mogablock", "MultiOrderDWConv"),
    "MultiOrderGatedAggregation": (".mogablock", "MultiOrderGatedAggregation"),
    "MogaBlock": (".mogablock", "MogaBlock"),
    "SHSA_GroupNorm": (".shsa", "SHSA_GroupNorm"),
    "SHSABlock_FFN": (".shsa", "SHSABlock_FFN"),
    "SHSA": (".shsa", "SHSA"),
    "SHSABlock": (".shsa", "SHSABlock"),
    "SHSABlock_CGLU": (".shsa", "SHSABlock_CGLU"),
    "DSM_SpatialGate": (".dsm", "DSM_SpatialGate"),
    "DSM_LocalAttention": (".dsm", "DSM_LocalAttention"),
    "DualDomainSelectionMechanism": (".dsm", "DualDomainSelectionMechanism"),
    "EdgeEnhancer": (".edge_msie", "EdgeEnhancer"),
    "MutilScaleEdgeInformationEnhance": (".edge_msie", "MutilScaleEdgeInformationEnhance"),
    "MutilScaleEdgeInformationSelect": (".edge_msie", "MutilScaleEdgeInformationSelect"),
    "DeepPoolLayer": (".resto_blocks", "DeepPoolLayer"),
    "MSMBlock": (".resto_blocks", "MSMBlock"),
    "CAB": (".resto_blocks", "CAB"),
    "HDRAB": (".resto_blocks", "HDRAB"),
    "RAB": (".resto_blocks", "RAB"),
    "ShiftConv2d0": (".resto_blocks", "ShiftConv2d0"),
    "ShiftConv2d1": (".resto_blocks", "ShiftConv2d1"),
    "LFE": (".resto_blocks", "LFE"),
    "FourierUnit": (".ffcm", "FourierUnit"),
    "Freq_Fusion": (".ffcm", "Freq_Fusion"),
    "Fused_Fourier_Conv_Mixer": (".ffcm", "Fused_Fourier_Conv_Mixer"),
    "Modulator": (".smaformer", "Modulator"),
    "SMA": (".smaformer", "SMA"),
    "E_MLP": (".smaformer", "E_MLP"),
    "SMAFormerBlock": (".smaformer", "SMAFormerBlock"),
    "SMAFormerBlock_CGLU": (".smaformer", "SMAFormerBlock_CGLU"),
    "InceptionDWConv2d": (".inceptionnext_blocks", "InceptionDWConv2d"),
    "MetaNeXtBlock": (".inceptionnext_blocks", "MetaNeXtBlock"),
    "CAMixer": (".camixer", "CAMixer"),
    "TransformerEncoderLayer": (".transformer_encoder_layer", "TransformerEncoderLayer"),
    "PSConv": (".ap", "PSConv"),
    "APBottleneck": (".ap", "APBottleneck"),
    "gConvBlock": (".gconv", "gConvBlock"),
    "FDT": (".fdt", "FDT"),
    "FCA": (".fca", "FCA"),
    "GlobalFilterBlock": (".global_filter", "GlobalFilterBlock"),
    "LSConv": (".lsnet_blocks", "LSConv"),
    "LSBlock": (".lsnet_blocks", "LSBlock"),
    "StripBlock": (".strip", "StripBlock"),
    "StripCGLU": (".strip", "StripCGLU"),
    "FDConv": (".fdconv", "FDConv"),
    "Bottleneck_FDConv": (".fdconv", "Bottleneck_FDConv"),
    "CSI": (".csi", "CSI"),
    "Bottleneck_FourierConv": (".fourier_conv_block", "Bottleneck_FourierConv"),
    "wConv2d": (".wconv", "wConv2d"),
    "Bottleneck_wConv": (".wconv", "Bottleneck_wConv"),
    # Batch 03 (C2f)
    "Partial_conv3": (".partial_conv", "Partial_conv3"),
    "SMPConv": (".smpconv", "SMPConv"),
    "SMPCGLU": (".smpcglu", "SMPCGLU"),
    "Partial_FDConv": (".faster_fdconv", "Partial_FDConv"),
    "FasterFDConv": (".faster_fdconv", "FasterFDConv"),
    "SFS_Conv": (".sfsconv", "SFS_Conv"),
    "Partial_SFSConv": (".faster_sfsconv", "Partial_SFSConv"),
    "FasterSFSConv": (".faster_sfsconv", "FasterSFSConv"),
    "KAN": (".faster_kan", "KAN"),
    "Faster_Block_KAN": (".faster_kan", "Faster_Block_KAN"),
    "FAT_Block": (".fat", "FAT_Block"),
    "DBlock": (".darkir", "DBlock"),
    "EBlock": (".darkir", "EBlock"),
    # Batch 04 (C2f)
    "AdditiveBlock": (".additive", "AdditiveBlock"),
    "AdditiveBlock_CGLU": (".additive", "AdditiveBlock_CGLU"),
    "IEL": (".iel", "IEL"),
    "DTAB": (".dtab", "DTAB"),
    "Bottleneck_PFDConv": (".pfdconv", "Bottleneck_PFDConv"),
    "Bottleneck_SFSConv": (".sfsconv_blocks", "Bottleneck_SFSConv"),
    "Bottleneck_PSFSConv": (".sfsconv_blocks", "Bottleneck_PSFSConv"),
    # Batch 04 (C2f) - no-CUDA extras
    "HFERB": (".hferb", "HFERB"),
    "JDPM": (".jdpm", "JDPM"),
    "ETB": (".etb", "ETB"),
    "SFHF_Block": (".sfhf", "SFHF_Block"),
    "ELGCA_EncoderBlock": (".elgca", "ELGCA_EncoderBlock"),
    "ELGCA_CGLU": (".elgca", "ELGCA_CGLU"),
    "LEGM": (".legm", "LEGM"),
    "LFE_Module": (".lfem", "LFE_Module"),
    "ESCBlock": (".esc", "ESCBlock"),
    "Kat": (".kat", "Kat"),
    # 注意力模块 (即插即用)
    "CoordAtt": (".ca", "CoordAtt"),
    "h_sigmoid": (".ca", "h_sigmoid"),
    "h_swish": (".ca", "h_swish"),
    "DeformableLKA": (".deformable_lka", "DeformableLKA"),
    "DeformConv": (".deformable_lka", "DeformConv"),
    "EMA_Attention": (".ema_attn", "EMA_Attention"),
    "LSKBlock_SA": (".lsk", "LSKBlock_SA"),
    "LSKBlock": (".lsk", "LSKBlock"),
    # 上采样/下采样模块已迁移至 ultralytics.nn.sample
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value  # 缓存，避免重复导入
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
