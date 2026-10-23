"""
Extraction Modules
提取并迁移的 C3k2、C2PSA 和 SPPF 变体模块（仅导出封装后的变体类）。

分组导出策略：
- 每个分组独立 try/except，任何一组失败不影响其它组使用；
- 仅导出封装后的变体（如 C3k2_*、C2PSA_* 等），不导出底层基础/辅助构件，避免重复对外暴露。
"""

__all__ = []

# ===================== C3k2 变体分组 =====================
try:
    from .c3k2_variants import (
        # Batch 1
        C3k2_Faster,
        C3k2_PConv,
        C3k2_ODConv,
        C3k2_Faster_EMA,
        C3k2_DBB,
        C3k2_WDBB,
        C3k2_DeepDBB,
        # Batch 2
        C3k2_CloAtt,
        C3k2_SCConv,
        C3k2_ScConv,
        C3k2_EMSC,
        C3k2_EMSCP,
        # Batch 3
        C3k2_ContextGuided,
        C3k2_MSBlock,
        C3k2_EMBC,
        C3k2_EMA,
        # Batch 4
        C3k2_DLKA,
        C3k2_DAttention,
        C3k2_Parc,
        C3k2_DWR,
        C3k2_RFAConv,
        # Batch 5
        C3k2_RFCBAMConv,
        C3k2_RFCAConv,
        C3k2_FocusedLinearAttention,
        C3k2_MLCA,
        C3k2_AKConv,
        # Batch 6
        C3k2_UniRepLKNetBlock,
        C3k2_DRB,
        C3k2_DWR_DRB,
        C3k2_AggregatedAtt,
        C3k2_SWC,
        # Batch 7
        C3k2_iRMB,
        C3k2_iRMB_Cascaded,
        C3k2_iRMB_DRB,
        C3k2_iRMB_SWC,
        C3k2_DynamicConv,
        # Batch 8
        C3k2_GhostDynamicConv,
        C3k2_RVB,
        C3k2_RVB_SE,
        C3k2_RVB_EMA,
        # Batch 9
        C3k2_PKIModule,
        C3k2_PPA,
        C3k2_Faster_CGLU,
        C3k2_Star,
        # Batch 10
        C3k2_Star_CAA,
        C3k2_EIEM,
        C3k2_DEConv,
        # Batch 11
        C3k2_gConv,
        C3k2_AdditiveBlock,
        C3k2_AdditiveBlock_CGLU,
        # Batch 12 (新增迁移)
        C3k2_RetBlock,
        C3k2_Heat,
        C3k2_WTConv,
        C3k2_FMB,
        C3k2_MSMHSA_CGLU,
        C3k2_MogaBlock,
        C3k2_SHSA,
        C3k2_SHSA_CGLU,
        C3k2_MutilScaleEdgeInformationEnhance,
        C3k2_MutilScaleEdgeInformationSelect,
        C3k2_FFCM,
        C3k2_SMAFB,
        C3k2_SMAFB_CGLU,
        C3k2_MSM,
        C3k2_HDRAB,
        C3k2_RAB,
        C3k2_LFE,
        C3k2_LFEM,
        C3k2_LEGM,
    )
    __all__ += [
        'C3k2_Faster','C3k2_PConv','C3k2_ODConv','C3k2_Faster_EMA',
        'C3k2_DBB','C3k2_WDBB','C3k2_DeepDBB',
        'C3k2_CloAtt','C3k2_SCConv','C3k2_ScConv','C3k2_EMSC','C3k2_EMSCP',
        'C3k2_ContextGuided','C3k2_MSBlock','C3k2_EMBC','C3k2_EMA',
        'C3k2_DLKA','C3k2_DAttention','C3k2_Parc','C3k2_DWR','C3k2_RFAConv',
        'C3k2_RFCBAMConv','C3k2_RFCAConv','C3k2_FocusedLinearAttention','C3k2_MLCA','C3k2_AKConv',
        'C3k2_UniRepLKNetBlock','C3k2_DRB','C3k2_DWR_DRB','C3k2_AggregatedAtt','C3k2_SWC',
        'C3k2_iRMB','C3k2_iRMB_Cascaded','C3k2_iRMB_DRB','C3k2_iRMB_SWC','C3k2_DynamicConv',
        'C3k2_GhostDynamicConv','C3k2_RVB','C3k2_RVB_SE','C3k2_RVB_EMA',
        'C3k2_PKIModule','C3k2_PPA','C3k2_Faster_CGLU','C3k2_Star',
        'C3k2_Star_CAA','C3k2_EIEM','C3k2_DEConv',
        'C3k2_gConv','C3k2_AdditiveBlock','C3k2_AdditiveBlock_CGLU',
        'C3k2_RetBlock','C3k2_Heat','C3k2_WTConv','C3k2_FMB',
        'C3k2_MSMHSA_CGLU','C3k2_MogaBlock','C3k2_SHSA','C3k2_SHSA_CGLU',
        'C3k2_MutilScaleEdgeInformationEnhance','C3k2_MutilScaleEdgeInformationSelect','C3k2_FFCM',
        'C3k2_SMAFB','C3k2_SMAFB_CGLU',
        'C3k2_MSM','C3k2_HDRAB','C3k2_RAB','C3k2_LFE',
        'C3k2_LFEM','C3k2_LEGM',
    ]
except Exception:  # 保持其他分组可用
    pass

# ===================== C2f 变体分组（Batch 01）=====================
from .c2f_variants import (
    C2f_CAMixer,
    C2f_Heat,
    C2f_FMB,
    C2f_MSMHSA_CGLU,
    C2f_MogaBlock,
    C2f_SHSA,
    C2f_SHSA_CGLU,
    C2f_HDRAB,
    C2f_RAB,
    C2f_FFCM,
    C2f_SMAFB,
    C2f_SMAFB_CGLU,
    # Batch 02
    C2f_AP,
    C2f_CSI,
    C2f_gConv,
    C2f_FCA,
    C2f_FDConv,
    C2f_FDT,
    C2f_FourierConv,
    C2f_GlobalFilter,
    C2f_LSBlock,
    C2f_Strip,
    C2f_StripCGLU,
    C2f_wConv,
    # Batch 03
    C2f_FasterFDConv,
    C2f_FasterSFSConv,
    C2f_Faster_KAN,
    C2f_FAT,
    C2f_SMPCGLU,
    C2f_DBlock,
    # Batch 04 (partial)
    C2f_AdditiveBlock,
    C2f_AdditiveBlock_CGLU,
    C2f_IEL,
    C2f_DTAB,
    C2f_PFDConv,
    C2f_SFSConv,
    C2f_PSFSConv,
    C2f_EBlock,
    # Batch 04 (no-CUDA extras)
    C2f_HFERB,
    C2f_JDPM,
    C2f_ETB,
    C2f_SFHF,
    C2f_MSM,
    C2f_ELGCA,
    C2f_ELGCA_CGLU,
    C2f_LEGM,
    C2f_LFEM,
    C2f_ESC,
    C2f_KAT,
)

__all__ += [
    "C2f_CAMixer",
    "C2f_Heat",
    "C2f_FMB",
    "C2f_MSMHSA_CGLU",
    "C2f_MogaBlock",
    "C2f_SHSA",
    "C2f_SHSA_CGLU",
    "C2f_HDRAB",
    "C2f_RAB",
    "C2f_FFCM",
    "C2f_SMAFB",
    "C2f_SMAFB_CGLU",
    "C2f_AP",
    "C2f_CSI",
    "C2f_gConv",
    "C2f_FCA",
    "C2f_FDConv",
    "C2f_FDT",
    "C2f_FourierConv",
    "C2f_GlobalFilter",
    "C2f_LSBlock",
    "C2f_Strip",
    "C2f_StripCGLU",
    "C2f_wConv",
    "C2f_FasterFDConv",
    "C2f_FasterSFSConv",
    "C2f_Faster_KAN",
    "C2f_FAT",
    "C2f_SMPCGLU",
    "C2f_DBlock",
    "C2f_AdditiveBlock",
    "C2f_AdditiveBlock_CGLU",
    "C2f_IEL",
    "C2f_DTAB",
    "C2f_PFDConv",
    "C2f_SFSConv",
    "C2f_PSFSConv",
    "C2f_EBlock",
    "C2f_HFERB",
    "C2f_JDPM",
    "C2f_ETB",
    "C2f_SFHF",
    "C2f_MSM",
    "C2f_ELGCA",
    "C2f_ELGCA_CGLU",
    "C2f_LEGM",
    "C2f_LFEM",
    "C2f_ESC",
    "C2f_KAT",
]

# ===================== C2PSA 变体分组 =====================
try:
    from .c2psa_variants import (
        # 仅导出封装后的模块，不导出 *Block / *lock 等辅助类
        C2PSA,
        C2fPSA,
        C2BRA,
        C2CGA,
        C2DA,
        C2DPB,
        C2Pola,
        C2TSSA,
        C2ASSA,
        C2MSLA,
        C2PSA_DYT,
        C2TSSA_DYT,
        C2Pola_DYT,
        # Batch 4 - FFN 增强
        C2PSA_FMFFN,
        C2PSA_CGLU,
        C2PSA_SEFN,
        C2PSA_SEFFN,
        C2PSA_EDFFN,
        # Batch 5 - Mona 复合增强
        C2PSA_Mona,
        C2TSSA_DYT_Mona,
        C2TSSA_DYT_Mona_SEFN,
        C2TSSA_DYT_Mona_SEFFN,
        C2TSSA_DYT_Mona_EDFFN,
    )
    __all__ += [
        'C2PSA','C2fPSA','C2BRA','C2CGA','C2DA','C2DPB','C2Pola','C2TSSA',
        'C2ASSA','C2MSLA','C2PSA_DYT','C2TSSA_DYT','C2Pola_DYT',
        'C2PSA_FMFFN','C2PSA_CGLU','C2PSA_SEFN','C2PSA_SEFFN','C2PSA_EDFFN',
        'C2PSA_Mona','C2TSSA_DYT_Mona','C2TSSA_DYT_Mona_SEFN','C2TSSA_DYT_Mona_SEFFN','C2TSSA_DYT_Mona_EDFFN',
    ]
except Exception:
    pass

# ===================== SPPF 及空间池化分组 =====================
try:
    from .sppf_base import (
        # 标准 SPP/SPPF
        SPP,
        SPPF,
        SPPELAN,
        SPP_DAMO,
        LSKA,
        SPPF_LSKA,
        # 金字塔池化/融合
        PyramidPoolAgg,
        PyramidPoolAgg_PCE,
        SimFusion_3in,
        SimFusion_4in,
        AdvPoolFusion,
        # 注入/小波模块
        IFM,
        InjectionMultiSum_Auto_pool,
        WaveletPool,
        WaveletUnPool,
    )
    __all__ += [
        'SPP','SPPF','SPPELAN','SPP_DAMO','LSKA','SPPF_LSKA',
        'PyramidPoolAgg','PyramidPoolAgg_PCE','SimFusion_3in','SimFusion_4in','AdvPoolFusion',
        'IFM','InjectionMultiSum_Auto_pool','WaveletPool','WaveletUnPool',
    ]
except Exception:
    pass

# ===================== Other Base 独立模块分组 =====================
try:
    from .other_base import (
        RFEM,
    )
    __all__ += [
        'RFEM',
    ]
except Exception:
    pass

# ===================== LoGStem 模块 =====================
try:
    from .logstem import (
        LoGStem,
        LoGStem2x,
        DRFD,
        Cut,
    )
    __all__ += [
        'LoGStem',
        'LoGStem2x',
        'DRFD',
        'Cut',
    ]
except Exception:
    pass

# ===================== 万物皆可融 Block 体系 =====================
try:
    from .block_fusion import C3k_Block, C3_Block, C2f_Block, C3k2_Block
    __all__ += ['C3k_Block', 'C3_Block', 'C2f_Block', 'C3k2_Block']
except Exception:
    pass
