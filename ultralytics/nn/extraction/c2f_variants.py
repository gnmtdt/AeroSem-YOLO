"""
C2f Extraction - Variant Exports

实现策略（与 AIFI 迁移规范一致）：
- 依赖可复用的底座类放在 `nn/public/`
- C2f 变体作为“组装导出类”放在 `nn/extraction`
"""

from __future__ import annotations

from ultralytics.nn.public import (
    CAMixer,
    HeatBlock,
    FMB,
    MSMHSA_CGLU,
    MogaBlock,
    SHSABlock,
    SHSABlock_CGLU,
    HDRAB,
    RAB,
    Fused_Fourier_Conv_Mixer,
    SMAFormerBlock,
    SMAFormerBlock_CGLU,
    APBottleneck,
    CSI,
    gConvBlock,
    FCA,
    FDT,
    GlobalFilterBlock,
    LSBlock,
    StripBlock,
    StripCGLU,
    Bottleneck_FourierConv,
    Bottleneck_wConv,
    Bottleneck_FDConv,
    SMPCGLU,
    FasterFDConv,
    FasterSFSConv,
    Faster_Block_KAN,
    FAT_Block,
    DBlock,
    EBlock,
    AdditiveBlock,
    AdditiveBlock_CGLU,
    IEL,
    DTAB,
    Bottleneck_PFDConv,
    Bottleneck_SFSConv,
    Bottleneck_PSFSConv,
    # Batch 04 (no-CUDA extras)
    HFERB,
    JDPM,
    ETB,
    SFHF_Block,
    ELGCA_EncoderBlock,
    ELGCA_CGLU,
    LEGM,
    LFE_Module,
    DeepPoolLayer,
    ESCBlock,
    Kat,
)

from .c2f_base import C2fVariantBase

__all__ = [
    # Batch 01
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
    # Batch 02
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
    # Batch 03
    "C2f_FasterFDConv",
    "C2f_FasterSFSConv",
    "C2f_Faster_KAN",
    "C2f_FAT",
    "C2f_SMPCGLU",
    "C2f_DBlock",
    # Batch 04 (partial)
    "C2f_AdditiveBlock",
    "C2f_AdditiveBlock_CGLU",
    "C2f_IEL",
    "C2f_DTAB",
    "C2f_PFDConv",
    "C2f_SFSConv",
    "C2f_PSFSConv",
    "C2f_EBlock",
    # Batch 04 (no-CUDA extras)
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


class C2f_CAMixer(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: CAMixer(self.c, window_size=4))


class C2f_Heat(C2fVariantBase):
    def __init__(self, c1, c2, n=1, feat_size=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        heat_res = 14 if feat_size is None else feat_size
        self._build_blocks(n, lambda: HeatBlock(self.c, heat_res))


class C2f_FMB(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: FMB(self.c))


class C2f_MSMHSA_CGLU(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: MSMHSA_CGLU(self.c))


class C2f_MogaBlock(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: MogaBlock(self.c))


class C2f_SHSA(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: SHSABlock(self.c, pdim=64))


class C2f_SHSA_CGLU(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: SHSABlock_CGLU(self.c, pdim=64))


class C2f_HDRAB(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: HDRAB(self.c, self.c))


class C2f_RAB(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: RAB(self.c, self.c))


class C2f_FFCM(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: Fused_Fourier_Conv_Mixer(self.c))


class C2f_SMAFB(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: SMAFormerBlock(self.c))


class C2f_SMAFB_CGLU(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: SMAFormerBlock_CGLU(self.c))


# ===================== Batch 02 =====================


class C2f_AP(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: APBottleneck(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=e))


class C2f_CSI(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: CSI(self.c))


class C2f_gConv(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: gConvBlock(self.c))


class C2f_FCA(C2fVariantBase):
    def __init__(self, c1, c2, n=1, reso=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        fca_reso = 64 if reso is None else reso
        self._build_blocks(n, lambda: FCA(self.c, reso=fca_reso))


class C2f_FDConv(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: Bottleneck_FDConv(self.c, self.c, shortcut=shortcut, g=g, e=e))


class C2f_FDT(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: FDT(self.c))


class C2f_FourierConv(C2fVariantBase):
    def __init__(self, c1, c2, n=1, size=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        if size is None:
            raise ValueError("C2f_FourierConv 需要显式提供 `size`（如 [20, 20]）以确保频域权重尺寸一致。")
        self._build_blocks(n, lambda: Bottleneck_FourierConv(self.c, self.c, shortcut=shortcut, g=g, size=size, e=e))


class C2f_GlobalFilter(C2fVariantBase):
    def __init__(self, c1, c2, n=1, size=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        if size is None:
            raise ValueError("C2f_GlobalFilter 需要显式提供 `size`（如 20）以构建频域权重。")
        self._build_blocks(n, lambda: GlobalFilterBlock(self.c, size=size))


class C2f_LSBlock(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: LSBlock(self.c, depth=1))


class C2f_Strip(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: StripBlock(self.c))


class C2f_StripCGLU(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: StripCGLU(self.c))


class C2f_wConv(C2fVariantBase):
    def __init__(self, c1, c2, n=1, den=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        if den is None:
            raise ValueError("C2f_wConv 需要显式提供 `den`（如 [0.9]）以构建权重核。")
        self._build_blocks(n, lambda: Bottleneck_wConv(self.c, self.c, shortcut=shortcut, g=g, den=den, e=e))


# ===================== Batch 03 =====================


class C2f_FasterFDConv(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: FasterFDConv(self.c, self.c))


class C2f_FasterSFSConv(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: FasterSFSConv(self.c, self.c))


class C2f_Faster_KAN(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: Faster_Block_KAN(self.c, self.c))


class C2f_FAT(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: FAT_Block(self.c))


class C2f_SMPCGLU(C2fVariantBase):
    def __init__(self, c1, c2, n=1, kernel_size=13, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: SMPCGLU(self.c, kernel_size))


class C2f_DBlock(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: DBlock(self.c))


# ===================== Batch 04（第一批）=====================


class C2f_AdditiveBlock(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: AdditiveBlock(self.c))


class C2f_AdditiveBlock_CGLU(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: AdditiveBlock_CGLU(self.c))


class C2f_IEL(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: IEL(self.c))


class C2f_DTAB(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: DTAB(self.c))


class C2f_PFDConv(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: Bottleneck_PFDConv(self.c, self.c, shortcut, g=g, e=e))


class C2f_SFSConv(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: Bottleneck_SFSConv(self.c, self.c, shortcut, g=g, k=(3, 3), e=1.0))


class C2f_PSFSConv(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: Bottleneck_PSFSConv(self.c, self.c, shortcut, g=g, e=e))


class C2f_EBlock(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: EBlock(self.c))


# ===================== Batch 04（第二批：无需CUDA扩展）=====================


class C2f_HFERB(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: HFERB(self.c))


class C2f_JDPM(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: JDPM(self.c))


class C2f_ETB(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: ETB(self.c))


class C2f_SFHF(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: SFHF_Block(self.c))


class C2f_MSM(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: DeepPoolLayer(self.c))


class C2f_ELGCA(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: ELGCA_EncoderBlock(self.c))


class C2f_ELGCA_CGLU(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: ELGCA_CGLU(self.c))


class C2f_LEGM(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: LEGM(self.c))


class C2f_LFEM(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: LFE_Module(self.c))


class C2f_ESC(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: ESCBlock(self.c))


class C2f_KAT(C2fVariantBase):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self._build_blocks(n, lambda: Kat(self.c))
