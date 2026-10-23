"""RTDETR/RTDETRMM backbone modules (multi-output backbones)."""

from .timm_backbone import TimmBackbone
from .convnextv2 import (
    convnextv2_atto,
    convnextv2_base,
    convnextv2_femto,
    convnextv2_huge,
    convnextv2_large,
    convnextv2_nano,
    convnextv2_pico,
    convnextv2_tiny,
)
from .repvit import repvit_m0_9, repvit_m1_0, repvit_m1_1, repvit_m1_5, repvit_m2_3
from .EfficientFormerV2 import efficientformerv2_l, efficientformerv2_s0, efficientformerv2_s1, efficientformerv2_s2
from .efficientViT import EfficientViT_M0, EfficientViT_M1, EfficientViT_M2, EfficientViT_M3, EfficientViT_M4, EfficientViT_M5
from .SwinTransformer import SwinTransformer_Tiny

__all__ = [
    "TimmBackbone",
    "convnextv2_atto",
    "convnextv2_femto",
    "convnextv2_pico",
    "convnextv2_nano",
    "convnextv2_tiny",
    "convnextv2_base",
    "convnextv2_large",
    "convnextv2_huge",
    "repvit_m0_9",
    "repvit_m1_0",
    "repvit_m1_1",
    "repvit_m1_5",
    "repvit_m2_3",
    "efficientformerv2_s0",
    "efficientformerv2_s1",
    "efficientformerv2_s2",
    "efficientformerv2_l",
    "EfficientViT_M0",
    "EfficientViT_M1",
    "EfficientViT_M2",
    "EfficientViT_M3",
    "EfficientViT_M4",
    "EfficientViT_M5",
    "SwinTransformer_Tiny",
]
