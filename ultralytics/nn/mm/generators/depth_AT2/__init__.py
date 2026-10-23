# ultralytics/nn/mm/generators/depth_AT2/__init__.py
"""
Depth-Anything-V2 module - fully integrated into ultralytics.

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under Apache License 2.0.
"""

from .dpt import DepthAnythingV2, DPTHead, ConvBlock
from .dinov2 import DINOv2, DinoVisionTransformer
from .transform import Resize, NormalizeImage, PrepareForNet
from .blocks import FeatureFusionBlock, ResidualConvUnit, _make_scratch

__all__ = [
    # Main model
    "DepthAnythingV2",
    "DPTHead",
    "ConvBlock",
    # Backbone
    "DINOv2",
    "DinoVisionTransformer",
    # Transforms
    "Resize",
    "NormalizeImage",
    "PrepareForNet",
    # Blocks
    "FeatureFusionBlock",
    "ResidualConvUnit",
    "_make_scratch",
]
