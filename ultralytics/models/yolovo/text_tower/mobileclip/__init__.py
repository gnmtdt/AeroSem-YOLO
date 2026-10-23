"""
YOLOVO MobileCLIP runtime package.

This package provides MobileCLIP text encoding capabilities for YOLOVO without
requiring pip install of the upstream apple/ml-mobileclip package.

The vendored source code is located under _vendor/mobileclip/ and is injected
into sys.path by factory.py before any mobileclip imports.

Usage:
    from ultralytics.models.yolovo.text_tower.mobileclip import build_mobileclip_bundle
    bundle = build_mobileclip_bundle(variant="s0", asset_dir="path/to/weights")
"""

from ultralytics.models.yolovo.text_tower.mobileclip.factory import (
    build_mobileclip_bundle,
    MobileCLIPVariantSpec,
    YOLOVOMobileCLIPBundle,
)

__all__ = ["build_mobileclip_bundle", "MobileCLIPVariantSpec", "YOLOVOMobileCLIPBundle"]
