"""
Multimodal segmentation adapters for YOLO family.

This package provides YOLOMM-style multimodal wrappers for the segmentation task,
mirroring the detection-side implementations while switching to segmentation
train/val/predict bases.
"""

from .train import MultiModalSegmentationTrainer
from .val import MultiModalSegmentationValidator
from .predict import MultiModalSegmentationPredictor
from .cocoval import MultiModalSegmentationCOCOValidator

__all__ = [
    "MultiModalSegmentationTrainer",
    "MultiModalSegmentationValidator",
    "MultiModalSegmentationPredictor",
    "MultiModalSegmentationCOCOValidator",
]
