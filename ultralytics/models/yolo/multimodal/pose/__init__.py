# Ultralytics YOLO, AGPL-3.0 license

"""
Multi-Modal Pose Task Entry Point.

Provides YOLOMM pose estimation training / validation components:
- MultiModalPoseTrainer
- MultiModalPoseValidator

Note: Predictor has been moved to mm_predictor.py as YOLOMMPosePredictor
"""

from .train import MultiModalPoseTrainer
from .val import MultiModalPoseValidator

__all__ = ["MultiModalPoseTrainer", "MultiModalPoseValidator"]
