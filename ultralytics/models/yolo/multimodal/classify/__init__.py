# Ultralytics YOLO, AGPL-3.0 license

"""
多模态分类模块

本模块提供 YOLOMM 多模态分类任务的训练和验证组件。

注意：Predictor 已迁移到 mm_predictor.py 作为 YOLOMMClassifyPredictor
"""

from .train import MultiModalClassificationTrainer
from .val import MultiModalClassificationValidator

__all__ = [
    "MultiModalClassificationTrainer",
    "MultiModalClassificationValidator",
]
