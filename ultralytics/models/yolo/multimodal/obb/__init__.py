# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
多模态 OBB 任务入口

提供 YOLOMM 旋转框训练 / 验证组件：
- MultiModalOBBTrainer
- MultiModalOBBValidator

注意：Predictor 已迁移到 mm_predictor.py 作为 YOLOMMOBBPredictor
"""

from .train import MultiModalOBBTrainer
from .val import MultiModalOBBValidator

__all__ = ["MultiModalOBBTrainer", "MultiModalOBBValidator"]
