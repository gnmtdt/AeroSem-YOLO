# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector.

RT-DETR offers real-time performance and high accuracy, excelling in accelerated backends like CUDA with TensorRT.
It features an efficient hybrid encoder and IoU-aware query selection for enhanced detection accuracy.

References:
    https://arxiv.org/pdf/2304.08069.pdf
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel
# 说明：可视化异常与工具已迁移至 visualize_core，此处无需引入旧路径
# from ultralytics.models.yolo.multimodal.modal_filling import generate_modality_filling
# 说明：模态填充与消融已统一下沉至 MultiModalRouter（ultralytics/nn/mm/router.py）。
# 这里不再直接调用 generate_modality_filling，旧实现保留为注释以便回溯。

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """
    Interface for Baidu's RT-DETR model, a Vision Transformer-based real-time object detector.

    This model provides real-time performance with high accuracy. It supports efficient hybrid encoding, IoU-aware
    query selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained model.

    Methods:
        task_map: Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

    Examples:
        Initialize RT-DETR with a pre-trained model
        >>> from ultralytics import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        """
        Initialize the RT-DETR model with the given pre-trained model file.

        Args:
            model (str): Path to the pre-trained model. Supports .pt, .yaml, and .yml formats.
        """
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """
        Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            (dict): A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }

    def cocoval(self, **kwargs):
        """
        Run COCO evaluation using RTDETRMMCOCOValidator.

        Args:
            **kwargs: Additional arguments to pass to the validator.

        Returns:
            dict: Validation metrics from the COCO evaluation.
        """
        from .cocoval import RTDETRMMCOCOValidator
        
        self._check_is_pytorch_model()
        
        args = {**self.overrides, **{"rect": True, "conf": 0.05}, **kwargs, **{"mode": "val"}}
        # 在验证前统一输出 canonical complexity summary；runtime-aware 统计如有需要仅保留 internal-only
        try:
            imgsz = int(args.get("imgsz", 640))
            from ultralytics.nn.mm.complexity import (
                compute_default_multimodal_complexity_report,
                log_default_complexity,
            )
            from ultralytics.utils import LOGGER as _LOGGER

            report = compute_default_multimodal_complexity_report(self.model, imgsz=imgsz)
            log_default_complexity(self.model, report, _LOGGER)
        except Exception:
            pass

        validator = RTDETRMMCOCOValidator(
            dataloader=None,
            save_dir=None,
            pbar=None,
            args=args,
            _callbacks=self.callbacks
        )
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics
