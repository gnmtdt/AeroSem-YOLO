# Ultralytics RTDETRMM Predictor Adapter
# 适配 BasePredictor 接口到新的 MultiModalPredictor 引擎
# Version: v1.0
# Date: 2026-01-13

from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.multimodal import MultiModalPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER


class RTDETRMMPredictor(BasePredictor):
    _tid = "debugger"
    """
    RTDETRMM 推理适配器（BasePredictor -> MultiModalPredictor 桥接）

    职责：
    - 从 Model.predict() 接收 BasePredictor 风格的参数
    - 转换为新推理引擎所需的格式并调用 MultiModalPredictor
    - 保持与现有 API 的兼容性

    设计原则：
    - 不继承 MultiModalPredictor（避免多重继承复杂度）
    - 作为适配层（Adapter），内部持有 MultiModalPredictor 实例
    - 遵循 BasePredictor 接口规范，确保可被 Model.predict() 正确调用
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化 RTDETRMM 推理适配器

        Args:
            cfg: 配置对象
            overrides: 覆盖参数
            _callbacks: 回调函数
        """
        super().__init__(cfg, overrides, _callbacks)
        self.predictor = None  # MultiModalPredictor 实例（延迟初始化）

    def setup_model(self, model, verbose=True, debug=False):
        """
        设置模型（BasePredictor 接口要求）

        Args:
            model: RTDETRMM 模型实例
            verbose: 是否输出详细日志（常规日志）
            debug: 是否输出DEBUG调试日志
        """
        super().setup_model(model, verbose)

        # 初始化新推理引擎
        self.predictor = MultiModalPredictor(
            model=self.model,
            imgsz=self.args.imgsz,
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            device=str(self.device),
            verbose=verbose,
            debug=debug,
            font_size=getattr(self.args, 'font_size', None),
            show_filename=getattr(self.args, 'show_filename', False)
        )

        if verbose:
            LOGGER.info("推理引擎初始化成功")

    def __call__(
        self,
        rgb_source: Union[str, Path, List[Union[str, Path]]] = None,
        x_source: Union[str, Path, List[Union[str, Path]]] = None,
        stream: bool = False,
        strict_match: bool = True,
        **kwargs: Any
    ):
        """
        执行推理（适配新API - 显式RGB和X模态输入）

        Args:
            rgb_source: RGB图像源
            x_source: X模态图像源
            stream: 是否流式返回结果
            strict_match: 批量推理时的匹配策略（默认严格）
            **kwargs: 其他参数

        Returns:
            stream=True: Generator[MultiModalResult]
            stream=False: List[MultiModalResult]
        """
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        # 旧式双输入拆包: source=[rgb, x] -> rgb_source, x_source
        if x_source is None and isinstance(rgb_source, (list, tuple)) and len(rgb_source) == 2:
            rgb_source, x_source = rgb_source[0], rgb_source[1]

        # 从 args 中读取保存参数
        save = kwargs.get('save', getattr(self.args, 'save', False))
        save_txt = kwargs.get('save_txt', getattr(self.args, 'save_txt', False))
        save_json = kwargs.get('save_json', getattr(self.args, 'save_json', False))
        save_dir = kwargs.get('save_dir', getattr(self.args, 'save_dir', None))
        font_size = kwargs.pop('font_size', None)
        show_filename = kwargs.pop('show_filename', None)

        if save_dir is None:
            save_dir = getattr(self, 'save_dir', None)

        # 调用新推理引擎
        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=stream,
            strict_match=strict_match,
            save=save,
            save_txt=save_txt,
            save_json=save_json,
            save_dir=save_dir,
            font_size=font_size,
            show_filename=show_filename,
            **kwargs
        )

    def predict_cli(self, rgb_source=None, x_source=None, strict_match: bool = True):
        """
        CLI 模式推理（BasePredictor 接口要求）

        Args:
            rgb_source: RGB图像源
            x_source: X模态图像源
            strict_match: 批量推理时的匹配策略（默认严格）

        Returns:
            推理结果列表
        """
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        # 旧式双输入拆包: source=[rgb, x] -> rgb_source, x_source
        if x_source is None and isinstance(rgb_source, (list, tuple)) and len(rgb_source) == 2:
            rgb_source, x_source = rgb_source[0], rgb_source[1]

        # CLI 模式默认保存
        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            strict_match=strict_match,
            stream=False,
            save=True,
            save_txt=getattr(self.args, 'save_txt', False),
            save_json=getattr(self.args, 'save_json', False),
            save_dir=getattr(self, 'save_dir', None)
        )
