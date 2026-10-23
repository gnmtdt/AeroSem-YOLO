# Ultralytics YOLO MultiModal Predictor Adapter
# 适配 BasePredictor 接口到新的 MultiModalPredictor 引擎
# Version: v1.0
# Date: 2026-01-13

from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.multimodal import MultiModalPredictor, MultiModalSegmentPredictor
from ultralytics.engine.multimodal import MultiModalOBBPredictor, MultiModalPosePredictor, MultiModalClassifyPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER


class YOLOMMPredictor(BasePredictor):
    """
    YOLOMM 推理适配器（BasePredictor -> MultiModalPredictor 桥接）

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
        初始化 YOLOMM 推理适配器

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
            model: YOLOMM 模型实例
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
            **kwargs: 其他参数（支持 conf, iou, max_det 等）

        Returns:
            stream=True: Generator[MultiModalResult]
            stream=False: List[MultiModalResult]
        """
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        # 旧式双输入拆包: source=[rgb, x] -> rgb_source, x_source
        if x_source is None and isinstance(rgb_source, (list, tuple)) and len(rgb_source) == 2:
            rgb_source, x_source = rgb_source[0], rgb_source[1]

        # 从 args 或 kwargs 中读取参数
        save = kwargs.pop('save', getattr(self.args, 'save', False))
        save_txt = kwargs.pop('save_txt', getattr(self.args, 'save_txt', False))
        save_json = kwargs.pop('save_json', getattr(self.args, 'save_json', False))
        save_dir = kwargs.pop('save_dir', getattr(self.args, 'save_dir', None))

        # 提取NMS参数（支持运行时覆盖）
        conf = kwargs.pop('conf', None)
        iou = kwargs.pop('iou', None)
        max_det = kwargs.pop('max_det', None)
        crop = kwargs.pop('crop', getattr(self.args, 'crop', False))
        font_size = kwargs.pop('font_size', None)
        show_filename = kwargs.pop('show_filename', None)

        if save_dir is None:
            save_dir = getattr(self, 'save_dir', None)

        # 调用新推理引擎（传递NMS参数）
        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=stream,
            strict_match=strict_match,
            save=save,
            save_txt=save_txt,
            save_json=save_json,
            save_dir=save_dir,
            conf=conf,
            iou=iou,
            max_det=max_det,
            crop=crop,
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


class YOLOMMSegPredictor(BasePredictor):
    """
    YOLOMM 分割推理适配器（BasePredictor -> MultiModalSegmentPredictor 桥接）

    职责：
    - 从 Model.predict() 接收 BasePredictor 风格的参数
    - 转换为新推理引擎所需的格式并调用 MultiModalSegmentPredictor
    - 保持与现有 API 的兼容性
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.predictor = None

    def setup_model(self, model, verbose=True, debug=False):
        super().setup_model(model, verbose)

        self.predictor = MultiModalSegmentPredictor(
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
            LOGGER.info("YOLOMMSegPredictor: 分割推理引擎已初始化")

    def __call__(
        self,
        rgb_source: Union[str, Path, List[Union[str, Path]]] = None,
        x_source: Union[str, Path, List[Union[str, Path]]] = None,
        stream: bool = False,
        **kwargs: Any
    ):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        save = kwargs.pop('save', getattr(self.args, 'save', False))
        save_txt = kwargs.pop('save_txt', getattr(self.args, 'save_txt', False))
        save_json = kwargs.pop('save_json', getattr(self.args, 'save_json', False))
        save_dir = kwargs.pop('save_dir', getattr(self.args, 'save_dir', None))

        conf = kwargs.pop('conf', None)
        iou = kwargs.pop('iou', None)
        max_det = kwargs.pop('max_det', None)
        crop = kwargs.pop('crop', getattr(self.args, 'crop', False))
        font_size = kwargs.pop('font_size', None)
        show_filename = kwargs.pop('show_filename', None)

        if save_dir is None:
            save_dir = getattr(self, 'save_dir', None)

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=stream,
            save=save,
            save_txt=save_txt,
            save_json=save_json,
            save_dir=save_dir,
            conf=conf,
            iou=iou,
            max_det=max_det,
            crop=crop,
            font_size=font_size,
            show_filename=show_filename,
            **kwargs
        )

    def predict_cli(self, rgb_source=None, x_source=None):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=False,
            save=True,
            save_txt=getattr(self.args, 'save_txt', False),
            save_json=getattr(self.args, 'save_json', False),
            save_dir=getattr(self, 'save_dir', None)
        )


class YOLOMMSemSegPredictor(BasePredictor):
    """YOLOMM 语义分割推理适配器"""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.predictor = None

    def setup_model(self, model, verbose=True, debug=False):
        super().setup_model(model, verbose)
        from ultralytics.engine.multimodal.predictor import MultiModalSemSegPredictor
        self.predictor = MultiModalSemSegPredictor(
            model=self.model,
            imgsz=self.args.imgsz,
            device=str(self.device),
            verbose=verbose,
            debug=debug,
            save_conf=getattr(self.args, 'save_conf', False),
            ignore_index=getattr(self.args, 'ignore_index', 255),
        )
        if verbose:
            LOGGER.info("YOLOMMSemSegPredictor: 语义分割推理引擎已初始化")

    def __call__(self, rgb_source=None, x_source=None, stream=False, **kwargs):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")
        save = kwargs.pop('save', getattr(self.args, 'save', False))
        save_json = kwargs.pop('save_json', getattr(self.args, 'save_json', False))
        save_dir = kwargs.pop('save_dir', getattr(self.args, 'save_dir', None))
        return self.predictor(
            rgb_source=rgb_source, x_source=x_source, stream=stream,
            save=save, save_json=save_json, save_dir=save_dir, **kwargs
        )

    def predict_cli(self, rgb_source=None, x_source=None):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")
        return self.predictor(
            rgb_source=rgb_source, x_source=x_source, stream=False,
            save=True, save_json=getattr(self.args, 'save_json', False),
            save_dir=getattr(self, 'save_dir', None)
        )


class YOLOMMOBBPredictor(BasePredictor):
    """
    YOLOMM OBB 推理适配器（BasePredictor -> MultiModalOBBPredictor 桥接）
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.predictor = None

    def setup_model(self, model, verbose=True, debug=False):
        super().setup_model(model, verbose)

        self.predictor = MultiModalOBBPredictor(
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
            LOGGER.info("YOLOMMOBBPredictor: OBB推理引擎已初始化")

    def __call__(
        self,
        rgb_source=None,
        x_source=None,
        stream: bool = False,
        **kwargs
    ):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        save = kwargs.pop('save', getattr(self.args, 'save', False))
        save_txt = kwargs.pop('save_txt', getattr(self.args, 'save_txt', False))
        save_json = kwargs.pop('save_json', getattr(self.args, 'save_json', False))
        save_dir = kwargs.pop('save_dir', getattr(self.args, 'save_dir', None))
        conf = kwargs.pop('conf', None)
        iou = kwargs.pop('iou', None)
        max_det = kwargs.pop('max_det', None)
        crop = kwargs.pop('crop', getattr(self.args, 'crop', False))
        font_size = kwargs.pop('font_size', None)
        show_filename = kwargs.pop('show_filename', None)

        if save_dir is None:
            save_dir = getattr(self, 'save_dir', None)

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=stream,
            save=save,
            save_txt=save_txt,
            save_json=save_json,
            save_dir=save_dir,
            conf=conf,
            iou=iou,
            max_det=max_det,
            crop=crop,
            font_size=font_size,
            show_filename=show_filename,
            **kwargs
        )

    def predict_cli(self, rgb_source=None, x_source=None):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=False,
            save=True,
            save_txt=getattr(self.args, 'save_txt', False),
            save_json=getattr(self.args, 'save_json', False),
            save_dir=getattr(self, 'save_dir', None)
        )


class YOLOMMPosePredictor(BasePredictor):
    """
    YOLOMM Pose 推理适配器（BasePredictor -> MultiModalPosePredictor 桥接）
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.predictor = None

    def setup_model(self, model, verbose=True, debug=False):
        super().setup_model(model, verbose)

        self.predictor = MultiModalPosePredictor(
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
            LOGGER.info("YOLOMMPosePredictor: Pose推理引擎已初始化")

    def __call__(
        self,
        rgb_source=None,
        x_source=None,
        stream: bool = False,
        **kwargs
    ):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        save = kwargs.pop('save', getattr(self.args, 'save', False))
        save_txt = kwargs.pop('save_txt', getattr(self.args, 'save_txt', False))
        save_json = kwargs.pop('save_json', getattr(self.args, 'save_json', False))
        save_dir = kwargs.pop('save_dir', getattr(self.args, 'save_dir', None))
        conf = kwargs.pop('conf', None)
        iou = kwargs.pop('iou', None)
        max_det = kwargs.pop('max_det', None)
        crop = kwargs.pop('crop', getattr(self.args, 'crop', False))
        font_size = kwargs.pop('font_size', None)
        show_filename = kwargs.pop('show_filename', None)

        if save_dir is None:
            save_dir = getattr(self, 'save_dir', None)

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=stream,
            save=save,
            save_txt=save_txt,
            save_json=save_json,
            save_dir=save_dir,
            conf=conf,
            iou=iou,
            max_det=max_det,
            crop=crop,
            font_size=font_size,
            show_filename=show_filename,
            **kwargs
        )

    def predict_cli(self, rgb_source=None, x_source=None):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=False,
            save=True,
            save_txt=getattr(self.args, 'save_txt', False),
            save_json=getattr(self.args, 'save_json', False),
            save_dir=getattr(self, 'save_dir', None)
        )


class YOLOMMClassifyPredictor(BasePredictor):
    """
    YOLOMM Classify 推理适配器（BasePredictor -> MultiModalClassifyPredictor 桥接）
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.predictor = None

    def setup_model(self, model, verbose=True, debug=False):
        super().setup_model(model, verbose)

        self.predictor = MultiModalClassifyPredictor(
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
            LOGGER.info("YOLOMMClassifyPredictor: 分类推理引擎已初始化")

    def __call__(
        self,
        rgb_source=None,
        x_source=None,
        stream: bool = False,
        **kwargs
    ):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        save = kwargs.pop('save', getattr(self.args, 'save', False))
        save_txt = kwargs.pop('save_txt', getattr(self.args, 'save_txt', False))
        save_json = kwargs.pop('save_json', getattr(self.args, 'save_json', False))
        save_dir = kwargs.pop('save_dir', getattr(self.args, 'save_dir', None))
        crop = kwargs.pop('crop', getattr(self.args, 'crop', False))
        font_size = kwargs.pop('font_size', None)
        show_filename = kwargs.pop('show_filename', None)

        if save_dir is None:
            save_dir = getattr(self, 'save_dir', None)

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=stream,
            save=save,
            save_txt=save_txt,
            save_json=save_json,
            save_dir=save_dir,
            crop=crop,
            font_size=font_size,
            show_filename=show_filename,
            **kwargs
        )

    def predict_cli(self, rgb_source=None, x_source=None):
        if self.predictor is None:
            raise RuntimeError("模型未初始化，请先调用 setup_model()")

        return self.predictor(
            rgb_source=rgb_source,
            x_source=x_source,
            stream=False,
            save=True,
            save_txt=getattr(self.args, 'save_txt', False),
            save_json=getattr(self.args, 'save_json', False),
            save_dir=getattr(self, 'save_dir', None)
        )


__all__ = [
    'YOLOMMPredictor',
    'YOLOMMSegPredictor',
    'YOLOMMSemSegPredictor',
    'YOLOMMOBBPredictor',
    'YOLOMMPosePredictor',
    'YOLOMMClassifyPredictor',
]
