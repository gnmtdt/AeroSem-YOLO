from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils.torch_utils import de_parallel


class YOLOVOPredictor(DetectionPredictor):
    """
    YOLOVO 开放词汇检测预测器。

        本类只负责 YOLOVO 的 text prompt 生命周期；图像预处理、模型调用、
        NMS、Results 构造继续复用 DetectionPredictor。

        职责边界：
        - predict 调用开始后、warmup 前准备 text cache；
        - 单图、列表、文件夹、stream 连续预测期间复用同一份 cache；
        - 本次 predict 调用结束后清理由 predictor 创建的 text cache。

    """

    def __init__(self, *args, **kwargs):
        """初始化 YOLOVO predictor 的一次性 prompt 状态。"""
        super().__init__(*args, **kwargs)
        self.prompts = None
        self._owns_text_cache = False
        self._previous_inner_names = None
        self._previous_backend_names = None

    @staticmethod
    def _normalize_prompts(prompts):
        """将 prompts 规范化为 list[str]；None 表示未显式传入。"""
        if prompts is None:
            return None
        if isinstance(prompts, str):
            normalized = [prompts]
        else:
            normalized = list(prompts)
        if not normalized:
            raise ValueError("YOLOVO predict prompts 不能为空。")
        return normalized

    def set_prompts(self, prompts):
        """接收 Model.predict(prompts=...) 传下来的本次预测 prompts。"""
        self.prompts = self._normalize_prompts(prompts)

    @staticmethod
    def _copy_names(names):
        """复制 names 容器，避免本次 prompts 污染后续预测状态。"""
        if isinstance(names, dict):
            return dict(names)
        if isinstance(names, list):
            return list(names)
        return names

    def _inner_yolovo_model(self):
        """返回 AutoBackend 内部的 YOLOVOModel，并校验接口存在。"""
        inner = getattr(self.model, "model", self.model)
        inner = de_parallel(inner)
        if not hasattr(inner, "set_input_text"):
            raise TypeError(f"YOLOVOPredictor requires a YOLOVOModel, got {type(inner).__name__}")
        return inner

    def _prepare_text_cache(self):
        """在 warmup 前准备 text cache。

        状态规则：
        - 有 self.prompts：本次 predict 使用用户传入 prompts；
        - 没有 prompts 但模型已有 text cache：尊重已有 cache，不接管清理；
        - 没有 prompts 且没有 cache：调用 set_input_text()，让模型从
          class_texts 或 names 生成默认文本，并在本次结束后清理。
        """
        inner = self._inner_yolovo_model()
        self._owns_text_cache = False
        self._previous_inner_names = None
        self._previous_backend_names = None
        if self.prompts is not None:
            self._previous_inner_names = self._copy_names(getattr(inner, "names", None))
            self._previous_backend_names = self._copy_names(getattr(self.model, "names", None))
            inner.set_input_text(self.prompts)
            self._owns_text_cache = True
        elif not inner.has_input("text"):
            self._previous_inner_names = self._copy_names(getattr(inner, "names", None))
            self._previous_backend_names = self._copy_names(getattr(self.model, "names", None))
            inner.set_input_text()
            self._owns_text_cache = True
        self.model.names = inner.names

    def _clear_text_cache(self):
        """清理本次 predictor 创建的 text cache，并重置一次性 prompts。"""
        try:
            if self._owns_text_cache:
                inner = self._inner_yolovo_model()
                inner.input_cache.pop("text", None)
                if self._previous_inner_names is not None:
                    inner.names = self._previous_inner_names
                if self._previous_backend_names is not None:
                    self.model.names = self._previous_backend_names
        finally:
            self._owns_text_cache = False
            self._previous_inner_names = None
            self._previous_backend_names = None
            self.prompts = None

    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """执行 YOLOVO 预测流，确保 warmup 前已有 text cache，结束后清理。"""
        if not self.model:
            self.setup_model(model)
        self._prepare_text_cache()
        try:
            yield from super().stream_inference(source, model, *args, **kwargs)
        finally:
            self._clear_text_cache()
