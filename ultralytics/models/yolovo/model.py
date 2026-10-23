from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import YOLOVOModel


def _configure_mobileclip_vendor_path():
    """Make vendored MobileCLIP importable before unpickling YOLOVO checkpoints."""
    import sys

    vendor_dir = Path(__file__).resolve().parent / "text_tower" / "mobileclip" / "_vendor"
    vendor_path = str(vendor_dir)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)


class YOLOVO(Model):
    """
    YOLOVO open-vocabulary multi-modal detection entry point.

        YOLOVO combines RGB detection with a MobileCLIP text tower to enable
        open-vocabulary object detection. Users provide text prompts describing
        target classes, and the model detects corresponding objects in images.

        Examples:
            Basic usage with text prompts::

                >>> model = YOLOVO("yolovo11n.yaml")
                >>> results = model.predict("image.jpg", text=["person", "car", "dog"])

    """

    def __init__(self, model="yolovo11n.yaml", task=None, verbose=False):
        # P1-3: 强制 task 为 detect，不支持的任务提前报错
        if task is not None and task != "detect":
            raise ValueError(f"YOLOVO currently only supports 'detect' task, got '{task}'")
        if isinstance(model, (str, Path)) and str(model).endswith(".pt"):
            _configure_mobileclip_vendor_path()
        super().__init__(model, task="detect", verbose=verbose)

    def predict(self, source, text=None, **kwargs):
        """Run prediction with optional YOLOVO text prompts.

        ``text`` only applies to this predict call. The wrapper passes it to
        YOLOVOPredictor through the ``prompts`` hook; predictor owns text cache
        setup and cleanup around warmup/inference.

        Args:
            source: Image path, numpy array, or other supported source.
            text: List of class name strings, e.g. ``["person", "car"]``.
            **kwargs: Additional arguments forwarded to ``Model.predict``.

        Returns:
            List of Results or generator if ``stream=True``.
        """
        if text is not None:
            if "prompts" in kwargs:
                raise ValueError("YOLOVO.predict() 请使用 text 参数，不要同时传入 prompts。")
            if isinstance(text, str):
                prompts = [text]
            else:
                prompts = list(text)
            if not prompts:
                raise ValueError("YOLOVO.predict(text=...) 至少需要一个文本 prompt。")
            kwargs["prompts"] = prompts
        return super().predict(source, **kwargs)

    def export(self, **kwargs):
        """Export model to ONNX/other formats with text features baked in."""
        self.model.prepare_export(kwargs.get("imgsz", 640))
        try:
            return super().export(**kwargs)
        finally:
            self.model.cleanup_export()

    @property
    def task_map(self):
        """Map tasks to their corresponding model, trainer, validator, and predictor classes."""
        from ultralytics.models.yolovo.predict import YOLOVOPredictor
        from ultralytics.models.yolovo.train import YOLOVOTrainer
        from ultralytics.models.yolovo.val import YOLOVOValidator

        return {
            "detect": {
                "model": YOLOVOModel,
                "predictor": YOLOVOPredictor,
                "trainer": YOLOVOTrainer,
                "validator": YOLOVOValidator,
            },
        }
