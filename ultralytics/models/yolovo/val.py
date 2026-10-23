import json

from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import (
    de_parallel,
    maybe_log_multimodal_canonical_summary,
    select_device,
    set_multimodal_complexity_owner_state,
    smart_inference_mode,
)


class YOLOVOValidator(DetectionValidator):
    """
    YOLOVO open-vocabulary detection validator.

        v1 has no X modality -- no modality parsing, ablation, or multi-channel
        visualization needed.  The addition over DetectionValidator is preparing
        the full class-name text cache before eval warmup and metric collection.

        Val loss is handled by ``YOLOVOModel.loss()`` which returns zeros in eval
        mode.

    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)

    @staticmethod
    def _normalize_names(names):
        """Return names as a contiguous class-id dictionary."""
        if names is None:
            return {}
        if isinstance(names, dict):
            def sort_key(k):
                try:
                    return 0, int(k)
                except (TypeError, ValueError):
                    return 1, str(k)

            return {i: names[k] for i, k in enumerate(sorted(names, key=sort_key))}
        return {i: name for i, name in enumerate(names)}

    @staticmethod
    def _inner_yolovo_model(model):
        """Return a YOLOVOModel-like object without confusing it with its Sequential body."""
        candidate = de_parallel(model)
        if hasattr(candidate, "set_input_text"):
            return candidate

        # Only AutoBackend(PyTorch) stores the real PyTorch model on `.model`.
        if isinstance(candidate, AutoBackend) or getattr(candidate, "pt", False):
            inner = getattr(candidate, "model", None)
            if inner is not None:
                inner = de_parallel(inner)
                if hasattr(inner, "set_input_text"):
                    return inner
        return None

    def _prepare_text_cache(self, model):
        """Encode validation class names into the underlying YOLOVOModel cache."""
        inner = self._inner_yolovo_model(model)
        data = getattr(self, "data", None) or {}
        names = data.get("names", getattr(inner or model, "names", None))
        if not names:
            raise ValueError("YOLOVO validation requires class names in data.yaml.")
        names = self._normalize_names(names)
        texts = list(names.values())
        if inner is None:
            model_names = self._normalize_names(getattr(model, "names", {}))
            if model_names != names:
                raise ValueError(
                    "YOLOVO validation cannot update text prompts for this non-PyTorch backend. "
                    "Use a .pt model or export with class names matching data.yaml."
                )
            return None

        inner.set_input_text(texts)
        if hasattr(model, "names"):
            model.names = inner.names
        return inner

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Execute validation, preparing YOLOVO text cache before external warmup."""
        if trainer is not None:
            return super().__call__(trainer=trainer, model=model)

        self.training = False
        augment = self.args.augment
        if str(self.args.model).endswith(".yaml") and model is None:
            LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
        callbacks.add_integration_callbacks(self)
        model = AutoBackend(
            weights=model or self.args.model,
            device=select_device(self.args.device, self.args.batch),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
        )
        self.device = model.device
        self.args.half = model.fp16
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_imgsz(self.args.imgsz, stride=stride)
        if engine:
            self.args.batch = model.batch_size
        elif not (pt or jit or getattr(model, "dynamic", False)):
            self.args.batch = model.metadata.get("batch", 1)
            LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

        if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
            self.data = check_det_dataset(self.args.data)
        else:
            raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0
        if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
            self.args.rect = False
        self.stride = model.stride
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

        model.eval()
        self._prepare_text_cache(model)
        set_multimodal_complexity_owner_state(model, direct_ingress=True)
        maybe_log_multimodal_canonical_summary(model, imgsz=imgsz, logger=LOGGER, direct_ingress=True)
        data_channels = int(self.data.get("channels", 3))
        model_channels = int(getattr(model, "ch", data_channels))
        warmup_channels = model_channels or data_channels
        if warmup_channels != data_channels:
            LOGGER.warning(
                f"Warmup通道数使用模型channels={warmup_channels} (data.yaml channels={data_channels})。"
                f"请确保数据加载输出通道与模型输入通道一致。"
            )
        model.warmup(imgsz=(1 if pt else self.args.batch, warmup_channels, imgsz, imgsz))

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            with dt[0]:
                batch = self.preprocess(batch)
            with dt[1]:
                preds = model(batch["img"], augment=augment)
            with dt[2]:
                pass
            with dt[3]:
                preds = self.postprocess(preds)
            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            self.run_callbacks("on_val_batch_end")

        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        LOGGER.info(
            "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                *tuple(self.speed.values())
            )
        )
        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                LOGGER.info(f"Saving {f.name}...")
                json.dump(self.jdict, f)
            stats = self.eval_json(stats)
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats

    def init_metrics(self, model):
        """Initialize metrics and encode full class names as text vectors.

        After ``DetectionValidator.init_metrics`` sets ``self.names``, encode
        the complete class-name list via ``model.set_input_text`` so that the
        eval-mode forward pass has the correct text embeddings cached.
        """
        self._prepare_text_cache(model)
        super().init_metrics(model)

    # v2 (has_x=True) needs build_dataset override to use YOLOVODataset with
    # mm_transforms.  v1 uses DetectionValidator.build_dataset → build_yolo_dataset
    # which returns YOLODataset -- functionally identical for v1 (augment=False).
