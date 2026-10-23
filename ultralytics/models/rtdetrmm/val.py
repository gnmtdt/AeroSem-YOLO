# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
RT-DETR MultiModal validator module.

This module provides the RTDETRMMValidator class for validating multi-modal RT-DETR models
with support for RGB+X modality inputs.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.utils import LOGGER, colorstr, TQDM
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.ops import Profile
from copy import copy
from ultralytics.nn.mm.complexity import (
    compute_default_multimodal_complexity_report,
    log_default_complexity,
)
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    get_modality_path_from_data,
    get_runtime_x_modality_name,
    normalize_modality_token,
    resolve_requested_modality_token,
    resolve_modality_pair,
)
from ultralytics.engine.afss.tasks.detect import build_detect_afss_score_row


class RTDETRMMValidator(DetectionValidator):
    _tid = "debugger"
    """
    A validator class for RT-DETR MultiModal (RTDETRMM) object detection models.

    This class extends RTDETRValidator to support multi-modal inputs (RGB + X modality)
    during the validation process. It handles evaluation metrics and performance assessment
    for multi-modal RT-DETR models.

    Attributes:
        args: Validation arguments and settings.
        model: The RTDETRMM model being validated.
        dataloader: Multi-modal validation dataloader.
        metrics: Validation metrics for multi-modal detection.

    Methods:
        preprocess: Preprocess batch data for multi-modal validation.
        init_metrics: Initialize metrics for multi-modal evaluation.
        get_dataloader: Create multi-modal validation dataloader.

    Examples:
        >>> validator = RTDETRMMValidator(args={'data': 'multimodal-dataset.yaml'})
        >>> validator(model=rtdetrmm_model)
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize RTDETRMMValidator for multi-modal validation.

        Args:
            dataloader: Multi-modal validation dataloader.
            save_dir (Path, optional): Directory to save validation results.
            pbar (tqdm, optional): Progress bar for validation.
            args (SimpleNamespace): Validation configuration arguments.
            _callbacks (list, optional): List of callback functions.
        """
        # Note: pbar 参数仅用于接口兼容，不传递给父类
        super().__init__(dataloader, save_dir, args, _callbacks)

        # Detect multimodal mode
        self.is_multimodal = self._detect_multimodal_mode()

        # Get modality parameter from args (consistent with trainer)
        if args:
            if isinstance(args, dict):
                self.modality = args.get('modality', None)
            else:
                self.modality = getattr(args, 'modality', None)
        else:
            self.modality = None

        # 仅对 rgb/x token 做归一化：rgb/RGB→RGB、x/X→X（其它模态名保持原样）
        self.modality = normalize_modality_token(self.modality)
        # 回写 args/self.args，确保训练内 copy(args) 与后续读取一致
        if args is not None:
            if isinstance(args, dict):
                args["modality"] = self.modality
            else:
                setattr(args, "modality", self.modality)
        if hasattr(self, "args") and self.args is not None:
            if isinstance(self.args, dict):
                self.args["modality"] = self.modality
            else:
                setattr(self.args, "modality", self.modality)

        # Initialize multimodal configuration (parsed later when data is available)
        self.multimodal_config = None

        # # Log initialization status
        # if self.is_multimodal:
        #     LOGGER.info(f"{colorstr('RTDETRMMValidator')}: 多模态验证模式已启用")
        #     if self.modality:
        #         LOGGER.info(f"单模态消融验证: {colorstr(self.modality)}")
        #     else:
        #         LOGGER.info("双模态验证模式")
        # RTDETRMMValidator 为多模态家族专用，严格 Fail-Fast（不做降级）

    def _detect_multimodal_mode(self) -> bool:
        """
        Detect if this is a multimodal validation session.
        
        RTDETRMMValidator is designed for multi-modal validation only.
        Always return True for data-driven configuration.

        Returns:
            bool: True (always enable multimodal mode)
        """
        return True

    @property
    def is_dual_modal(self) -> bool:
        """Check if in dual-modal mode"""
        return self.is_multimodal and self.modality is None

    @property
    def is_single_modal(self) -> bool:
        """Check if in single-modal mode"""
        return self.is_multimodal and self.modality is not None

    def _get_non_distill_loss_names(self, trainer):
        """Return validation loss names with all distillation-only items removed."""
        _DISTILL_PREFIXES = ('distill_', 'd_out', 'd_feat')
        return [
            name
            for name in getattr(trainer, 'loss_names', ())
            if not any(str(name).startswith(p) for p in _DISTILL_PREFIXES)
        ]

    def _label_val_loss_items(self, loss_items: torch.Tensor, prefix: str = "val"):
        """Build a labeled validation loss dict without relying on trainer.loss_names shape."""
        keys = [f"{prefix}/{x}" for x in getattr(self, '_val_loss_names', ())]
        values = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, values))

    def _get_complexity_flag(self, name: str, default: bool = False) -> bool:
        """Read complexity handshake flag from model or trainer args (private attrs only)."""
        from ultralytics.utils.torch_utils import (
            MM_COMPLEXITY_OWNER_EMITTED, MM_COMPLEXITY_OWNER_PENDING, MM_COMPLEXITY_DIRECT_INGRESS,
        )
        flag_map = {
            "mm_complexity_owner_emitted": MM_COMPLEXITY_OWNER_EMITTED,
            "mm_complexity_owner_pending": MM_COMPLEXITY_OWNER_PENDING,
            "mm_complexity_direct_ingress": MM_COMPLEXITY_DIRECT_INGRESS,
        }
        attr = flag_map.get(name, f"_mm_{name}")
        # 优先从 model/trainer 读私有属性
        for source in (getattr(self, 'model', None), getattr(self, 'trainer', None)):
            if source is not None:
                val = getattr(source, attr, None)
                if val is not None:
                    return bool(val)
        return default

    def _set_complexity_flag(self, name: str, value: bool) -> None:
        """Write complexity handshake flag to model (private attr only, never to args dict)."""
        from ultralytics.utils.torch_utils import (
            MM_COMPLEXITY_OWNER_EMITTED, MM_COMPLEXITY_OWNER_PENDING, MM_COMPLEXITY_DIRECT_INGRESS,
        )
        flag_map = {
            "mm_complexity_owner_emitted": MM_COMPLEXITY_OWNER_EMITTED,
            "mm_complexity_owner_pending": MM_COMPLEXITY_OWNER_PENDING,
            "mm_complexity_direct_ingress": MM_COMPLEXITY_DIRECT_INGRESS,
        }
        attr = flag_map.get(name, f"_mm_{name}")
        for target in (getattr(self, 'model', None), getattr(self, 'trainer', None)):
            if target is not None:
                setattr(target, attr, value)

    @staticmethod
    def _unwrap_complexity_model(model: Any) -> Any:
        """Prefer the underlying PyTorch model for complexity accounting when AutoBackend wraps it."""
        if hasattr(model, "pt") and getattr(model, "pt", False) and hasattr(model, "model"):
            return getattr(model, "model", None) or model
        return model

    def _maybe_log_direct_ingress_complexity(self, model: Any, imgsz: int) -> None:
        """Own the human-visible summary only when validation bypasses model-level owners."""
        owner_emitted = self._get_complexity_flag("mm_complexity_owner_emitted", default=False)
        owner_pending = self._get_complexity_flag("mm_complexity_owner_pending", default=False)
        direct_ingress = self._get_complexity_flag("mm_complexity_direct_ingress", default=not self.training)
        if owner_emitted or owner_pending or not direct_ingress:
            return

        complexity_model = self._unwrap_complexity_model(model)
        if complexity_model is None:
            return

        try:
            report = compute_default_multimodal_complexity_report(complexity_model, imgsz=imgsz)
            log_default_complexity(complexity_model, report, LOGGER)
        except Exception as e:
            LOGGER.warning(f"RTDETRMMValidator: 复杂度摘要输出失败（可忽略，不影响验证）：{e}")
            return

        self._set_complexity_flag("mm_complexity_owner_emitted", True)
        self._set_complexity_flag("mm_complexity_owner_pending", False)
        self._set_complexity_flag("mm_complexity_direct_ingress", False)

    def __call__(self, trainer=None, model=None):
        """
        执行多模态RT-DETR验证过程，支持动态通道数warmup。
        
        重写基类方法以支持动态通道数warmup和多模态数据处理。
        参考YOLOMM的实现但改进为支持动态Xch配置。
        
        Args:
            trainer: Training instance (if called during training)
            model: Model to validate (if called independently)
            
        Returns:
            dict: Validation metrics
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        
        if self.training:
            self.device = trainer.device
            # 关键修复：从trainer获取data配置（包含更新后的channels）
            if self.data is None:
                self.data = trainer.data
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self._val_loss_names = self._get_non_distill_loss_names(trainer)
            self.loss = torch.zeros(len(self._val_loss_names), device=trainer.device, dtype=trainer.loss_items.dtype)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
            # 注入runtime模态参数到路由器（训练态）
            if hasattr(model, 'mm_router') and model.mm_router and self.modality:
                model.mm_router.set_runtime_params(
                    self.modality,
                    strategy=getattr(self.args, 'ablation_strategy', None),
                    seed=getattr(self.args, 'seed', None),
                )
        else:
            # 独立验证模式：使用传入的模型或加载模型
            from ultralytics.utils import callbacks, emojis
            from ultralytics.utils.checks import check_imgsz
            from ultralytics.nn.autobackend import AutoBackend
            from ultralytics.utils.torch_utils import select_device
            from ultralytics.data.utils import check_det_dataset
            
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)
                LOGGER.info(f"Setting batch={self.args.batch} for RT-DETR validation")

            self._maybe_log_direct_ingress_complexity(model, int(imgsz))

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                # 如果没有data配置，从args.data加载
                if not hasattr(self, 'data') or self.data is None:
                    self.data = check_det_dataset(self.args.data)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            # 注入runtime模态参数到路由器（仅在PyTorch后端）
            try:
                if hasattr(model, 'pt') and model.pt and hasattr(model, 'model') and hasattr(model.model, 'mm_router') and model.model.mm_router and self.modality:
                    model.model.mm_router.set_runtime_params(
                        self.modality,
                        strategy=getattr(self.args, 'ablation_strategy', None),
                        seed=getattr(self.args, 'seed', None),
                    )
            except Exception:
                pass
            
            # Dynamic channel warmup based on data configuration
            if hasattr(self, 'data') and self.data and 'Xch' in self.data:
                x_channels = self.data.get('Xch', 3)
                total_channels = 3 + x_channels
                LOGGER.info(f"执行{total_channels}通道多模态RT-DETR模型warmup (RGB:3 + X:{x_channels})")
                model.warmup(imgsz=(1 if pt else self.args.batch, total_channels, imgsz, imgsz))
            else:
                # Use 6-channel default for backward compatibility
                LOGGER.info("执行6通道多模态RT-DETR模型warmup (默认)")
                model.warmup(imgsz=(1 if pt else self.args.batch, 6, imgsz, imgsz))

        # 继续执行标准验证流程
        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    loss_items = model.loss(batch, preds)[1]
                    if loss_items.numel() != len(self._val_loss_names):
                        raise RuntimeError(
                            f"Validation loss dimension mismatch: got {loss_items.numel()} items from model.loss(), "
                            f"but validator expects {len(self._val_loss_names)} non-distill items {self._val_loss_names}."
                        )
                    self.loss += loss_items

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader) * 1E3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **self._label_val_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}
        else:
            LOGGER.info("Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image" %
                       tuple(self.speed.values()))
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build multi-modal dataset using YOLOMultiModalImageDataset.

        This method follows YOLOMM's successful pattern by overriding build_dataset
        to ensure proper initialization timing and data access for validation.

        Args:
            img_path (str): Path to images
            mode (str): Dataset mode ('val', 'test')
            batch (int, optional): Batch size for validation

        Returns:
            Dataset: YOLOMultiModalImageDataset for multi-modal, standard dataset otherwise
        """
        if not self.is_multimodal:
            # Fall back to standard RT-DETR dataset
            return super().build_dataset(img_path, mode, batch)

        # Get model stride parameter (consistent with DetectionValidator)
        stride = getattr(self, 'stride', 32)
        if hasattr(self, 'model') and self.model:
            stride = max(int(de_parallel(self.model).stride.max() if hasattr(self.model, 'stride') else 0), 32)

        # Lazy loading: parse multi-modal configuration on demand
        if self.multimodal_config is None:
            self.multimodal_config = self._parse_multimodal_config()
            LOGGER.info(f"多模态验证配置解析完成 - 模态: {self.multimodal_config['models']}")

        # Use parsed modality configuration
        modalities = self.multimodal_config['models']
        modalities_dict = self.multimodal_config['modalities']

        # 获取X模态信息（关键修复）
        _rgb_modality, x_modality = resolve_modality_pair(modalities, strict=True)
        x_modality_dir = modalities_dict.get(x_modality) if x_modality else None

        LOGGER.info(f"构建多模态验证数据集 - 模式: {mode}, 路径: {img_path}, 模态: {modalities}")

        # If single-modal validation is enabled, log modality info and validate compatibility
        if self.modality:
            self._validate_modality_compatibility()
            LOGGER.info(f"启用单模态验证: {self.modality}-only，将应用智能模态填充")

        # Call build_yolo_dataset with multi_modal_image=True to enable multi-modal dataset
        return build_yolo_dataset(
            self.args, img_path, batch, self.data,
            mode=mode,
            rect=False,  # RT-DETR validation uses fixed-shape inference (align with RTDETRValidator)
            stride=stride,
            multi_modal_image=True,  # Key parameter: enable YOLOMultiModalImageDataset
            x_modality=x_modality,  # Pass X modality type
            x_modality_dir=x_modality_dir,  # Pass X modality directory
            modalities=modalities,  # Pass modality configuration (backward compatibility)
            precision_report_dir=self.save_dir,
        )

    def postprocess(self, preds):
        """
        RT-DETR 输出后处理（独立拷贝版，避免依赖 ultralytics.models.rtdetr.*）。

        Args:
            preds (torch.Tensor | list | tuple): 原始预测输出。

        Returns:
            List[Dict[str, torch.Tensor]]: 每张图像的预测字典，包含 bboxes/conf/cls。
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        # RT-DETR bbox 默认是归一化 xywh（0-1）。如果输入不是方形（如 rect=True），必须用实际 (h,w) 缩放。
        # 这里优先使用 preprocess() 记录的本批次推理尺寸，避免只按单一 imgsz 缩放导致 bbox 严重错位。
        imgsz_hw = getattr(self, "_imgsz_hw", None)
        if imgsz_hw is not None and len(imgsz_hw) == 2:
            h, w = int(imgsz_hw[0]), int(imgsz_hw[1])
            scale = torch.tensor([w, h, w, h], device=bboxes.device, dtype=bboxes.dtype)
            bboxes = bboxes * scale
        else:
            bboxes = bboxes * float(self.args.imgsz)

        # outputs: list[Tensor(N,6)] where 6=(x1,y1,x2,y2,conf,cls)
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        from ultralytics.utils import ops as _ops

        for i, bbox in enumerate(bboxes):  # (num_queries, 4)
            bbox = _ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (num_queries,)
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred[score > self.args.conf]

        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]} for x in outputs]

    def init_metrics(self, model):
        """Initialize metrics and capture runtime X modality name from loaded weights."""
        super().init_metrics(model)
        self.x_modality_name = get_runtime_x_modality_name(model)

    def preprocess(self, batch):
        """
        Preprocess batch data for multi-modal validation.

        Uses shared dtype policy instead of inheriting blanket all-channel /255
        from the parent DetectionValidator. Ensures consistency with the training
        phase preprocessing pipeline.

        Args:
            batch (dict): Batch data containing images and labels

        Returns:
            dict: Preprocessed batch data
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype

        if not self.is_multimodal:
            # Non-multimodal: standard preprocessing
            batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255.0
        else:
            # Multimodal: contract-driven preprocessing
            # legacy+legacy 恢复 29e006c^; mixed 按 per-modality 处理
            img = batch["img"].float()
            img = apply_dataset_precision_contract(img, extract_batch_precision_contract(batch), raw_dtype=raw_dtype)
            batch["img"] = img
            if self.args.half:
                from ultralytics.nn.mm.utils import check_tensor_dtype_for_half
                check_tensor_dtype_for_half(batch["img"], context="RTDETRMM validation preprocess")
                batch["img"] = batch["img"].half()

        for k in ("batch_idx", "cls", "bboxes"):
            batch[k] = batch[k].to(self.device)

        # Multi-modal specific preprocessing
        if self.is_multimodal and "img" in batch:
            try:
                self._imgsz_hw = tuple(batch["img"].shape[2:4])
            except Exception:
                self._imgsz_hw = None

            x_channels = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
            expected_channels = 3 + x_channels

            if batch["img"].shape[1] == expected_channels:
                pass  # 模态消融由路由层统一完成
            elif batch["img"].shape[1] == 3:
                pass  # Standard 3-channel
            else:
                LOGGER.warning(f"意外的通道数: {batch['img'].shape[1]}，期望{expected_channels}或3通道")

        return batch

    def _apply_modality_ablation(self, batch):
        """
        Apply single-modality ablation by zeroing out non-selected modality channels.
        
        Supports dynamic channel configuration based on data['Xch'].

        Args:
            batch (dict): Batch data to modify
        """
        if not self.modality:
            return

        images = batch["img"]  # [B, total_channels, H, W]
        x_channels = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        total_channels = 3 + x_channels

        if images.shape[1] != total_channels:
            LOGGER.warning(f"Channel mismatch: expected {total_channels}, got {images.shape[1]}")
            return

        if self.modality == "RGB":
            # Zero out X modality (channels 3:3+Xch)
            images[:, 3:3+x_channels, :, :] = 0
            LOGGER.debug(f"单模态RGB验证: X模态通道({x_channels}ch)已置零")
        else:
            # Zero out RGB modality (channels 0:3)
            images[:, 0:3, :, :] = 0
            LOGGER.debug(f"单模态{self.modality}验证: RGB通道已置零")

        batch["img"] = images

    def _parse_multimodal_config(self):
        """
        Parse and validate multi-modal configuration from data.yaml.

        Uses safe data access pattern from YOLOMM to avoid AttributeError during initialization.
        Maintains consistency with RTDETRMMTrainer configuration parsing.

        Returns:
            dict: Parsed multi-modal configuration
        """
        # Safe data access - use getattr to avoid AttributeError during initialization
        data = getattr(self, 'data', None)

        # Priority 1: User-specified modality parameter (single-modal validation)
        if self.modality:
            rgb_modality, x_modality = resolve_modality_pair(data, default_secondary=self._determine_x_modality_from_data())
            resolved_modality = resolve_requested_modality_token(
                self.modality, [rgb_modality, x_modality], default_x=x_modality, strict=True
            )
            config = {
                'models': [rgb_modality, x_modality],
                'modalities': {
                    rgb_modality: get_modality_path_from_data(data, rgb_modality),
                    x_modality: self._get_x_modality_path(x_modality),
                }
            }
            LOGGER.info(
                "Single-modal validation bridge complete: requested=%s -> resolved=%s, pair=%s",
                self.modality,
                resolved_modality,
                config['models'],
            )
            return config

        # Priority 2: Dual-modal validation (use original configuration parsing logic)
        return self._get_default_multimodal_config()

    def _get_default_multimodal_config(self):
        """
        Get default multi-modal configuration from data.yaml.

        Returns:
            dict: Default multi-modal configuration.
        """
        # Check if data is available
        data = getattr(self, 'data', None)

        # Priority 1: modality_used field
        if data and 'modality_used' in data:
            modality_used = data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                rgb_modality, x_modality = resolve_modality_pair(modality_used, strict=True)
                config = {
                    'models': [rgb_modality, x_modality],
                    'modalities': {
                        rgb_modality: get_modality_path_from_data(data, rgb_modality),
                        x_modality: get_modality_path_from_data(data, x_modality),
                    }
                }

                LOGGER.info(f"Loaded multi-modal validation config: {modality_used}")
                return config

        # Priority 2: models field (backward compatibility)
        if data and 'models' in data:
            models = data['models']
            if isinstance(models, list) and len(models) >= 2:
                rgb_modality, x_modality = resolve_modality_pair(models, strict=True)
                config = {
                    'models': [rgb_modality, x_modality],
                    'modalities': {
                        rgb_modality: get_modality_path_from_data(data, rgb_modality),
                        x_modality: get_modality_path_from_data(data, x_modality),
                    }
                }
                return config

        # Determine default modality
        x_modality = self._determine_x_modality_from_data()
        rgb_modality, x_modality = resolve_modality_pair(data, default_secondary=x_modality)
        config = {
            'models': [rgb_modality, x_modality],
            'modalities': {
                rgb_modality: get_modality_path_from_data(data, rgb_modality),
                x_modality: get_modality_path_from_data(data, x_modality)
            }
        }
        LOGGER.info(f"Using default multi-modal validation config: rgb+{x_modality}")
        return config

    def _get_x_modality_path(self, modality_name):
        """
        获取指定模态的实际路径。
        
        优先从data.yaml的modality字段读取，
        如果不存在则使用默认格式 'images_{modality_name}'。
        
        Args:
            modality_name (str): 模态名称（如 'ir', 'depth', 'thermal'）
            
        Returns:
            str: 模态对应的目录路径
        """
        # 优先从data.yaml的modality字段读取
        data = getattr(self, 'data', None)
        if data and 'modality' in data:
            modality_paths = data['modality']
            if isinstance(modality_paths, dict) and modality_name in modality_paths:
                return modality_paths[modality_name]
        
        # 向后兼容：检查modalities字段
        if data and 'modalities' in data:
            modalities = data['modalities']
            if isinstance(modalities, dict) and modality_name in modalities:
                return modalities[modality_name]
        
        return get_modality_path_from_data(data, modality_name)

    def _determine_x_modality_from_data(self):
        """
        Intelligently determine X modality type from data configuration.

        Returns:
            str: X modality identifier (e.g., 'depth', 'thermal', 'ir').
        """
        # Safe data access - use getattr to avoid AttributeError during initialization
        data = getattr(self, 'data', None)

        # Check data.yaml for modality information
        if data:
            # Check modality_used
            if 'modality_used' in data:
                modality_used = data['modality_used']
                x_modality = extract_x_modality_name_from_data(data)
                if x_modality != 'unknown':
                    LOGGER.info(f"从数据配置解析X模态: {x_modality}")
                    return x_modality

        # Use depth as default when undetermined
        LOGGER.warning("无法自动确定X模态类型，使用默认值: depth")
        return 'depth'

    def _validate_modality_compatibility(self):
        """
        Validate compatibility between user-specified modality parameter and data configuration.

        Raises:
            ValueError: When modality parameter is incompatible with available data
        """
        if not self.modality:
            return

        # Get available modalities
        available_modalities = []
        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            available_modalities = self.multimodal_config.get('models', [])
        elif hasattr(self, 'data') and self.data and 'models' in self.data:
            available_modalities = self.data['models']

        # Validate modality compatibility
        if available_modalities:
            try:
                resolved = resolve_requested_modality_token(
                    self.modality, available_modalities, default_x=self._determine_x_modality_from_data(), strict=True
                )
            except ValueError as exc:
                raise ValueError(
                    f"指定的验证modality '{self.modality}' 不在可用模态列表中: {available_modalities}。"
                    f"请检查数据配置或modality参数。"
                ) from exc
            LOGGER.info(f"✅ 验证模态兼容性通过: '{self.modality}' 映射到 {resolved}")
        else:
            # If unable to get available modalities, just give a warning
            LOGGER.warning(f"⚠️  无法验证modality '{self.modality}' 的兼容性，未找到可用模态配置")

    def plot_val_samples(self, batch, ni):
        """
        绘制验证样本，支持多模态可视化。
        
        使用统一的复用组件实现，遵循[RGB, X]通道顺序，
        实现RGB、X模态、多模态并排三种可视化输出。
        
        Args:
            batch (dict): 批次数据
            ni (int): 批次索引
        """
        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
            prepare_rgb_for_display,
            split_modalities, visualize_x_to_3ch, concat_side_by_side,
            duplicate_bboxes_for_side_by_side, ensure_batch_idx_long, resolve_x_modality
        )
        
        # 获取多模态图像数据
        multimodal_images = batch["img"]  # Shape: (batch, 3+Xch, H, W)
        
        # 获取动态通道数配置
        x_channels = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        
        # 拆分模态：实际通道顺序[RGB, X]
        rgb_images, x_images = split_modalities(multimodal_images, x_channels)
        precision_contract = extract_precision_contract_from_batch(batch)
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])
        
        # 获取X模态类型
        x_modality = getattr(self, 'x_modality_name', 'unknown')
        
        # 确保batch_idx类型正确
        batch_idx = ensure_batch_idx_long(batch["batch_idx"])
        batch["batch_idx"] = batch_idx
        
        # 根据验证模式决定输出
        if self.modality:
            # 单模态验证：仅输出指定模态
            if self.modality == "RGB":
                # RGB单模态
                plot_images(
                    rgb_visual,
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
            else:
                # X模态单模态
                x_visual = visualize_x_to_3ch(
                    x_images,
                    colorize=False,
                    x_modality=x_modality,
                    precision=precision_contract["x"],
                )
                plot_images(
                    x_visual,
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_labels_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
        else:
            # 双模态验证：输出三种图像
            try:
                # 1. RGB模态验证样本
                plot_images(
                    rgb_visual,
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
                # 2. X模态验证样本（默认灰度复制，不伪彩）
                x_visual = visualize_x_to_3ch(
                    x_images,
                    colorize=False,
                    x_modality=x_modality,
                    precision=precision_contract["x"],
                )
                plot_images(
                    x_visual,
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_labels_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
                # 3. 多模态并排对比图 - 使用duplicate函数为两侧绘制bbox
                side_by_side_images = concat_side_by_side(rgb_visual, x_visual)
                # 复制bbox到两侧：左半(RGB) + 右半(X)
                batch_ids_dup, cls_ids_dup, bboxes_dup, _ = duplicate_bboxes_for_side_by_side(
                    batch["batch_idx"], batch["cls"].squeeze(-1), batch["bboxes"], None
                )
                plot_images(
                    side_by_side_images,
                    batch_ids_dup,
                    cls_ids_dup,
                    bboxes_dup,
                    paths=[p.replace('.jpg', '_multimodal.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_labels_multimodal.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
            except Exception as e:
                LOGGER.warning(f"绘制{x_modality}模态验证样本失败: {e}")
    
    def plot_predictions(self, batch, preds, ni):
        """
        绘制预测结果，支持多模态可视化。
        
        统一坐标制处理：output_to_target→像素xywh→归一化xywh→绘图，
        确保并排图bbox正确缩放到左半区域。
        
        Args:
            batch (dict): 批次数据
            preds (list): 预测结果
            ni (int): 批次索引
        """
        from ultralytics.utils.plotting import plot_images, output_to_target
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
            prepare_rgb_for_display,
            split_modalities, visualize_x_to_3ch, concat_side_by_side,
            to_norm_xywh_for_plot, duplicate_bboxes_for_side_by_side, resolve_x_modality,
            ensure_batch_idx_long, clip_boxes_norm_xywh
        )
        
        # 获取多模态图像数据
        multimodal_images = batch["img"]
        
        # 获取动态通道数配置
        x_channels = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        
        # 拆分模态：实际通道顺序[RGB, X]
        rgb_images, x_images = split_modalities(multimodal_images, x_channels)
        precision_contract = extract_precision_contract_from_batch(batch)
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])
        
        # 获取X模态类型
        x_modality = getattr(self, 'x_modality_name', 'unknown')
        
        # 统一坐标制：output_to_target → 像素xywh → 归一化xywh
        batch_ids, cls_ids, boxes_xywh_px, confs = output_to_target(preds, max_det=self.args.max_det)
        
        # 获取图像尺寸用于坐标归一化
        _, _, H, W = rgb_images.shape
        img_hw = (H, W)
        
        # 转换为归一化坐标用于plot_images
        batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm = to_norm_xywh_for_plot(
            batch_ids, cls_ids, boxes_xywh_px, confs, img_hw
        )
        
        # 确保batch_idx类型正确
        batch_ids_norm = ensure_batch_idx_long(batch_ids_norm)
        # 先做几何裁剪到单图域[0,1]，防止xywh分量clamp不生效导致的越界
        if (isinstance(boxes_norm, torch.Tensor) and boxes_norm.numel() > 0) or (
            isinstance(boxes_norm, np.ndarray) and boxes_norm.size > 0
        ):
            boxes_norm = clip_boxes_norm_xywh(boxes_norm, 0.0, 1.0, 0.0, 1.0)
        
        # 根据验证模式决定输出
        if self.modality:
            # 单模态验证：仅输出指定模态
            if self.modality == "RGB":
                # RGB单模态预测
                plot_images(
                    rgb_visual,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
            else:
                # X模态单模态预测
                x_visual = visualize_x_to_3ch(
                    x_images,
                    colorize=False,
                    x_modality=x_modality,
                    precision=precision_contract["x"],
                )
                plot_images(
                    x_visual,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_pred_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
        else:
            # 双模态验证：输出三种预测图
            try:
                # 1. RGB预测结果
                plot_images(
                    rgb_visual,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
                # 2. X模态预测结果
                x_visual = visualize_x_to_3ch(
                    x_images,
                    colorize=False,
                    x_modality=x_modality,
                    precision=precision_contract["x"],
                )
                plot_images(
                    x_visual,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_pred_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
                # 3. 多模态并排预测图 - 使用duplicate函数为两侧绘制bbox
                side_by_side_images = concat_side_by_side(rgb_visual, x_visual)
                # 复制bbox到两侧：左半(RGB) + 右半(X)
                batch_ids_dup, cls_ids_dup, boxes_dup, confs_dup = duplicate_bboxes_for_side_by_side(
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm
                )
                plot_images(
                    side_by_side_images,
                    batch_ids_dup, cls_ids_dup, boxes_dup, confs_dup,
                    paths=[p.replace('.jpg', '_multimodal.jpg') for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_pred_multimodal.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
                
            except Exception as e:
                LOGGER.warning(f"绘制{x_modality}模态预测结果失败: {e}")

    # ------------------------------------------------------------------
    # AFSS per-sample scoring helpers
    # ------------------------------------------------------------------

    def afss_score_sample(self, pred, batch, si):
        """Build one AFSS score row for detect task from validator-native primitives."""
        if "im_file" not in batch:
            raise KeyError("AFSS sample scoring requires batch['im_file']")
        pbatch = self._prepare_batch(si, batch)
        predn = self._prepare_pred(pred, pbatch)
        result = self._process_batch(predn, pbatch)
        tp = result["tp"]
        matched = int(tp[:, 0].sum()) if len(tp) else 0
        return build_detect_afss_score_row(
            im_file=str(batch["im_file"][si]),
            matched=matched,
            pred_count=int(len(predn["cls"])),
            gt_count=int(len(pbatch["cls"])),
            task_name="detect",
        )

    def afss_score_batch(self, preds, batch):
        """Score all samples in a batch for AFSS detect adapter reuse."""
        return [self.afss_score_sample(pred, batch, si) for si, pred in enumerate(preds)]

    def score_sample(self, pred, batch, si):
        """Backward-compatible AFSS sample helper for legacy scorer calls."""
        return self.afss_score_sample(pred, batch, si)

    def score_batch(self, preds, batch):
        """Backward-compatible AFSS batch helper for legacy scorer calls."""
        return self.afss_score_batch(preds, batch)
