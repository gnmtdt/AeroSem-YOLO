# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
RT-DETR MultiModal trainer module.

This module provides the RTDETRMMTrainer class for training multi-modal RT-DETR models
with support for RGB+X modality inputs.
"""

import torch
import warnings
from copy import copy
from pathlib import Path
from typing import Optional

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.dataset import YOLOMultiModalImageDataset
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.nn.mm.complexity import (
    compute_default_multimodal_complexity_report,
    log_default_complexity,
)
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.patches import torch_load
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    get_modality_path_from_data,
    normalize_modality_token,
    resolve_requested_modality_token,
    resolve_modality_pair,
    set_x_modality_name_in_ckpt_metadata,
    sync_x_modality_name_to_model,
)
from ultralytics.engine.afss import AFSSConfig, AFSSRuntime
from ultralytics.utils.torch_utils import torch_distributed_zero_first

from .utils import detect_mm, require_multimodal

class RTDETRMMTrainer(DetectionTrainer):
    _tid = "debugger"
    """
    A trainer class for RT-DETR MultiModal (RTDETRMM) object detection models.

    This class extends DetectionTrainer to support multi-modal inputs (RGB + X modality)
    during the training process. It integrates seamlessly with YOLOMM's data pipeline
    while maintaining RT-DETR's training strategies.

    Key Features:
        - Supports RGB+X dual-modal training with 6-channel input
        - 通过 YAML/CKPT 内容判据识别多模态结构（不依赖文件名后缀）
        - Integration with YOLOMultiModalImageDataset
        - Support for single-modal ablation training via modality parameter
        - 严格 Fail-Fast：不满足多模态判据时直接报错

    Attributes:
        args: Training arguments and hyperparameters.
        model: The RTDETRMM model being trained.
        modality: Specific modality for single-modal training (None for dual-modal).
        is_multimodal: Whether the model is configured for multi-modal operation.
        multimodal_config: Configuration for multi-modal training.

    Methods:
        get_dataset: Build multi-modal dataset using YOLOMultiModalImageDataset.
        get_model: Initialize RT-DETR model with multi-modal support.
        preprocess_batch: Preprocess batch data for multi-modal inputs.
        get_validator: Return multi-modal compatible validator.

    Examples:
        >>> # Dual-modal training
        >>> trainer = RTDETRMMTrainer(overrides={'model': 'rtdetr-r18-mm.yaml', 'data': 'multimodal-dataset.yaml'})
        >>> trainer.train()
        
        >>> # Single-modal ablation
        >>> trainer = RTDETRMMTrainer(overrides={'model': 'rtdetr-r18-mm.yaml', 'modality': 'thermal'})
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize RTDETRMMTrainer for multi-modal training.

        Args:
            cfg (str | DictConfig, optional): Configuration file path or configuration dictionary.
            overrides (dict, optional): Dictionary to override default training arguments.
            _callbacks (list, optional): List of callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "detect"  # Ensure task type is correct
        
        # Pre-initialize multi-modal attributes before parent init
        # This is necessary because parent init calls get_dataset
        cfg_model = ""
        if isinstance(cfg, dict):
            cfg_model = str(cfg.get("model", ""))
        else:
            cfg_model = str(getattr(cfg, "model", "")) if cfg is not None else ""
        model_name = str(overrides.get("model", cfg_model))

        # Fail-Fast：RTDETRMMTrainer 仅服务多模态结构（不依赖 -mm 文件名）
        self.mm_evidence = require_multimodal(model_name, who="RTDETRMMTrainer").to_dict()
        self.is_multimodal = True
        # 仅对 rgb/x token 做归一化：rgb/RGB→RGB、x/X→X（其它模态名保持原样）
        self.modality = normalize_modality_token(overrides.get("modality", None))
        overrides["modality"] = self.modality
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        self.multimodal_config = None
        
        # Initialize parent trainer
        super().__init__(cfg, overrides, _callbacks)

        # grid_sample backward 无确定性 CUDA 实现，逐操作屏蔽噪音警告
        warnings.filterwarnings(
            "ignore",
            message="grid_sampler_2d_backward_cuda does not have a deterministic implementation",
            category=UserWarning,
        )
        
        # Update modality from args after parent init
        self.modality = normalize_modality_token(getattr(self.args, "modality", None))
        # 回写 args，确保训练内 validator/copy(args) 看到一致 token
        self.args.modality = self.modality
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        
        # # Log initialization with modality information
        # if self.modality:
        #     LOGGER.info(f"Initializing RTDETRMMTrainer - Single-modal training mode: {self.modality}-only")
        # else:
        #     LOGGER.info("Initializing RTDETRMMTrainer - Dual-modal training mode (RGB+X)")

        # ---- AFSS sampling integration ----
        self._sync_afss_task_name()
        self.afss_config = AFSSConfig.from_args(self.args)
        self.afss_runtime = None
        if self.afss_config.enabled:
            self.add_callback("on_train_start", self._afss_on_train_start)
            self.add_callback("on_train_epoch_start", self._afss_on_train_epoch_start)
            self.add_callback("on_train_epoch_end", self._afss_on_train_epoch_end)
            LOGGER.info(
                "AFSS 配置已启用: task=%s warmup=%d, update_interval=%d, score_conf=%.3f, score_iou=%.3f, "
                "ema_alpha=%.3f, state_dir=%s",
                self.afss_config.task_name,
                self.afss_config.warmup_epochs,
                self.afss_config.state_update_interval,
                self.afss_config.score_conf,
                self.afss_config.score_iou,
                self.afss_config.state_ema_alpha,
                self.afss_config.state_dir,
            )

        # ---- Distillation runtime (lazy-initialized after model is built) ----
        self.distill_runtime = None
        self._distill_student_collector = None
        self._distill_cfg = self._parse_distill_arg()

    def _sync_afss_task_name(self):
        """Keep AFSS task wiring aligned with trainer task for adapter-based runtime."""
        task_name = str(getattr(self.args, "task", "detect"))
        setattr(self.args, "afss_task_name", task_name)

    def _ensure_afss_runtime(self, dataset):
        """Initialize AFSS runtime once training dataset is available."""
        if not self.afss_config.enabled or self.afss_runtime is not None:
            return self.afss_runtime
        self._sync_afss_task_name()
        self.afss_config = AFSSConfig.from_args(self.args)
        self.afss_runtime = AFSSRuntime.from_dataset(
            dataset=dataset,
            args=self.args,
            save_dir=self.save_dir,
            resume=bool(getattr(self.args, "resume", False)),
        )
        return self.afss_runtime

    def _afss_on_train_start(self, trainer):
        """Callback: announce AFSS bootstrap runtime."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_start()

    def _afss_on_train_epoch_start(self, trainer):
        """Callback: refresh AFSS bootstrap epoch selection."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_start(trainer.epoch)

    def _afss_on_train_epoch_end(self, trainer):
        """Callback: persist AFSS state snapshot and trigger scheduled scoring."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_end(
                trainer.epoch,
                trainer=trainer,
                validator=trainer.validator,
            )

    def _setup_train(self, world_size):
        """Build model, then initialize distillation runtime if configured."""
        super()._setup_train(world_size)
        self._init_distill_runtime()
        self._distill_loss_names = ()
        self._distill_epoch_accum = {}
        self._distill_epoch_count = 0
        if self.distill_runtime is not None:
            from ultralytics.nn.mm.distill.adapters import RTDETRMMDetectDistillAdapter
            self._distill_adapter = RTDETRMMDetectDistillAdapter(
                runtime=self.distill_runtime,
                config=self.distill_runtime.config,
                student_model=de_parallel(self.model),
                trainer=self,
            )
            _, mode = self._distill_cfg
            distill_names = []
            if mode in ("output", "both"):
                distill_names.extend(["d_out", "d_out_cls", "d_out_box"])
            if mode in ("feature", "both"):
                distill_names.extend(["d_feat", "d_feat_fg", "d_feat_bg", "d_feat_cwd", "d_feat_ctx"])
            self._distill_loss_names = tuple(distill_names)
            self.add_callback("on_train_epoch_end", self._log_distill_epoch_summary)

    def compute_batch_loss(self, batch):
        """Compute batch loss, injecting distillation loss if enabled."""
        if self.distill_runtime is None:
            return self.model(batch)

        from ultralytics.utils.torch_utils import de_parallel

        student_model = de_parallel(self.model)
        self._distill_student_collector.reset()

        td = getattr(self, "td", None)
        if td is None or not td.enabled:
            det_loss, det_items, student_preds = student_model.distill_forward(batch)
            teacher_outputs = self.distill_runtime.run_teachers(batch)
            self._distill_adapter.set_epoch_state(self.epoch, self.epochs)
            student_features = self._distill_student_collector.features
            distill_loss, distill_items = self._distill_adapter.compute_distill_loss(
                student_preds, student_features, teacher_outputs
            )
            total_loss = det_loss + distill_loss
            self._accumulate_distill_items_rt(distill_items)
            return total_loss, det_items

        with td.span("batch.forward.student", sync_cuda=True, epoch=self.epoch):
            det_loss, det_items, student_preds = student_model.distill_forward(batch)

        with td.span("batch.forward.teacher", sync_cuda=True, epoch=self.epoch):
            teacher_outputs = self.distill_runtime.run_teachers(batch)

        self._distill_adapter.set_epoch_state(self.epoch, self.epochs)
        student_features = self._distill_student_collector.features
        with td.span("batch.forward.distill", sync_cuda=True, epoch=self.epoch):
            distill_loss, distill_items = self._distill_adapter.compute_distill_loss(
                student_preds, student_features, teacher_outputs
            )

        total_loss = det_loss + distill_loss
        self._accumulate_distill_items_rt(distill_items)
        return total_loss, det_items

    def _accumulate_distill_items_rt(self, distill_items):
        """Accumulate distill items for epoch-level summary and CSV (RTDETRMM variant)."""
        _zero = 0.0
        distill_values = {}
        if "d_out" in distill_items:
            distill_values["d_out"] = float(distill_items["d_out"])
            distill_values["d_out_cls"] = float(distill_items.get("distill_output_cls", _zero))
            distill_values["d_out_box"] = float(distill_items.get("distill_output_box", _zero))
        if "distill_feature" in distill_items:
            distill_values["d_feat"] = float(distill_items["distill_feature"])
            distill_values["d_feat_fg"] = float(distill_items.get("distill_feature_fg", _zero))
            distill_values["d_feat_bg"] = float(distill_items.get("distill_feature_bg", _zero))
            distill_values["d_feat_cwd"] = float(distill_items.get("distill_feature_cwd", _zero))
            distill_values["d_feat_ctx"] = float(distill_items.get("distill_feature_ctx", _zero))
        for k, v in distill_values.items():
            self._distill_epoch_accum[k] = self._distill_epoch_accum.get(k, 0.0) + v
        self._distill_epoch_count += 1

    def _log_distill_epoch_summary(self, trainer):
        """Callback: log epoch-level distill summary after training batches."""
        if not self._distill_loss_names or self._distill_epoch_count == 0:
            return
        n = self._distill_epoch_count
        parts = [f"{name}={self._distill_epoch_accum.get(name, 0.0) / n:.4f}"
                 for name in self._distill_loss_names]
        LOGGER.info(f"  Distill Epoch {self.epoch + 1}/{self.epochs}: {'  '.join(parts)}")

    def save_metrics(self, metrics):
        """Override: inject distill epoch averages into CSV for plotting."""
        if self._distill_loss_names and self._distill_epoch_count > 0:
            n = self._distill_epoch_count
            for name in self._distill_loss_names:
                key = f"train/{name}"
                if key not in metrics:
                    metrics[key] = round(self._distill_epoch_accum.get(name, 0.0) / n, 5)
        super().save_metrics(metrics)
        self._distill_epoch_accum = {}
        self._distill_epoch_count = 0

    def _check_multimodal_model(self):
        """
        Check if the model is configured for multi-modal operation.
        
        Returns:
            bool: True if model content matches multimodal spec
        """
        model_name = str(self.args.model)
        return detect_mm(model_name).is_multimodal
    
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build multi-modal dataset using YOLOMultiModalImageDataset.

        This method follows YOLOMM's successful pattern by overriding build_dataset
        instead of get_dataset, ensuring proper initialization timing and data access.

        Args:
            img_path (str): Path to images
            mode (str): Dataset mode ('train', 'val', 'test')
            batch (int, optional): Batch size for rectangle training

        Returns:
            Dataset: YOLOMultiModalImageDataset for multi-modal, standard dataset otherwise
        """
        if not self.is_multimodal:
            raise ValueError(
                "RTDETRMMTrainer is designed for multi-modal RT-DETR models only. "
                "For standard RT-DETR training, use RTDETRTrainer instead. "
                "RTDETRMM 的判定基于 YAML/CKPT 的内容判据（RGB/X/Dual 路由标记或 ckpt 元信息），不依赖文件名后缀。"
            )

        # Get model stride parameter (consistent with DetectionTrainer)
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)

        # Lazy loading: parse multi-modal configuration on demand
        if not hasattr(self, 'multimodal_config') or self.multimodal_config is None:
            self.multimodal_config = self._parse_multimodal_config()
            LOGGER.info(f"多模态配置解析完成 - 模态: {self.multimodal_config['models']}")

        # Use parsed modality configuration
        modalities = self.multimodal_config['models']
        
        # 🔑 关键修复：明确提取X模态信息（与YOLOMM保持一致）
        _rgb_modality, x_modality = resolve_modality_pair(self.multimodal_config['models'], strict=True)
        x_modality_dir = self.multimodal_config['modalities'][x_modality]

        LOGGER.info(f"构建多模态数据集 - 模式: {mode}, 路径: {img_path}, 模态: {modalities}")

        # If single-modal training is enabled, log modality padding info and validate compatibility
        if self.modality:
            self._validate_modality_compatibility()
            LOGGER.info(f"启用单模态训练: {self.modality}-only，将应用智能模态填充")

        # Call build_yolo_dataset with multi_modal_image=True to enable multi-modal dataset
        return build_yolo_dataset(
            self.args, img_path, batch, self.data,
            mode=mode,
            rect=False,  # RT-DETR validation uses fixed-shape inference (align with RTDETRValidator)
            stride=gs,
            multi_modal_image=True,  # Key parameter: enable YOLOMultiModalImageDataset
            x_modality=x_modality,  # ✅ 修复：传递模态类型
            x_modality_dir=x_modality_dir,  # ✅ 修复：传递模态目录
            enable_self_modal_generation=getattr(self.args, 'enable_self_modal_generation', False),  # ✅ 修复：添加自动模态生成支持
            precision_report_dir=self.save_dir,
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct dataloader with optional AFSS sampler injection for training."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        sampler_override = None
        if mode == "train" and self.afss_config.enabled:
            runtime = self._ensure_afss_runtime(dataset)
            sampler_override = runtime.create_sampler(rank=rank, shuffle=shuffle)
        return build_dataloader(
            dataset,
            batch_size,
            workers,
            shuffle,
            rank,
            sampler_override=sampler_override,
        )

    def get_model(self, cfg: Optional[dict] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Initialize and return RT-DETR model with multi-modal support.
        
        This method creates RTDETRDetectionModel with correct channel configuration,
        following the same pattern as YOLOMM for consistency.
        
        Args:
            cfg (dict, optional): Model configuration.
            weights (str, optional): Path to pre-trained model weights.
            verbose (bool): Verbose logging if True.
            
        Returns:
            RTDETRDetectionModel: Initialized model ready for multi-modal routing.
        """
        from ultralytics.utils import RANK
        
        # 计算正确的输入通道数（双模态训练 与 模态消融 均按 Pch+Xch 构建）
        x_channels = self.data.get('Xch', 3)
        channels = 3 + x_channels  # RGB(3) + X(Xch)
        if verbose and RANK in {-1, 0}:
            tag = "消融" if self.modality else "多模态"
            LOGGER.info(f"{tag}RT-DETR模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch")
        
        # 🔧 关键修复：更新data配置以修复验证器通道数不匹配问题
        self.data["channels"] = channels
        if verbose and RANK in {-1, 0}:
            LOGGER.info(f"已更新data[\"channels\"]为{channels}，确保验证器兼容性")
        
        # Create RT-DETR model（在构建前将dataset_config注入yaml以在解析期识别x_modality）
        cfg_in = cfg or self.args.model
        from ultralytics.nn.mm.utils import prepare_multimodal_cfg_dict

        cfg_dict = prepare_multimodal_cfg_dict(
            cfg_in, self.data,
            mm_input_role=getattr(self.args, 'mm_input_role', None)
        )
        model = RTDETRDetectionModel(
            cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1
        )
        
        # 加载权重（如果提供）
        if weights:
            model.load(weights)
        # 同步多模态路由配置（与 YOLO 路径保持一致）
        try:
            if hasattr(model, 'multimodal_router') and model.multimodal_router:
                model.multimodal_router.update_dataset_config(self.data)
        except Exception as e:
            LOGGER.warning(f"RTDETRMMTrainer: update_dataset_config failed: {e}")

        sync_x_modality_name_to_model(model, extract_x_modality_name_from_data(getattr(self, "data", None)))

        # Unified multimodal complexity logs
        try:
            imgsz = int(getattr(self.args, 'imgsz', 640))
            report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
            log_default_complexity(model, report, LOGGER)
        except Exception as e:
            LOGGER.warning(f"模型复杂度统计失败（可忽略，不影响训练）：{e}")

        return model
    
    def preprocess_batch(self, batch):
        """
        Preprocess batch data for multi-modal inputs.

        Handles both standard (3-channel) and multi-modal (6-channel) inputs,
        with support for single-modal ablation training.

        Uses shared dtype policy instead of inheriting blanket all-channel /255
        from the parent DetectionTrainer.

        Args:
            batch (dict): Batch data dictionary containing images and labels.

        Returns:
            dict: Preprocessed batch ready for training.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype

        if not self.is_multimodal:
            # Non-multimodal: standard detect preprocessing
            batch["img"] = batch["img"].float() / 255.0
        else:
            # Multimodal: contract-driven preprocessing
            # legacy+legacy 恢复 29e006c^; mixed 按 per-modality 处理
            img = batch["img"].float()
            img = apply_dataset_precision_contract(img, extract_batch_precision_contract(batch), raw_dtype=raw_dtype)
            batch["img"] = img

        # Multi-modal specific preprocessing
        # Router handles modality ablation at runtime
        if self.is_multimodal:
            try:
                if self.modality and hasattr(self, 'model') and self.model is not None:
                    if hasattr(self.model, 'mm_router') and self.model.mm_router:
                        self.model.mm_router.set_runtime_params(
                            self.modality,
                            strategy=getattr(self.args, 'ablation_strategy', None),
                            seed=getattr(self.args, 'seed', None),
                        )
            except Exception:
                pass

        return batch
    
    def get_validator(self):
        """
        Return validator compatible with multi-modal RT-DETR.

        Returns:
            RTDETRMMValidator: Multi-modal validator instance.
        """
        # Validator always uses standard RT-DETR loss_names (no distill columns).
        # Distillation loss columns are only for training progress bar, set in
        # _setup_train after runtime init.
        self.loss_names = ("giou_loss", "cls_loss", "l1_loss")

        from .val import RTDETRMMValidator

        # 创建验证器并传递更新后的data配置
        validator = RTDETRMMValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args)
        )

        # 传递trainer的data配置给验证器
        # 确保验证器使用包含更新后channels信息的data配置
        validator.data = self.data

        return validator
    
    def _parse_multimodal_config(self):
        """
        Parse and validate multi-modal configuration from data.yaml.
        
        Adapted from YOLOMM's implementation to maintain consistency.
        
        Returns:
            dict: Parsed multi-modal configuration.
        """
        data = getattr(self, 'data', None)

        # Single-modal training configuration
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
                "Single-modal training bridge complete: requested=%s -> resolved=%s, pair=%s",
                self.modality,
                resolved_modality,
                config['models'],
            )
            return config
        
        # Dual-modal training configuration
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

                LOGGER.info(f"Loaded multi-modal config: {modality_used}")
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

        # Determine default configuration
        x_modality = self._determine_x_modality_from_data()
        rgb_modality, x_modality = resolve_modality_pair(data, default_secondary=x_modality)
        config = {
            'models': [rgb_modality, x_modality],
            'modalities': {
                rgb_modality: get_modality_path_from_data(data, rgb_modality),
                x_modality: get_modality_path_from_data(data, x_modality)
            }
        }
        LOGGER.info(f"Using default multi-modal config: rgb+{x_modality}")
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
        return get_modality_path_from_data(self.data, modality_name)
    
    def _determine_x_modality_from_data(self):
        """
        Intelligently determine X modality type from data configuration.

        Returns:
            str: X modality identifier (e.g., 'depth', 'thermal', 'ir').
        """
        # Safe data access - use getattr to avoid AttributeError during initialization
        data = getattr(self, 'data', None)

        x_modality = extract_x_modality_name_from_data(data)
        if x_modality != 'unknown':
            return x_modality
        
        # Use depth as default when undetermined
        LOGGER.warning("Cannot determine X modality type, using default: depth")
        return 'depth'
    
    def _validate_modality_compatibility(self):
        """
        Validate that specified modality is compatible with available data.
        
        Raises:
            ValueError: When modality parameter is incompatible with data configuration.
        """
        if not self.modality:
            return
        
        # Get available modalities
        available_modalities = []
        if self.multimodal_config:
            available_modalities = self.multimodal_config.get('models', [])
        
        # Validate modality compatibility
        if available_modalities:
            try:
                resolved = resolve_requested_modality_token(
                    self.modality, available_modalities, default_x=self._determine_x_modality_from_data(), strict=True
                )
            except ValueError as exc:
                raise ValueError(
                    f"Specified modality '{self.modality}' not in available modalities: {available_modalities}. "
                    f"Please check data configuration or modality parameter."
                ) from exc
            LOGGER.info(f"✅ Modality compatibility validated: '{self.modality}' -> {resolved} in {available_modalities}")
        else:
            # 如果无法获取可用模态，仅给出警告
            LOGGER.warning(f"⚠️  无法验证modality '{self.modality}' 的兼容性，未找到可用模态配置")
    
    def save_model(self):
        """
        Save model with multi-modal configuration information.
        
        Extends parent's save_model to include multi-modal metadata.
        """
        # Call parent's model saving
        super().save_model()
        
        # Save multi-modal configuration to checkpoint
        if self.is_multimodal:
            x_modality_name = extract_x_modality_name_from_data(getattr(self, "data", None))
            # Update last checkpoint
            ckpt = torch_load(self.last, map_location='cpu')
            set_x_modality_name_in_ckpt_metadata(ckpt, x_modality_name)
            if hasattr(self, 'multimodal_config'):
                ckpt['multimodal_config'] = self.multimodal_config
            ckpt['modality'] = self.modality
            ckpt['is_multimodal'] = True
            torch.save(ckpt, self.last)
            
            # Update best checkpoint if exists
            if self.best.exists():
                ckpt_best = torch_load(self.best, map_location='cpu')
                set_x_modality_name_in_ckpt_metadata(ckpt_best, x_modality_name)
                if hasattr(self, 'multimodal_config'):
                    ckpt_best['multimodal_config'] = self.multimodal_config
                ckpt_best['modality'] = self.modality
                ckpt_best['is_multimodal'] = True
                torch.save(ckpt_best, self.best)
    
    def final_eval(self):
        """
        Perform final evaluation with multi-modal specific logging.
        """
        # Execute standard evaluation
        super().final_eval()

        # Export LLM-friendly JSON format validation results
        from ultralytics.utils.llm_export import export_final_val_llm_json

        try:
            export_final_val_llm_json(self)
        except Exception as e:
            LOGGER.warning(f"LLM JSON export failed: {e}")

        # Log multi-modal specific information
        if self.is_multimodal and hasattr(self, 'multimodal_config') and self.multimodal_config:
            _rgb_modality, x_modality = resolve_modality_pair(self.multimodal_config['models'], strict=True)
            if self.modality:
                LOGGER.info(f"Final evaluation complete - Single-modal training: {self.modality}-only")
            else:
                LOGGER.info(f"Final evaluation complete - Dual-modal training: RGB+{x_modality}")
        else:
            LOGGER.info("Final evaluation complete - Standard RT-DETR training")

    # ---- Distillation helpers ------------------------------------------------

    def _parse_distill_arg(self):
        """Parse ``self.args.distill`` and return ``(yaml_path, mode)`` or None."""
        distill = getattr(self.args, "distill", None)
        if distill is None:
            return None
        if not isinstance(distill, (list, tuple)) or len(distill) != 2:
            raise ValueError(
                "distill must be [yaml_path, mode] (list/tuple of length 2), "
                f"got {type(distill).__name__}: {distill}"
            )
        yaml_path, mode = str(distill[0]), str(distill[1]).lower()
        if mode not in ("output", "feature", "both"):
            raise ValueError(f"distill mode must be output/feature/both, got '{mode}'")
        LOGGER.info(f"Distillation enabled: yaml={yaml_path}, mode={mode}")
        return yaml_path, mode

    def _init_distill_runtime(self):
        """Build distill runtime after student model is available (called in train())."""
        if self._distill_cfg is None:
            return
        yaml_path, mode = self._distill_cfg
        from ultralytics.nn.mm.distill.schema import load_distill_config
        from ultralytics.nn.mm.distill.runtime import DistillRuntime
        from ultralytics.utils.torch_utils import de_parallel

        config = load_distill_config(yaml_path)
        student = de_parallel(self.model)
        self.distill_runtime = DistillRuntime(
            config=config,
            mode=mode,
            family="rtdetrmm",
            student_model=student,
            device=student.device if hasattr(student, 'device') else next(student.parameters()).device,
        )
        self._distill_student_collector = self.distill_runtime.register_student_hooks(student)

    def plot_training_samples(self, batch, ni):
        """
        绘制多模态训练样本，支持精度契约的三视图输出：
        - 多模态训练（未传 modality）：输出 RGB、X（灰度可视化）与并排对比图
        - 模态消融（传入 modality）：rgb 仅输出 RGB；X 仅输出对应 X 模态

        Args:
            batch (dict): 训练批次数据
            ni (int): 当前迭代次数
        """
        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
            split_modalities,
            prepare_rgb_for_display,
            visualize_x_to_3ch,
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
            ensure_batch_idx_long,
        )

        # 批数据
        images = batch["img"]  # [B, 3+Xch, H, W]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]

        # 统一 batch_idx 类型
        batch_idx = ensure_batch_idx_long(batch["batch_idx"])
        batch["batch_idx"] = batch_idx

        # 动态 X 通道数
        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        x_modality = self.data.get("x_modality", "unknown")

        # 拆分模态（严格 [RGB, X] 顺序）
        rgb_images, x_images = split_modalities(images, xch)
        precision_contract = extract_precision_contract_from_batch(batch)
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])

        # 模态消融：仅输出指定模态
        if self.modality:
            if self.modality == "RGB":
                plot_images(
                    rgb_visual,
                    batch_idx,
                    cls,
                    bboxes,
                    paths=paths,
                    fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
                    on_plot=self.on_plot,
                )
            else:
                # X-only，灰度可视化
                x_visual = visualize_x_to_3ch(
                    x_images,
                    colorize=False,
                    x_modality=x_modality,
                    precision=precision_contract["x"],
                )
                plot_images(
                    x_visual,
                    batch_idx,
                    cls,
                    bboxes,
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
                    fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
                    on_plot=self.on_plot,
                )
            return

        # 检测 X 通道是否为空（单模态 ablation，X 通道全零）
        x_is_empty = x_images.abs().max() < 1e-6

        # 多模态训练：输出 RGB 图
        plot_images(
            rgb_visual,
            batch_idx,
            cls,
            bboxes,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
            on_plot=self.on_plot,
        )

        if x_is_empty:
            # X 通道为空时仅输出 RGB 图，跳过 X 和并排图
            return

        # X（灰度）
        x_visual = visualize_x_to_3ch(
            x_images,
            colorize=False,
            x_modality=x_modality,
            precision=precision_contract["x"],
        )
        plot_images(
            x_visual,
            batch_idx,
            cls,
            bboxes,
            paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
            fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
            on_plot=self.on_plot,
        )

        # 并排对比（复制 bbox 到左右半幅）
        side_by_side_images = concat_side_by_side(rgb_visual, x_visual)
        batch_ids_dup, cls_ids_dup, bboxes_dup, _ = duplicate_bboxes_for_side_by_side(
            batch_idx, cls, bboxes, None
        )
        plot_images(
            side_by_side_images,
            batch_ids_dup,
            cls_ids_dup,
            bboxes_dup,
            paths=[p.replace('.jpg', '_multimodal.jpg') for p in paths],
            fname=self.save_dir / f"train_batch{ni}_labels_multimodal.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """绘制训练指标图表，蒸馏训练时额外生成独立的蒸馏指标图。"""
        from ultralytics.utils.plotting import plot_distill_results

        # 调用父类的指标绘制（生成标准 results.png）
        super().plot_metrics()

        # 蒸馏训练时生成独立的蒸馏指标图
        if self._distill_cfg is not None:
            plot_distill_results(file=self.csv, family="rtdetrmm", on_plot=self.on_plot)
