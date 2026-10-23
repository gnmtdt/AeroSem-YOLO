# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random

import torch
import torch.nn as nn
from copy import copy

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK
from ultralytics.nn.mm.pruning.trainability import (
    find_frozen_floating_parameters,
    restore_parameter_trainability,
)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.dataset import YOLOMultiModalImageDataset
from ultralytics.nn.mm.complexity import (
    compute_default_multimodal_complexity_report,
    log_default_complexity,
)
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


class MultiModalDetectionTrainer(DetectionTrainer):
    _tid = "debugger"
    """
    多模态检测训练器，基于Input字段路由系统的RGB+X模态训练流程。

    核心特色:
    - 支持配置驱动的多模态路由 (通过第5字段: 'RGB', 'X', 'Dual')
    - 早期融合: 6通道RGB+X输入统一处理
    - 中期融合: 独立RGB和X路径特征提取后融合
    - 晚期融合: 高层语义特征融合
    
    这个类继承DetectionTrainer，集成MultiModalRouter实现灵活的多模态数据流控制。
    支持RGB+深度、RGB+热红外等任意X模态组合的完整训练流程。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化多模态检测训练器。
        
        Args:
            cfg (str | DictConfig, optional): 配置文件路径或配置字典
            overrides (dict, optional): 配置覆盖参数
            _callbacks (list, optional): 回调函数列表
        """
        if overrides is None:
            overrides = {}
        overrides.setdefault("task", "detect")  # 仅当子类未设置时默认为detect（如semseg会设置'semseg'）
        super().__init__(cfg, overrides, _callbacks)
        
        # Get modality parameter from standard cfg system (与推理器保持一致)
        # Modality validation is handled by cfg system, no local validation needed
        self.modality = normalize_modality_token(getattr(self.args, "modality", None))
        # 回写 args，确保训练内 validator/copy(args) 看到一致 token
        self.args.modality = self.modality
        
        # Initialize modality-specific attributes
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

        # Initialize logging control flags
        self._multimodal_config_logged = False  # 控制多模态配置日志只记录一次

        # # Log initialization with modality information
        # if self.modality:
        #     LOGGER.info(f"初始化MultiModalDetectionTrainer - 单模态训练模式: {self.modality}-only")
        # else:
        #     LOGGER.info("初始化MultiModalDetectionTrainer - 双模态训练模式")

        # ---- Distillation runtime (lazy-initialized after model is built) ----
        self.distill_runtime = None
        self._distill_student_collector = None
        self._distill_cfg = self._parse_distill_arg()
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

    def _sync_afss_task_name(self):
        """Keep AFSS task wiring aligned with trainer task for adapter-based runtime."""
        task_name = str(getattr(self.args, "task", "detect"))
        setattr(self.args, "afss_task_name", task_name)

    def _setup_train(self, world_size):
        """Build model, then initialize distillation runtime if configured."""
        import time
        td = getattr(self, "td", None)

        super()._setup_train(world_size)

        t0 = time.time()
        self._init_distill_runtime()
        # Distillation epoch-level accumulator (always present for safety)
        self._distill_loss_names = ()
        self._distill_epoch_accum = {}
        self._distill_epoch_count = 0
        if self.distill_runtime is not None:
            from ultralytics.nn.mm.distill.adapters import YOLOMMDetectDistillAdapter
            self._distill_adapter = YOLOMMDetectDistillAdapter(
                runtime=self.distill_runtime,
                config=self.distill_runtime.config,
                student_model=de_parallel(self.model),
                trainer=self,
            )
            # Do NOT extend self.loss_names -- progress bar only shows base detection loss.
            # Store distill column names separately for CSV and epoch summary.
            _, mode = self._distill_cfg
            distill_names = []
            if mode in ("output", "both"):
                distill_names.extend(["d_out", "d_out_cls", "d_out_loc"])
            if mode in ("feature", "both"):
                distill_names.extend(["d_feat", "d_feat_fg", "d_feat_bg", "d_feat_cwd", "d_feat_ctx"])
            self._distill_loss_names = tuple(distill_names)
            self.add_callback("on_train_epoch_end", self._log_distill_epoch_summary)
        if td is not None and td.enabled:
            td.log_event("setup.init_distill_runtime", elapsed=time.time() - t0)

    def compute_batch_loss(self, batch):
        """Compute batch loss, injecting distillation loss if enabled.

        Returns only base detection loss items for progress bar display.
        Distill items are accumulated separately for epoch-level summary.
        """
        if self.distill_runtime is None:
            return self.model(batch)

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
            self._accumulate_distill_items(distill_items)
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
        self._accumulate_distill_items(distill_items)
        return total_loss, det_items

    def _accumulate_distill_items(self, distill_items):
        """Accumulate distill items for epoch-level summary and CSV."""
        _zero = 0.0
        distill_values = {}
        if "d_out" in distill_items:
            distill_values["d_out"] = float(distill_items["d_out"])
            distill_values["d_out_cls"] = float(distill_items.get("distill_output_cls", _zero))
            distill_values["d_out_loc"] = float(distill_items.get("distill_output_loc", _zero))
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

    def _parse_multimodal_config(self):
        """
        解析和验证数据配置文件中的多模态设置。
        
        解析data.yaml中的modalities和models字段，确保配置正确性，
        提供默认配置和友好的错误信息。
        
        优先支持用户指定的单模态训练参数。
        
        Returns:
            dict: 解析后的多模态配置
            
        Raises:
            ValueError: 当多模态配置不正确时
        """
        if self.modality:
            primary_modality, secondary_modality = resolve_modality_pair(
                getattr(self, "data", None), default_secondary=self._determine_x_modality_from_data()
            )
            if self.modality and self.modality.lower() == primary_modality.lower():
                config = {
                    'models': [primary_modality, secondary_modality],
                    'modalities': {
                        primary_modality: get_modality_path_from_data(self.data, primary_modality),
                        secondary_modality: get_modality_path_from_data(self.data, secondary_modality),
                    },
                }
                LOGGER.info(f"{primary_modality}单模态训练，动态确定secondary模态: {secondary_modality}")
            else:
                actual_secondary = resolve_requested_modality_token(
                    self.modality,
                    getattr(self, "data", None),
                    default_x=secondary_modality,
                )
                config = {
                    'models': [primary_modality, actual_secondary],
                    'modalities': {
                        primary_modality: get_modality_path_from_data(self.data, primary_modality),
                        actual_secondary: get_modality_path_from_data(self.data, actual_secondary),
                    },
                }
                if self.modality == "X":
                    LOGGER.info(f"X模态单模态训练: {actual_secondary}-only (从'X'解析)")
                else:
                    LOGGER.info(f"X模态单模态训练: {actual_secondary}-only")
            return config

        config = self._get_default_multimodal_config()

        if not self.data:
            LOGGER.warning("训练器未提供数据配置，使用默认多模态配置: rgb+depth")
            return config

        if 'modality_used' in self.data:
            models = self.data['modality_used']
            if not isinstance(models, list):
                raise ValueError(f"data.yaml中的'modality_used'必须是列表格式，当前为: {type(models)}")
            rgb_modality, x_modality = resolve_modality_pair(models, strict=True)
            config['models'] = [rgb_modality, x_modality]
            LOGGER.info(f"从data.yaml的modality_used读取模态组合: {config['models']}")
        elif 'models' in self.data:
            models = self.data['models']
            if not isinstance(models, list):
                raise ValueError(f"data.yaml中的'models'必须是列表格式，当前为: {type(models)}")
            rgb_modality, x_modality = resolve_modality_pair(models, strict=True)
            config['models'] = [rgb_modality, x_modality]
            LOGGER.info(f"使用配置中的模态组合: {config['models']}")
        else:
            LOGGER.debug(f"未找到'modality_used'或'models'配置，使用默认组合: {config['models']}")

        config['modalities'] = {
            modality: get_modality_path_from_data(self.data, modality)
            for modality in config['models']
        }
        LOGGER.info(f"使用配置驱动的模态路径映射: {config['modalities']}")

        _rgb_modality, x_modality = resolve_modality_pair(config['models'], strict=True)
        LOGGER.info(f"✅ 使用用户配置的X模态: {x_modality} (配置驱动，支持任意模态类型)")
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
        智能确定X模态类型，避免硬编码depth。

        优先级:
        1. 从共享解析组件读取已配置的真实X模态
        2. 从数据目录结构推断
        3. 最后使用depth作为默认值

        Returns:
            str: X模态类型标识符
        """
        x_modality = extract_x_modality_name_from_data(getattr(self, "data", None))
        if x_modality != 'unknown':
            LOGGER.info(f"从数据配置解析X模态: {x_modality}")
            return x_modality

        if self.data and 'path' in self.data:
            try:
                import os
                data_path = self.data['path']
                if os.path.exists(data_path):
                    for item in os.listdir(data_path):
                        if item.startswith('images_') and item != 'images':
                            x_modality = item.replace('images_', '')
                            LOGGER.info(f"从目录结构推断X模态: {x_modality}")
                            return x_modality
            except Exception as e:
                LOGGER.debug(f"目录结构推断失败: {e}")

        LOGGER.warning("无法自动确定X模态类型，使用默认值: depth")
        return 'depth'

    def _get_default_multimodal_config(self):
        """
        获取默认的多模态配置，优先从数据配置文件读取。

        Returns:
            dict: 默认多模态配置
        """
        if self.data and 'modality_used' in self.data:
            modality_used = self.data['modality_used']
            if isinstance(modality_used, list) and len(modality_used) >= 2:
                rgb_modality, x_modality = resolve_modality_pair(modality_used, strict=True)
                resolved_models = [rgb_modality, x_modality]
                LOGGER.info(f"从data.yaml读取模态组合: {resolved_models}")
                config = {
                    'models': resolved_models,
                    'modalities': {
                        mod: get_modality_path_from_data(self.data, mod)
                        for mod in resolved_models
                    },
                }
                LOGGER.info(f"从data.yaml读取路径映射: {config['modalities']}")
                return config

        if self.data and 'models' in self.data:
            models = self.data['models']
            if isinstance(models, list) and len(models) >= 2:
                rgb_modality, x_modality = resolve_modality_pair(models, strict=True)
                resolved_models = [rgb_modality, x_modality]
                LOGGER.info(f"从数据配置读取模态组合: {resolved_models}")
                return {
                    'models': resolved_models,
                    'modalities': {
                        modality: get_modality_path_from_data(self.data, modality)
                        for modality in resolved_models
                    },
                }

        rgb_modality, x_modality = resolve_modality_pair(
            getattr(self, "data", None), default_secondary=self._determine_x_modality_from_data()
        )
        config = {
            'models': [rgb_modality, x_modality],
            'modalities': {
                rgb_modality: get_modality_path_from_data(self.data, rgb_modality),
                x_modality: get_modality_path_from_data(self.data, x_modality),
            },
        }
        LOGGER.info(f"生成默认多模态配置: {rgb_modality}+{x_modality}")
        return config

    def _validate_modality_compatibility(self):
        """
        验证用户指定的modality参数与数据配置的兼容性。
        
        Raises:
            ValueError: 当modality参数与可用数据不兼容时
        """
        if not self.modality:
            return

        available_modalities = []
        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            available_modalities = self.multimodal_config.get('models', [])
        elif self.data and 'models' in self.data:
            available_modalities = self.data['models']
        elif self.data and 'modality_used' in self.data:
            available_modalities = self.data['modality_used']

        if available_modalities:
            try:
                resolved = resolve_requested_modality_token(
                    self.modality,
                    available_modalities,
                    strict=True,
                    default_x=resolve_modality_pair(available_modalities)[1],
                )
            except ValueError as exc:
                raise ValueError(
                    f"指定的modality '{self.modality}' 不在可用模态列表中: {available_modalities}。"
                    f"请检查数据配置或modality参数。"
                ) from exc

            if self.modality == "X":
                LOGGER.info(f"✅ 模态兼容性验证通过: '{self.modality}' 映射到 {resolved}")
            else:
                LOGGER.info(f"✅ 模态兼容性验证通过: {self.modality} → {resolved}，可用模态 {available_modalities} 中")
        else:
            LOGGER.warning(f"⚠️  无法验证modality '{self.modality}' 的兼容性，未找到可用模态配置")

    def preprocess_batch(self, batch):
        """多模态感知的预处理。

        Legacy+legacy: 恢复 29e006c^ 语义 — uint8 全通道 /255; float 仅 RGB /255。
        Mixed/high_precision: 按 contract 逐模态处理。
        顺序: to(device) -> contract-aware preprocess -> half() -> multi-scale
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype  # 捕获原始 dtype（uint8/float）再转 float
        img = batch["img"].float()

        if self.args.debug:
            cond = img[:, :3].max() > 1.5
            LOGGER.warning(
                f"[PLOT-DBG] preprocess condition: img[:,:3].max()={img[:, :3].max():.4f} > 1.5 = {cond}"
            )
            for idx in range(img.shape[0]):
                ri = img[idx, :3]
                xi = img[idx, 3:]
                flag = " *** BLACK ***" if ri.max() < 0.01 else ""
                LOGGER.warning(
                    f"[PLOT-DBG] PRE  idx={idx} | RGB [{ri.min():.4f}, {ri.max():.4f}]{flag} | "
                    f"X [{xi.min():.4f}, {xi.max():.4f}] | dtype={ri.dtype}"
                )

        # contract-driven preprocessing: legacy+legacy 恢复 29e006c^; mixed 按 per-modality 处理
        img = apply_dataset_precision_contract(img, extract_batch_precision_contract(batch), raw_dtype=raw_dtype)
        batch["img"] = img

        if self.args.debug:
            for idx in range(img.shape[0]):
                ri = img[idx, :3]
                xi = img[idx, 3:]
                flag = " *** BLACK ***" if ri.max() < 0.01 else ""
                LOGGER.warning(
                    f"[PLOT-DBG] POST idx={idx} | RGB [{ri.min():.4f}, {ri.max():.4f}]{flag} | "
                    f"X [{xi.min():.4f}, {xi.max():.4f}] | dtype={ri.dtype}"
                )

        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs

        return batch

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        构建多模态数据集，支持RGB+X模态的图像数据。

        Args:
            img_path (str): RGB图像路径
            mode (str): 数据集模式 ('train', 'val', 'test')
            batch (int, optional): 批次大小

        Returns:
            Dataset: 多模态数据集实例
        """
        # 解析多模态配置
        self.multimodal_config = self._parse_multimodal_config()

        # 验证modality兼容性
        self._validate_modality_compatibility()

        # 获取X模态信息
        _rgb_modality, x_modality = resolve_modality_pair(self.multimodal_config['models'], strict=True)
        x_modality_dir = self.multimodal_config['modalities'][x_modality]

        # 构建多模态数据集
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)

        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
            multi_modal_image=True,  # 启用图像多模态
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=getattr(self.args, 'enable_self_modal_generation', False),
            precision_report_dir=self.save_dir,
        )

    def get_validator(self):
        """
        获取多模态检测验证器。

        Returns:
            MultiModalDetectionValidator: 多模态验证器实例
        """
        # Validator always uses standard detection loss_names (no distill columns).
        # Distillation loss columns are only for training progress bar, set in
        # _setup_train after runtime init.
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")

        # 延迟导入避免循环依赖
        from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator

        return MultiModalDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """
        绘制多模态训练样本，改为使用统一复用组件，支持：
        - 多模态训练（未传modality）：输出 RGB、X（伪彩可视化）与并排对比图
        - 模态消融（传入modality）：rgb仅输出RGB；X仅输出对应X模态

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
            resolve_x_modality,
        )

        # 批数据
        images = batch["img"]  # [B, 3+Xch, H, W]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]

        # 统一 batch_idx 类型
        batch_idx = ensure_batch_idx_long(batch["batch_idx"])
        batch["batch_idx"] = batch_idx

        # 动态 X 通道数（与验证器一致）
        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3

        # 拆分模态（严格 [RGB, X] 顺序）
        rgb_images, x_images = split_modalities(images, xch)
        precision_contract = extract_precision_contract_from_batch(batch)
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])

        if self.args.debug:
            import numpy as np
            for idx in range(rgb_images.shape[0]):
                r = rgb_images[idx]
                x = x_images[idx]
                r_min, r_max = r.min().item(), r.max().item()
                x_min, x_max = x.min().item(), x.max().item()
                flag = " *** BLACK ***" if r_max < 0.01 else ""
                LOGGER.warning(
                    f"[PLOT-DBG] SPLIT idx={idx} ni={ni} | RGB [{r_min:.4f}, {r_max:.4f}]{flag} | "
                    f"X [{x_min:.4f}, {x_max:.4f}] | dtype={r.dtype}"
                )
            # 检查 plot_images 入口处 numpy 转换后的数据
            rgb_np = rgb_images.cpu().float().numpy()
            for idx in range(rgb_np.shape[0]):
                img_i = rgb_np[idx]
                flag = " *** BLACK ***" if img_i.max() < 0.01 else ""
                LOGGER.warning(
                    f"[PLOT-DBG] TO_PLOT idx={idx} | max={img_i.max():.6f} min={img_i.min():.6f}{flag}"
                )

        # 解析 X 模态类型（支持 'X'/具体模态/None）
        x_modality = resolve_x_modality(self.modality, getattr(self, 'data', None))

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
                # X-only，默认灰度可视化（与val一致，伪彩关闭）
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

        # 多模态训练：输出 RGB、X、并排三类图片
        # 1) RGB
        plot_images(
            rgb_visual,
            batch_idx,
            cls,
            bboxes,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
            on_plot=self.on_plot,
        )

        # 2) X（灰度）
        x_visual = visualize_x_to_3ch(
            x_images,
            colorize=False,
            x_modality=x_modality,
            precision=precision_contract["x"],
        )

        if self.args.debug:
            LOGGER.info(
                f"[MM-DEBUG] visualize_x_to_3ch | "
                f"x_visual dtype={x_visual.dtype} shape={x_visual.shape} "
                f"[{x_visual.min():.4f}, {x_visual.max():.4f}]"
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

        # 3) 并排对比（复制bbox到左右半幅）
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
        """
        绘制多模态训练指标图表。

        继承父类的指标绘制功能，添加多模态特定的指标可视化。
        """
        from ultralytics.utils.plotting import plot_distill_results

        # 调用父类的指标绘制（生成标准 results.png）
        super().plot_metrics()

        # 蒸馏训练时生成独立的蒸馏指标图
        if self._distill_cfg is not None:
            plot_distill_results(file=self.csv, family="yolomm", on_plot=self.on_plot)

        LOGGER.info("多模态训练指标绘制完成")
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        获取多模态检测模型，确保使用正确的通道数。
        
        重写父类方法以支持多模态输入的动态通道数配置。
        
        Args:
            cfg (str, optional): 模型配置文件路径
            weights (str, optional): 预训练权重路径
            verbose (bool): 是否打印详细信息
            
        Returns:
            DetectionModel: 配置了正确通道数的检测模型
        """
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK
        
        # 计算正确的输入通道数（双模态训练 与 模态消融 均按 Pch+Xch 构建）
        x_channels = self.data.get('Xch', 3)
        channels = 3 + x_channels  # RGB(3) + X(Xch)
        if verbose and RANK in {-1, 0}:
            tag = "消融" if self.modality else "多模态"
            LOGGER.info(f"{tag}模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch总输入")
        
        # 构建多模态 cfg_dict（注入 dataset_config + input 字段）
        from ultralytics.nn.mm.utils import prepare_multimodal_cfg_dict

        cfg_dict = prepare_multimodal_cfg_dict(
            cfg, self.data,
            mm_input_role=getattr(self.args, 'mm_input_role', None)
        )
        model = DetectionModel(cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        
        # 更新multimodal_router的dataset_config（如果存在）
        if hasattr(model, 'multimodal_router') and model.multimodal_router:
            model.multimodal_router.update_dataset_config(self.data)
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"已更新MultiModalRouter的数据集配置，Xch={self.data.get('Xch', 3)}")
        # 注入运行时模态消融参数到路由器
        if hasattr(model, 'mm_router') and model.mm_router and self.modality:
            # 仅设置参数，不在训练预处理阶段做置零/拼接，统一由路由处理
            model.mm_router.set_runtime_params(
                self.modality,
                strategy=getattr(self.args, 'ablation_strategy', None),
                seed=getattr(self.args, 'seed', None),
            )
        
        if weights:
            model.load(weights)

        sync_x_modality_name_to_model(model, extract_x_modality_name_from_data(getattr(self, "data", None)))

        try:
            imgsz = int(getattr(self.args, "imgsz", 640))
            report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
            log_default_complexity(model, report, LOGGER)
        except Exception as e:
            LOGGER.warning(f"模型复杂度统计失败（可忽略，不影响训练）：{e}")
        
        return model

    def save_model(self):
        """
        保存多模态模型，包含模态配置信息。

        重写父类方法，确保多模态配置信息被正确保存。
        """
        # 调用父类的模型保存
        super().save_model()

        # 保存多模态配置到模型检查点
        x_modality_name = extract_x_modality_name_from_data(getattr(self, "data", None))
        ckpt = torch_load(self.last, map_location='cpu')
        set_x_modality_name_in_ckpt_metadata(ckpt, x_modality_name)
        if hasattr(self, 'multimodal_config'):
            ckpt['multimodal_config'] = self.multimodal_config
        ckpt['modality'] = self.modality  # 保存单模态训练信息
        torch.save(ckpt, self.last)

        # 如果存在best模型，也更新它
        if self.best.exists():
            ckpt_best = torch_load(self.best, map_location='cpu')
            set_x_modality_name_in_ckpt_metadata(ckpt_best, x_modality_name)
            if hasattr(self, 'multimodal_config'):
                ckpt_best['multimodal_config'] = self.multimodal_config
            ckpt_best['modality'] = self.modality
            torch.save(ckpt_best, self.best)

    def final_eval(self):
        """
        执行最终评估，包含多模态特定的评估指标。

        注意：父类final_eval()没有返回值，所以这里主要是执行评估并记录多模态信息。
        """
        # 执行标准评估（父类方法没有返回值）
        super().final_eval()

        # 导出 LLM 友好的 JSON 格式验证结果
        from ultralytics.utils.llm_export import export_final_val_llm_json

        try:
            export_final_val_llm_json(self)
        except Exception as e:
            LOGGER.warning(f"LLM JSON export failed: {e}")

        # 记录多模态特定信息
        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            # 记录模态信息
            _rgb_modality, x_modality = resolve_modality_pair(self.multimodal_config['models'], strict=True)
            if self.modality:
                LOGGER.info(f"最终评估完成 - 单模态训练: {self.modality}-only")
            else:
                LOGGER.info(f"最终评估完成 - 双模态训练: RGB+{x_modality}")
        else:
            LOGGER.info("最终评估完成 - 多模态训练")

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

        config = load_distill_config(yaml_path)
        student = de_parallel(self.model)
        self.distill_runtime = DistillRuntime(
            config=config,
            mode=mode,
            family="yolomm",
            student_model=student,
            device=student.device if hasattr(student, 'device') else next(student.parameters()).device,
        )
        self._distill_student_collector = self.distill_runtime.register_student_hooks(student)


# ---------------------------------------------------------------------------
# Pruned model fine-tuning
# ---------------------------------------------------------------------------

def require_pruned_checkpoint(ckpt: dict) -> dict:
    """Validate that a checkpoint carries the minimal pruned marker.

    Args:
        ckpt: Loaded checkpoint dict.

    Returns:
        The ``prune_info`` sub-dict.

    Raises:
        ValueError: If the checkpoint is not a valid pruned checkpoint.
    """
    prune_info = ckpt.get("prune_info")
    if not isinstance(prune_info, dict) or prune_info.get("is_pruned") is not True:
        raise ValueError(
            'finetrain 仅接受带 `prune_info={"is_pruned": True}` 的剪枝权重。'
        )
    return prune_info

class PrunedMultiModalDetectionTrainer(MultiModalDetectionTrainer):
    """Trainer specialized for fine-tuning pruned YOLOMM models.

    Key differences from ``MultiModalDetectionTrainer``:
    - Loads the pruned model structure directly from the checkpoint (no YAML rebuild).
    - Validates that the checkpoint carries ``prune_info = {"is_pruned": True}``.
    - Propagates ``prune_info`` into every checkpoint saved during training.
    - Rejects ``model_scale``/``scale`` arguments that would trigger architecture rebuild.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.prune_info = None

        # Reject structure-rebuilding arguments
        if getattr(self.args, "model_scale", None):
            raise ValueError(
                "finetrain 不支持 model_scale/scale; 剪枝后结构必须直接来自 checkpoint。"
            )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Use the already-injected pruned model; skip YAML-based rebuild."""
        if isinstance(self.model, torch.nn.Module):
            # Model was pre-injected by YOLOMM.finetrain()
            if not isinstance(self.prune_info, dict) or self.prune_info.get("is_pruned") is not True:
                raise ValueError("剪枝后训练器收到内存模型，但缺少合法 prune_info。")
            return self.model

        # Fallback: load from .pt path
        if not str(self.model).endswith(".pt"):
            raise ValueError("finetrain 仅接受带 prune_info 的剪枝后 .pt 权重。")

        from ultralytics.nn.tasks import attempt_load_one_weight

        weights_obj, ckpt = attempt_load_one_weight(self.model)
        self.prune_info = require_pruned_checkpoint(ckpt)
        self.model = weights_obj
        return self.model

    def setup_model(self):
        """Override to prevent YAML-based model construction for pruned models.

        This method ensures the loaded pruned model has proper trainability state
        by restoring any frozen floating parameters from legacy checkpoints.
        """
        if not isinstance(self.model, torch.nn.Module):
            self.model = self.get_model()

        if not isinstance(self.prune_info, dict) or self.prune_info.get("is_pruned") is not True:
            raise ValueError("剪枝后训练器收到模型，但缺少合法 prune_info。")

        # Normalize trainability for legacy pruned checkpoints.
        # New pruned.pt exports will have restored == [] (already clean).
        restored = restore_parameter_trainability(self.model)
        remaining = find_frozen_floating_parameters(self.model)
        if remaining:
            raise RuntimeError(
                "Legacy pruned checkpoint still contains frozen floating parameters "
                f"after finetrain normalization: {remaining[:20]}"
            )
        if restored:
            LOGGER.info(
                f"[PrunedFinetrain] Restored trainability for {len(restored)} parameters "
                "from legacy pruned checkpoint before trainer freeze policy is applied."
            )

        try:
            imgsz = int(getattr(self.args, "imgsz", 640))
            report = compute_default_multimodal_complexity_report(self.model, imgsz=imgsz)
            log_default_complexity(self.model, report, LOGGER)
        except Exception as e:
            LOGGER.warning(f"剪枝后训练模型复杂度统计失败（可忽略，不影响训练）：{e}")

    def save_model(self):
        """Save checkpoint with ``prune_info`` preserved."""
        super().save_model()

        if not isinstance(self.prune_info, dict) or self.prune_info.get("is_pruned") is not True:
            raise ValueError("PrunedMultiModalDetectionTrainer.save_model() 缺少合法 prune_info。")

        targets = [self.last]
        if self.best.exists():
            targets.append(self.best)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            epoch_pt = self.wdir / f"epoch{self.epoch}.pt"
            if epoch_pt.exists():
                targets.append(epoch_pt)

        for path in targets:
            if path.exists():
                ckpt = torch_load(path, map_location="cpu")
                # Preserve full prune_info dict (not just the marker)
                ckpt["prune_info"] = self.prune_info
                torch.save(ckpt, path)
