# Ultralytics YOLO, AGPL-3.0 license

"""
多模态分类训练器

本模块提供多模态分类任务的训练器，支持 RGB+X 模态的图像分类训练。
采用 YAML 配置 + txt 标签文件的数据管理范式，与检测任务保持一致。
"""

from copy import copy
from typing import Any, Dict, Optional

import torch

from ultralytics.data import YOLOMultiModalClassifyDataset, build_dataloader
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.afss import AFSSConfig, AFSSRuntime
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.models.utils.multimodal.vis import (
    concat_side_by_side,
    extract_precision_contract_from_batch,
    prepare_rgb_for_display,
    split_modalities,
    visualize_x_to_3ch,
)
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import is_parallel, strip_optimizer
from ultralytics.nn.mm.complexity import compute_default_multimodal_complexity_report, log_default_complexity
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    normalize_modality_token,
    set_x_modality_name_in_ckpt_metadata,
    sync_x_modality_name_to_model,
)


class MultiModalClassificationTrainer(BaseTrainer):
    """
    多模态分类训练器

    继承 BaseTrainer，集成多模态数据加载和训练流程，
    支持 RGB+X 模态的图像分类任务。

    Attributes:
        model: 分类模型
        data: 数据集配置字典
        loss_names: 损失项名称列表
        modality: 当前训练模态（None=双模态，'rgb'/'depth'等=单模态）

    Methods:
        get_model: 返回配置好的分类模型
        build_dataset: 构建多模态分类数据集
        get_dataloader: 返回数据加载器
        get_validator: 返回验证器

    Examples:
        >>> from ultralytics.models.yolo.multimodal.classify import MultiModalClassificationTrainer
        >>> trainer = MultiModalClassificationTrainer(overrides={"data": "mm_cls.yaml"})
        >>> trainer.train()
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: Optional[Dict[str, Any]] = None,
        _callbacks=None
    ):
        """
        初始化多模态分类训练器

        Args:
            cfg: 默认配置
            overrides: 配置覆盖
            _callbacks: 回调函数列表
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"

        # 默认图像尺寸
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224

        # 预先初始化模态属性：BaseTrainer.__init__ 内会调用 self.get_dataset()，需要先具备该字段
        self.modality = normalize_modality_token(overrides.get("modality", None))
        overrides["modality"] = self.modality
        self.is_dual_modal = self.modality is None

        super().__init__(cfg, overrides, _callbacks)

        # 以最终 args 为准（允许通过 cfg/CLI 覆盖）
        self.modality = normalize_modality_token(getattr(self.args, "modality", self.modality))
        # 回写 args，确保训练内 validator/copy(args) 看到一致 token
        self.args.modality = self.modality
        self.is_dual_modal = self.modality is None

        # 日志
        if self.modality:
            LOGGER.info(f"初始化 MultiModalClassificationTrainer - 单模态训练: {self.modality}-only")
        else:
            LOGGER.info("初始化 MultiModalClassificationTrainer - 双模态训练")

        self._sync_afss_task_name()
        self._ensure_default_afss_classify_overrides()
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
        """Keep AFSS task name aligned with current trainer task."""
        setattr(self.args, "afss_task_name", "classify")

    def _ensure_default_afss_classify_overrides(self):
        """Inject classification-specific AFSS semantic defaults when missing."""
        raw_overrides = getattr(self.args, "afss_task_overrides", None)
        if raw_overrides is None:
            raw_overrides = {}
        elif hasattr(raw_overrides, "__dict__"):
            raw_overrides = vars(raw_overrides).copy()
        elif not isinstance(raw_overrides, dict):
            raw_overrides = dict(raw_overrides)

        classify_override = raw_overrides.get("classify")
        if classify_override is None:
            classify_override = {}
        elif hasattr(classify_override, "__dict__"):
            classify_override = vars(classify_override).copy()
        elif not isinstance(classify_override, dict):
            classify_override = dict(classify_override)

        classify_override.setdefault("sufficiency_mode", "top1_prob_if_correct")
        raw_overrides["classify"] = classify_override
        setattr(self.args, "afss_task_overrides", raw_overrides)

    def _ensure_afss_runtime(self, dataset):
        """Initialize AFSS runtime once training dataset is available."""
        if not self.afss_config.enabled or self.afss_runtime is not None:
            return self.afss_runtime
        self._sync_afss_task_name()
        self._ensure_default_afss_classify_overrides()
        self.afss_config = AFSSConfig.from_args(self.args)
        self.afss_runtime = AFSSRuntime.from_dataset(
            dataset=dataset,
            args=self.args,
            save_dir=self.save_dir,
            resume=bool(getattr(self.args, "resume", False)),
        )
        return self.afss_runtime

    def _afss_on_train_start(self, trainer):
        """Callback: announce AFSS runtime bootstrap state."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_start()

    def _afss_on_train_epoch_start(self, trainer):
        """Callback: refresh AFSS selection plan for current epoch."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_start(trainer.epoch)

    def _afss_on_train_epoch_end(self, trainer):
        """Callback: persist AFSS state and perform scheduled scoring."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_end(
                trainer.epoch,
                trainer=trainer,
                validator=trainer.validator,
            )

    def get_dataset(self):
        """
        获取数据集配置

        使用 check_det_dataset 解析 YAML 配置文件，
        与检测任务保持一致的数据管理范式。
        """
        # BaseTrainer.__init__ 已创建 self.args，此处统一确定模态（避免 __init__ 时序问题）
        self.modality = normalize_modality_token(getattr(self.args, "modality", getattr(self, "modality", None)))
        # 回写 args，避免后续读取产生 token 不一致
        self.args.modality = self.modality
        self.is_dual_modal = self.modality is None

        # 使用检测任务的数据集解析（支持 YAML 格式）
        self.data = check_det_dataset(self.args.data)

        # 确保必要字段存在
        if "nc" not in self.data:
            self.data["nc"] = len(self.data.get("names", {}))

        # 计算通道数
        self._calculate_channels()

        # BaseTrainer.__init__ 会用返回值覆盖 self.data，必须显式返回
        return self.data

    def _calculate_channels(self):
        """计算输入通道数"""
        xch = self.data.get("Xch", 3)

        if self.modality:
            # 单模态训练
            self.data["channels"] = 3
        else:
            # 双模态训练
            self.data["channels"] = 3 + xch

        LOGGER.info(f"输入通道数: {self.data['channels']} (Xch={xch})")

    def set_model_attributes(self):
        """设置模型的类别名称"""
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """
        返回配置好的分类模型

        Args:
            cfg: 模型配置
            weights: 预训练权重
            verbose: 是否显示模型信息

        Returns:
            ClassificationModel: 分类模型
        """
        from ultralytics.nn.mm.utils import prepare_multimodal_cfg_dict

        cfg_dict = prepare_multimodal_cfg_dict(
            cfg, self.data,
            mm_input_role=getattr(self.args, 'mm_input_role', None)
        )
        model = ClassificationModel(
            cfg_dict,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1
        )

        if weights:
            model.load(weights)

        # 初始化参数
        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout

        # 确保所有参数可训练
        for p in model.parameters():
            p.requires_grad = True

        sync_x_modality_name_to_model(model, extract_x_modality_name_from_data(getattr(self, "data", None)))
        model.is_multimodal = True
        try:
            imgsz = int(getattr(self.args, "imgsz", 224))
            report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
            log_default_complexity(model, report, LOGGER)
        except Exception as e:
            LOGGER.warning(f"分类模型复杂度统计失败（可忽略，不影响训练）：{e}")

        return model

    def setup_model(self):
        """
        加载或创建模型

        Returns:
            模型检查点（如果有）
        """
        import torchvision

        if str(self.model) in torchvision.models.__dict__:
            # 使用 torchvision 预训练模型
            self.model = torchvision.models.__dict__[self.model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
            ckpt = None
        else:
            ckpt = super().setup_model()

        # 调整输出层
        ClassificationModel.reshape_outputs(self.model, self.data["nc"])
        return ckpt

    def save_model(self):
        """保存分类模型并写入 canonical x_modality metadata。"""
        from ultralytics.utils.patches import torch_load

        super().save_model()

        x_modality_name = extract_x_modality_name_from_data(getattr(self, "data", None))

        ckpt = torch_load(self.last, map_location="cpu")
        set_x_modality_name_in_ckpt_metadata(ckpt, x_modality_name)
        torch.save(ckpt, self.last)

        if self.best.exists():
            ckpt_best = torch_load(self.best, map_location="cpu")
            set_x_modality_name_in_ckpt_metadata(ckpt_best, x_modality_name)
            torch.save(ckpt_best, self.best)

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """
        构建多模态分类数据集

        Args:
            img_path: 图像路径
            mode: 模式（train/val/test）
            batch: 批次信息（未使用）

        Returns:
            YOLOMultiModalClassifyDataset: 多模态分类数据集
        """
        return YOLOMultiModalClassifyDataset(
            img_path=img_path,
            data=self.data,
            args=self.args,
            augment=mode == "train",
            prefix=mode,
            precision_report_dir=self.save_dir,
        )

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train"
    ):
        """
        返回数据加载器

        Args:
            dataset_path: 数据集路径
            batch_size: 批次大小
            rank: 进程排名（分布式训练）
            mode: 模式（train/val/test）

        Returns:
            DataLoader: 数据加载器
        """
        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)

        shuffle = mode == "train"
        sampler_override = None
        if mode == "train" and self.afss_config.enabled:
            runtime = self._ensure_afss_runtime(dataset)
            sampler_override = runtime.create_sampler(rank=rank, shuffle=shuffle)

        loader = build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=shuffle,
            rank=rank,
            sampler_override=sampler_override,
        )

        # 附加推理 transforms
        if mode != "train" and hasattr(loader.dataset, "torch_transforms"):
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms

        return loader

    def preprocess_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预处理批次数据，contract-aware 精度归一化。"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        batch["img"] = apply_dataset_precision_contract(
            batch["img"].float(),
            extract_batch_precision_contract(batch),
            raw_dtype=raw_dtype,
        )
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def progress_string(self) -> str:
        """返回训练进度字符串"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        """返回多模态分类验证器"""
        self.loss_names = ["loss"]

        from ultralytics.models.yolo.multimodal.classify.val import (
            MultiModalClassificationValidator
        )

        return MultiModalClassificationValidator(
            self.test_loader,
            self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )

    def label_loss_items(
        self,
        loss_items: Optional[torch.Tensor] = None,
        prefix: str = "train"
    ):
        """返回带标签的损失字典"""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_metrics(self):
        """绘制指标图"""
        plot_results(file=self.csv, classify=True, on_plot=self.on_plot)

    def final_eval(self):
        """最终评估"""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.data = self.args.data
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

        # 导出 LLM 友好的 JSON 格式验证结果
        from ultralytics.utils.llm_export import export_final_val_llm_json

        try:
            export_final_val_llm_json(self)
        except Exception as e:
            LOGGER.warning(f"LLM JSON export failed: {e}")

    def plot_training_samples(self, batch: Dict[str, torch.Tensor], ni: int):
        """绘制训练样本，多模态 RGB/X/并排三组可视化。"""
        x_modality = self.data.get("x_modality", "unknown")
        xch = self.data.get("Xch", 3)
        batch_idx = torch.arange(len(batch["img"]))

        # 拆分模态
        rgb_images, x_images = split_modalities(batch["img"], xch)
        precision_contract = extract_precision_contract_from_batch(batch)

        # RGB 可视化
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])
        plot_images(
            images=rgb_visual,
            batch_idx=batch_idx,
            cls=batch["cls"],
            fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
            on_plot=self.on_plot,
        )

        # X 模态可视化
        x_visual = visualize_x_to_3ch(
            x_images, colorize=False, x_modality=x_modality, precision=precision_contract["x"]
        )
        plot_images(
            images=x_visual,
            batch_idx=batch_idx,
            cls=batch["cls"],
            fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
            on_plot=self.on_plot,
        )

        # 并排 multimodal 可视化
        side = concat_side_by_side(rgb_visual, x_visual)
        plot_images(
            images=side,
            batch_idx=batch_idx,
            cls=batch["cls"],
            fname=self.save_dir / f"train_batch{ni}_labels_multimodal.jpg",
            on_plot=self.on_plot,
        )
