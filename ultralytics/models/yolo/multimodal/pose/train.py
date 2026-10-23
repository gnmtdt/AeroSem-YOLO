# Ultralytics YOLO, AGPL-3.0 license

"""
Multi-Modal Pose Trainer.

Provides MultiModalPoseTrainer for RGB+X pose estimation training.
"""

from copy import copy

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.nn.mm.complexity import compute_default_multimodal_complexity_report, log_default_complexity
from ultralytics.nn.tasks import PoseModel
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    get_modality_path_from_data,
    normalize_modality_token,
    resolve_rgb_x_pair,
    set_x_modality_name_in_ckpt_metadata,
    sync_x_modality_name_to_model,
)
from ultralytics.engine.afss import AFSSConfig, AFSSRuntime


class MultiModalPoseTrainer(PoseTrainer):
    """
    Multi-modal pose estimation trainer (RGB+X).

    Reuses YOLOMM routing with 6+ channel input for pose keypoint prediction.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize MultiModalPoseTrainer with multi-modal configuration."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

        # Modality control consistent with YOLOMM detect/segment/obb
        self.modality = normalize_modality_token(getattr(self.args, "modality", None))
        # 回写 args，确保训练内 validator/copy(args) 看到一致 token
        self.args.modality = self.modality
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

        if self.modality:
            LOGGER.info(f"MultiModalPoseTrainer initialized - single modal: {self.modality}-only")
        else:
            LOGGER.info("MultiModalPoseTrainer initialized - dual modal training")

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
        task_name = str(getattr(self.args, "task", "pose"))
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
        """Callback: announce AFSS runtime bootstrap."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_start()

    def _afss_on_train_epoch_start(self, trainer):
        """Callback: refresh AFSS adaptive selection plan."""
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

    def preprocess_batch(self, batch):
        """多模态 Pose 感知的预处理。

        与 MultiModalDetectionTrainer.preprocess_batch 保持一致的 contract 归一化策略，
        并保留 Pose 任务所需的 keypoints 处理。
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        img = batch["img"].float()

        # contract-driven preprocessing
        img = apply_dataset_precision_contract(img, extract_batch_precision_contract(batch), raw_dtype=raw_dtype)
        batch["img"] = img

        # keypoints 处理（Pose 任务特有）
        if "keypoints" in batch:
            batch["keypoints"] = batch["keypoints"].to(self.device).float()

        return batch

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build multi-modal pose dataset with RGB+X image pipeline.

        Args:
            img_path: Path to images directory.
            mode: Dataset mode ('train' or 'val').
            batch: Batch size.

        Returns:
            Multi-modal dataset instance.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
            multi_modal_image=True,
            x_modality=self._determine_x_modality_from_data(),
            x_modality_dir=self._get_x_modality_path(self._determine_x_modality_from_data()),
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
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

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize PoseModel with dynamic channel count based on modality.

        Args:
            cfg: Model configuration file or dict.
            weights: Path to pretrained weights.
            verbose: Whether to print model info.

        Returns:
            Initialized PoseModel instance.
        """
        # 计算正确的输入通道数（双模态训练 与 模态消融 均按 Pch+Xch 构建）
        x_channels = self.data.get("Xch", 3)
        channels = 3 + x_channels  # RGB(3) + X(Xch)
        if verbose and RANK in {-1, 0}:
            tag = "消融" if self.modality else "多模态"
            LOGGER.info(f"{tag}Pose模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch")

        # 检查 kpt_shape 配置，提示 kobj_loss 计算条件
        kpt_shape = self.data.get("kpt_shape", [17, 3])
        if verbose and RANK in {-1, 0}:
            kpt_dim = kpt_shape[1] if len(kpt_shape) > 1 else 2
            if kpt_dim == 3:
                LOGGER.info(f"关键点配置: {kpt_shape[0]}点 × {kpt_dim}维 (x,y,可见性) → kobj_loss: 已启用")
            else:
                LOGGER.warning(
                    f"关键点配置: {kpt_shape[0]}点 × {kpt_dim}维 (仅x,y坐标) → kobj_loss: 已禁用 (数据集无可见性标注)"
                )

        cfg_in = cfg or (self.args.model if hasattr(self, 'args') else None)
        from ultralytics.nn.mm.utils import prepare_multimodal_cfg_dict

        cfg_dict = prepare_multimodal_cfg_dict(
            cfg_in, self.data,
            mm_input_role=getattr(self.args, 'mm_input_role', None)
        )
        model = PoseModel(cfg_dict, nc=self.data["nc"], ch=channels,
                          data_kpt_shape=kpt_shape, verbose=verbose and RANK == -1)

        # Configure multi-modal router for single modality mode
        if hasattr(model, "mm_router") and model.mm_router and self.modality:
            model.mm_router.set_runtime_params(
                self.modality,
                strategy=getattr(self.args, "ablation_strategy", None),
                seed=getattr(self.args, "seed", None),
            )

        if weights:
            model.load(weights)

        sync_x_modality_name_to_model(model, extract_x_modality_name_from_data(getattr(self, "data", None)))

        # Unified multimodal complexity logs
        try:
            imgsz = int(getattr(self.args, "imgsz", 640))
            report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
            log_default_complexity(model, report, LOGGER)
        except Exception as e:
            LOGGER.warning(f"模型复杂度统计失败（可忽略，不影响训练）：{e}")

        return model

    def get_validator(self):
        """Return multi-modal pose validator with consistent loss names."""
        from .val import MultiModalPoseValidator

        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return MultiModalPoseValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def plot_training_samples(self, batch, ni):
        """
        绘制多模态 Pose 训练样本，支持三视图模式（RGB / X / multimodal）。

        - keypoints 只在单图域（RGB 和 X 分别绘制），并排图仅绘制 bbox；
        - 单模态消融时仅输出对应模态；
        - 使用 contract-aware 可视化预处理。
        """
        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
            prepare_rgb_for_display,
            split_modalities,
            visualize_x_to_3ch,
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
            ensure_batch_idx_long,
            resolve_x_modality,
        )

        images = batch["img"]  # [B, 3+Xch, H, W]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]

        batch_idx = ensure_batch_idx_long(batch["batch_idx"])
        batch["batch_idx"] = batch_idx

        # keypoints（Pose 任务特有）
        kpts = batch.get("keypoints", None)

        # X 通道数
        xch = self.data.get("Xch", 3) if hasattr(self, "data") and self.data else 3

        # 拆分模态
        rgb_images, x_images = split_modalities(images, xch)
        precision_contract = extract_precision_contract_from_batch(batch)
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])

        # X 模态类型
        x_modality = resolve_x_modality(self.modality, getattr(self, "data", None))

        # 单模态消融
        if self.modality:
            if self.modality == "RGB":
                plot_images(
                    rgb_visual,
                    batch_idx,
                    cls,
                    bboxes,
                    kpts=kpts,
                    paths=paths,
                    fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
                    on_plot=self.on_plot,
                )
            else:
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
                    kpts=kpts,
                    paths=[p.replace(".jpg", f"_{x_modality}.jpg") for p in paths],
                    fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
                    on_plot=self.on_plot,
                )
            return

        # 双模态：1) RGB（带 keypoints）
        plot_images(
            rgb_visual,
            batch_idx,
            cls,
            bboxes,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
            on_plot=self.on_plot,
        )

        # 2) X（带 keypoints）
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
            kpts=kpts,
            paths=[p.replace(".jpg", f"_{x_modality}.jpg") for p in paths],
            fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
            on_plot=self.on_plot,
        )

        # 3) 并排对比（仅 bbox，不传 keypoints 以避免坐标系问题）
        side_by_side_images = concat_side_by_side(rgb_visual, x_visual)
        batch_ids_dup, cls_ids_dup, bboxes_dup, _ = duplicate_bboxes_for_side_by_side(
            batch_idx, cls, bboxes, None
        )
        plot_images(
            side_by_side_images,
            batch_ids_dup,
            cls_ids_dup,
            bboxes_dup,
            paths=[p.replace(".jpg", "_multimodal.jpg") for p in paths],
            fname=self.save_dir / f"train_batch{ni}_labels_multimodal.jpg",
            on_plot=self.on_plot,
        )

    def _determine_x_modality_from_data(self):
        """
        Determine X modality name from data.yaml configuration.

        Returns:
            X modality name string (e.g., 'depth', 'thermal', 'ir').
        """
        data = getattr(self, "data", {}) or {}
        x_modality = extract_x_modality_name_from_data(data)
        return x_modality if x_modality != "unknown" else "depth"

    def _get_x_modality_path(self, x_modality: str):
        """
        Get X modality directory path from data.yaml modalities mapping.

        Args:
            x_modality: X modality name.

        Returns:
            Directory path for X modality images.
        """
        data = getattr(self, "data", {}) or {}
        return get_modality_path_from_data(data, x_modality)

    def final_eval(self):
        """Execute final evaluation and export LLM-friendly JSON results."""
        super().final_eval()

        # Export LLM-friendly JSON format validation results
        from ultralytics.utils.llm_export import export_final_val_llm_json

        try:
            export_final_val_llm_json(self)
        except Exception as e:
            LOGGER.warning(f"LLM JSON export failed: {e}")

        # Log multi-modal specific information
        _rgb_modality, x_modality = resolve_rgb_x_pair(getattr(self, 'data', {}) or {}, default_x=self._determine_x_modality_from_data())
        if self.modality:
            LOGGER.info(f"Final evaluation complete - single modal: {self.modality}-only")
        else:
            LOGGER.info(f"Final evaluation complete - dual modal: RGB+{x_modality}")

    def save_model(self):
        """Save model and persist canonical x_modality metadata."""
        import torch
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
