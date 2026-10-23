from copy import copy, deepcopy
from typing import Optional

import math
import random
import torch.nn as nn

from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.nn.mm.complexity import compute_default_multimodal_complexity_report, log_default_complexity
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    get_modality_path_from_data,
    normalize_modality_token,
    resolve_rgb_x_pair,
    set_x_modality_name_in_ckpt_metadata,
    sync_x_modality_name_to_model,
)
from ultralytics.engine.afss import AFSSConfig, AFSSRuntime


class MultiModalSegmentationTrainer(SegmentationTrainer):
    """
    Multimodal segmentation trainer for RGB+X inputs.

    This mirrors the YOLOMM detection trainer while switching the task to
    segmentation and using SegmentationModel for the head.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

        self.modality = normalize_modality_token(getattr(self.args, "modality", None))
        # 回写 args，确保训练内 validator/copy(args) 看到一致 token
        self.args.modality = self.modality
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

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

    # -----------------
    # Dataset building
    # -----------------
    def build_dataset(self, img_path, mode: str = "train", batch: Optional[int] = None):
        # Resolve x-modality info from data config
        x_modality, x_dir = self._resolve_x_modality_and_dir()

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
            x_modality=x_modality,
            x_modality_dir=x_dir,
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
            precision_report_dir=self.save_dir,
        )

    # -----------
    # Model init
    # -----------
    def get_model(self, cfg: str | dict | None = None, weights: str | None = None, verbose: bool = True):
        # 计算正确的输入通道数（双模态训练 与 模态消融 均按 Pch+Xch 构建）
        x_channels = self.data.get("Xch", 3)
        channels = 3 + x_channels  # RGB(3) + X(Xch)
        if verbose and RANK in {-1, 0}:
            tag = "消融" if self.modality else "多模态"
            LOGGER.info(f"{tag}分割模型初始化: RGB(3ch) + X({x_channels}ch) = {channels}ch")

        # 构建多模态 cfg_dict（注入 dataset_config + input 字段）
        from ultralytics.nn.mm.utils import prepare_multimodal_cfg_dict

        cfg_dict = prepare_multimodal_cfg_dict(
            cfg, self.data,
            mm_input_role=getattr(self.args, 'mm_input_role', None)
        )
        model = SegmentationModel(cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)

        if hasattr(model, "multimodal_router") and model.multimodal_router:
            model.multimodal_router.update_dataset_config(self.data)
            if verbose and RANK in {-1, 0}:
                LOGGER.info(f"已更新MultiModalRouter的数据集配置，Xch={self.data.get('Xch', 3)}")

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
        from ultralytics.models.yolo.multimodal.segment.val import MultiModalSegmentationValidator

        # Keep the same ordering as v8SegmentationLoss: box, seg, cls, dfl, semseg
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "semseg_loss"
        return MultiModalSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _sync_afss_task_name(self):
        """Keep AFSS task wiring aligned with trainer task for adapter-based runtime."""
        setattr(self.args, "afss_task_name", str(getattr(self.args, "task", "segment")))

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
        """Callback: refresh AFSS epoch selection."""
        if trainer.afss_runtime is not None:
            trainer.afss_runtime.on_train_epoch_start(trainer.epoch)

    def _afss_on_train_epoch_end(self, trainer):
        """Callback: persist AFSS state snapshot and run scheduled scoring."""
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

    # -----------------
    # Batch preprocessing
    # -----------------
    def preprocess_batch(self, batch):
        """多模态分割感知的预处理。

        与 MultiModalDetectionTrainer.preprocess_batch 保持一致的 contract 归一化策略，
        并保留分割任务所需的 masks 处理。
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        img = batch["img"].float()

        # contract-driven preprocessing
        img = apply_dataset_precision_contract(img, extract_batch_precision_contract(batch), raw_dtype=raw_dtype)
        batch["img"] = img

        # masks 处理（分割任务特有）
        if "masks" in batch:
            batch["masks"] = batch["masks"].to(self.device).float()

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

    # -----------------
    # Helper utilities
    # -----------------
    def _resolve_x_modality_and_dir(self) -> tuple[str, Optional[str]]:
        data = getattr(self, "data", {}) or {}
        _rgb_modality, x_mod = resolve_rgb_x_pair(data, default_x=extract_x_modality_name_from_data(data) or "depth")
        if x_mod == "unknown":
            LOGGER.warning("无法自动确定X模态类型，使用默认值: depth")
            x_mod = "depth"
        return x_mod, get_modality_path_from_data(data, x_mod)

    # -----------------
    # Visualization
    # -----------------
    def plot_training_samples(self, batch, ni):
        """
        绘制多模态训练样本，支持分割任务的掩码叠加。

        - 多模态（未传入单模态 ablation）: 分别输出 RGB、X（灰度可视化）与并排对比图；
        - 单模态（传入 modality）: 仅输出该模态；
        - 分割叠加：在 RGB 与 X 单独图像上叠加 mask；并排图由于坐标系变化，暂不叠加 mask。
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

        batch_idx = ensure_batch_idx_long(batch["batch_idx"]) if "batch_idx" in batch else None
        if batch_idx is None:
            # fallback to zeros if missing
            import torch

            batch_idx = ensure_batch_idx_long(torch.zeros(cls.shape[0], dtype=torch.long))
            batch["batch_idx"] = batch_idx

        # masks（可选）
        masks = batch.get("masks", None)

        # X 通道数
        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        rgb_images, x_images = split_modalities(images, xch)
        x_modality = resolve_x_modality(self.modality, getattr(self, 'data', None))

        # contract-aware 可视化预处理
        precision_contract = extract_precision_contract_from_batch(batch)
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])

        # 单模态
        if self.modality:
            if self.modality == "RGB":
                plot_images(
                    rgb_visual,
                    batch_idx,
                    cls,
                    bboxes,
                    masks=masks,
                    paths=paths,
                    fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
                    on_plot=self.on_plot,
                )
            else:
                x_visual = visualize_x_to_3ch(
                    x_images, colorize=False, x_modality=x_modality,
                    precision=precision_contract["x"],
                )
                plot_images(
                    x_visual,
                    batch_idx,
                    cls,
                    bboxes,
                    masks=masks,
                    paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
                    fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
                    on_plot=self.on_plot,
                )
            return

        # 双模态：RGB
        plot_images(
            rgb_visual,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
            on_plot=self.on_plot,
        )

        # 双模态：X（灰度）
        x_visual = visualize_x_to_3ch(
            x_images, colorize=False, x_modality=x_modality,
            precision=precision_contract["x"],
        )
        plot_images(
            x_visual,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
            fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
            on_plot=self.on_plot,
        )

        # 并排对比（仅 bbox 以避免 mask 坐标系偏移）
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

    # -----------------
    # Checkpoint I/O
    # -----------------
    def save_model(self):
        """保存模型并写入多模态元信息（与检测版一致）。"""
        from ultralytics.utils.patches import torch_load
        import torch

        super().save_model()

        x_modality_name = extract_x_modality_name_from_data(getattr(self, "data", None))
        ckpt = torch_load(self.last, map_location='cpu')
        set_x_modality_name_in_ckpt_metadata(ckpt, x_modality_name)
        if hasattr(self, 'multimodal_config'):
            ckpt['multimodal_config'] = getattr(self, 'multimodal_config', None)
        ckpt['modality'] = self.modality
        torch.save(ckpt, self.last)

        if self.best.exists():
            ckpt_best = torch_load(self.best, map_location='cpu')
            set_x_modality_name_in_ckpt_metadata(ckpt_best, x_modality_name)
            if hasattr(self, 'multimodal_config'):
                ckpt_best['multimodal_config'] = getattr(self, 'multimodal_config', None)
            ckpt_best['modality'] = self.modality
            torch.save(ckpt_best, self.best)

    def final_eval(self):
        """最终评估后记录多模态信息。"""
        super().final_eval()

        # 导出 LLM 友好的 JSON 格式验证结果
        from ultralytics.utils.llm_export import export_final_val_llm_json

        try:
            export_final_val_llm_json(self)
        except Exception as e:
            LOGGER.warning(f"LLM JSON export failed: {e}")

        if hasattr(self, 'multimodal_config') and self.multimodal_config:
            _rgb_modality, x_modality = resolve_rgb_x_pair(self.multimodal_config['models'], strict=True)
            if self.modality:
                LOGGER.info(f"最终评估完成 - 单模态训练: {self.modality}-only")
            else:
                LOGGER.info(f"最终评估完成 - 双模态训练: RGB+{x_modality}")
        else:
            LOGGER.info("最终评估完成 - 多模态分割")
