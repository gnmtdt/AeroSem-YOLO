# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch

from ultralytics.models.yolo.multimodal.train import MultiModalDetectionTrainer
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.nn.tasks import SemSegModel
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_results
from ultralytics.nn.mm.complexity import compute_default_multimodal_complexity_report, log_default_complexity
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    sync_x_modality_name_to_model,
)
from .val import MultiModalSemSegValidator


class MultiModalSemSegTrainer(MultiModalDetectionTrainer):
    """Trainer for YOLOMM semantic segmentation.

    Inherits MultiModalDetectionTrainer for modality ablation + AFSS + distill.
    Overrides only the task-specific methods below.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = overrides or {}
        overrides["task"] = "semseg"
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        self.args.ignore_index = int(self.data.get("ignore_index", getattr(self.args, "ignore_index", 255)))
        x_modality = self._determine_x_modality_from_data()
        x_dir = self._get_x_modality_path(x_modality)
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=False,  # SemSeg 不需要矩形批处理，mask 像素严格对齐
            stride=gs,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_dir,
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
            precision_report_dir=self.save_dir,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        x_channels = self.data.get("Xch", 3)
        channels = 3 + x_channels
        from ultralytics.nn.mm.utils import prepare_multimodal_cfg_dict

        cfg_dict = prepare_multimodal_cfg_dict(
            cfg, self.data,
            mm_input_role=getattr(self.args, 'mm_input_role', None)
        )
        model = SemSegModel(cfg_dict, nc=self.data["nc"], ch=channels, verbose=verbose and RANK == -1)
        if model.mm_router and self.modality:
            model.mm_router.set_runtime_params(
                self.modality,
                strategy=getattr(self.args, "ablation_strategy", None),
                seed=getattr(self.args, "seed", None),
            )
        if weights:
            model.load(weights)
        sync_x_modality_name_to_model(model, extract_x_modality_name_from_data(self.data))
        if verbose and RANK in {-1, 0}:
            try:
                imgsz = int(getattr(self.args, "imgsz", 640))
                report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
                log_default_complexity(model, report, LOGGER)
            except Exception as e:
                LOGGER.warning(f"模型复杂度统计失败（可忽略，不影响训练）：{e}")
        return model

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        batch["img"] = apply_dataset_precision_contract(
            batch["img"].float(), extract_batch_precision_contract(batch), raw_dtype=raw_dtype
        )
        batch["sem_mask"] = batch["sem_mask"].to(self.device, non_blocking=True).long()
        return batch

    def get_validator(self):
        criterion = getattr(self.model, "criterion", None) or self.model.init_criterion()
        self.loss_names = getattr(criterion, "names", ("semseg_ce_loss", "semseg_dice_loss", "semseg_aux_loss"))
        return MultiModalSemSegValidator(
            dataloader=self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def save_model(self):
        import torch
        from datetime import datetime

        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": de_parallel(self.model).half(),
            "ema": de_parallel(self.ema.ema).half() if self.ema else None,
            "optimizer": self.optimizer.state_dict(),
            "train_args": self.args,
            "yaml": self.model.yaml,
            "nc": self.data["nc"],
            "names": self.data["names"],
            "x_modality_name": getattr(self.model, "x_modality_name", None),
            "Xch": self.data.get("Xch", 3),
            "ignore_index": self.args.ignore_index,
            "date": datetime.now().isoformat(),
        }
        torch.save(ckpt, self.last)
        if self.best_fitness == self.last_fitness:
            torch.save(ckpt, self.best)
        del ckpt

    def plot_metrics(self):
        from ultralytics.utils.plotting import plot_distill_results
        plot_results(file=self.csv, semseg=True, on_plot=self.on_plot)
        if self._distill_cfg is not None:
            plot_distill_results(file=self.csv, family="yolomm", on_plot=self.on_plot)

    def _sync_afss_task_name(self):
        self.args.afss_task_name = "semseg"

    def plot_training_samples(self, batch, ni):
        """Plot multimodal semseg training samples with semantic mask overlays.

        Outputs (dual-modal): RGB+mask overlay, X+mask overlay, side-by-side.
        Ablation (single-modal): RGB+mask or X+mask only.
        """
        from ultralytics.utils.plotting import plot_images, colors
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
            split_modalities,
            prepare_rgb_for_display,
            visualize_x_to_3ch,
            concat_side_by_side,
            resolve_x_modality,
        )

        images = batch["img"]
        paths = batch["im_file"]
        sem_mask = batch["sem_mask"]

        xch = self.data.get("Xch", 3) if hasattr(self, "data") and self.data else 3
        rgb_images, x_images = split_modalities(images, xch)
        precision_contract = extract_precision_contract_from_batch(batch)
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])
        x_modality = resolve_x_modality(self.modality, getattr(self, "data", None))

        # Build alpha-blended sem_mask overlay on RGB
        rgb_overlay = self._overlay_sem_mask(rgb_visual, sem_mask, alpha=0.4)
        empty_idx = torch.zeros(0, dtype=torch.long)
        empty_cls = torch.zeros(0)
        empty_bbox = torch.zeros((0, 4))

        if self.modality:
            if self.modality == "RGB":
                plot_images(
                    rgb_overlay, empty_idx, empty_cls, empty_bbox, paths=paths,
                    fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
                    on_plot=self.on_plot,
                )
            else:
                x_visual = visualize_x_to_3ch(
                    x_images, colorize=False, x_modality=x_modality,
                    precision=precision_contract["x"],
                )
                x_overlay = self._overlay_sem_mask(x_visual, sem_mask, alpha=0.4)
                plot_images(
                    x_overlay, empty_idx, empty_cls, empty_bbox,
                    paths=[p.replace(".jpg", f"_{x_modality}.jpg") for p in paths],
                    fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
                    on_plot=self.on_plot,
                )
            return

        # Dual-modal
        plot_images(
            rgb_overlay, empty_idx, empty_cls, empty_bbox, paths=paths,
            fname=self.save_dir / f"train_batch{ni}_labels_rgb.jpg",
            on_plot=self.on_plot,
        )

        x_visual = visualize_x_to_3ch(
            x_images, colorize=False, x_modality=x_modality,
            precision=precision_contract["x"],
        )
        x_overlay = self._overlay_sem_mask(x_visual, sem_mask, alpha=0.4)
        plot_images(
            x_overlay, empty_idx, empty_cls, empty_bbox,
            paths=[p.replace(".jpg", f"_{x_modality}.jpg") for p in paths],
            fname=self.save_dir / f"train_batch{ni}_labels_{x_modality}.jpg",
            on_plot=self.on_plot,
        )

        side_by_side = concat_side_by_side(rgb_overlay, x_overlay)
        plot_images(
            side_by_side, empty_idx, empty_cls, empty_bbox,
            paths=[p.replace(".jpg", "_multimodal.jpg") for p in paths],
            fname=self.save_dir / f"train_batch{ni}_labels_multimodal.jpg",
            on_plot=self.on_plot,
        )

    @staticmethod
    def _overlay_sem_mask(images, sem_mask, alpha=0.5):
        """Blend dense class-id sem_mask as colored overlay on images.

        Args:
            images: [B, 3, H, W] float tensor in [0, 1] or [0, 255]
            sem_mask: [B, H, W] long tensor with class IDs (ignore_index pixels excluded)
            alpha: overlay opacity (0=transparent, 1=opaque)

        Returns:
            [B, 3, H, W] uint8 numpy array
        """
        import numpy as np
        from ultralytics.utils.plotting import colors

        B, _, H, W = images.shape
        nc = int(sem_mask[sem_mask != 255].max().item()) + 1 if (sem_mask != 255).any() else 1

        im_np = images.detach().cpu().float().numpy()
        if im_np.max() <= 1.0:
            im_np = im_np * 255.0
        im_np = im_np.clip(0, 255).astype(np.uint8).transpose(0, 2, 3, 1)  # B,H,W,3

        mask_np = sem_mask.cpu().numpy()
        overlay_rgb = np.zeros((B, H, W, 3), dtype=np.uint8)
        color_lut = np.array([colors(c, bgr=False) for c in range(nc)], dtype=np.uint8)  # (nc, 3)
        valid = mask_np < nc
        overlay_rgb[valid] = color_lut[mask_np[valid]]

        alpha_mask = valid.astype(np.float32)[..., None]
        blended = (im_np * (1 - alpha * alpha_mask) + overlay_rgb * alpha * alpha_mask)
        return blended.clip(0, 255).astype(np.uint8).transpose(0, 3, 1, 2)
