# Ultralytics YOLO, AGPL-3.0 license

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

from ultralytics.models.yolo.multimodal.val import MultiModalDetectionValidator
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.metrics import SemSegMetrics


class MultiModalSemSegValidator(MultiModalDetectionValidator):
    """Validator for YOLOMM semantic segmentation with bincount-based metrics."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar=None, args=args, _callbacks=_callbacks)
        self.args.task = "semseg"
        self.metrics = SemSegMetrics()
        self._target_shape = None
        self.results_dir = None

    def init_metrics(self, model):
        self.names = model.names
        self.nc = len(self.names)
        self.metrics = SemSegMetrics(names=self.names)
        self.seen = 0
        self.dataset = getattr(self.dataloader, "dataset", None)
        if self.args.save_json:
            self.results_dir = self.save_dir / "results"
            self.results_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        batch["img"] = apply_dataset_precision_contract(
            batch["img"].float(), extract_batch_precision_contract(batch), raw_dtype=raw_dtype
        )
        batch["sem_mask"] = batch["sem_mask"].to(self.device, non_blocking=True).long()
        self._target_shape = tuple(batch["sem_mask"].shape[-2:])
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        return batch

    def postprocess(self, preds):
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if preds.shape[-2:] != self._target_shape:
            preds = F.interpolate(preds, size=self._target_shape, mode="bilinear", align_corners=False)
        if self.nc > 1:
            return preds.argmax(dim=1).to(torch.int32)
        else:
            return preds.gt(0).squeeze(1).to(torch.int32)

    def update_metrics(self, preds, batch):
        if self.args.save_json:
            self._save_pred_masks(preds, batch)
        self.metrics.update_stats(preds, batch["sem_mask"])
        self.seen += preds.shape[0]

    def get_stats(self):
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        return self.metrics.results_dict

    def gather_stats(self):
        """DDP: reduce confusion matrix to rank 0."""
        import torch.distributed as dist
        if RANK == -1 or not dist.is_available() or not dist.is_initialized():
            return
        if self.metrics.matrix is None:
            cm_nc = self.metrics.cm_nc
            self.metrics.matrix = torch.zeros((cm_nc, cm_nc), device=self.device, dtype=torch.float32)
        dist.reduce(self.metrics.matrix, dst=0, op=dist.ReduceOp.SUM)
        if RANK == 0:
            gathered = [None] * dist.get_world_size()
            dist.gather_object(self.metrics.nt_per_image, gathered, dst=0)
            self.metrics.nt_per_image = np.sum(gathered, axis=0)
        elif RANK > 0:
            dist.gather_object(self.metrics.nt_per_image, None, dst=0)

    def get_desc(self):
        return ("%22s" + "%11s" * 4) % ("Class", "Images", "Pixels", "mIoU", "PixAcc")

    def print_results(self):
        super().print_results()
        stats = self.metrics
        LOGGER.info(
            f"{colorstr('SemSeg:')} mIoU={stats.miou:.4f}, "
            f"pixel_acc={stats.pixel_accuracy:.4f}"
        )
        # Per-class detail
        summary = stats.summary()
        if summary:
            LOGGER.info(
                ("%22s" + "%11s" * 3) % ("Class", "Images", "Pixels", "IoU")
            )
            for row in summary:
                LOGGER.info(
                    ("%22s" + "%11d" * 2 + "%11.4f") % (
                        row["Class"], row["Images"], row["Pixels"], row["IoU"]
                    )
                )
        if self.args.save_json and self.results_dir:
            LOGGER.info(f"Semantic prediction masks saved to {self.results_dir}")

    def _save_pred_masks(self, preds, batch):
        """Save predicted masks as PNG files, with inverse label_mapping if applicable."""
        if self.results_dir is None:
            return
        im_files = batch.get("im_file", [])
        if not im_files:
            return
        preds_np = preds.cpu().numpy()
        if self.dataset and hasattr(self.dataset, "label_mapping") and self.dataset.label_mapping:
            preds_np = self.dataset.convert_label(preds_np, inverse=True)
        preds_np = preds_np.astype(np.uint8, copy=False)
        for pred, im_file in zip(preds_np, im_files):
            save_path = self.results_dir / Path(im_file).with_suffix(".png").name
            Image.fromarray(pred).save(save_path)

    def finalize_metrics(self, *args, **kwargs):
        pass
