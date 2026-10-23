# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import numpy as np
from pathlib import Path

from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.utils import LOGGER, callbacks, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import (
    de_parallel,
    maybe_log_multimodal_canonical_summary,
    select_device,
    set_multimodal_complexity_owner_state,
    smart_inference_mode,
)
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.ops import Profile
from ultralytics.utils import TQDM
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    get_modality_path_from_data,
    get_runtime_x_modality_name,
    normalize_modality_token,
    resolve_rgb_x_pair,
)


class MultiModalOBBValidator(OBBValidator):
    """
    多模态 OBB 验证器：继承 OBBValidator，加入 6+ 通道 warmup 与 runtime 模态注入。
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        # 适配父类签名
        super().__init__(dataloader, save_dir, args, _callbacks)

        # 多模态标记
        if args:
            if isinstance(args, dict):
                self.modality = args.get("modality", None)
            else:
                self.modality = getattr(args, "modality", None)
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
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None

        if self.modality:
            LOGGER.info(f"初始化 MultiModalOBBValidator - 单模态验证: {self.modality}-only")
        else:
            LOGGER.info("初始化 MultiModalOBBValidator - 双模态验证")

    def build_dataset(self, img_path, mode="val", batch=None):
        """构建多模态 OBB 数据集，与 MultiModalOBBTrainer.build_dataset 对齐。"""
        stride = int(getattr(self, "stride", 32) or 32)
        x_modality = self._determine_x_modality_from_data()
        x_modality_dir = self._get_x_modality_path(x_modality)

        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=stride,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=getattr(self.args, "enable_self_modal_generation", False),
            precision_report_dir=self.save_dir,
        )

    def _determine_x_modality_from_data(self) -> str:
        """解析 data.yaml 中的 X 模态名称。"""
        data = getattr(self, "data", {}) or {}
        x_modality = extract_x_modality_name_from_data(data)
        return x_modality if x_modality != "unknown" else "unknown"

    def _get_x_modality_path(self, x_modality: str) -> str:
        """根据 data.yaml modalities 映射获取 X 模态目录。"""
        data = getattr(self, "data", {}) or {}
        return get_modality_path_from_data(data, x_modality)

    def preprocess(self, batch):
        """多模态 OBB 感知的预处理。

        与 MultiModalDetectionValidator.preprocess 保持一致的 contract 归一化策略。
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        img = batch["img"].float()
        # contract-driven preprocessing
        img = apply_dataset_precision_contract(img, extract_batch_precision_contract(batch), raw_dtype=raw_dtype)
        batch["img"] = img

        if self.args.half:
            batch["img"] = batch["img"].half()

        for k in ("batch_idx", "cls", "bboxes"):
            batch[k] = batch[k].to(self.device)

        return batch

    def init_metrics(self, model):  # type: ignore[override]
        """Initialize metrics and capture runtime X modality name from loaded weights."""
        super().init_metrics(model)
        self.x_modality_name = get_runtime_x_modality_name(model)

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        执行验证流程，支持 6+ 通道多模态输入与旋转框评估。
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)

        if self.training:
            self.device = trainer.device
            if self.data is None:
                self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
            if hasattr(model, "mm_router") and model.mm_router and self.modality:
                model.mm_router.set_runtime_params(
                    self.modality,
                    strategy=getattr(self.args, "ablation_strategy", None),
                    seed=getattr(self.args, "seed", None),
                )
        else:
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
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 6, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0
            if not pt:
                self.args.rect = False
            self.stride = model.stride
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            # runtime 模态注入
            try:
                if hasattr(model, "pt") and model.pt and hasattr(model, "model") and hasattr(model.model, "mm_router") and model.model.mm_router and self.modality:
                    model.model.mm_router.set_runtime_params(
                        self.modality,
                        strategy=getattr(self.args, "ablation_strategy", None),
                        seed=getattr(self.args, "seed", None),
                    )
            except Exception:
                pass

            # 多模态 warmup
            if hasattr(self, "data") and self.data and "Xch" in self.data:
                x_channels = self.data.get("Xch", 3)
                total_channels = 3 + x_channels
                LOGGER.info(f"执行 {total_channels} 通道多模态 OBB 模型 warmup (RGB:3 + X:{x_channels})")
                model.warmup(imgsz=(1 if pt else self.args.batch, total_channels, imgsz, imgsz))
            else:
                LOGGER.info("执行 6 通道多模态 OBB 模型 warmup (默认)")
                model.warmup(imgsz=(1 if pt else self.args.batch, 6, imgsz, imgsz))

            set_multimodal_complexity_owner_state(model, direct_ingress=True)
            maybe_log_multimodal_canonical_summary(model, imgsz=imgsz, logger=LOGGER, direct_ingress=True)

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
                if self.training:
                    orig_mode = model.training
                    try:
                        model.train()
                        self.loss += model.loss(batch, preds)[1]
                    finally:
                        if not orig_mode:
                            model.eval()
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            self.run_callbacks("on_val_batch_end")

        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
        return self.metrics.results_dict

    # ------------------------------------------------------------------
    # Visualization (GT samples + predictions)
    # ------------------------------------------------------------------

    def plot_val_samples(self, batch, ni):
        """
        绘制 OBB 验证样本，支持三视图模式（RGB / X / multimodal）。

        - OBB 使用 5 维 rotated boxes，plot_images 已支持；
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

        multimodal_images = batch["img"]  # [B, 3+Xch, H, W]
        xch = self.data.get("Xch", 3) if hasattr(self, "data") and self.data else 3

        rgb_images, x_images = split_modalities(multimodal_images, xch)
        precision_contract = extract_precision_contract_from_batch(batch)
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])

        x_modality = getattr(self, "x_modality_name", "unknown")
        batch_idx = ensure_batch_idx_long(batch["batch_idx"])
        batch["batch_idx"] = batch_idx

        # 单模态验证
        if self.modality:
            if self.modality == "RGB":
                plot_images(
                    rgb_visual,
                    batch_idx,
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg",
                    names=self.names,
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
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=[p.replace(".jpg", f"_{x_modality}.jpg") for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_labels_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
            return

        # 双模态：1) RGB
        plot_images(
            rgb_visual,
            batch_idx,
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

        # 2) X
        x_visual = visualize_x_to_3ch(
            x_images,
            colorize=False,
            x_modality=x_modality,
            precision=precision_contract["x"],
        )
        plot_images(
            x_visual,
            batch_idx,
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=[p.replace(".jpg", f"_{x_modality}.jpg") for p in batch["im_file"]],
            fname=self.save_dir / f"val_batch{ni}_labels_{x_modality}.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

        # 3) 并排对比
        side_by_side_images = concat_side_by_side(rgb_visual, x_visual)
        batch_ids_dup, cls_ids_dup, bboxes_dup, _ = duplicate_bboxes_for_side_by_side(
            batch_idx, batch["cls"].squeeze(-1), batch["bboxes"], None
        )
        plot_images(
            side_by_side_images,
            batch_ids_dup,
            cls_ids_dup,
            bboxes_dup,
            paths=[p.replace(".jpg", "_multimodal.jpg") for p in batch["im_file"]],
            fname=self.save_dir / f"val_batch{ni}_labels_multimodal.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """
        绘制 OBB 预测结果，支持三视图模式（RGB / X / multimodal）。

        - 使用统一坐标制：output_to_target -> 像素xywh -> 归一化xywh -> 绘图；
        - OBB 使用 5 维 rotated boxes。
        """
        from ultralytics.utils.plotting import plot_images, output_to_target
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
            prepare_rgb_for_display,
            split_modalities,
            visualize_x_to_3ch,
            concat_side_by_side,
            to_norm_xywh_for_plot,
            duplicate_bboxes_for_side_by_side,
            resolve_x_modality,
            ensure_batch_idx_long,
            clip_boxes_norm_xywh,
        )

        multimodal_images = batch["img"]
        xch = self.data.get("Xch", 3) if hasattr(self, "data") and self.data else 3

        rgb_images, x_images = split_modalities(multimodal_images, xch)
        precision_contract = extract_precision_contract_from_batch(batch)
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])

        x_modality = getattr(self, "x_modality_name", "unknown")

        # 统一坐标制：output_to_target -> 像素xywh -> 归一化xywh
        batch_ids, cls_ids, boxes_xywh_px, confs = output_to_target(preds, max_det=self.args.max_det)
        _, _, H, W = rgb_images.shape
        img_hw = (H, W)
        batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm = to_norm_xywh_for_plot(
            batch_ids, cls_ids, boxes_xywh_px, confs, img_hw
        )
        batch_ids_norm = ensure_batch_idx_long(batch_ids_norm)
        if (isinstance(boxes_norm, torch.Tensor) and boxes_norm.numel() > 0) or (
            isinstance(boxes_norm, np.ndarray) and boxes_norm.size > 0
        ):
            boxes_norm = clip_boxes_norm_xywh(boxes_norm, 0.0, 1.0, 0.0, 1.0)

        # 单模态验证
        if self.modality:
            if self.modality == "RGB":
                plot_images(
                    rgb_visual,
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg",
                    names=self.names,
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
                    batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
                    paths=[p.replace(".jpg", f"_{x_modality}.jpg") for p in batch["im_file"]],
                    fname=self.save_dir / f"val_batch{ni}_pred_{x_modality}.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                )
            return

        # 双模态：1) RGB 预测
        plot_images(
            rgb_visual,
            batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

        # 2) X 预测
        x_visual = visualize_x_to_3ch(
            x_images,
            colorize=False,
            x_modality=x_modality,
            precision=precision_contract["x"],
        )
        plot_images(
            x_visual,
            batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm,
            paths=[p.replace(".jpg", f"_{x_modality}.jpg") for p in batch["im_file"]],
            fname=self.save_dir / f"val_batch{ni}_pred_{x_modality}.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

        # 3) 并排对比
        side_by_side_images = concat_side_by_side(rgb_visual, x_visual)
        batch_ids_dup, cls_ids_dup, boxes_dup, confs_dup = duplicate_bboxes_for_side_by_side(
            batch_ids_norm, cls_ids_norm, boxes_norm, confs_norm
        )
        plot_images(
            side_by_side_images,
            batch_ids_dup, cls_ids_dup, boxes_dup, confs_dup,
            paths=[p.replace(".jpg", "_multimodal.jpg") for p in batch["im_file"]],
            fname=self.save_dir / f"val_batch{ni}_pred_multimodal.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    # ------------------------------------------------------------------
    # AFSS per-sample scoring helpers
    # ------------------------------------------------------------------

    def afss_score_sample(self, pred, batch, si):
        """Score a single OBB sample with rotated-box matching primitives."""
        if "im_file" not in batch:
            raise KeyError("AFSS sample scoring requires batch['im_file']")
        pbatch = self._prepare_batch(si, batch)
        predn = self._prepare_pred(pred, pbatch)
        result = self._process_batch(predn, pbatch)
        tp = result["tp"]
        matched = int(tp[:, 0].sum()) if len(tp) else 0
        pred_count = int(len(predn["cls"]))
        gt_count = int(len(pbatch["cls"]))

        if gt_count == 0 and pred_count == 0:
            obb_precision = 1.0
            obb_recall = 1.0
        elif gt_count == 0:
            obb_precision = 0.0
            obb_recall = 1.0
        elif pred_count == 0:
            obb_precision = 0.0
            obb_recall = 0.0
        else:
            obb_precision = matched / pred_count
            obb_recall = matched / gt_count

        precision_op = float(obb_precision)
        recall_op = float(obb_recall)
        sufficiency_raw = min(precision_op, recall_op)
        im_file = str(batch["im_file"][si])
        return {
            "sample_key": str(Path(im_file).resolve()),
            "im_file": im_file,
            "task_name": "obb",
            "precision": precision_op,
            "recall": recall_op,
            "precision_op": precision_op,
            "recall_op": recall_op,
            "sufficiency_raw": sufficiency_raw,
            "valid_for_afss": True,
            "task_metrics": {
                "obb_precision": precision_op,
                "obb_recall": recall_op,
                "matched_count": int(matched),
                "pred_count": int(pred_count),
                "gt_count": int(gt_count),
            },
            # Keep legacy counters for compatibility with current state-store expectations.
            "matched_count": int(matched),
            "pred_count": int(pred_count),
            "gt_count": int(gt_count),
        }

    def afss_score_batch(self, preds, batch):
        """Score all OBB samples in a batch for AFSS adapter reuse."""
        return [self.afss_score_sample(pred, batch, si) for si, pred in enumerate(preds)]

    def score_sample(self, pred, batch, si):
        """Backward-compatible AFSS sample helper for legacy scorer calls."""
        return self.afss_score_sample(pred, batch, si)

    def score_batch(self, preds, batch):
        """Backward-compatible AFSS batch helper for legacy scorer calls."""
        return self.afss_score_batch(preds, batch)
