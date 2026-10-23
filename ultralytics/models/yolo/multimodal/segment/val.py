from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import callbacks
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.mm.utils import (
    extract_x_modality_name_from_data,
    get_modality_path_from_data,
    get_runtime_x_modality_name,
    normalize_modality_token,
    resolve_rgb_x_pair,
)


class MultiModalSegmentationValidator(SegmentationValidator):
    """
    Multimodal segmentation validator for RGB+X inputs.

    Provides channel-aware warmup and accepts a runtime modality parameter,
    mirroring the detection-side multimodal validator behavior.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)

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

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):  # type: ignore[override]
        self.training = trainer is not None
        if self.training:
            # Align with training setup
            self.device = trainer.device
            if self.data is None:
                self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            model.eval()
            # Inject runtime modality to router during train-time val
            try:
                if hasattr(model, "mm_router") and model.mm_router and self.modality:
                    model.mm_router.set_runtime_params(self.modality)
            except Exception:
                pass
        else:
            # 方案A：不在子类中手动创建 AutoBackend，避免二次包裹与错误 fuse 调用。
            # 委托父类创建并管理 AutoBackend（包括设备、数据、warmup 等）。
            pass

        # Defer to parent implementation for the main loop
        return super().__call__(trainer=trainer, model=model)

    def init_metrics(self, model):  # type: ignore[override]
        """Initialize metrics and capture runtime X modality name from loaded weights."""
        super().init_metrics(model)
        self.x_modality_name = get_runtime_x_modality_name(model)

    def preprocess(self, batch):
        """多模态分割感知的预处理。

        与 MultiModalDetectionValidator.preprocess 保持一致的 contract 归一化策略，
        并保留分割任务所需的 masks 处理。
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

        # masks 处理（分割任务特有）
        if "masks" in batch:
            batch["masks"] = batch["masks"].to(self.device).float()

        return batch

    # -----------------
    # Dataset building
    # -----------------
    def build_dataset(self, img_path, mode: str = "val", batch=None):
        """
        构建多模态分割验证数据集。

        通过传递 multi_modal_image=True 启用 YOLOMultiModalImageDataset，
        并从 data.yaml/配置中解析 x_modality 与目录映射。严格遵循分割任务，
        使用 task='segment' 使 use_segments=True。
        """
        # 解析模态组合
        data = getattr(self, "data", {}) or {}
        modalities = None
        mod_map = None
        if "modality_used" in data and isinstance(data["modality_used"], list):
            modalities = data["modality_used"]
        elif "models" in data and isinstance(data["models"], list):
            modalities = data["models"]
        mod_map = data.get("modality") or data.get("modalities")

        # 提取 X 模态
        _rgb_modality, x_modality = resolve_rgb_x_pair(data, default_x=extract_x_modality_name_from_data(data))
        x_modality_dir = get_modality_path_from_data(data, x_modality)

        # stride
        stride = getattr(self, "stride", 32) or 32

        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=True,
            stride=stride,
            multi_modal_image=True,
            x_modality=x_modality,
            x_modality_dir=x_modality_dir,
            enable_self_modal_generation=False,  # 验证默认关闭自体生成
            precision_report_dir=self.save_dir,
        )

    # -----------------
    # Visualization (GT samples)
    # -----------------
    def plot_val_samples(self, batch, ni):
        """多模态分割验证：绘制 GT（RGB/X/并排 + mask）。"""
        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
            prepare_rgb_for_display,
            split_modalities,
            visualize_x_to_3ch,
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
            ensure_batch_idx_long,
        )

        images = batch["img"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        masks = batch.get("masks", None)

        batch_idx = ensure_batch_idx_long(batch.get("batch_idx")) if "batch_idx" in batch else None
        if batch_idx is None:
            import torch
            batch_idx = ensure_batch_idx_long(torch.zeros(cls.shape[0], dtype=torch.long))

        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        rgb_images, x_images = split_modalities(images, xch)
        x_modality = getattr(self, 'x_modality_name', 'unknown')

        # contract-aware 可视化预处理
        precision_contract = extract_precision_contract_from_batch(batch)
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])

        # RGB
        plot_images(
            rgb_visual,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            paths=paths,
            fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg",
            on_plot=self.on_plot,
        )
        # X
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
            fname=self.save_dir / f"val_batch{ni}_labels_{x_modality}.jpg",
            on_plot=self.on_plot,
        )
        # 并排（仅 bbox）
        side = concat_side_by_side(rgb_visual, x_visual)
        bidx2, cls2, bb2, _ = duplicate_bboxes_for_side_by_side(batch_idx, cls, bboxes, None)
        plot_images(
            side,
            bidx2,
            cls2,
            bb2,
            paths=[p.replace('.jpg', '_multimodal.jpg') for p in paths],
            fname=self.save_dir / f"val_batch{ni}_labels_multimodal.jpg",
            on_plot=self.on_plot,
        )

    # -----------------
    # Visualization (Preds)
    # -----------------
    def plot_predictions(self, batch, preds, ni):
        """多模态分割验证：绘制预测（RGB/X/并排 + mask）。

        批量绘制策略：与YOLOMM检测验证器保持一致的输出风格。
        - 按batch绘制，而非逐图绘制
        - 输出格式：val_batch{ni}_pred_{modality}.jpg
        - 每图最多100个实例，避免性能问题
        """

        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            extract_precision_contract_from_batch,
            prepare_rgb_for_display,
            split_modalities,
            visualize_x_to_3ch,
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
        )
        import torch
        from ultralytics.utils import ops

        CAP = 100  # 每图绘制上限

        images = batch["img"]
        paths = batch["im_file"]
        B, C, H, W = images.shape

        # 1. 合并batch内所有预测（参考检测验证器策略）
        for i, p in enumerate(preds):
            p["batch_idx"] = torch.ones_like(p["cls"]) * i

        keys = preds[0].keys()
        batched_preds = {}

        # 合并所有预测字段
        for k in keys:
            if k == "masks":
                # masks特殊处理：限制每图最多CAP个
                masks_list = []
                for p in preds:
                    m = p["masks"][:CAP] if p["masks"].numel() else p["masks"]
                    masks_list.append(m.to(torch.uint8).cpu())
                batched_preds[k] = torch.cat(masks_list, dim=0) if masks_list and masks_list[0].numel() else torch.zeros((0, H, W), dtype=torch.uint8)
            else:
                batched_preds[k] = torch.cat([p[k][:CAP] for p in preds], dim=0)

        # 2. 转换坐标格式：xyxy -> xywhn
        batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])
        batched_preds["bboxes"][:, 0] /= W
        batched_preds["bboxes"][:, 1] /= H
        batched_preds["bboxes"][:, 2] /= W
        batched_preds["bboxes"][:, 3] /= H

        # 3. 拆分模态
        xch = self.data.get('Xch', 3) if hasattr(self, 'data') and self.data else 3
        rgb_images, x_images = split_modalities(images, xch)
        x_modality = getattr(self, 'x_modality_name', 'unknown')

        # contract-aware 可视化预处理
        precision_contract = extract_precision_contract_from_batch(batch)
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])

        # 4. 批量绘制三视图
        # RGB视图
        plot_images(
            rgb_visual,
            batched_preds["batch_idx"].long(),
            batched_preds["cls"],
            batched_preds["bboxes"],
            confs=batched_preds.get("conf"),
            masks=batched_preds.get("masks"),
            paths=paths,
            fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

        # X视图
        x_visual = visualize_x_to_3ch(
            x_images, colorize=False, x_modality=x_modality,
            precision=precision_contract["x"],
        )
        plot_images(
            x_visual,
            batched_preds["batch_idx"].long(),
            batched_preds["cls"],
            batched_preds["bboxes"],
            confs=batched_preds.get("conf"),
            masks=batched_preds.get("masks"),
            paths=[p.replace('.jpg', f'_{x_modality}.jpg') for p in paths],
            fname=self.save_dir / f"val_batch{ni}_pred_{x_modality}.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

        # 并排视图
        side = concat_side_by_side(rgb_visual, x_visual)
        bidx2, cls2, bb2, conf2 = duplicate_bboxes_for_side_by_side(
            batched_preds["batch_idx"].long(),
            batched_preds["cls"],
            batched_preds["bboxes"],
            batched_preds.get("conf")
        )
        masks_side = torch.cat([batched_preds["masks"], batched_preds["masks"]], dim=2) if batched_preds["masks"].numel() else batched_preds["masks"]
        plot_images(
            side,
            bidx2,
            cls2,
            bb2,
            confs=conf2,
            masks=masks_side,
            paths=[p.replace('.jpg', '_multimodal.jpg') for p in paths],
            fname=self.save_dir / f"val_batch{ni}_pred_multimodal.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    # ------------------------------------------------------------------
    # AFSS per-sample scoring helpers
    # ------------------------------------------------------------------
    def afss_score_sample(self, pred, batch, si):
        """Build one AFSS score row for segment task using validator-native primitives."""
        if "im_file" not in batch:
            raise KeyError("AFSS sample scoring requires batch['im_file']")

        pbatch = self._prepare_batch(si, batch)
        predn = self._prepare_pred(pred, pbatch)
        result = self._process_batch(predn, pbatch)

        tp_box = result.get("tp")
        tp_mask = result.get("tp_m")
        matched_box = int(tp_box[:, 0].sum()) if tp_box is not None and len(tp_box) else 0
        matched_mask = int(tp_mask[:, 0].sum()) if tp_mask is not None and len(tp_mask) else 0
        pred_count = int(len(predn["cls"]))
        gt_count = int(len(pbatch["cls"]))
        pred_mask_count = int(predn["masks"].shape[0]) if "masks" in predn else 0
        gt_mask_count = int(pbatch["masks"].shape[0]) if "masks" in pbatch else 0

        if gt_count == 0 and pred_count == 0:
            box_precision = 1.0
            box_recall = 1.0
            mask_precision = 1.0
            mask_recall = 1.0
            empty_case = "no_pred_no_label"
        elif gt_count == 0:
            box_precision = 0.0
            box_recall = 1.0
            mask_precision = 0.0
            mask_recall = 1.0
            empty_case = "pred_without_label"
        elif pred_count == 0:
            box_precision = 0.0
            box_recall = 0.0
            mask_precision = 0.0
            mask_recall = 0.0
            empty_case = "label_without_pred"
        else:
            box_precision = matched_box / pred_count
            box_recall = matched_box / gt_count
            mask_precision = matched_mask / pred_count
            mask_recall = matched_mask / gt_count
            empty_case = "normal"

        box_sufficiency = min(box_precision, box_recall)
        mask_sufficiency = min(mask_precision, mask_recall)
        sufficiency_raw = min(box_sufficiency, mask_sufficiency)

        return {
            "im_file": batch["im_file"][si],
            "task_name": "segment",
            "precision_op": box_precision,
            "recall_op": box_recall,
            "sufficiency_raw": sufficiency_raw,
            "valid_for_afss": True,
            "task_metrics": {
                "box_precision": float(box_precision),
                "box_recall": float(box_recall),
                "mask_precision": float(mask_precision),
                "mask_recall": float(mask_recall),
                "box_sufficiency": float(box_sufficiency),
                "mask_sufficiency": float(mask_sufficiency),
                "matched_box_count": int(matched_box),
                "matched_mask_count": int(matched_mask),
                "pred_count": int(pred_count),
                "gt_count": int(gt_count),
                "pred_mask_count": int(pred_mask_count),
                "gt_mask_count": int(gt_mask_count),
                "overlap_mask": int(bool(getattr(self.args, "overlap_mask", False))),
                "empty_case": empty_case,
            },
            # Keep legacy compatibility fields for state store and older tooling.
            "precision": float(box_precision),
            "recall": float(box_recall),
            "matched_count": int(matched_box),
            "pred_count": int(pred_count),
            "gt_count": int(gt_count),
        }

    def afss_score_batch(self, preds, batch):
        """Score all samples in a batch for segment AFSS adapter reuse."""
        return [self.afss_score_sample(pred, batch, si) for si, pred in enumerate(preds)]

    def score_sample(self, pred, batch, si):
        """Backward-compatible AFSS sample helper for legacy scorer calls."""
        return self.afss_score_sample(pred, batch, si)

    def score_batch(self, preds, batch):
        """Backward-compatible AFSS batch helper for legacy scorer calls."""
        return self.afss_score_batch(preds, batch)
