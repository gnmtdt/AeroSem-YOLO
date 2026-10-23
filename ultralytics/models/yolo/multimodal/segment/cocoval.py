# Ultralytics YOLO 🚀, AGPL-3.0 license

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.multimodal.segment.val import MultiModalSegmentationValidator
from ultralytics.utils import LOGGER, colorstr, ops
from ultralytics.utils.coco_eval_bbox_mm import COCOevalBBoxMM
from ultralytics.utils.coco_eval_segm_mm import COCOevalSegmMM, rle_encode, rle_area
from ultralytics.utils.coco_metrics import COCOMetrics, COCO_AREA_SMALL, COCO_AREA_MEDIUM
from ultralytics.nn.mm.complexity import compute_default_multimodal_complexity_report, build_default_complexity_summary
from ultralytics.utils.torch_utils import _complexity_base_model


class MultiModalSegmentationCOCOValidator(MultiModalSegmentationValidator):
    """
    多模态实例分割 COCO 验证器（纯本地实现，不依赖 pycocotools / faster-coco-eval）。

    输出两套 12 项 COCO 指标：
    - Mask（segm）：主指标（符合 COCO segmentation 的主评测口径）
    - Box（bbox）：辅助指标（便于对齐检测性能）

    重要约束（Fail-Fast）：
    - 必须提供预测 masks 与 GT masks，否则直接报错（不做静默降级）。
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)

        self.coco_metrics_mask = None
        self.coco_metrics_box = None

        self._img_shapes = []
        self._dt_img = []
        self._gt_img = []

        self._dt_cls = []
        self._dt_conf = []
        self._dt_bbox_xyxy = []
        self._dt_mask_rle = []
        self._dt_mask_area = []

        self._gt_cls = []
        self._gt_bbox_xyxy = []
        self._gt_mask_rle = []
        self._gt_mask_area = []

        self.gt_size_counts_mask = {"small": 0, "medium": 0, "large": 0}
        self.gt_size_counts_box = {"small": 0, "medium": 0, "large": 0}

        self.complexity_report = None
        self.gflops_total = None
        self.stage_gflops = {}

    def init_metrics(self, model):
        super().init_metrics(model)

        self.model = model
        self.coco_metrics_mask = COCOMetrics(save_dir=self.save_dir, names=getattr(model, "names", {}), plot=False)
        self.coco_metrics_box = COCOMetrics(save_dir=self.save_dir, names=getattr(model, "names", {}), plot=False)

        self._img_shapes = []
        self._dt_img = []
        self._gt_img = []
        self._dt_cls = []
        self._dt_conf = []
        self._dt_bbox_xyxy = []
        self._dt_mask_rle = []
        self._dt_mask_area = []
        self._gt_cls = []
        self._gt_bbox_xyxy = []
        self._gt_mask_rle = []
        self._gt_mask_area = []

        self.gt_size_counts_mask = {"small": 0, "medium": 0, "large": 0}
        self.gt_size_counts_box = {"small": 0, "medium": 0, "large": 0}

        try:
            imgsz = int(getattr(self.args, "imgsz", 640))
            base = _complexity_base_model(model)
            self.complexity_report = compute_default_multimodal_complexity_report(base, imgsz=imgsz)
            summary = build_default_complexity_summary(base, self.complexity_report)
            self.gflops_total = summary["gflops_total"]
            self.stage_gflops = summary["stage_gflops"]
        except Exception:
            self.complexity_report = None
            self.gflops_total = None
            self.stage_gflops = {}

    def get_desc(self):
        # 进度表头只保留 COCO 指标列，避免出现 "RGB+XCOCO-..." 这类粘连与冗余信息
        return ("%22s" + "%11s" * 3) % ("Class", "Images", "Instances", "COCO-Mask mAP@.5:.95")

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)

        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred, pbatch)

            ori_shape = pbatch["ori_shape"]  # (h, w)
            self._img_shapes.append((int(ori_shape[0]), int(ori_shape[1])))
            img_id = len(self._img_shapes) - 1

            # ---------- GT ----------
            gt_cls = pbatch.get("cls", None)
            gt_bboxes = pbatch.get("bboxes", None)
            gt_masks = pbatch.get("masks", None)
            if gt_cls is None or gt_bboxes is None or gt_masks is None:
                raise ValueError("分割 COCO 评测需要 batch 中包含 cls/bboxes/masks（GT）。")

            gt_cls = gt_cls.view(-1)
            if gt_cls.numel() > 0:
                gt_masks_bin = self._get_instance_masks(gt_cls, gt_masks, overlap_mask=bool(self.args.overlap_mask))
                gt_masks_rle, gt_mask_area = self._scale_and_encode_masks(
                    gt_masks_bin, ori_shape, pbatch.get("ratio_pad", None)
                )
                gt_bboxes_xyxy = gt_bboxes.detach().cpu().numpy()  # xyxy，已是原图坐标
                gt_cls_np = gt_cls.detach().cpu().numpy()

                if len(gt_masks_rle) != len(gt_cls_np):
                    raise ValueError(
                        f"GT masks 数量与 GT cls 数量不一致：masks={len(gt_masks_rle)} cls={len(gt_cls_np)}"
                    )

                for i in range(len(gt_cls_np)):
                    self._gt_img.append(img_id)
                    self._gt_cls.append(int(gt_cls_np[i]))
                    self._gt_bbox_xyxy.append(gt_bboxes_xyxy[i])
                    self._gt_mask_rle.append(gt_masks_rle[i])
                    self._gt_mask_area.append(float(gt_mask_area[i]))

                # GT 尺寸分布（mask area）
                for a in gt_mask_area.tolist():
                    if a < COCO_AREA_SMALL:
                        self.gt_size_counts_mask["small"] += 1
                    elif a < COCO_AREA_MEDIUM:
                        self.gt_size_counts_mask["medium"] += 1
                    else:
                        self.gt_size_counts_mask["large"] += 1

                # GT 尺寸分布（box area）
                gt_box_area = self._xyxy_area_np(gt_bboxes_xyxy)
                for a in gt_box_area.tolist():
                    if a < COCO_AREA_SMALL:
                        self.gt_size_counts_box["small"] += 1
                    elif a < COCO_AREA_MEDIUM:
                        self.gt_size_counts_box["medium"] += 1
                    else:
                        self.gt_size_counts_box["large"] += 1

            # ---------- DT ----------
            dt_bboxes = predn.get("bboxes", None)
            dt_cls = predn.get("cls", None)
            dt_conf = predn.get("conf", None)
            dt_masks = predn.get("masks", None)
            if dt_masks is None:
                raise ValueError("分割 COCO 评测需要 preds 中包含 masks（预测）。")

            if dt_bboxes is not None and dt_bboxes.numel() > 0:
                dt_masks_rle, dt_mask_area = self._scale_and_encode_masks(
                    dt_masks, ori_shape, pbatch.get("ratio_pad", None)
                )

                dt_bboxes_xyxy = dt_bboxes.detach().cpu().numpy()
                dt_cls_np = dt_cls.detach().cpu().numpy()
                dt_conf_np = dt_conf.detach().cpu().numpy()

                if len(dt_masks_rle) != len(dt_conf_np):
                    raise ValueError(
                        f"Pred masks 数量与 conf 数量不一致：masks={len(dt_masks_rle)} conf={len(dt_conf_np)}"
                    )

                for i in range(len(dt_conf_np)):
                    self._dt_img.append(img_id)
                    self._dt_cls.append(int(dt_cls_np[i]))
                    self._dt_conf.append(float(dt_conf_np[i]))
                    self._dt_bbox_xyxy.append(dt_bboxes_xyxy[i])
                    self._dt_mask_rle.append(dt_masks_rle[i])
                    self._dt_mask_area.append(float(dt_mask_area[i]))

    def get_stats(self):
        if self.coco_metrics_mask is None or self.coco_metrics_box is None:
            raise RuntimeError("COCO 指标容器未初始化，请先运行 init_metrics().")

        # 先计算 COCO（避免 super().get_stats() 清空 stats）
        coco_mask_dict, coco_box_dict = self._compute_coco_metrics()

        # 再计算/返回标准分割指标（Box+Mask P/R/mAP50/mAP50-95）
        base_stats = super().get_stats()

        # 统一输出（主指标为 Mask COCO）
        out = {}
        out.update(base_stats)
        out.update({f"metrics/coco_mask/{k}": v for k, v in coco_mask_dict.items()})
        out.update({f"metrics/coco_box/{k}": v for k, v in coco_box_dict.items()})

        out["metrics/coco/AP"] = coco_mask_dict.get("AP", 0.0)
        out["metrics/coco/AP50"] = coco_mask_dict.get("AP50", 0.0)
        out["metrics/coco/AP75"] = coco_mask_dict.get("AP75", 0.0)
        out["metrics/coco/AR100"] = coco_mask_dict.get("AR100", 0.0)

        out["fitness"] = float(coco_mask_dict.get("AP", 0.0))
        out["val/modality"] = self.modality if self.modality else "multimodal"
        if self.gflops_total:
            out["model/GFLOPs_default_total"] = float(self.gflops_total)
            out["model/GFLOPs_rgb_branch"] = float(self.stage_gflops.get("rgb_branch", 0.0))
            out["model/GFLOPs_x_branch"] = float(self.stage_gflops.get("x_branch", 0.0))
            out["model/GFLOPs_fusion"] = float(self.stage_gflops.get("fusion", 0.0))
            out["model/GFLOPs_head"] = float(self.stage_gflops.get("head", 0.0))

        return out

    def print_results(self):
        if self.coco_metrics_mask is None or self.coco_metrics_box is None:
            return

        # 先输出标准 val 摘要（与标准分割验证器保持一致）
        super().print_results()

        print(f"\n{colorstr('blue', 'bold', '=' * 80)}")
        print(f"{colorstr('blue', 'bold', '多模态实例分割 COCO 评估结果（纯本地实现）')}")
        print(f"{colorstr('blue', 'bold', '=' * 80)}")
        if self.modality:
            print(f"验证模式: {colorstr('cyan', f'{self.modality.upper()}-only')} (单模态)")
        else:
            print(f"验证模式: {colorstr('cyan', 'RGB+X')} (双模态)")
        print(f"数据集: {getattr(self.args, 'data', 'N/A')}")
        print(f"类别数: {getattr(self, 'nc', 'N/A')}")
        print(f"图像数: {getattr(self, 'seen', 0)}")

        # Mask 12项（主）
        print(f"\n{colorstr('green', 'bold', 'COCO Mask（主指标）')}")
        self._print_12(self.coco_metrics_mask)
        print(f"GT(mask) size: small={self.gt_size_counts_mask['small']}, medium={self.gt_size_counts_mask['medium']}, large={self.gt_size_counts_mask['large']}")

        # Box 12项（辅）
        print(f"\n{colorstr('green', 'bold', 'COCO Box（辅助）')}")
        self._print_12(self.coco_metrics_box)
        print(f"GT(box) size: small={self.gt_size_counts_box['small']}, medium={self.gt_size_counts_box['medium']}, large={self.gt_size_counts_box['large']}")

        print(f"{colorstr('blue', 'bold', '=' * 80)}")

        self._save_csv_results()
        if getattr(self.args, "save_json", True):
            self.save_json()

    def save_json(self, save_dir=None, filename=None):
        if self.coco_metrics_mask is None or self.coco_metrics_box is None:
            raise RuntimeError("COCO 指标尚未计算，无法保存 JSON。")

        save_dir = Path(save_dir or self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            modality_suffix = f"_{self.modality}" if self.modality else "_multimodal"
            filename = f"coco_seg_results{modality_suffix}.json"
        save_path = save_dir / filename

        results = {
            "evaluation_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "validator_type": "MultiModalSegmentationCOCOValidator",
                "modality": self.modality if self.modality else "RGB+X",
                "dataset": getattr(self.args, "data", "N/A"),
                "num_classes": getattr(self, "nc", None),
                "num_images": int(len(self._img_shapes)),
            },
            "coco_mask": self._metrics_to_dict(self.coco_metrics_mask),
            "coco_box": self._metrics_to_dict(self.coco_metrics_box),
            "gflops": {
                "total_default": float(self.gflops_total) if self.gflops_total else None,
                "stages": dict(self.stage_gflops or {}),
            },
            "gt_size_counts": {
                "mask": dict(self.gt_size_counts_mask),
                "box": dict(self.gt_size_counts_box),
            },
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        LOGGER.info(f"COCO(seg) 结果已保存: {save_path}")
        return save_path

    # -----------------
    # Internal helpers
    # -----------------
    @staticmethod
    def _xyxy_area_np(xyxy: np.ndarray) -> np.ndarray:
        xyxy = np.asarray(xyxy, dtype=np.float32)
        w = np.clip(xyxy[:, 2] - xyxy[:, 0], 0.0, None)
        h = np.clip(xyxy[:, 3] - xyxy[:, 1], 0.0, None)
        return w * h

    @staticmethod
    def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = xyxy.tolist()
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        return np.array([float(x1), float(y1), w, h], dtype=np.float32)

    @staticmethod
    def _metrics_to_dict(m: COCOMetrics) -> dict:
        return {
            "AP": float(getattr(m, "AP", 0.0)),
            "AP50": float(getattr(m, "AP50", 0.0)),
            "AP75": float(getattr(m, "AP75", 0.0)),
            "APsmall": float(getattr(m, "APsmall", 0.0)),
            "APmedium": float(getattr(m, "APmedium", 0.0)),
            "APlarge": float(getattr(m, "APlarge", 0.0)),
            "AR1": float(getattr(m, "AR1", 0.0)),
            "AR10": float(getattr(m, "AR10", 0.0)),
            "AR100": float(getattr(m, "AR100", 0.0)),
            "ARsmall": float(getattr(m, "ARsmall", 0.0)),
            "ARmedium": float(getattr(m, "ARmedium", 0.0)),
            "ARlarge": float(getattr(m, "ARlarge", 0.0)),
        }

    @staticmethod
    def _print_12(m: COCOMetrics):
        rows = [
            ("AP", m.AP),
            ("AP50", m.AP50),
            ("AP75", m.AP75),
            ("APsmall", m.APsmall),
            ("APmedium", m.APmedium),
            ("APlarge", m.APlarge),
            ("AR1", getattr(m, "AR1", 0.0)),
            ("AR10", getattr(m, "AR10", 0.0)),
            ("AR100", getattr(m, "AR100", 0.0)),
            ("ARsmall", getattr(m, "ARsmall", 0.0)),
            ("ARmedium", getattr(m, "ARmedium", 0.0)),
            ("ARlarge", getattr(m, "ARlarge", 0.0)),
        ]
        for k, v in rows:
            print(f"{k:<12}: {v:>8.3f}")

    @staticmethod
    def _get_instance_masks(gt_cls: torch.Tensor, gt_masks: torch.Tensor, overlap_mask: bool) -> torch.Tensor:
        """
        统一得到 shape [N, H, W] 的 per-instance 二值 GT masks。
        """
        if gt_cls.numel() == 0:
            return gt_masks.new_zeros((0, gt_masks.shape[-2], gt_masks.shape[-1]))

        if not overlap_mask:
            if gt_masks.ndim != 3:
                raise ValueError(f"Expected GT masks shape [N,H,W] when overlap_mask=False, got {gt_masks.shape}")
            return gt_masks.gt(0.5).to(torch.uint8)

        # overlap_mask=True: gt_masks 为单张图的 instance-id mask（形如 [1,H,W] 或 [H,W]）
        if gt_masks.ndim == 3 and gt_masks.shape[0] == 1:
            m = gt_masks[0]
        elif gt_masks.ndim == 2:
            m = gt_masks
        else:
            raise ValueError(f"Unexpected GT masks shape for overlap_mask=True: {gt_masks.shape}")

        nl = int(gt_cls.numel())
        index = torch.arange(nl, device=m.device).view(nl, 1, 1) + 1
        mm = m.repeat(nl, 1, 1)
        return (mm == index).to(torch.uint8)

    @staticmethod
    def _scale_and_encode_masks(masks: torch.Tensor, ori_shape, ratio_pad):
        """
        将 masks（[N,H,W] uint8/bool/float）缩放回原图尺寸并编码为 RLE。
        返回：
        - rles: list[dict]
        - areas: np.ndarray, shape [N]
        """
        if masks is None:
            return [], np.zeros((0,), dtype=np.float32)
        if isinstance(masks, torch.Tensor):
            if masks.numel() == 0:
                return [], np.zeros((0,), dtype=np.float32)
            if masks.dtype in (torch.float16, torch.float32, torch.float64):
                masks_u8 = masks.gt(0.5).to(torch.uint8).detach().cpu().numpy()  # [N,H,W]
            else:
                masks_u8 = masks.gt(0).to(torch.uint8).detach().cpu().numpy()  # [N,H,W]
        else:
            masks_u8 = np.asarray(masks, dtype=np.uint8)
            if masks_u8.size == 0:
                return [], np.zeros((0,), dtype=np.float32)
        if masks_u8.ndim != 3:
            raise ValueError(f"Expected masks shape [N,H,W], got {masks_u8.shape}")

        hwc = np.transpose(masks_u8, (1, 2, 0))  # [H,W,N]
        scaled = ops.scale_image(hwc, ori_shape, ratio_pad=ratio_pad)
        if scaled.ndim == 2:
            scaled = scaled[:, :, None]
        nchw = np.transpose(scaled, (2, 0, 1))  # [N,H0,W0]
        nchw = (nchw > 0).astype(np.uint8)

        rles = []
        areas = np.zeros((nchw.shape[0],), dtype=np.float32)
        for i in range(nchw.shape[0]):
            rle = rle_encode(nchw[i])
            rles.append(rle)
            areas[i] = float(rle_area(rle))
        return rles, areas

    def _compute_coco_metrics(self):
        # 基础一致性检查
        if len(self._img_shapes) == 0:
            raise ValueError("未收集到任何图像信息，无法计算 COCO 指标。")

        if len(self._dt_conf) != len(self._dt_bbox_xyxy) or len(self._dt_conf) != len(self._dt_cls) or len(self._dt_conf) != len(self._dt_mask_rle):
            raise ValueError("预测采集的 conf/cls/bbox/mask 数量不一致，请检查采集逻辑。")
        if len(self._gt_cls) != len(self._gt_bbox_xyxy) or len(self._gt_cls) != len(self._gt_mask_rle):
            raise ValueError("GT 采集的 cls/bbox/mask 数量不一致，请检查采集逻辑。")

        imgIds = list(range(len(self._img_shapes)))

        # -----------------
        # Mask（segm）
        # -----------------
        segm_dts = []
        for i in range(len(self._dt_conf)):
            segm_dts.append(
                {
                    "image_id": int(self._dt_img[i]),
                    "category_id": int(self._dt_cls[i]),
                    "score": float(self._dt_conf[i]),
                    "segmentation": self._dt_mask_rle[i],
                    "area": float(self._dt_mask_area[i]),
                    "id": i + 1,
                }
            )
        segm_gts = []
        for j in range(len(self._gt_cls)):
            segm_gts.append(
                {
                    "image_id": int(self._gt_img[j]),
                    "category_id": int(self._gt_cls[j]),
                    "segmentation": self._gt_mask_rle[j],
                    "area": float(self._gt_mask_area[j]),
                    "iscrowd": 0,
                    "ignore": 0,
                    "id": j + 1,
                }
            )

        catIds = sorted(list({d["category_id"] for d in segm_dts} | {g["category_id"] for g in segm_gts}))
        if not catIds:
            # 无 GT 时不允许继续（会导致“全 0”误解）
            raise ValueError("未采集到任何 GT 类别，无法计算 COCO(segm) 指标。")

        segm_eval = COCOevalSegmMM()
        segm_eval.set_data(gts=segm_gts, dts=segm_dts, imgIds=imgIds, catIds=catIds)
        segm_eval.evaluate()
        segm_eval.accumulate()
        segm_stats = segm_eval.summarize()
        segm_per_class = segm_eval.compute_per_class_metrics()

        for k, v in list(segm_stats.items()):
            if v == -1:
                segm_stats[k] = 0.0
        self.coco_metrics_mask.update(segm_stats)
        self.coco_metrics_mask.per_class_metrics = segm_per_class  # 供外部分析/保存

        # -----------------
        # Box（bbox）
        # -----------------
        box_dts = []
        for i in range(len(self._dt_conf)):
            xywh = self._xyxy_to_xywh(self._dt_bbox_xyxy[i])
            box_dts.append(
                {
                    "image_id": int(self._dt_img[i]),
                    "category_id": int(self._dt_cls[i]),
                    "bbox": xywh.tolist(),
                    "score": float(self._dt_conf[i]),
                    "area": float(xywh[2] * xywh[3]),
                    "id": i + 1,
                }
            )
        box_gts = []
        for j in range(len(self._gt_cls)):
            xywh = self._xyxy_to_xywh(self._gt_bbox_xyxy[j])
            box_gts.append(
                {
                    "image_id": int(self._gt_img[j]),
                    "category_id": int(self._gt_cls[j]),
                    "bbox": xywh.tolist(),
                    "iscrowd": 0,
                    "ignore": 0,
                    "area": float(xywh[2] * xywh[3]),
                    "id": j + 1,
                }
            )

        box_eval = COCOevalBBoxMM()
        box_eval.set_data(gts=box_gts, dts=box_dts, imgIds=imgIds, catIds=catIds)
        box_eval.evaluate()
        box_eval.accumulate()
        box_stats = box_eval.summarize()
        box_per_class = box_eval.compute_per_class_metrics()

        for k, v in list(box_stats.items()):
            if v == -1:
                box_stats[k] = 0.0
        self.coco_metrics_box.update(box_stats)
        self.coco_metrics_box.per_class_metrics = box_per_class

        return segm_stats, box_stats

    def _save_csv_results(self):
        try:
            save_dir = Path(self.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Overall
            overall = save_dir / "coco_seg_metrics_overall.csv"
            with open(overall, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Type", "Metric", "Value"])
                for typ, m in [("Mask", self.coco_metrics_mask), ("Box", self.coco_metrics_box)]:
                    d = self._metrics_to_dict(m)
                    for k, v in d.items():
                        w.writerow([typ, k, f"{v:.6f}"])
                w.writerow(["Meta", "GFLOPs(total[default])", f"{self.gflops_total:.3f}" if self.gflops_total else "N/A"])
                for key in ("rgb_branch", "x_branch", "fusion", "head"):
                    w.writerow(["Meta", f"GFLOPs({key})", f"{self.stage_gflops.get(key, 0.0):.3f}" if self.stage_gflops else "N/A"])
                w.writerow(["Meta", "Modality", self.modality if self.modality else "multimodal"])

            LOGGER.info(f"COCO(seg) CSV 已保存: {overall}")
        except Exception as e:
            LOGGER.warning(f"保存 COCO(seg) CSV 时出错: {e}")
