# Ultralytics YOLO - AGPL-3.0 license
"""多模态 Pose COCO 验证器：Pose(OKS) + Box 双维度 COCO 评估。"""

from ultralytics.models.yolo.multimodal.pose.val import MultiModalPoseValidator
from ultralytics.utils.coco_metrics import COCOMetrics, COCO_AREA_SMALL, COCO_AREA_MEDIUM
from ultralytics.utils.coco_eval_pose_mm import COCOevalPoseMM
from ultralytics.utils.coco_eval_bbox_mm import COCOevalBBoxMM
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import de_parallel, smart_inference_mode, _complexity_base_model
from ultralytics.nn.mm.complexity import compute_default_multimodal_complexity_report, build_default_complexity_summary
import torch
import numpy as np
import csv
from pathlib import Path


class MultiModalPoseCOCOValidator(MultiModalPoseValidator):
    """
    多模态 Pose COCO 验证器。

    继承 MultiModalPoseValidator，在标准 Pose P/R/mAP 之上添加：
    - Pose(OKS): 12 项 COCO 关键点指标（使用 OKS 替代 IoU）
    - Box: 12 项 COCO 检测框指标（标准 IoU）

    接口与 MultiModalCOCOValidator（检测）风格一致。
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

        self.coco_metrics_pose = None
        self.coco_metrics_box = None

        # 累积采集容器
        self._img_shapes = []
        self._dt_img, self._gt_img = [], []
        self._dt_cls, self._dt_conf = [], []
        self._dt_bbox_xyxy, self._dt_kpts = [], []
        self._gt_cls, self._gt_bbox_xyxy, self._gt_kpts = [], [], []

        self.gt_size_counts = {"small": 0, "medium": 0, "large": 0}
        self.complexity_report = None
        self.gflops_total = None
        self.stage_gflops = {}
        self._coco_computed = False
        self._enriched_stats = {}

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def init_metrics(self, model):
        self.model = model
        super().init_metrics(model)

        self.coco_metrics_pose = COCOMetrics(
            save_dir=self.save_dir,
            names=getattr(model, 'names', {}),
            plot=getattr(self.args, 'plots', False),
            on_plot=getattr(self, 'on_plot', None),
        )
        self.coco_metrics_box = COCOMetrics(
            save_dir=self.save_dir,
            names=getattr(model, 'names', {}),
            plot=False,
            on_plot=None,
        )

        # 重置采集容器
        self._img_shapes = []
        self._dt_img, self._gt_img = [], []
        self._dt_cls, self._dt_conf = [], []
        self._dt_bbox_xyxy, self._dt_kpts = [], []
        self._gt_cls, self._gt_bbox_xyxy, self._gt_kpts = [], [], []
        self.gt_size_counts = {"small": 0, "medium": 0, "large": 0}
        self._coco_computed = False
        self._enriched_stats = {}

        # Canonical multimodal complexity summary（通过引擎 API 统一计算）
        self.complexity_report = None
        self.gflops_total = None
        self.stage_gflops = {}
        try:
            imgsz = int(getattr(self.args, 'imgsz', 640))
            base = _complexity_base_model(model)
            self.complexity_report = compute_default_multimodal_complexity_report(base, imgsz=imgsz)
            summary = build_default_complexity_summary(base, self.complexity_report)
            self.gflops_total = summary["gflops_total"]
            self.stage_gflops = summary["stage_gflops"]
        except Exception:
            self.complexity_report = None
            self.gflops_total = None
            self.stage_gflops = {}

    # ------------------------------------------------------------------
    # 进度条表头
    # ------------------------------------------------------------------
    def get_desc(self):
        return ("%22s" + "%11s" * 3) % ("Class", "Images", "Instances", "COCO-Pose mAP@.5:.95")

    # ------------------------------------------------------------------
    # 数据采集
    # ------------------------------------------------------------------
    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)

        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred, pbatch)

            ori_shape = pbatch["ori_shape"]
            self._img_shapes.append((int(ori_shape[0]), int(ori_shape[1])))
            img_id = len(self._img_shapes) - 1

            # ---- GT ----
            gt_cls = pbatch.get("cls", None)
            gt_bboxes = pbatch.get("bboxes", None)
            gt_kpts = pbatch.get("keypoints", None)

            if gt_cls is not None and gt_cls.numel() > 0:
                gt_cls_np = gt_cls.view(-1).detach().cpu().numpy()
                gt_bboxes_np = gt_bboxes.detach().cpu().numpy()
                gt_kpts_np = gt_kpts.detach().cpu().numpy()

                for i in range(len(gt_cls_np)):
                    self._gt_img.append(img_id)
                    self._gt_cls.append(int(gt_cls_np[i]))
                    self._gt_bbox_xyxy.append(gt_bboxes_np[i])
                    self._gt_kpts.append(gt_kpts_np[i].reshape(-1).tolist())

                    x1, y1, x2, y2 = gt_bboxes_np[i]
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    if area < COCO_AREA_SMALL:
                        self.gt_size_counts["small"] += 1
                    elif area < COCO_AREA_MEDIUM:
                        self.gt_size_counts["medium"] += 1
                    else:
                        self.gt_size_counts["large"] += 1

            # ---- DT ----
            dt_bboxes = predn.get("bboxes", None)
            dt_conf = predn.get("conf", None)
            dt_cls = predn.get("cls", None)
            dt_kpts = predn.get("keypoints", None)

            if dt_bboxes is not None and dt_bboxes.numel() > 0:
                dt_bboxes_np = dt_bboxes.detach().cpu().numpy()
                dt_cls_np = dt_cls.detach().cpu().numpy()
                dt_conf_np = dt_conf.detach().cpu().numpy()
                dt_kpts_np = dt_kpts.detach().cpu().numpy()

                for i in range(len(dt_conf_np)):
                    self._dt_img.append(img_id)
                    self._dt_cls.append(int(dt_cls_np[i]))
                    self._dt_conf.append(float(dt_conf_np[i]))
                    self._dt_bbox_xyxy.append(dt_bboxes_np[i])
                    self._dt_kpts.append(dt_kpts_np[i].reshape(-1).tolist())

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------
    def get_stats(self):
        if not self._coco_computed:
            self._compute_coco_metrics()
            self._coco_computed = True

        base_stats = super().get_stats()
        out = dict(base_stats) if isinstance(base_stats, dict) else {}

        # COCO Pose(OKS) metrics
        if self.coco_metrics_pose:
            for k in ('AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
                       'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'):
                out[f'metrics/coco_pose/{k}'] = getattr(self.coco_metrics_pose, k, 0.0)
        # COCO Box metrics
        if self.coco_metrics_box:
            for k in ('AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
                       'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'):
                out[f'metrics/coco_box/{k}'] = getattr(self.coco_metrics_box, k, 0.0)

        # 主指标（以 Pose/OKS 为准）
        out['metrics/coco/AP'] = getattr(self.coco_metrics_pose, 'AP', 0.0) if self.coco_metrics_pose else 0.0
        out['metrics/coco/AP50'] = getattr(self.coco_metrics_pose, 'AP50', 0.0) if self.coco_metrics_pose else 0.0
        out['metrics/coco/AP75'] = getattr(self.coco_metrics_pose, 'AP75', 0.0) if self.coco_metrics_pose else 0.0
        out['metrics/coco/AR100'] = getattr(self.coco_metrics_pose, 'AR100', 0.0) if self.coco_metrics_pose else 0.0
        out['fitness'] = float(out.get('metrics/coco/AP', 0.0))
        out['val/modality'] = self.modality if self.modality else 'multimodal'

        if self.gflops_total:
            out['model/GFLOPs_default_total'] = float(self.gflops_total)
            out['model/GFLOPs_rgb_branch'] = float(self.stage_gflops.get('rgb_branch', 0.0))
            out['model/GFLOPs_x_branch'] = float(self.stage_gflops.get('x_branch', 0.0))
            out['model/GFLOPs_fusion'] = float(self.stage_gflops.get('fusion', 0.0))
            out['model/GFLOPs_head'] = float(self.stage_gflops.get('head', 0.0))

        self._enriched_stats = out
        return out

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        super().__call__(trainer, model)
        if not self.training:
            return self._enriched_stats
        return self.metrics.results_dict

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    def print_results(self):
        if self.coco_metrics_pose is None or self.coco_metrics_box is None:
            return

        # 标准 Pose 摘要
        super().print_results()
        LOGGER.info("")

        # COCO Pose (OKS) 段
        pose_f1 = self.metrics.pose.f1 if hasattr(self.metrics, 'pose') and hasattr(self.metrics.pose, 'f1') else []
        pose_ap_idx = self.metrics.pose.ap_class_index if hasattr(self.metrics, 'pose') and hasattr(self.metrics.pose, 'ap_class_index') else []
        self._print_coco_section("Pose/OKS", self.coco_metrics_pose, pose_f1, pose_ap_idx)

        # COCO Box 段
        box_f1 = self.metrics.box.f1 if hasattr(self.metrics, 'box') and hasattr(self.metrics.box, 'f1') else []
        box_ap_idx = self.metrics.box.ap_class_index if hasattr(self.metrics, 'box') and hasattr(self.metrics.box, 'ap_class_index') else []
        self._print_coco_section("Box", self.coco_metrics_box, box_f1, box_ap_idx)

        # CSV
        self._save_csv_results()

    def _print_coco_section(self, title, coco_m, f1_array, ap_class_index):
        """输出一个 COCO 段（Per-Class + AP Summary + AR Summary + Size Breakdown）。"""
        # F1 映射
        f1_map = {}
        if hasattr(f1_array, '__len__') and len(f1_array) and hasattr(ap_class_index, '__len__') and len(ap_class_index):
            for i, c in enumerate(ap_class_index):
                f1_map[int(c)] = float(f1_array[i])
        mf = float(np.mean(list(f1_map.values()))) if f1_map else 0.0

        hdr4 = "%22s" + "%11s" * 4
        row4 = "%22s" + "%11.3g" * 4

        # --- Per-Class ---
        rows = []
        if hasattr(coco_m, 'class_stats') and coco_m.class_stats and 'ap' in coco_m.class_stats:
            ap_array = coco_m.class_stats['ap']
            unique_cls = coco_m.class_stats['unique_classes']
            for ci, cidx in enumerate(unique_cls):
                cidx = int(cidx)
                name = self.names[cidx] if cidx < len(self.names) else f"class_{cidx}"
                if ci < ap_array.shape[0]:
                    ap = float(ap_array[ci].mean())
                    ap50 = float(ap_array[ci, 0]) if 0 < ap_array.shape[1] else 0.0
                    ap75 = float(ap_array[ci, 5]) if 5 < ap_array.shape[1] else 0.0
                else:
                    ap = ap50 = ap75 = 0.0
                rows.append((name, ap, ap50, ap75, f1_map.get(cidx, 0.0)))
        elif hasattr(coco_m, 'per_class_metrics') and isinstance(coco_m.per_class_metrics, dict) and coco_m.per_class_metrics:
            for cid in sorted(coco_m.per_class_metrics.keys()):
                ci = int(cid)
                name = self.names[ci] if ci < len(self.names) else f"class_{ci}"
                m = coco_m.per_class_metrics[cid]
                rows.append((name, float(m.get('AP', 0.0)), float(m.get('AP50', 0.0)), float(m.get('AP75', 0.0)), f1_map.get(ci, 0.0)))

        if rows:
            LOGGER.info(f"COCO Per-Class Metrics ({title}):")
            LOGGER.info(hdr4 % ("Class", "AP", "AP50", "AP75", "F1"))
            ap_all = np.mean([r[1] for r in rows])
            ap50_all = np.mean([r[2] for r in rows])
            ap75_all = np.mean([r[3] for r in rows])
            f1_all = np.mean([r[4] for r in rows])
            LOGGER.info(row4 % ("all", ap_all, ap50_all, ap75_all, f1_all))
            for name, ap, ap50, ap75, f1 in rows:
                LOGGER.info(row4 % (name, ap, ap50, ap75, f1))

        # --- AP Summary + mF1 ---
        hdr7 = "%22s" + "%11s" * 7
        row7 = "%22s" + "%11.3g" * 7
        LOGGER.info(f"COCO AP Summary ({title}):")
        LOGGER.info(hdr7 % ("Overall", "AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "mF1"))
        LOGGER.info(row7 % (
            "all",
            getattr(coco_m, 'AP', 0.0), getattr(coco_m, 'AP50', 0.0), getattr(coco_m, 'AP75', 0.0),
            getattr(coco_m, 'APsmall', 0.0), getattr(coco_m, 'APmedium', 0.0), getattr(coco_m, 'APlarge', 0.0),
            mf,
        ))

        # --- AR Summary ---
        hdr6 = "%22s" + "%11s" * 6
        row6 = "%22s" + "%11.3g" * 6
        LOGGER.info(f"COCO AR Summary ({title}):")
        LOGGER.info(hdr6 % ("Overall", "AR@1", "AR@10", "AR@100", "ARsmall", "ARmedium", "ARlarge"))
        LOGGER.info(row6 % (
            "all",
            getattr(coco_m, 'AR1', 0.0), getattr(coco_m, 'AR10', 0.0), getattr(coco_m, 'AR100', 0.0),
            getattr(coco_m, 'ARsmall', 0.0), getattr(coco_m, 'ARmedium', 0.0), getattr(coco_m, 'ARlarge', 0.0),
        ))

        # --- Size Breakdown ---
        size_hdr = ("%22s" + "%11s" * 5) % ("Size", "AP", "AP50", "AP75", "AR@100", "GTs")
        size_row = "%22s" + "%11.3g" * 4 + "%11d"
        LOGGER.info(f"COCO Size Breakdown ({title}):")
        LOGGER.info(size_hdr)
        for tag in ("small", "medium", "large"):
            LOGGER.info(size_row % (
                tag.capitalize(),
                getattr(coco_m, f'AP{tag}', 0.0),
                getattr(coco_m, f'AP{tag}50', 0.0),
                getattr(coco_m, f'AP{tag}75', 0.0),
                getattr(coco_m, f'AR{tag}', 0.0),
                self.gt_size_counts.get(tag, 0),
            ))
        LOGGER.info("")

    # ------------------------------------------------------------------
    # COCO 计算核心
    # ------------------------------------------------------------------
    def _compute_coco_metrics(self):
        """运行 COCOevalPoseMM（OKS）+ COCOevalBBoxMM（IoU）。"""
        if len(self._img_shapes) == 0:
            self._set_default_stats()
            return

        imgIds = list(range(len(self._img_shapes)))
        catIds = sorted(list({c for c in self._dt_cls} | {c for c in self._gt_cls}))
        if not catIds:
            self._set_default_stats()
            return

        # ---- Pose (OKS) ----
        pose_gts, pose_dts = [], []
        for j in range(len(self._gt_cls)):
            bbox_xywh = self._xyxy_to_xywh(self._gt_bbox_xyxy[j])
            area = bbox_xywh[2] * bbox_xywh[3]
            pose_gts.append({
                "image_id": int(self._gt_img[j]),
                "category_id": int(self._gt_cls[j]),
                "keypoints": self._gt_kpts[j],
                "bbox": bbox_xywh,
                "area": float(area),
                "iscrowd": 0, "ignore": 0,
                "id": j + 1,
            })
        for i in range(len(self._dt_conf)):
            bbox_xywh = self._xyxy_to_xywh(self._dt_bbox_xyxy[i])
            area = bbox_xywh[2] * bbox_xywh[3]
            pose_dts.append({
                "image_id": int(self._dt_img[i]),
                "category_id": int(self._dt_cls[i]),
                "keypoints": self._dt_kpts[i],
                "bbox": bbox_xywh,
                "score": float(self._dt_conf[i]),
                "area": float(area),
                "id": i + 1,
            })

        pose_eval = COCOevalPoseMM()
        pose_eval.set_data(gts=pose_gts, dts=pose_dts, imgIds=imgIds, catIds=catIds)
        pose_eval.evaluate()
        pose_eval.accumulate()
        pose_stats = pose_eval.summarize()
        pose_per_class = pose_eval.compute_per_class_metrics()
        for k, v in list(pose_stats.items()):
            if v == -1:
                pose_stats[k] = 0.0
        self.coco_metrics_pose.update(pose_stats)
        self.coco_metrics_pose.per_class_metrics = pose_per_class

        # ---- Box ----
        box_gts, box_dts = [], []
        for j in range(len(self._gt_cls)):
            bbox_xywh = self._xyxy_to_xywh(self._gt_bbox_xyxy[j])
            box_gts.append({
                "image_id": int(self._gt_img[j]),
                "category_id": int(self._gt_cls[j]),
                "bbox": bbox_xywh,
                "area": float(bbox_xywh[2] * bbox_xywh[3]),
                "iscrowd": 0, "ignore": 0,
                "id": j + 1,
            })
        for i in range(len(self._dt_conf)):
            bbox_xywh = self._xyxy_to_xywh(self._dt_bbox_xyxy[i])
            box_dts.append({
                "image_id": int(self._dt_img[i]),
                "category_id": int(self._dt_cls[i]),
                "bbox": bbox_xywh,
                "score": float(self._dt_conf[i]),
                "area": float(bbox_xywh[2] * bbox_xywh[3]),
                "id": i + 1,
            })

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

    def _set_default_stats(self):
        defaults = {
            'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
            'APsmall': 0.0, 'APmedium': 0.0, 'APlarge': 0.0,
            'AR1': 0.0, 'AR10': 0.0, 'AR100': 0.0,
            'ARsmall': 0.0, 'ARmedium': 0.0, 'ARlarge': 0.0,
        }
        if self.coco_metrics_pose:
            self.coco_metrics_pose.update(defaults)
        if self.coco_metrics_box:
            self.coco_metrics_box.update(defaults)

    # ------------------------------------------------------------------
    # CSV 保存
    # ------------------------------------------------------------------
    def _save_csv_results(self):
        try:
            save_dir = Path(self.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # F1 映射
            pose_f1_map, box_f1_map = {}, {}
            if hasattr(self.metrics, 'pose') and hasattr(self.metrics.pose, 'f1') and len(self.metrics.pose.f1):
                for i, c in enumerate(self.metrics.pose.ap_class_index):
                    pose_f1_map[int(c)] = float(self.metrics.pose.f1[i])
            if hasattr(self.metrics, 'box') and hasattr(self.metrics.box, 'f1') and len(self.metrics.box.f1):
                for i, c in enumerate(self.metrics.box.ap_class_index):
                    box_f1_map[int(c)] = float(self.metrics.box.f1[i])

            # 1. 按类别指标
            class_csv = save_dir / "coco_pose_metrics_by_class.csv"
            with open(class_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=[
                    'Class', 'Pose_AP', 'Pose_AP50', 'Pose_AP75', 'Pose_F1',
                    'Box_AP', 'Box_AP50', 'Box_AP75', 'Box_F1',
                ])
                w.writeheader()
                pose_pc = getattr(self.coco_metrics_pose, 'per_class_metrics', {}) or {}
                box_pc = getattr(self.coco_metrics_box, 'per_class_metrics', {}) or {}
                all_cids = sorted(set(pose_pc.keys()) | set(box_pc.keys()))
                for cid in all_cids:
                    ci = int(cid)
                    name = self.names[ci] if ci < len(self.names) else f"class_{ci}"
                    pm = pose_pc.get(cid, {})
                    bm = box_pc.get(cid, {})
                    w.writerow({
                        'Class': name,
                        'Pose_AP': f"{pm.get('AP', 0.0):.3f}",
                        'Pose_AP50': f"{pm.get('AP50', 0.0):.3f}",
                        'Pose_AP75': f"{pm.get('AP75', 0.0):.3f}",
                        'Pose_F1': f"{pose_f1_map.get(ci, 0.0):.3f}",
                        'Box_AP': f"{bm.get('AP', 0.0):.3f}",
                        'Box_AP50': f"{bm.get('AP50', 0.0):.3f}",
                        'Box_AP75': f"{bm.get('AP75', 0.0):.3f}",
                        'Box_F1': f"{box_f1_map.get(ci, 0.0):.3f}",
                    })

            # 2. 总体指标
            overall_csv = save_dir / "coco_pose_metrics_overall.csv"
            with open(overall_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['Metric', 'Pose_Value', 'Box_Value'])
                w.writeheader()
                for k in ('AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
                           'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'):
                    pv = getattr(self.coco_metrics_pose, k, 0.0)
                    bv = getattr(self.coco_metrics_box, k, 0.0)
                    w.writerow({'Metric': k, 'Pose_Value': f"{pv:.3f}", 'Box_Value': f"{bv:.3f}"})
                pose_mf = float(np.mean(list(pose_f1_map.values()))) if pose_f1_map else 0.0
                box_mf = float(np.mean(list(box_f1_map.values()))) if box_f1_map else 0.0
                w.writerow({'Metric': 'mF1', 'Pose_Value': f"{pose_mf:.3f}", 'Box_Value': f"{box_mf:.3f}"})
                w.writerow({'Metric': 'Modality', 'Pose_Value': self.modality or 'multimodal', 'Box_Value': ''})

            LOGGER.info(f"CSV 结果已保存: {class_csv.name}, {overall_csv.name}")

        except Exception as e:
            LOGGER.warning(f"保存 CSV 文件时出错: {e}")

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _xyxy_to_xywh(box_xyxy):
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        return [x1, y1, x2 - x1, y2 - y1]
