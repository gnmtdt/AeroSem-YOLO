# Ultralytics Multimodal Inference - Results Container
# Multimodal-aware result container with visualization support
# Version: v1.0
# Date: 2026-01-13

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from ultralytics.data.multimodal.precision_contract import HIGH_PRECISION, normalize_precision_contract
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils import ops


class MultiModalResults:
    """
    多模态推理结果容器（语义完整）

    核心字段：
    - boxes: 检测框 [N, 6] (x1, y1, x2, y2, conf, cls)
    - paths: {'rgb': Path, 'x': Path}
    - orig_imgs: {'rgb': np.ndarray, 'x': np.ndarray}
    - meta: {id, x_modality, xch, ori_shape, imgsz}

    可视化规则：
    - 永远输出 RGB 可视化
    - 只有当 xch ∈ {1,3} 时才允许输出 X 可视化
    - 当 xch > 3: plot() 仅返回 RGB 结果
    """

    def __init__(
        self,
        boxes: np.ndarray,
        paths: Dict[str, Path],
        orig_imgs: Dict[str, np.ndarray],
        meta: Dict,
        names: Optional[Dict[int, str]] = None
    ):
        """
        初始化多模态结果容器

        Args:
            boxes: 检测框 [N, 6] (x1, y1, x2, y2, conf, cls)
            paths: 图像路径字典
            orig_imgs: 原始图像字典
            meta: 元数据字典
            names: 类别名称字典 {class_id: class_name}
        """
        self.boxes = boxes
        self.paths = paths
        self.orig_imgs = orig_imgs
        self.meta = meta
        self.names = names or {}

        # 检测数量
        self.num_dets = len(boxes)

        # 可视化条件
        self.xch = meta.get('xch', 3)
        self.can_visualize_x = self.xch in {1, 3}

        # 速度数据（由 predictor 在 speed=True 时填充）
        self.speed = {}

    def class_counts(self) -> List[Dict[str, Any]]:
        """
        返回单样本最终检测结果的"按类别计数"统计。

        统计口径：以 self.boxes 为准（已完成前处理/推理/后处理后的最终 boxes）。
        输出结构：按 class_id 升序的列表，每项包含 class_id/class_name/count。

        Returns:
            [{"class_id": 0, "class_name": "person", "count": 2}, ...]
        """
        if self.num_dets == 0:
            return []

        cls_ids = self.boxes[:, 5].astype(int)
        unique_ids, counts = np.unique(cls_ids, return_counts=True)
        items = []
        for cls_id, cnt in zip(unique_ids.tolist(), counts.tolist()):
            items.append({
                "class_id": int(cls_id),
                "class_name": str(self.names.get(int(cls_id), str(cls_id))),
                "count": int(cnt),
            })
        return items

    def _get_filename_text(self) -> str:
        """
        获取用于在可视化图上显示的源文件名

        优先使用 RGB 路径的文件名，其次使用 X 路径的文件名。
        如果路径为空或无效则返回空字符串。

        Returns:
            文件名字符串（不含目录），如 "image001.jpg"
        """
        for key in ('rgb', 'x'):
            p = self.paths.get(key)
            if p is not None:
                try:
                    return Path(p).name
                except (TypeError, ValueError):
                    continue
        return ""

    def _prepare_display_image(self, img: np.ndarray, modality: str) -> np.ndarray:
        """Normalize a raw modality image into a uint8 display image using the point-level contract."""
        contract = normalize_precision_contract(self.meta.get("precision_contract"))
        precision = contract.get(modality, "legacy")
        arr = np.asarray(img)
        if not np.issubdtype(arr.dtype, np.number):
            return arr

        arr_f = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if precision == HIGH_PRECISION:
            min_v = float(arr_f.min()) if arr_f.size else 0.0
            max_v = float(arr_f.max()) if arr_f.size else 0.0
            if max_v > min_v:
                arr_f = (arr_f - min_v) / (max_v - min_v)
            else:
                arr_f.fill(0.0)
        else:
            if arr.dtype == np.uint8:
                arr_f = arr_f / 255.0
            elif arr.dtype == np.uint16:
                arr_f = arr_f / 65535.0
            else:
                arr_f = np.clip(arr_f, 0.0, 1.0)

        return (np.clip(arr_f, 0.0, 1.0) * 255.0).astype(np.uint8)

    def plot(
        self,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True,
        show_filename: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        绘制检测结果

        Args:
            conf: 是否显示置信度
            line_width: 线宽
            font_size: 字体大小
            labels: 是否显示标签
            show_filename: 是否在结果图上显示源文件名

        Returns:
            {'rgb': annotated_rgb, 'x': annotated_x} 或 {'rgb': annotated_rgb}

        Note:
            - plot() 始终返回 'rgb' key，即使单X模态推理时也会返回黑底占位图
            - 单模态推理时，缺失的模态不会出现在返回字典中
        """
        results = {}

        # 预取原图（单模态推理时可能为 None）
        rgb0 = self.orig_imgs.get('rgb', None)
        x0 = self.orig_imgs.get('x', None)

        # 1. RGB 可视化（必出：为兼容保存流程，RGB缺失时使用黑底占位图）
        if rgb0 is not None:
            rgb_img = self._prepare_display_image(rgb0.copy(), modality='rgb')
        else:
            # X-only 推理：使用黑底占位图（不显示X内容）
            if 'ori_shape' not in self.meta:
                raise ValueError("meta 缺少 ori_shape，无法为缺失的RGB构造占位画布")
            h, w = self.meta['ori_shape']
            rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

        rgb_annotated = self._annotate_image(
            rgb_img,
            boxes=self.boxes,
            conf=conf,
            line_width=line_width,
            font_size=font_size,
            labels=labels,
            show_filename=show_filename
        )
        results['rgb'] = rgb_annotated

        # 2. X 模态可视化（仅当 xch ∈ {1,3} 且 X 真实存在）
        if self.can_visualize_x and x0 is not None:
            x_img = self._prepare_display_image(x0.copy(), modality='x')

            # 处理 X 模态图像（确保是3通道BGR）
            if len(x_img.shape) == 2:  # 灰度图
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)
            elif x_img.shape[2] == 1:  # 单通道
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)

            x_annotated = self._annotate_image(
                x_img,
                boxes=self.boxes,
                conf=conf,
                line_width=line_width,
                font_size=font_size,
                labels=labels,
                show_filename=show_filename
            )
            results['x'] = x_annotated

        return results

    def _annotate_image(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True,
        show_filename: bool = False
    ) -> np.ndarray:
        """
        在图像上标注检测框

        Args:
            img: 原始图像（BGR格式）
            boxes: 检测框 [N, 6]
            conf: 是否显示置信度
            line_width: 线宽
            font_size: 字体大小
            labels: 是否显示标签

        Returns:
            标注后的图像
        """
        annotator = Annotator(
            img,
            line_width=line_width,
            font_size=font_size,
            pil=False  # 使用cv2模式
        )

        for box in boxes:
            x1, y1, x2, y2, confidence, cls = box
            cls = int(cls)

            # 构造标签
            if labels:
                class_name = self.names.get(cls, str(cls))
                if conf:
                    label = f"{class_name} {confidence:.2f}"
                else:
                    label = class_name
            else:
                label = ""

            # 绘制框和标签
            color = colors(cls, True)  # 根据类别获取颜色
            annotator.box_label(
                box=[x1, y1, x2, y2],
                label=label,
                color=color
            )

        # 绘制源文件名（如果 show_filename 为 True）
        # cv2 模式下 annotator.sf 是 fontScale（浮点数），annotator.tf 是 thickness（整数）
        if show_filename:
            fname = self._get_filename_text()
            if fname and hasattr(annotator, "sf"):
                fs = annotator.sf   # fontScale (float)
                ft = annotator.tf   # thickness (int)
                tw, th = cv2.getTextSize(fname, 0, fs, ft)[0]
                pad = 5
                img = annotator.im
                cv2.rectangle(img, (pad, pad), (pad + tw + 4, pad + th + 8), (0, 0, 0), -1)
                cv2.putText(img, fname, (pad + 2, pad + th + 2), 0, fs, (255, 255, 255), ft, cv2.LINE_AA)

        return annotator.result()

    def plot_merged(
        self,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True,
        show_filename: bool = False
    ) -> Optional[np.ndarray]:
        """
        绘制双模态并排合并图（不带任何标题和装饰性文字）

        Args:
            conf: 是否显示置信度
            line_width: 线宽
            font_size: 字体大小
            labels: 是否显示标签
            show_filename: 是否在结果图上显示源文件名

        Returns:
            并排合并图，如果不满足合并条件（RGB和X都存在且X可视化）则返回 None
        """
        # 仅当 RGB 与 X 都真实存在，且 X 可视化成立时才允许合并
        has_rgb = self.orig_imgs.get('rgb', None) is not None
        has_x = self.orig_imgs.get('x', None) is not None
        if not (has_rgb and has_x and self.can_visualize_x):
            return None

        # 获取标注后的图像
        annotated = self.plot(
            conf=conf,
            line_width=line_width,
            font_size=font_size,
            labels=labels,
            show_filename=show_filename
        )

        rgb_img = annotated['rgb']
        x_img = annotated['x']

        # 确保两图高度一致
        h_rgb, w_rgb = rgb_img.shape[:2]
        h_x, w_x = x_img.shape[:2]

        if h_rgb != h_x:
            # 以RGB高度为准，resize X
            x_img = cv2.resize(x_img, (w_x, h_rgb))

        # 并排拼接（RGB在左，X在右）
        merged = np.hstack([rgb_img, x_img])

        return merged

    def save_txt(
        self,
        save_path: Path,
        save_conf: bool = False
    ):
        """
        保存YOLO格式的txt标签文件

        Args:
            save_path: 保存路径
            save_conf: 是否保存置信度
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.num_dets == 0:
            # 创建空文件
            save_path.write_text("")
            return

        # 坐标归一化到 RGB 尺寸
        h, w = self.meta['ori_shape']

        with open(save_path, 'w') as f:
            for box in self.boxes:
                x1, y1, x2, y2, confidence, cls = box

                # YOLO 格式：class x_center y_center width height [conf]
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                if save_conf:
                    f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")
                else:
                    f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def save_json(
        self,
        save_path: Path
    ):
        """
        保存JSON格式的推理结果

        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 构造JSON数据
        rgb_path = self.paths.get('rgb', None)
        x_path = self.paths.get('x', None)
        has_rgb = self.orig_imgs.get('rgb', None) is not None
        has_x = self.orig_imgs.get('x', None) is not None

        # 推理模态标识
        if has_rgb and has_x:
            modality = 'rgb+x'
        elif has_rgb:
            modality = 'rgb'
        elif has_x:
            modality = 'x'
        else:
            modality = 'none'

        # RGB 可视化来源（与 plot() 逻辑保持一致）
        if has_rgb:
            rgb_rendered_from = 'rgb'
        else:
            rgb_rendered_from = 'blank'  # X-only: 黑底占位图

        data = {
            'id': self.meta['id'],
            'paths': {
                'rgb': str(rgb_path) if rgb_path is not None else None,
                'x': str(x_path) if x_path is not None else None
            },
            'meta': {
                'x_modality': self.meta['x_modality'],
                'xch': self.meta['xch'],
                'ori_shape': self.meta['ori_shape'],
                'imgsz': self.meta['imgsz'],
                'modality': modality,
                'modalities': {'rgb': has_rgb, 'x': has_x},
                'visualization': {
                    'rgb_rendered_from': rgb_rendered_from,
                    'x_visualizable': bool(has_x and self.xch in {1, 3})
                }
            },
            'detections': [],
            'summary': {
                'num_dets': int(self.num_dets),
                'class_counts': self.class_counts(),
            }
        }

        # 添加检测结果
        h, w = self.meta['ori_shape']
        for box in self.boxes:
            x1, y1, x2, y2, confidence, cls = box
            cls = int(cls)

            detection = {
                'class': cls,
                'class_name': self.names.get(cls, str(cls)),
                'confidence': float(confidence),
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                },
                'bbox_normalized': {
                    'x_center': float((x1 + x2) / 2 / w),
                    'y_center': float((y1 + y2) / 2 / h),
                    'width': float((x2 - x1) / w),
                    'height': float((y2 - y1) / h)
                }
            }
            data['detections'].append(detection)

        # 写入JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _get_instance_bbox(self, idx: int, padding_ratio: float = 0.05) -> tuple:
        """
        获取指定实例的外接水平矩形（带padding）

        Args:
            idx: 实例索引
            padding_ratio: padding比例（相对于bbox尺寸）

        Returns:
            (x1, y1, x2, y2) 裁切区域坐标
        """
        box = self.boxes[idx]
        x1, y1, x2, y2 = box[:4].tolist()

        # 计算padding
        w, h = x2 - x1, y2 - y1
        pad_x = w * padding_ratio
        pad_y = h * padding_ratio

        # 应用padding并裁剪到图像边界
        ori_h, ori_w = self.meta['ori_shape']
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(ori_w, x2 + pad_x)
        y2 = min(ori_h, y2 + pad_y)

        return int(x1), int(y1), int(x2), int(y2)

    def _crop_annotate_instance(
        self,
        img: np.ndarray,
        idx: int,
        crop_bbox: tuple,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None
    ) -> np.ndarray:
        """
        在裁切图上标注单个实例（基类实现：绘制BBOX）

        Args:
            img: 裁切后的图像
            idx: 实例索引
            crop_bbox: 裁切区域坐标 (x1, y1, x2, y2)
            line_width: 线宽
            font_size: 字体大小

        Returns:
            标注后的图像
        """
        annotator = Annotator(img, line_width=line_width, font_size=font_size, pil=False)

        box = self.boxes[idx]
        orig_xyxy = box[:4].tolist()

        # 转换为裁切图坐标系
        x1_crop, y1_crop = crop_bbox[0], crop_bbox[1]
        local_xyxy = [
            orig_xyxy[0] - x1_crop,
            orig_xyxy[1] - y1_crop,
            orig_xyxy[2] - x1_crop,
            orig_xyxy[3] - y1_crop
        ]

        cls_id = int(box[5])
        conf = float(box[4])
        cls_name = self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)
        label = f"{cls_name} {conf:.2f}"

        color = colors(cls_id, True)
        annotator.box_label(local_xyxy, label, color)

        return annotator.result()

    def save_crop(
        self,
        save_dir: Path,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None
    ) -> List[Path]:
        """
        保存实例裁切图

        Args:
            save_dir: 保存根目录
            line_width: 线宽
            font_size: 字体大小

        Returns:
            保存的文件路径列表
        """
        if self.boxes is None or len(self.boxes) == 0:
            return []

        saved_paths = []
        sample_id = self.meta.get('id', 'unknown')

        # 创建目录: crops/{sample_id}/
        crop_dir = Path(save_dir) / 'crops' / sample_id
        crop_dir.mkdir(parents=True, exist_ok=True)

        # 获取原图
        rgb0 = self.orig_imgs.get('rgb', None)
        x0 = self.orig_imgs.get('x', None)

        for idx in range(len(self.boxes)):
            box = self.boxes[idx]
            cls_id = int(box[5])
            conf = float(box[4])
            cls_name = self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)

            # 文件名前缀
            name_prefix = f"{cls_name}_{idx}_{conf:.2f}"

            # 获取裁切区域
            crop_bbox = self._get_instance_bbox(idx)
            x1, y1, x2, y2 = crop_bbox

            crops = {}

            # RGB 裁切
            if rgb0 is not None:
                rgb_crop = rgb0[y1:y2, x1:x2].copy()
                rgb_crop = self._crop_annotate_instance(
                    rgb_crop, idx, crop_bbox, line_width, font_size
                )
                crops['rgb'] = rgb_crop

            # X 模态裁切
            if self.can_visualize_x and x0 is not None:
                x_crop = x0[y1:y2, x1:x2].copy()
                # 确保3通道
                if len(x_crop.shape) == 2:
                    x_crop = cv2.cvtColor(x_crop, cv2.COLOR_GRAY2BGR)
                elif x_crop.shape[2] == 1:
                    x_crop = cv2.cvtColor(x_crop, cv2.COLOR_GRAY2BGR)
                x_crop = self._crop_annotate_instance(
                    x_crop, idx, crop_bbox, line_width, font_size
                )
                crops['x'] = x_crop

            # 保存裁切图
            if 'rgb' in crops:
                rgb_path = crop_dir / f"{name_prefix}_rgb.jpg"
                cv2.imwrite(str(rgb_path), crops['rgb'])
                saved_paths.append(rgb_path)

            if 'x' in crops:
                x_path = crop_dir / f"{name_prefix}_x.jpg"
                cv2.imwrite(str(x_path), crops['x'])
                saved_paths.append(x_path)

            # 双模态时保存拼接图
            if 'rgb' in crops and 'x' in crops:
                # 调整高度一致后横向拼接
                h1, h2 = crops['rgb'].shape[0], crops['x'].shape[0]
                if h1 != h2:
                    target_h = max(h1, h2)
                    if h1 < target_h:
                        crops['rgb'] = cv2.resize(crops['rgb'], (int(crops['rgb'].shape[1] * target_h / h1), target_h))
                    if h2 < target_h:
                        crops['x'] = cv2.resize(crops['x'], (int(crops['x'].shape[1] * target_h / h2), target_h))

                merged = np.concatenate([crops['rgb'], crops['x']], axis=1)
                merged_path = crop_dir / f"{name_prefix}_merged.jpg"
                cv2.imwrite(str(merged_path), merged)
                saved_paths.append(merged_path)

        return saved_paths


class MultiModalSegmentResults(MultiModalResults):
    """
    多模态分割推理结果容器

    继承自 MultiModalResults，扩展支持实例分割 masks。

    新增字段：
    - masks: 分割掩码 [N, H, W] (numpy array, uint8, 0-255)

    可视化规则：
    - 继承父类的 RGB/X 双模态可视化
    - 在检测框下方叠加半透明分割掩码
    """

    def __init__(
        self,
        boxes: np.ndarray,
        paths: Dict[str, Path],
        orig_imgs: Dict[str, np.ndarray],
        meta: Dict,
        names: Optional[Dict[int, str]] = None,
        masks: Optional[np.ndarray] = None
    ):
        """
        初始化多模态分割结果容器

        Args:
            boxes: 检测框 [N, 6] (x1, y1, x2, y2, conf, cls)
            paths: 图像路径字典
            orig_imgs: 原始图像字典
            meta: 元数据字典
            names: 类别名称字典 {class_id: class_name}
            masks: 分割掩码 [N, H, W] (可选)
        """
        super().__init__(boxes, paths, orig_imgs, meta, names)
        self.masks = masks  # [N, H, W] or None

    def plot(
        self,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True,
        mask_alpha: float = 0.5,
        show_filename: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        绘制分割结果（masks + boxes）

        Args:
            conf: 是否显示置信度
            line_width: 线宽
            font_size: 字体大小
            labels: 是否显示标签
            mask_alpha: mask透明度 (0.0-1.0)
            show_filename: 是否在结果图上显示源文件名

        Returns:
            {'rgb': annotated_rgb, 'x': annotated_x} 或 {'rgb': annotated_rgb}
        """
        results = {}

        # 预取原图（单模态推理时可能为 None）
        rgb0 = self.orig_imgs.get('rgb', None)
        x0 = self.orig_imgs.get('x', None)

        # 1. RGB 可视化
        if rgb0 is not None:
            rgb_img = self._prepare_display_image(rgb0.copy(), modality='rgb')
        else:
            if 'ori_shape' not in self.meta:
                raise ValueError("meta 缺少 ori_shape，无法为缺失的RGB构造占位画布")
            h, w = self.meta['ori_shape']
            rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

        # 先绘制 masks，再绘制 boxes
        rgb_with_masks = self._draw_masks(rgb_img, mask_alpha)
        rgb_annotated = self._annotate_image(
            rgb_with_masks,
            boxes=self.boxes,
            conf=conf,
            line_width=line_width,
            font_size=font_size,
            labels=labels,
            show_filename=show_filename
        )
        results['rgb'] = rgb_annotated

        # 2. X 模态可视化
        if self.can_visualize_x and x0 is not None:
            x_img = self._prepare_display_image(x0.copy(), modality='x')
            if len(x_img.shape) == 2:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)
            elif x_img.shape[2] == 1:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)

            x_with_masks = self._draw_masks(x_img, mask_alpha)
            x_annotated = self._annotate_image(
                x_with_masks,
                boxes=self.boxes,
                conf=conf,
                line_width=line_width,
                font_size=font_size,
                labels=labels,
                show_filename=show_filename
            )
            results['x'] = x_annotated

        return results

    def _draw_masks(
        self,
        img: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        在图像上绘制半透明分割掩码

        Args:
            img: 原始图像（BGR格式）
            alpha: 透明度 (0.0-1.0)

        Returns:
            叠加 masks 后的图像
        """
        if self.masks is None or len(self.masks) == 0:
            return img

        result = img.copy()
        h, w = img.shape[:2]

        for i, mask in enumerate(self.masks):
            # 获取类别对应的颜色
            if len(self.boxes) > i:
                cls = int(self.boxes[i, 5])
            else:
                cls = 0
            color = colors(cls, True)

            # 确保 mask 尺寸与图像匹配
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)

            # 创建彩色掩码
            mask_bool = mask > 127 if mask.dtype == np.uint8 else mask > 0.5
            colored_mask = np.zeros_like(result)
            colored_mask[mask_bool] = color

            # 叠加到原图
            result = cv2.addWeighted(result, 1.0, colored_mask, alpha, 0)

        return result

    def plot_merged(
        self,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True,
        mask_alpha: float = 0.5,
        show_filename: bool = False
    ) -> Optional[np.ndarray]:
        """
        绘制双模态并排合并图（带分割掩码）

        Args:
            conf: 是否显示置信度
            line_width: 线宽
            font_size: 字体大小
            labels: 是否显示标签
            mask_alpha: mask透明度 (0.0-1.0)
            show_filename: 是否在结果图上显示源文件名

        Returns:
            并排合并图，如果不满足合并条件则返回 None
        """
        # 仅当 RGB 与 X 都真实存在，且 X 可视化成立时才允许合并
        has_rgb = self.orig_imgs.get('rgb', None) is not None
        has_x = self.orig_imgs.get('x', None) is not None
        if not (has_rgb and has_x and self.can_visualize_x):
            return None

        # 获取标注后的图像（使用子类的 plot 方法，传递 mask_alpha）
        annotated = self.plot(
            conf=conf,
            line_width=line_width,
            font_size=font_size,
            labels=labels,
            mask_alpha=mask_alpha,
            show_filename=show_filename
        )

        rgb_img = annotated['rgb']
        x_img = annotated['x']

        # 确保两图高度一致
        h_rgb, w_rgb = rgb_img.shape[:2]
        h_x, w_x = x_img.shape[:2]

        if h_rgb != h_x:
            x_img = cv2.resize(x_img, (w_x, h_rgb))

        # 并排拼接（RGB在左，X在右）
        merged = np.hstack([rgb_img, x_img])

        return merged

    def _crop_annotate_instance(
        self,
        img: np.ndarray,
        idx: int,
        crop_bbox: tuple,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None
    ) -> np.ndarray:
        """
        在裁切图上标注单个实例（分割任务：绘制mask + BBOX）
        """
        x1_crop, y1_crop, x2_crop, y2_crop = crop_bbox
        crop_h, crop_w = img.shape[:2]

        # 绘制mask
        if self.masks is not None and idx < len(self.masks):
            mask = self.masks[idx]
            # 裁切mask到对应区域
            mask_crop = mask[y1_crop:y2_crop, x1_crop:x2_crop]
            if mask_crop.shape[:2] != (crop_h, crop_w):
                mask_crop = cv2.resize(mask_crop.astype(np.uint8), (crop_w, crop_h))

            cls_id = int(self.boxes[idx][5])
            color = colors(cls_id, True)

            # 半透明mask叠加
            colored_mask = np.zeros_like(img)
            colored_mask[mask_crop > 0.5] = color
            img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

        annotator = Annotator(img, line_width=line_width, font_size=font_size, pil=False)

        # 绘制BBOX
        box = self.boxes[idx]
        orig_xyxy = box[:4].tolist()
        local_xyxy = [
            orig_xyxy[0] - x1_crop,
            orig_xyxy[1] - y1_crop,
            orig_xyxy[2] - x1_crop,
            orig_xyxy[3] - y1_crop
        ]

        cls_id = int(box[5])
        conf = float(box[4])
        cls_name = self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)
        label = f"{cls_name} {conf:.2f}"

        color = colors(cls_id, True)
        annotator.box_label(local_xyxy, label, color)

        return annotator.result()


class MultiModalOBB:
    """
    多模态 OBB 数据封装类 (numpy-only)

    提供与标准 OBB 类兼容的接口，用于旋转框数据的访问和转换。

    Attributes:
        data: 原始 OBB 数据 [N, 7] 格式 [x, y, w, h, angle, conf, cls]
        orig_shape: 原图尺寸 (height, width)

    Properties:
        xywhr: 中心点+宽高+角度格式 [N, 5]
        conf: 置信度 [N]
        cls: 类别 [N]
        xyxyxyxy: 4角点坐标格式 [N, 4, 2]
        xyxyxyxyn: 归一化的4角点坐标 [N, 4, 2]
        xyxy: 外接水平矩形 [N, 4]
    """

    def __init__(self, data: np.ndarray, orig_shape: tuple):
        """
        初始化 MultiModalOBB

        Args:
            data: OBB 数据 [N, 7] 格式 [x, y, w, h, angle, conf, cls]
            orig_shape: 原图尺寸 (height, width)
        """
        if data.ndim == 1:
            data = data[None, :]
        assert data.shape[-1] == 7, f"Expected 7 values per box but got {data.shape[-1]}"
        self.data = data
        self.orig_shape = orig_shape

    def __len__(self) -> int:
        """返回 OBB 数量"""
        return len(self.data)

    def __getitem__(self, idx) -> 'MultiModalOBB':
        """索引访问，返回新的 MultiModalOBB 实例"""
        return MultiModalOBB(self.data[idx], self.orig_shape)

    def __iter__(self):
        """迭代器，逐个返回 OBB"""
        for i in range(len(self)):
            yield self[i]

    @property
    def xywhr(self) -> np.ndarray:
        """返回 [x_center, y_center, width, height, rotation] 格式 [N, 5]"""
        return self.data[:, :5]

    @property
    def conf(self) -> np.ndarray:
        """返回置信度 [N]"""
        return self.data[:, 5]

    @property
    def cls(self) -> np.ndarray:
        """返回类别 [N]"""
        return self.data[:, 6]

    @property
    def xyxyxyxy(self) -> np.ndarray:
        """
        转换为4角点坐标格式 [N, 4, 2]

        使用 ops.xywhr2xyxyxyxy 进行转换
        """
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @property
    def xyxyxyxyn(self) -> np.ndarray:
        """
        返回归一化的4角点坐标 [N, 4, 2]

        坐标相对于 orig_shape 进行归一化
        """
        xyxyxyxyn = self.xyxyxyxy.copy()
        xyxyxyxyn[..., 0] /= self.orig_shape[1]  # width
        xyxyxyxyn[..., 1] /= self.orig_shape[0]  # height
        return xyxyxyxyn

    @property
    def xyxy(self) -> np.ndarray:
        """
        返回外接水平矩形 [N, 4] 格式 [x1, y1, x2, y2]

        计算每个旋转框的最小外接水平矩形
        """
        corners = self.xyxyxyxy  # [N, 4, 2]
        x = corners[..., 0]  # [N, 4]
        y = corners[..., 1]  # [N, 4]
        return np.stack([x.min(axis=1), y.min(axis=1), x.max(axis=1), y.max(axis=1)], axis=-1)


class MultiModalOBBResults(MultiModalResults):
    """
    多模态 OBB 推理结果

    继承自 MultiModalResults，扩展旋转框检测的可视化和保存功能。

    Attributes:
        obb: MultiModalOBB 实例，封装旋转框数据
    """

    def __init__(
        self,
        obb: np.ndarray,
        paths: Dict[str, Path],
        orig_imgs: Dict[str, np.ndarray],
        meta: Dict,
        names: Optional[Dict[int, str]] = None
    ):
        """
        初始化 MultiModalOBBResults

        Args:
            obb: OBB 数据 [N, 7] 格式 [x, y, w, h, angle, conf, cls]
            paths: 图像路径字典 {'rgb': path, 'x': path}
            orig_imgs: 原始图像字典 {'rgb': img, 'x': img}
            meta: 元数据字典，必须包含 'ori_shape'
            names: 类别名称字典 {id: name}
        """
        # 获取原图尺寸
        ori_shape = meta.get('ori_shape', (640, 640))

        # 创建 MultiModalOBB 实例
        if len(obb) > 0:
            self._obb = MultiModalOBB(obb, ori_shape)
            # 父类 boxes 使用外接水平矩形，用于兼容
            obb_xyxy = self._obb.xyxy  # [N, 4]
            boxes = np.column_stack([obb_xyxy, self._obb.conf, self._obb.cls])  # [N, 6]
        else:
            self._obb = MultiModalOBB(np.zeros((0, 7)), ori_shape)
            boxes = np.zeros((0, 6))

        super().__init__(
            boxes=boxes,
            paths=paths,
            orig_imgs=orig_imgs,
            meta=meta,
            names=names
        )

    @property
    def obb(self) -> MultiModalOBB:
        """返回 MultiModalOBB 实例"""
        return self._obb

    def _annotate_obb_image(
        self,
        img: np.ndarray,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True,
        debug: bool = False,
        show_filename: bool = False
    ) -> np.ndarray:
        """
        在图像上标注旋转框

        Args:
            img: 原始图像（BGR格式）
            conf: 是否显示置信度
            line_width: 线宽
            font_size: 字体大小
            labels: 是否显示标签
            debug: 是否输出调试信息
            show_filename: 是否在结果图上显示源文件名

        Returns:
            标注后的图像
        """
        annotator = Annotator(
            img,
            line_width=line_width,
            font_size=font_size,
            pil=False
        )

        if len(self._obb) == 0:
            return annotator.result()

        # 获取4角点坐标
        corners = self._obb.xyxyxyxy  # [N, 4, 2]

        for i in range(len(self._obb)):
            box_corners = corners[i].tolist()  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            confidence = float(self._obb.conf[i])
            cls_id = int(self._obb.cls[i])

            # 构建标签
            if labels:
                class_name = self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)
                if conf:
                    label = f"{class_name} {confidence:.2f}"
                else:
                    label = class_name
            else:
                label = ""

            # DEBUG: 单个框绘制信息
            if debug:
                from ultralytics.utils import LOGGER
                cls_name = self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)
                pts_str = ",".join([f"[{p[0]:.1f},{p[1]:.1f}]" for p in box_corners])
                LOGGER.info(f"[DEBUG][OBB][Draw] #{i} {cls_name}: 绘制角点[{pts_str}]")

            # 绘制旋转框和标签
            color = colors(cls_id, True)
            annotator.box_label(
                box=box_corners,
                label=label,
                color=color
            )

        # 绘制源文件名（如果 show_filename 为 True）
        # cv2 模式下 annotator.sf 是 fontScale（浮点数），annotator.tf 是 thickness（整数）
        if show_filename:
            fname = self._get_filename_text()
            if fname and hasattr(annotator, "sf"):
                fs = annotator.sf   # fontScale (float)
                ft = annotator.tf   # thickness (int)
                tw, th = cv2.getTextSize(fname, 0, fs, ft)[0]
                pad = 5
                img = annotator.im
                cv2.rectangle(img, (pad, pad), (pad + tw + 4, pad + th + 8), (0, 0, 0), -1)
                cv2.putText(img, fname, (pad + 2, pad + th + 2), 0, fs, (255, 255, 255), ft, cv2.LINE_AA)

        return annotator.result()

    def plot(
        self,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True,
        debug: bool = False,
        show_filename: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        绘制 OBB 检测结果

        Args:
            conf: 是否显示置信度
            line_width: 线宽
            font_size: 字体大小
            labels: 是否显示标签
            debug: 是否输出调试信息
            show_filename: 是否在结果图上显示源文件名

        Returns:
            {'rgb': annotated_rgb, 'x': annotated_x} 或 {'rgb': annotated_rgb}
        """
        results = {}

        # 预取原图
        rgb0 = self.orig_imgs.get('rgb', None)
        x0 = self.orig_imgs.get('x', None)

        # 1. RGB 可视化
        if rgb0 is not None:
            rgb_img = self._prepare_display_image(rgb0.copy(), modality='rgb')
        else:
            # X-only 推理：使用黑底占位图
            if 'ori_shape' not in self.meta:
                raise ValueError("meta 缺少 ori_shape，无法为缺失的RGB构造占位画布")
            h, w = self.meta['ori_shape']
            rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

        # DEBUG: 绘图坐标信息
        if debug and len(self._obb) > 0:
            from ultralytics.utils import LOGGER
            LOGGER.info(f"[DEBUG][OBB][Plot] 画布尺寸: {rgb_img.shape}")
            corners = self._obb.xyxyxyxy  # [N, 4, 2]
            for i in range(len(self._obb)):
                cls_id = int(self._obb.cls[i])
                cls_name = self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)
                pts = corners[i].tolist()
                pts_str = ",".join([f"[{p[0]:.1f},{p[1]:.1f}]" for p in pts])
                LOGGER.info(f"[DEBUG][OBB][Plot] #{i} {cls_name}: 角点[{pts_str}]")

        rgb_annotated = self._annotate_obb_image(
            rgb_img,
            conf=conf,
            line_width=line_width,
            font_size=font_size,
            labels=labels,
            debug=debug,
            show_filename=show_filename
        )
        results['rgb'] = rgb_annotated

        # 2. X 模态可视化
        if self.can_visualize_x and x0 is not None:
            x_img = self._prepare_display_image(x0.copy(), modality='x')

            # 处理 X 模态图像（确保是3通道BGR）
            if len(x_img.shape) == 2:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)
            elif x_img.shape[2] == 1:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)

            x_annotated = self._annotate_obb_image(
                x_img,
                conf=conf,
                line_width=line_width,
                font_size=font_size,
                labels=labels,
                debug=debug,
                show_filename=show_filename
            )
            results['x'] = x_annotated

        return results

    def save_txt(
        self,
        save_path: Path,
        save_conf: bool = False
    ):
        """
        保存 OBB 格式的 txt 标签文件

        格式: class x1 y1 x2 y2 x3 y3 x4 y4 [conf]
        坐标为归一化的4角点坐标

        Args:
            save_path: 保存路径
            save_conf: 是否保存置信度
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if len(self._obb) == 0:
            save_path.write_text("")
            return

        # 获取归一化的4角点坐标
        xyxyxyxyn = self._obb.xyxyxyxyn  # [N, 4, 2]

        with open(save_path, 'w') as f:
            for i in range(len(self._obb)):
                cls_id = int(self._obb.cls[i])
                corners = xyxyxyxyn[i].flatten()  # [8] - x1,y1,x2,y2,x3,y3,x4,y4

                # 格式: class x1 y1 x2 y2 x3 y3 x4 y4 [conf]
                line = f"{cls_id}"
                for coord in corners:
                    line += f" {coord:.6f}"

                if save_conf:
                    conf_val = float(self._obb.conf[i])
                    line += f" {conf_val:.6f}"

                f.write(line + "\n")

    def save_json(
        self,
        save_path: Path
    ):
        """
        保存 OBB JSON 结果，保留水平框兼容字段并新增旋转框字段。

        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        rgb_path = self.paths.get('rgb', None)
        x_path = self.paths.get('x', None)
        has_rgb = self.orig_imgs.get('rgb', None) is not None
        has_x = self.orig_imgs.get('x', None) is not None

        if has_rgb and has_x:
            modality = 'rgb+x'
        elif has_rgb:
            modality = 'rgb'
        elif has_x:
            modality = 'x'
        else:
            modality = 'none'

        data = {
            'id': self.meta['id'],
            'task': 'obb',
            'paths': {
                'rgb': str(rgb_path) if rgb_path is not None else None,
                'x': str(x_path) if x_path is not None else None
            },
            'meta': {
                'x_modality': self.meta['x_modality'],
                'xch': self.meta['xch'],
                'ori_shape': self.meta['ori_shape'],
                'imgsz': self.meta['imgsz'],
                'modality': modality,
                'modalities': {'rgb': has_rgb, 'x': has_x},
                'visualization': {
                    'rgb_rendered_from': 'rgb' if has_rgb else 'blank',
                    'x_visualizable': bool(has_x and self.xch in {1, 3})
                }
            },
            'detections': [],
            'summary': {
                'num_dets': int(self.num_dets),
                'class_counts': self.class_counts(),
            }
        }

        h, w = self.meta['ori_shape']
        xywhr = self.obb.xywhr
        xyxy = self.obb.xyxy
        corners = self.obb.xyxyxyxy
        corners_n = self.obb.xyxyxyxyn

        for i in range(len(self.obb)):
            cls = int(self.obb.cls[i])
            conf = float(self.obb.conf[i])
            cx, cy, bw, bh, angle = xywhr[i].tolist()
            x1, y1, x2, y2 = xyxy[i].tolist()

            detection = {
                'class': cls,
                'class_name': self.names.get(cls, str(cls)),
                'confidence': conf,
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                },
                'bbox_normalized': {
                    'x_center': float((x1 + x2) / 2 / w),
                    'y_center': float((y1 + y2) / 2 / h),
                    'width': float((x2 - x1) / w),
                    'height': float((y2 - y1) / h)
                },
                'obb': {
                    'xywhr': {
                        'x_center': float(cx),
                        'y_center': float(cy),
                        'width': float(bw),
                        'height': float(bh),
                        'rotation': float(angle),
                    },
                    'corners': [
                        {'x': float(pt[0]), 'y': float(pt[1])}
                        for pt in corners[i].tolist()
                    ],
                    'corners_normalized': [
                        {'x': float(pt[0]), 'y': float(pt[1])}
                        for pt in corners_n[i].tolist()
                    ],
                },
            }
            data['detections'].append(detection)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _get_instance_bbox(self, idx: int, padding_ratio: float = 0.05) -> tuple:
        """
        获取OBB实例的外接水平矩形（带padding）
        """
        # 使用 obb.xyxy 获取外接水平矩形
        xyxy = self._obb.xyxy[idx]
        x1, y1, x2, y2 = xyxy.tolist()

        # 计算padding
        w, h = x2 - x1, y2 - y1
        pad_x = w * padding_ratio
        pad_y = h * padding_ratio

        # 应用padding并裁剪到图像边界
        ori_h, ori_w = self.meta['ori_shape']
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(ori_w, x2 + pad_x)
        y2 = min(ori_h, y2 + pad_y)

        return int(x1), int(y1), int(x2), int(y2)

    def _crop_annotate_instance(
        self,
        img: np.ndarray,
        idx: int,
        crop_bbox: tuple,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None
    ) -> np.ndarray:
        """
        在裁切图上标注单个OBB实例（绘制旋转框）
        """
        annotator = Annotator(img, line_width=line_width, font_size=font_size, pil=False)

        x1_crop, y1_crop = crop_bbox[0], crop_bbox[1]

        # 获取旋转框角点并转换到裁切坐标系
        corners = self._obb.xyxyxyxy[idx]  # [4, 2]
        local_corners = corners.copy()
        local_corners[:, 0] -= x1_crop
        local_corners[:, 1] -= y1_crop

        cls_id = int(self._obb.cls[idx])
        conf = float(self._obb.conf[idx])
        cls_name = self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)
        label = f"{cls_name} {conf:.2f}"

        color = colors(cls_id, True)
        annotator.box_label(local_corners.tolist(), label, color)

        return annotator.result()

    def save_crop(
        self,
        save_dir: Path,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None
    ) -> List[Path]:
        """
        保存OBB实例裁切图
        """
        if self._obb is None or len(self._obb) == 0:
            return []

        saved_paths = []
        sample_id = self.meta.get('id', 'unknown')

        crop_dir = Path(save_dir) / 'crops' / sample_id
        crop_dir.mkdir(parents=True, exist_ok=True)

        rgb0 = self.orig_imgs.get('rgb', None)
        x0 = self.orig_imgs.get('x', None)

        for idx in range(len(self._obb)):
            cls_id = int(self._obb.cls[idx])
            conf = float(self._obb.conf[idx])
            cls_name = self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)

            name_prefix = f"{cls_name}_{idx}_{conf:.2f}"
            crop_bbox = self._get_instance_bbox(idx)
            x1, y1, x2, y2 = crop_bbox

            crops = {}

            if rgb0 is not None:
                rgb_crop = rgb0[y1:y2, x1:x2].copy()
                rgb_crop = self._crop_annotate_instance(rgb_crop, idx, crop_bbox, line_width, font_size)
                crops['rgb'] = rgb_crop

            if self.can_visualize_x and x0 is not None:
                x_crop = x0[y1:y2, x1:x2].copy()
                if len(x_crop.shape) == 2:
                    x_crop = cv2.cvtColor(x_crop, cv2.COLOR_GRAY2BGR)
                elif x_crop.shape[2] == 1:
                    x_crop = cv2.cvtColor(x_crop, cv2.COLOR_GRAY2BGR)
                x_crop = self._crop_annotate_instance(x_crop, idx, crop_bbox, line_width, font_size)
                crops['x'] = x_crop

            if 'rgb' in crops:
                rgb_path = crop_dir / f"{name_prefix}_rgb.jpg"
                cv2.imwrite(str(rgb_path), crops['rgb'])
                saved_paths.append(rgb_path)

            if 'x' in crops:
                x_path = crop_dir / f"{name_prefix}_x.jpg"
                cv2.imwrite(str(x_path), crops['x'])
                saved_paths.append(x_path)

            if 'rgb' in crops and 'x' in crops:
                h1, h2 = crops['rgb'].shape[0], crops['x'].shape[0]
                if h1 != h2:
                    target_h = max(h1, h2)
                    if h1 < target_h:
                        crops['rgb'] = cv2.resize(crops['rgb'], (int(crops['rgb'].shape[1] * target_h / h1), target_h))
                    if h2 < target_h:
                        crops['x'] = cv2.resize(crops['x'], (int(crops['x'].shape[1] * target_h / h2), target_h))

                merged = np.concatenate([crops['rgb'], crops['x']], axis=1)
                merged_path = crop_dir / f"{name_prefix}_merged.jpg"
                cv2.imwrite(str(merged_path), merged)
                saved_paths.append(merged_path)

        return saved_paths


class MultiModalPoseResults(MultiModalResults):
    """多模态姿态估计推理结果"""

    def __init__(
        self,
        boxes: np.ndarray,
        keypoints: Optional[np.ndarray],
        paths: Dict[str, Path],
        orig_imgs: Dict[str, np.ndarray],
        meta: Dict,
        names: Optional[Dict[int, str]] = None
    ):
        super().__init__(
            boxes=boxes,
            paths=paths,
            orig_imgs=orig_imgs,
            meta=meta,
            names=names
        )
        self.keypoints = keypoints  # [N, K, 3] (x, y, conf)

    def _crop_annotate_instance(
        self,
        img: np.ndarray,
        idx: int,
        crop_bbox: tuple,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None
    ) -> np.ndarray:
        """
        在裁切图上标注单个Pose实例（绘制BBOX + 关键点骨架）
        """
        annotator = Annotator(img, line_width=line_width, font_size=font_size, pil=False)

        x1_crop, y1_crop = crop_bbox[0], crop_bbox[1]

        # 绘制BBOX
        box = self.boxes[idx]
        orig_xyxy = box[:4].tolist()
        local_xyxy = [
            orig_xyxy[0] - x1_crop,
            orig_xyxy[1] - y1_crop,
            orig_xyxy[2] - x1_crop,
            orig_xyxy[3] - y1_crop
        ]

        cls_id = int(box[5])
        conf = float(box[4])
        cls_name = self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)
        label = f"{cls_name} {conf:.2f}"

        color = colors(cls_id, True)
        annotator.box_label(local_xyxy, label, color)

        # 绘制关键点
        if self.keypoints is not None and idx < len(self.keypoints):
            kpts = self.keypoints[idx].copy()
            # 转换到裁切坐标系
            kpts[:, 0] -= x1_crop
            kpts[:, 1] -= y1_crop
            annotator.kpts(kpts, shape=img.shape[:2])

        return annotator.result()


class MultiModalClassifyResults(MultiModalResults):
    """多模态分类推理结果"""

    def __init__(
        self,
        probs: np.ndarray,
        paths: Dict[str, Path],
        orig_imgs: Dict[str, np.ndarray],
        meta: Dict,
        names: Optional[Dict[int, str]] = None
    ):
        super().__init__(
            boxes=np.zeros((0, 6)),  # 分类无边界框
            paths=paths,
            orig_imgs=orig_imgs,
            meta=meta,
            names=names
        )
        self.probs = probs  # [1, num_classes]

    @property
    def top1(self) -> int:
        """返回 top-1 类别索引"""
        return int(self.probs.argmax())

    @property
    def top5(self) -> list:
        """返回 top-5 类别索引"""
        return self.probs.argsort()[0][-5:][::-1].tolist()

    @property
    def top1_conf(self) -> float:
        """返回 top-1 置信度"""
        return float(self.probs.max())


class MultiModalSemSegResults:
    """Semantic segmentation result with class-id mask and optional confidence map."""

    def __init__(self, sem_mask, paths, orig_imgs, meta, names=None, conf_map=None):
        self.sem_mask = sem_mask.astype(np.uint16 if int(np.max(sem_mask, initial=0)) > 255 else np.uint8)
        self.paths = paths
        self.orig_imgs = orig_imgs
        self.meta = meta
        self.names = names or {}
        self.conf_map = conf_map

    def plot(self, show_labels=True, show_conf=True, alpha=0.5, show_filename=False):
        overlays = {}
        if self.orig_imgs.get("rgb") is not None:
            overlays["rgb"] = self._overlay(self.orig_imgs["rgb"], self.sem_mask, alpha, show_filename)
        if self.orig_imgs.get("x") is not None and self.meta.get("xch", 3) in {1, 3}:
            overlays["x"] = self._overlay(self._prepare_x_for_display(self.orig_imgs["x"]), self.sem_mask, alpha, show_filename)
        return overlays

    def _overlay(self, image, mask, alpha=0.5, show_filename=False):
        import cv2
        from .predictor import generate_color_map
        nc = int(np.max(mask)) + 1
        colors = generate_color_map(nc)
        overlay = image.copy()
        for c in range(nc):
            overlay[mask == c] = (overlay[mask == c] * (1 - alpha) + np.array(colors[c]) * alpha).astype(np.uint8)
        return overlay

    def _prepare_x_for_display(self, x_img):
        if x_img.ndim == 2:
            return cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)
        if x_img.shape[-1] == 1:
            return cv2.cvtColor(x_img[..., 0], cv2.COLOR_GRAY2BGR)
        return x_img[..., :3]

    def save_mask(self, path):
        import cv2
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.sem_mask)

    def save_json(self, path):
        import json
        from pathlib import Path
        hist = np.bincount(self.sem_mask.reshape(-1), minlength=len(self.names)).tolist()
        payload = {
            "paths": {k: str(v) for k, v in self.paths.items()},
            "orig_shape": list(self.meta.get("orig_shape", self.sem_mask.shape)),
            "input_shape": list(self.meta.get("input_shape", [])),
            "mask_shape": list(self.sem_mask.shape),
            "ignore_index": int(self.meta.get("ignore_index", 255)),
            "class_histogram": hist,
            "names": self.names,
            "meta": self.meta,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
