# Ultralytics Multimodal Inference - Saver Component
# Handles saving of multimodal inference results
# Version: v1.0
# Date: 2026-01-13

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from .results import MultiModalResults


class MultiModalSaver:
    """
    多模态推理结果保存器（显式且可预测）

    输出文件命名规则（以 RGB 为主路径基准）：
    - {id}_rgb.jpg: 仅当存在真实 RGB 原图时输出
    - {id}_{x_modality}.jpg: 仅当 xch ∈ {1,3} 出
    - {id}_multimodal.jpg: 仅当 RGB 与 X 都可视化时出（并排）
    - labels/{id}.txt: 可选（坐标以 RGB 尺寸归一化）
    - {id}.json: 可选（paths + boxes + meta）
    """

    def __init__(
        self,
        save_dir: Path,
        save: bool = True,
        save_img: Optional[bool] = None,
        save_txt: bool = False,
        save_json: bool = False,
        save_conf: bool = False,
        crop: bool = False
    ):
        """
        初始化保存器

        Args:
            save_dir: 保存根目录
            save: 是否保存可视化图像（兼容字段；等价于 save_img）
            save_img: 是否保存可视化图像（推荐字段）
            save_txt: 是否保存txt标签
            save_json: 是否保存json结果
            save_conf: txt中是否包含置信度
            crop: 是否保存实例裁切图
        """
        self.save_dir = Path(save_dir)
        if save_img is None:
            save_img = save
        self.save_img = save_img
        self.save_txt = save_txt
        self.save_json = save_json
        self.save_conf = save_conf
        self.crop = crop

        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.save_txt:
            (self.save_dir / 'labels').mkdir(parents=True, exist_ok=True)

        if self.save_json:
            (self.save_dir / 'json').mkdir(parents=True, exist_ok=True)

    def save(
        self,
        result: MultiModalResults,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        labels: bool = True,
        show_filename: bool = False
    ):
        """
        保存单个推理结果

        Args:
            result: MultiModalResults 实例
            conf: 可视化时是否显示置信度
            line_width: 线宽
            font_size: 字体大小
            labels: 是否显示标签
            show_filename: 是否在结果图上显示源文件名
        """
        sample_id = result.meta['id']
        x_modality = result.meta['x_modality']
        has_real_rgb = result.orig_imgs.get('rgb', None) is not None
        has_real_x = result.orig_imgs.get('x', None) is not None

        # 1. 保存可视化图像（受 save 参数控制）
        if self.save_img:
            annotated = result.plot(
                conf=conf,
                line_width=line_width,
                font_size=font_size,
                labels=labels,
                show_filename=show_filename
            )

            # 1.1 RGB 可视化
            if has_real_rgb:
                rgb_path = self.save_dir / f"{sample_id}_rgb.jpg"
                cv2.imwrite(str(rgb_path), annotated['rgb'])

            # 1.2 X 模态可视化（仅当 xch in {1,3}）
            if 'x' in annotated:
                x_path = self.save_dir / f"{sample_id}_{x_modality}.jpg"
                cv2.imwrite(str(x_path), annotated['x'])

            # 1.3 双模态并排合并图
            if has_real_rgb and has_real_x:
                merged = result.plot_merged(
                    conf=conf,
                    line_width=line_width,
                    font_size=font_size,
                    labels=labels,
                    show_filename=show_filename
                )
                if merged is not None:
                    merged_path = self.save_dir / f"{sample_id}_multimodal.jpg"
                    cv2.imwrite(str(merged_path), merged)

        # 2. 保存txt标签（可选）
        if self.save_txt:
            txt_path = self.save_dir / 'labels' / f"{sample_id}.txt"
            result.save_txt(txt_path, save_conf=self.save_conf)

        # 3. 保存json结果（可选）
        if self.save_json:
            json_path = self.save_dir / 'json' / f"{sample_id}.json"
            result.save_json(json_path)

        # 4. 保存实例裁切图（可选）
        if self.crop:
            result.save_crop(
                save_dir=self.save_dir,
                line_width=line_width,
                font_size=font_size
            )


class MultiModalSemSegSaver:
    """Save SemSeg masks, overlays, confidence maps, and JSON metadata."""

    def __init__(self, save_dir, save_img=False, save_json=False, save_conf=False):
        self.save_dir = Path(save_dir)
        self.save_img = save_img
        self.save_json = save_json
        self.save_conf = save_conf

    def save(self, result, show_labels=True, show_conf=True, show_filename=False):
        import cv2
        sample_id = result.meta["id"]
        result.save_mask(self.save_dir / "pred_mask" / f"{sample_id}.png")
        if self.save_conf and result.conf_map is not None:
            conf = np.clip(result.conf_map * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(self.save_dir / "pred_mask" / f"{sample_id}_conf.png"), conf)
        if self.save_img:
            overlays = result.plot(show_labels=show_labels, show_conf=show_conf, show_filename=show_filename)
            for key, image in overlays.items():
                out = self.save_dir / "overlay" / f"{sample_id}_{key}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), image)
        if self.save_json:
            result.save_json(self.save_dir / "json" / f"{sample_id}.json")
