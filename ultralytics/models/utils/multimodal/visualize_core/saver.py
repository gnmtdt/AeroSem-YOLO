"""Unified saver for visualization results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .types import CoreVisualizationResult


class Saver:
    @staticmethod
    def _ensure_dir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _save_array(img: np.ndarray, path: Path, *, scale: float = 1.0) -> None:
        # Ensure parent dir exists (supports nested relative keys like "rgb/original")
        path.parent.mkdir(parents=True, exist_ok=True)

        s = float(scale)
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"save_scale 必须为正数，收到: {scale}")

        out = img
        if s != 1.0:
            if not isinstance(out, np.ndarray) or out.ndim < 2:
                raise ValueError(f"无法缩放保存：输入类型/维度异常: {type(out)} shape={getattr(out, 'shape', None)}")
            h, w = int(out.shape[0]), int(out.shape[1])
            new_w = max(1, int(round(w * s)))
            new_h = max(1, int(round(h * s)))
            max_side = max(new_w, new_h)
            if max_side > 16384:
                raise ValueError(
                    f"导出尺寸过大：{new_w}x{new_h}（save_scale={s}）。"
                    f"请降低 export_scale/save_scale 或使用更小的输入尺寸。"
                )
            interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
            out = cv2.resize(out, (new_w, new_h), interpolation=interp)

        # Assume RGB input; convert to BGR for OpenCV
        if out.ndim == 3 and out.shape[2] == 3:
            cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(path), out)

    @classmethod
    def save(cls, *, results: List[CoreVisualizationResult], out_dir: Path, method: str) -> List[str]:
        cls._ensure_dir(out_dir)
        saved: List[str] = []
        for r in results:
            layer_idx = r.meta.get("layer_idx", "na")
            modality = r.meta.get("modality", "auto")
            # feature map may be a grid image, heat is overlay or colormap
            base = f"{method}_layer{layer_idx}_{modality}"
            # Determine destination directory (support per-sample subdir via img_key)
            base_dir = out_dir
            img_key = r.meta.get('img_key', None)
            if isinstance(img_key, str) and len(img_key) > 0:
                base_dir = out_dir / img_key
                cls._ensure_dir(base_dir)
            # Optional per-result subdir (e.g., per-layer folder)
            subdir = r.meta.get('subdir', None)
            if isinstance(subdir, str) and len(subdir) > 0:
                base_dir = base_dir / subdir
                cls._ensure_dir(base_dir)

            save_scale = float(r.meta.get('save_scale', 1.0))
            if r.type == 'feature_tiles':
                # Save per-channel tiles under sub-directory per layer
                dst_dir = base_dir
                channels = r.meta.get('channels', None)
                if isinstance(r.data, list):
                    for i, v in enumerate(r.data):
                        if isinstance(v, np.ndarray):
                            if channels and i < len(channels):
                                fname = f"{base}_ch{int(channels[i])}.png"
                            else:
                                fname = f"{base}_{i:03d}.png"
                            path = dst_dir / fname
                            cls._save_array(v, path, scale=save_scale)
                            saved.append(str(path))
                # Proceed next result
                continue
            # Default naming for non-tiles
            if method == 'feature':
                base = f"{base}_grid"

            if isinstance(r.data, dict):
                for k, v in r.data.items():
                    if isinstance(v, np.ndarray):
                        # Allow nested relative paths for structured exports, e.g. "rgb/original"
                        k_str = str(k)
                        if ("/" in k_str) or ("\\" in k_str):
                            rel = Path(k_str)
                            if rel.is_absolute() or ".." in rel.parts:
                                raise ValueError(f"非法导出路径 key={k_str!r}（必须为相对路径且不可包含 '..'）")
                            path = base_dir / rel
                            if path.suffix == "":
                                path = path.with_suffix(".png")
                        else:
                            path = base_dir / f"{base}_{k_str}.png"
                        cls._save_array(v, path, scale=save_scale)
                        saved.append(str(path))
            elif isinstance(r.data, list):
                for i, v in enumerate(r.data):
                    if isinstance(v, np.ndarray):
                        path = base_dir / f"{base}_{i:03d}.png"
                        cls._save_array(v, path, scale=save_scale)
                        saved.append(str(path))
            elif isinstance(r.data, np.ndarray):
                path = base_dir / f"{base}.png"
                cls._save_array(r.data, path, scale=save_scale)
                saved.append(str(path))
        return saved
