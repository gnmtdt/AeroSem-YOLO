"""Unified Renderer for visualize_core.

Provides consistent rendering helpers for heatmap overlays and (future) feature
grids. For heatmaps, applies a consistent alpha blending and optional colormap.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import cv2


class Renderer:
    @staticmethod
    def identity(x: Any) -> Any:
        return x

    @staticmethod
    def _ensure_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[2] == 1:
            return np.repeat(img, 3, axis=2)
        if img.ndim == 3 and img.shape[2] >= 3:
            return img[:, :, :3]
        raise ValueError(f"Unsupported image shape for RGB conversion: {img.shape}")

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """Render-boundary conversion to uint8 for display/export.

        This is the accepted export-edge conversion. All earlier high-precision
        semantics should have been preserved or explicitly rejected before reaching here.
        """
        if img.dtype == np.uint8:
            return img
        x = img
        if x.max() <= 1.0:
            x = (x * 255.0).clip(0, 255)
        else:
            x = np.clip(x, 0, 255)
        return x.astype(np.uint8)

    @staticmethod
    def heat_overlay(
        original: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay heatmap onto original RGB image with consistent alpha.

        - If heatmap is single-channel float, apply colormap first.
        - If heatmap is already color (H,W,3), blend directly.
        - Original/X 保持真实通道；只对可见化渲染阶段进行 RGB 叠加。
        """
        ori = Renderer._to_uint8(Renderer._ensure_rgb(original))
        hm = heatmap
        # 尺寸对齐：若热图大小与原图不同，先缩放到原图尺寸
        if hm.ndim == 2:
            if hm.shape[0] != ori.shape[0] or hm.shape[1] != ori.shape[1]:
                hm = cv2.resize(hm, (ori.shape[1], ori.shape[0]), interpolation=cv2.INTER_CUBIC)
        elif hm.ndim == 3 and hm.shape[2] in (1, 3):
            if hm.shape[0] != ori.shape[0] or hm.shape[1] != ori.shape[1]:
                hm = cv2.resize(hm, (ori.shape[1], ori.shape[0]), interpolation=cv2.INTER_CUBIC)
        if hm.ndim == 2:
            hm = Renderer._to_uint8(hm)
            hm = cv2.applyColorMap(hm, colormap)
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        elif hm.ndim == 3 and hm.shape[2] == 3:
            hm = Renderer._to_uint8(hm)
            # 约定：3 通道 heatmap 输入应为 RGB；此处不再进行 BGR→RGB 自动猜测转换
        else:
            raise ValueError(f"Unsupported heatmap shape: {heatmap.shape}")
        return cv2.addWeighted(ori, 1 - alpha, hm, alpha, 0)

    @staticmethod
    def heat_overlay_multimodal(
        originals: Dict[str, np.ndarray],
        heatmaps: Dict[str, np.ndarray],
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k in originals.keys():
            if k in heatmaps:
                out[k] = Renderer.heat_overlay(originals[k], heatmaps[k], alpha=alpha, colormap=colormap)
        return out

    @staticmethod
    def heat_triptych(
        original: np.ndarray,
        heatmap: np.ndarray,
        overlay: np.ndarray | None = None,
        *,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
        scale: float = 1.0,
        pad: int = 8,
        title: bool = True,
    ) -> np.ndarray:
        """
        Create a human-friendly 3-panel image: original | heatmap | overlay.

        All outputs are RGB uint8.
        """
        if not isinstance(scale, (int, float)) or float(scale) <= 0:
            raise ValueError(f"scale must be > 0, got: {scale}")
        s = float(scale)
        pad = int(pad)

        ori = Renderer._to_uint8(Renderer._ensure_rgb(original))

        # Prepare heatmap visualization (RGB)
        hm = heatmap
        if hm.ndim == 2:
            hm_u8 = Renderer._to_uint8(hm)
            hm_color = cv2.applyColorMap(hm_u8, colormap)
            hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
        elif hm.ndim == 3 and hm.shape[2] == 1:
            hm_u8 = Renderer._to_uint8(hm[:, :, 0])
            hm_color = cv2.applyColorMap(hm_u8, colormap)
            hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
        elif hm.ndim == 3 and hm.shape[2] == 3:
            hm_color = Renderer._to_uint8(hm)
        else:
            raise ValueError(f"Unsupported heatmap shape: {heatmap.shape}")

        # Resize heatmap to original if needed
        if hm_color.shape[:2] != ori.shape[:2]:
            hm_color = cv2.resize(hm_color, (ori.shape[1], ori.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Overlay (RGB)
        if overlay is None:
            ov = cv2.addWeighted(ori, 1 - float(alpha), hm_color, float(alpha), 0)
        else:
            ov = Renderer._to_uint8(Renderer._ensure_rgb(overlay))
            if ov.shape[:2] != ori.shape[:2]:
                ov = cv2.resize(ov, (ori.shape[1], ori.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Optional scaling
        if s != 1.0:
            new_wh = (int(round(ori.shape[1] * s)), int(round(ori.shape[0] * s)))
            ori = cv2.resize(ori, new_wh, interpolation=cv2.INTER_LINEAR)
            hm_color = cv2.resize(hm_color, new_wh, interpolation=cv2.INTER_LINEAR)
            ov = cv2.resize(ov, new_wh, interpolation=cv2.INTER_LINEAR)

        h, w = ori.shape[:2]
        title_h = 24 if title else 0
        canvas = np.ones((h + title_h + pad * 2, w * 3 + pad * 4, 3), dtype=np.uint8) * 255

        y0 = pad + title_h
        xs = [pad, pad * 2 + w, pad * 3 + w * 2]
        canvas[y0:y0 + h, xs[0]:xs[0] + w] = ori
        canvas[y0:y0 + h, xs[1]:xs[1] + w] = hm_color
        canvas[y0:y0 + h, xs[2]:xs[2] + w] = ov

        if title:
            labels = ["original", "heatmap", "overlay"]
            for x, t in zip(xs, labels):
                cv2.putText(
                    canvas,
                    t,
                    (x, pad + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        return canvas
