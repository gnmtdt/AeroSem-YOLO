"""Preprocessor for visualize_core.

Model ingress only:
- resolve expected PRIMARY/SECONDARY/FUSED channel contract from model/router
- align channels with explicit {1,3} conversion rules only
- preserve display-only projection for vis.py / renderer.py

Display-only 3ch projection (pseudocolor, colormap) stays in vis.py / renderer.py
and must not flow back into model ingress.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np
import cv2

from ultralytics.data.multimodal.precision_contract import HIGH_PRECISION, LEGACY
from ultralytics.data.multimodal.utils import align_and_validate_x
from ultralytics.data.multimodal.precision_contract import resolve_precision_contract


class Preprocessor:
    @staticmethod
    def passthrough(x: Any) -> Any:
        return x

    @staticmethod
    def model_input_channels(model: Any, default: int = 3) -> int:
        """
        Infer expected input channel count for visualization preprocessing.

        Notes:
        - Prefer multimodal router's Dual channel config (3+Xch) when available, because some
          multimodal architectures (e.g., mm-mid) keep the first Conv in_channels=3 but still
          accept a Dual input tensor that is routed internally by the router.
        - Fallback to the first Conv2d.in_channels for non-multimodal models.
        """

        # 1) Prefer multimodal router config (Dual = 3 + Xch)
        router = None
        for chain in (
            ("mm_router",),
            ("multimodal_router",),
            ("model", "mm_router"),
            ("model", "multimodal_router"),
        ):
            m = model
            ok = True
            for a in chain:
                if hasattr(m, a):
                    m = getattr(m, a)
                else:
                    ok = False
                    break
            if ok and m is not None:
                router = m
                break

        if router is not None:
            try:
                sources = getattr(router, "INPUT_SOURCES", None)
                if isinstance(sources, dict) and "FUSED" in sources:
                    return int(sources["FUSED"])
            except Exception:
                pass

        # 2) Fallback: first Conv2d.in_channels (typical for standard YOLO)
        try:
            import torch.nn as nn  # local import to keep module lightweight

            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    return int(m.in_channels)

            # Generic fallback: any module exposing in_channels/out_channels
            for m in model.modules():
                if hasattr(m, "in_channels") and hasattr(m, "out_channels"):
                    ic = int(getattr(m, "in_channels"))
                    if ic > 0:
                        return ic
        except Exception:
            pass

        return int(default)

    @staticmethod
    def model_input_size(model: Any, default: int = 640) -> int:
        try:
            if hasattr(model, 'args') and hasattr(model.args, 'imgsz'):
                return int(model.args.imgsz)
        except Exception:
            pass
        return int(default)

    @staticmethod
    def _letterbox(im: np.ndarray, new_shape: Union[int, Tuple[int, int]] = 640, color=(114, 114, 114), stride: int = 32) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int, int, int]]:
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (top, bottom, left, right)

    @staticmethod
    def _legacy_to_float01(img: np.ndarray) -> np.ndarray:
        """Legacy 路径: 按 dtype 归一化到 [0,1]。"""
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        if img.dtype == np.uint16:
            return img.astype(np.float32) / 65535.0
        if img.dtype == np.float32:
            return img
        if img.dtype == np.float64:
            return img.astype(np.float32)
        if np.issubdtype(img.dtype, np.integer):
            img_f = img.astype(np.float32)
            max_val = img_f.max()
            if max_val > 1.0:
                img_f = img_f / max_val if max_val > 0 else img_f
            return img_f
        raise ValueError(
            f"可视化预处理器 legacy 路径收到不支持的dtype={img.dtype}。"
            f"支持: uint8, uint16, float32, float64, integer types。"
        )

    @staticmethod
    def _high_precision_to_float01(img: np.ndarray) -> np.ndarray:
        """High-precision 路径: min-max 归一化到 [0,1]。"""
        img_f = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mn = float(img_f.min())
        mx = float(img_f.max())
        if mx <= mn:
            return np.zeros_like(img_f, dtype=np.float32)
        return (img_f - mn) / (mx - mn)

    @staticmethod
    def _to_float01(img: np.ndarray, precision: str = LEGACY) -> np.ndarray:
        """Convert image to float32 [0,1], 按 precision 分流。

        - LEGACY: 按 dtype 归一化 (uint8/255, uint16/65535, float passthrough)
        - HIGH_PRECISION: min-max 归一化（适用于任意值域的高精度数据）
        """
        return Preprocessor._high_precision_to_float01(img) if precision == HIGH_PRECISION else Preprocessor._legacy_to_float01(img)

    @staticmethod
    def letterbox_single(image: np.ndarray, size: int, precision: str = LEGACY) -> np.ndarray:
        lb, _, _ = Preprocessor._letterbox(image, new_shape=size)
        return Preprocessor._to_float01(lb, precision=precision)

    @staticmethod
    def letterbox_dual(
        primary: np.ndarray,
        secondary: np.ndarray,
        size: int,
        primary_precision: str = LEGACY,
        secondary_precision: str = LEGACY,
    ) -> np.ndarray:
        # First image defines padding for consistency
        primary_lb, _, padding = Preprocessor._letterbox(primary, new_shape=size)
        secondary_lb, _, _ = Preprocessor._letterbox(secondary, new_shape=size)
        # Normalize
        primary_f = Preprocessor._to_float01(primary_lb, precision=primary_precision)
        secondary_f = Preprocessor._to_float01(secondary_lb, precision=secondary_precision)
        # Ensure channel dims
        if primary_f.ndim == 2:
            primary_f = primary_f[:, :, None]
        if secondary_f.ndim == 2:
            secondary_f = secondary_f[:, :, None]
        # Concatenate channels [Primary(3), Secondary(Xch)] without pseudocolor
        return np.concatenate([primary_f, secondary_f], axis=2)

    @staticmethod
    def letterbox_dual_aligned(
        primary: np.ndarray,
        secondary: np.ndarray,
        size: int,
        align_base: str = "primary",
        stride: int = 32,
        primary_precision: str = LEGACY,
        secondary_precision: str = LEGACY,
    ) -> np.ndarray:
        """
        Letterbox two modalities with the SAME ratio/padding derived from the base image.

        This enforces pixel-wise alignment on the final canvas by reusing the scaling ratio and
        padding computed from the base modality (primary or secondary) for the other modality.

        Args:
            primary: Primary (RGB) image (HWC)
            secondary: Secondary (X) modality image (HWC or HW or HWC with 1+ channels)
            size: square target size (int)
            align_base: 'primary' or 'secondary' indicating which modality defines ratio/padding
            stride: model stride for padding alignment (default: 32)

        Returns:
            Concatenated HWC float32 array with channels [Primary(3), Secondary(Xch)] in [0,1].
        """
        base = primary if str(align_base).lower() in ("primary", "rgb") else secondary

        # Compute base letterbox once to obtain target padding
        base_lb, ratio, pad = Preprocessor._letterbox(base, new_shape=size, stride=stride)
        r = ratio[0]
        top, bottom, left, right = pad

        def _apply_with_ratio(im: np.ndarray) -> np.ndarray:
            shape = im.shape[:2]
            # Resize with base ratio r
            new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
            if shape[::-1] != new_unpad:
                im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            # Apply the same padding tuple as base
            return cv2.copyMakeBorder(
                im,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )

        if base is primary:
            primary_lb = base_lb
            secondary_lb = _apply_with_ratio(secondary)
        else:
            secondary_lb = base_lb
            primary_lb = _apply_with_ratio(primary)

        # Normalize to [0,1]
        primary_f = Preprocessor._to_float01(primary_lb, precision=primary_precision)
        secondary_f = Preprocessor._to_float01(secondary_lb, precision=secondary_precision)

        # Ensure explicit channel dims
        if primary_f.ndim == 2:
            primary_f = primary_f[:, :, None]
        if secondary_f.ndim == 2:
            secondary_f = secondary_f[:, :, None]

        return np.concatenate([primary_f, secondary_f], axis=2)

    @staticmethod
    def prepare_inputs(inputs: Dict[str, np.ndarray], model: Any) -> np.ndarray:
        """Backward-compatible model-ingress entry point.

        Historically this helper performed lightweight letterbox/normalization
        directly.  Keep the public API, but route model-bound inputs through
        ``prepare_model_hwc()`` so callers cannot bypass the router-derived
        PRIMARY/SECONDARY/FUSED channel contract.
        """
        normalized_inputs: Dict[str, np.ndarray] = {}
        source_paths = inputs.get('__source_paths__')
        if source_paths is not None:
            normalized_inputs['__source_paths__'] = source_paths

        if 'rgb' in inputs:
            normalized_inputs['rgb'] = inputs['rgb']
        elif 'PRIMARY' in inputs:
            normalized_inputs['rgb'] = inputs['PRIMARY']

        if 'x' in inputs:
            normalized_inputs['x'] = inputs['x']
        elif 'SECONDARY' in inputs:
            normalized_inputs['x'] = inputs['SECONDARY']

        if not any(k in normalized_inputs for k in ('rgb', 'x')):
            raise ValueError("未检测到有效的 primary/secondary 输入。")

        hwc, _ = Preprocessor.prepare_model_hwc(normalized_inputs, model)
        return hwc

    # ------------------------------------------------------------------
    # Model-ingress helpers: channel contract + unified HWC assembly
    # ------------------------------------------------------------------

    @staticmethod
    def model_channel_contract(model: Any) -> Dict[str, int]:
        """Return the model's PRIMARY/SECONDARY/FUSED channel contract.

        Walks the same attribute chains as ``model_input_channels`` but
        returns the per-role breakdown instead of just the total.
        """
        router = None
        for chain in (
            ("mm_router",),
            ("multimodal_router",),
            ("model", "mm_router"),
            ("model", "multimodal_router"),
        ):
            m = model
            ok = True
            for attr in chain:
                if hasattr(m, attr):
                    m = getattr(m, attr)
                else:
                    ok = False
                    break
            if ok and m is not None:
                router = m
                break

        if router is not None:
            sources = getattr(router, "INPUT_SOURCES", {}) or {}
            primary_ch = int(sources.get("PRIMARY", sources.get("RGB", getattr(router, "primary_ch", 3))))
            secondary_ch = int(sources.get("SECONDARY", sources.get("X", getattr(router, "secondary_ch", 0))))
            fused_ch = int(sources.get("FUSED", sources.get("Dual", primary_ch + secondary_ch)))
            # Legacy fallback: only "Dual" key present, no PRIMARY/SECONDARY
            if secondary_ch == 0 and fused_ch > primary_ch:
                secondary_ch = fused_ch - primary_ch
            return {"primary_ch": primary_ch, "secondary_ch": secondary_ch, "fused_ch": fused_ch}

        total = Preprocessor.model_input_channels(model, default=3)
        return {"primary_ch": min(3, total), "secondary_ch": max(total - min(3, total), 0), "fused_ch": total}

    @staticmethod
    def _coerce_channels(arr: np.ndarray, expected_ch: int, *, label: str) -> np.ndarray:
        """Adapt *arr* to *expected_ch* channels.

        Only ``{1, 3}`` explicit conversion is allowed.
        When ``expected_ch > 3`` the actual channel count must match exactly.
        """
        if arr.ndim == 2:
            arr = arr[:, :, None]
        actual_ch = int(arr.shape[2])
        if actual_ch == expected_ch:
            return arr
        if {actual_ch, expected_ch} <= {1, 3}:
            if actual_ch == 1 and expected_ch == 3:
                return np.repeat(arr, 3, axis=2)
            if actual_ch == 3 and expected_ch == 1:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                return gray[:, :, None]
        raise ValueError(
            f"{label} 通道数不匹配：期望 {expected_ch}，实际 {actual_ch}。"
            f"仅允许 1<->3 显式转换；当通道数 >3 时必须严格匹配。"
        )

    @staticmethod
    def prepare_model_hwc(
        inputs: Dict[str, np.ndarray],
        model: Any,
        *,
        ablation_fill: str = "zeros",
        align_base: str = "primary",
        size: int | None = None,
    ) -> tuple:
        """Unified model-ingress HWC assembly.

        Assembles a single HWC float32 tensor that satisfies the model's
        PRIMARY/SECONDARY/FUSED channel contract.

        Args:
            inputs: dict with keys ``'rgb'`` / ``'x'`` (arrays) and
                optional ``'__source_paths__'`` for precision probing.
            model: the model object (used to query router + imgsz).
            ablation_fill: ``'zeros'`` or ``'half'`` for missing-modality fill.
            align_base: ``'primary'`` or ``'secondary'`` for dual letterbox.

        Returns:
            ``(hwc_array, contract_dict)`` where *contract_dict* has keys
            ``primary_ch``, ``secondary_ch``, ``fused_ch``.
        """
        size = size if size is not None else Preprocessor.model_input_size(model)
        contract = Preprocessor.model_channel_contract(model)

        def _contract_error(exc: Exception) -> ValueError:
            return ValueError(
                f"VIS ingress 与模型契约不一致：primary={contract['primary_ch']} "
                f"secondary={contract['secondary_ch']} fused={contract['fused_ch']}。"
                f"请检查输入文件通道数与模型 YAML/router 配置。原始错误: {exc}"
            )

        fill_val = 0.0 if ablation_fill == "zeros" else 0.5
        source_paths = inputs.get("__source_paths__", {})
        precision_contract = resolve_precision_contract(
            {
                "rgb_path": source_paths.get("rgb"),
                "x_path": source_paths.get("x"),
                "rgb_array": inputs.get("rgb"),
                "x_array": inputs.get("x"),
            },
            mode="single_point",
        )["contracts"]

        primary = inputs.get("rgb")
        secondary = inputs.get("x")
        has_primary = primary is not None
        has_secondary = secondary is not None

        if has_primary and has_secondary:
            # Dual modality: coerce primary, then align secondary via shared utility
            try:
                primary = Preprocessor._coerce_channels(primary, contract["primary_ch"], label="primary")
                secondary = align_and_validate_x(secondary, primary, expected_xch=contract["secondary_ch"])
            except ValueError as exc:
                raise _contract_error(exc) from exc
            hwc = Preprocessor.letterbox_dual_aligned(
                primary,
                secondary,
                size=size,
                align_base=align_base,
                primary_precision=precision_contract["rgb"],
                secondary_precision=precision_contract["x"],
            )
        elif has_primary:
            # Primary-only (ablation): fill secondary with zeros
            try:
                primary = Preprocessor._coerce_channels(primary, contract["primary_ch"], label="primary")
            except ValueError as exc:
                raise _contract_error(exc) from exc
            primary_f = Preprocessor.letterbox_single(primary, size, precision=precision_contract["rgb"])
            secondary_f = np.full(
                (primary_f.shape[0], primary_f.shape[1], contract["secondary_ch"]),
                fill_val,
                dtype=np.float32,
            )
            hwc = np.concatenate([primary_f, secondary_f], axis=2)
        elif has_secondary:
            # Secondary-only (ablation): fill primary with zeros
            try:
                secondary = Preprocessor._coerce_channels(secondary, contract["secondary_ch"], label="secondary")
            except ValueError as exc:
                raise _contract_error(exc) from exc
            secondary_f = Preprocessor.letterbox_single(secondary, size, precision=precision_contract["x"])
            primary_f = np.full(
                (secondary_f.shape[0], secondary_f.shape[1], contract["primary_ch"]),
                0.0,
                dtype=np.float32,
            )
            hwc = np.concatenate([primary_f, secondary_f], axis=2)
        else:
            raise ValueError("未检测到有效的 primary/secondary 输入。")

        if hwc.shape[2] != contract["fused_ch"]:
            raise ValueError(
                f"模型输入通道数不一致：预处理后={hwc.shape[2]}，期望={contract['fused_ch']}。"
            )
        return hwc, contract
