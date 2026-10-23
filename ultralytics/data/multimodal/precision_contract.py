from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from ultralytics.utils import LOGGER

LEGACY = "legacy"
HIGH_PRECISION = "high_precision"
DATASET_RUN = "dataset_run"
SINGLE_POINT = "single_point"

HIGH_PRECISION_SUFFIXES = {".npy", ".npz", ".tif", ".tiff"}
LEGACY_IMAGE_SUFFIXES = {".bmp", ".dng", ".heic", ".jpeg", ".jpg", ".mpo", ".png", ".webp"}


def normalize_precision_contract(contract: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """Normalize a contract-like object to {'rgb': ..., 'x': ...} with legacy defaults."""
    base = {"rgb": LEGACY, "x": LEGACY}
    if not isinstance(contract, Mapping):
        return base
    for key in ("rgb", "x"):
        value = contract.get(key, base[key])
        base[key] = HIGH_PRECISION if value == HIGH_PRECISION else LEGACY
    return base


def inspect_modality_precision(path: Any = None, array: Any = None, modality: str = "rgb") -> Dict[str, Any]:
    """Classify one modality source as legacy or high_precision using path + lightweight dtype evidence."""
    resolved_path = Path(path) if path not in (None, "") else None
    suffix = resolved_path.suffix.lower() if resolved_path else ""

    if array is not None:
        arr = np.asarray(array)
        dtype = str(arr.dtype)
    else:
        arr = None
        dtype = None

    if resolved_path is None and arr is None:
        return {
            "modality": modality,
            "path": None,
            "suffix": "",
            "dtype": None,
            "available": False,
            "precision": LEGACY,
            "reason": "missing",
        }

    if suffix in HIGH_PRECISION_SUFFIXES:
        precision, reason = HIGH_PRECISION, f"suffix:{suffix}"
    elif suffix in LEGACY_IMAGE_SUFFIXES:
        precision, reason = LEGACY, f"suffix:{suffix}"
    elif arr is not None and arr.dtype != np.uint8:
        precision, reason = HIGH_PRECISION, f"dtype:{arr.dtype}"
    elif arr is not None and arr.dtype == np.uint8:
        precision, reason = LEGACY, "dtype:uint8"
    elif suffix:
        # 未知后缀默认 LEGACY — 不再假设未知格式为高精度
        precision, reason = LEGACY, f"suffix:{suffix}(unknown->legacy)"
    else:
        precision, reason = LEGACY, "default:legacy"

    return {
        "modality": modality,
        "path": str(resolved_path) if resolved_path else None,
        "suffix": suffix,
        "dtype": dtype,
        "available": True,
        "precision": precision,
        "reason": reason,
    }


def _sample_indices(total: int, sample_size: int) -> list[int]:
    if total <= 0 or sample_size <= 0:
        return []
    if sample_size >= total:
        return list(range(total))
    return sorted({int(round(x)) for x in np.linspace(0, total - 1, num=sample_size)})


def _resolve_x_path(dataset: Any, im_file: str) -> Optional[str]:
    try:
        x_path = dataset._find_corresponding_x_image(im_file)
    except Exception:
        return None
    return x_path if x_path and Path(x_path).exists() else None


def _modality_ratio(inspections: Sequence[Mapping[str, Any]]) -> float:
    if not inspections:
        return 0.0
    hp = sum(1 for item in inspections if item.get("precision") == HIGH_PRECISION)
    return hp / len(inspections)


def probe_dataset_precision(dataset: Any, sample_size: int = 50, threshold: float = 0.8) -> Dict[str, Any]:
    """Resolve dataset-run contract from sampled multimodal file evidence."""
    total = len(getattr(dataset, "im_files", []) or [])
    indices = _sample_indices(total, min(sample_size, total))
    rgb_reports = []
    x_reports = []
    for idx in indices:
        im_file = dataset.im_files[idx]
        rgb_reports.append(inspect_modality_precision(im_file, modality="rgb"))
        x_reports.append(inspect_modality_precision(_resolve_x_path(dataset, im_file), modality="x"))

    rgb_ratio = _modality_ratio(rgb_reports)
    x_ratio = _modality_ratio(x_reports)
    contracts = {
        "rgb": HIGH_PRECISION if rgb_ratio > threshold else LEGACY,
        "x": HIGH_PRECISION if x_ratio > threshold else LEGACY,
    }

    return {
        "mode": DATASET_RUN,
        "sampled": len(indices),
        "threshold": float(threshold),
        "contracts": contracts,
        "ratios": {
            "rgb_high_precision": round(rgb_ratio, 6),
            "x_high_precision": round(x_ratio, 6),
        },
        "modalities": {
            "primary": getattr(dataset, "primary_modality", "rgb"),
            "secondary": getattr(dataset, "secondary_modality", None) or getattr(dataset, "x_modality", "x"),
            "x_name": getattr(dataset, "x_modality", "unknown"),
        },
        "sample_details": {
            "rgb": rgb_reports,
            "x": x_reports,
        },
    }


def filter_dataset_by_contract(dataset: Any, contract: Mapping[str, Any]) -> tuple[list[int], list[Dict[str, Any]]]:
    """Filter dataset indices that do not match the resolved dataset-run contract."""
    contract = normalize_precision_contract(contract)
    keep_indices: list[int] = []
    filtered: list[Dict[str, Any]] = []
    allow_generated_x = bool(getattr(dataset, "enable_self_modal_generation", False))

    for idx, im_file in enumerate(getattr(dataset, "im_files", [])):
        rgb_report = inspect_modality_precision(im_file, modality="rgb")
        x_path = _resolve_x_path(dataset, im_file)
        x_report = inspect_modality_precision(x_path, modality="x")
        reasons: list[str] = []

        if rgb_report["precision"] != contract["rgb"]:
            reasons.append("rgb_mismatch")

        if x_report["available"]:
            if x_report["precision"] != contract["x"]:
                reasons.append("x_mismatch")
        elif not allow_generated_x:
            reasons.append("x_missing")
        elif contract["x"] == HIGH_PRECISION:
            reasons.append("x_missing_for_high_precision")

        if reasons:
            filtered.append(
                {
                    "im_file": im_file,
                    "x_file": x_path,
                    "reason": "|".join(reasons),
                    "rgb_precision": rgb_report["precision"],
                    "x_precision": x_report["precision"],
                }
            )
        else:
            keep_indices.append(idx)

    return keep_indices, filtered


def resolve_single_point_precision_contract(
    rgb_path: Any = None,
    x_path: Any = None,
    rgb_array: Any = None,
    x_array: Any = None,
) -> Dict[str, Any]:
    """Resolve point-level precision contract for predict/vis-like single-point inputs."""
    rgb_report = inspect_modality_precision(rgb_path, rgb_array, modality="rgb")
    x_report = inspect_modality_precision(x_path, x_array, modality="x")
    return {
        "mode": SINGLE_POINT,
        "contracts": {
            "rgb": rgb_report["precision"],
            "x": x_report["precision"],
        },
        "inspections": {
            "rgb": rgb_report,
            "x": x_report,
        },
    }


def resolve_precision_contract(source: Any, mode: str) -> Dict[str, Any]:
    """Unified precision contract entry point."""
    if mode == DATASET_RUN:
        return probe_dataset_precision(source)
    if mode == SINGLE_POINT:
        if isinstance(source, Mapping):
            return resolve_single_point_precision_contract(
                rgb_path=source.get("rgb_path"),
                x_path=source.get("x_path"),
                rgb_array=source.get("rgb_array"),
                x_array=source.get("x_array"),
            )
        raise TypeError("single_point precision contract expects a mapping-like source")
    raise ValueError(f"Unsupported precision contract mode: {mode}")


def extract_batch_precision_contract(batch: Mapping[str, Any]) -> Dict[str, str]:
    """Extract a batch-level contract from collated dataset samples."""
    contract = batch.get("precision_contract")
    if isinstance(contract, (list, tuple)) and contract:
        contract = contract[0]
    return normalize_precision_contract(contract)


def resolve_modal_precision(labels: dict) -> tuple:
    """从 labels dict 解析 (rgb_precision, x_precision) 元组。

    用于 augment 阶段快速获取每个模态的精度语义，
    而不必在增强算子内部靠 dtype/max-value 猜测值域。
    """
    contract = normalize_precision_contract(labels.get("precision_contract"))
    return contract["rgb"], contract["x"]


def normalize_high_precision_tensor(t: Any, modality: str = "x") -> Any:
    """Shared high-precision normalization for tensor data (BCHW).

    For high-precision modalities that were loaded as raw arrays (uint16/float/etc.)
    and then tensorized. Applies appropriate normalization based on original dtype
    information that should be tracked separately.

    This is the single shared helper for ALL high-precision normalization in
    train / val / predict / vis. No caller may implement its own version.
    """
    import torch

    if not isinstance(t, torch.Tensor):
        return t
    # high-precision data: tensorize 时已经按 dtype 做了初步转换
    # 这里确保值域在合理范围。如果已经在 [0,1] 范围内则直通。
    t_flat = t.float()
    t_max = t_flat.max()
    if t_max <= 1.0:
        return t_flat
    # uint8 风格 [0,255] -> /255
    if t_max <= 255.0:
        return t_flat / 255.0
    # uint16 风格 [0,65535] -> /65535
    if t_max <= 65535.0:
        return t_flat / 65535.0
    # 更大范围 -> min-max normalize
    t_min = t_flat.min()
    denom = t_max - t_min
    if denom > 0:
        return (t_flat - t_min) / denom
    return t_flat


def preprocess_bchw_by_precision_contract(img: Any, contract: Optional[Mapping[str, Any]], raw_dtype: Any = None) -> Any:
    """Contract-aware BCHW tensor preprocessing.

    Replaces the old apply_dataset_precision_contract(). Handles three paths:
    - legacy+legacy: Restores 29e006c^ semantics (uint8 -> all /255; float -> RGB /255, X as-is)
    - high_precision+high_precision: Both modalities use high-precision path
    - mixed: Per-modality handling, no longer relies on whole-tensor dtype

    Args:
        img: BCHW tensor (B, C, H, W) where C = 3+Xch
        contract: precision contract dict {'rgb': ..., 'x': ...}
        raw_dtype: optional original dtype hint before any conversion (for legacy path)

    Returns:
        Preprocessed float tensor
    """
    import torch

    contract = normalize_precision_contract(contract)
    if img is None or getattr(img, "ndim", 0) < 4 or img.shape[1] < 3:
        return img

    raw_dtype = raw_dtype or img.dtype
    img = img.float()
    rgb_contract = contract["rgb"]
    x_contract = contract["x"]

    # --- legacy + legacy: restore 29e006c^ semantics ---
    if rgb_contract == LEGACY and x_contract == LEGACY:
        if raw_dtype == torch.uint8 or (not isinstance(raw_dtype, torch.dtype) and raw_dtype == np.uint8):
            # uint8 batch: all channels /255
            return img / 255.0
        # float batch (e.g. uint16 cast to float at tensorize): RGB /255, X keeps original values
        img[:, :3] = img[:, :3] / 255.0
        return img

    # --- mixed or high_precision paths: per-modality ---
    if rgb_contract == LEGACY:
        img[:, :3] = img[:, :3] / 255.0
    else:
        img[:, :3] = normalize_high_precision_tensor(img[:, :3], modality="rgb")

    if img.shape[1] > 3:
        if x_contract == LEGACY:
            img[:, 3:] = img[:, 3:] / 255.0
        else:
            img[:, 3:] = normalize_high_precision_tensor(img[:, 3:], modality="x")

    return img


# Backward-compatible alias
apply_dataset_precision_contract = preprocess_bchw_by_precision_contract


def write_precision_report(report: Mapping[str, Any], save_dir: Any, split: str) -> Optional[Path]:
    """Persist a structured dataset-run precision report if save_dir is available."""
    if save_dir in (None, ""):
        return None
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / f"precision_contract_{split}.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report_path


def log_dataset_precision_summary(report: Mapping[str, Any], report_path: Optional[Path] = None) -> None:
    """Emit the one-time dataset-run summary warning required by the contract design."""
    contracts = normalize_precision_contract(report.get("contracts"))
    ratios = report.get("ratios", {})
    sampled = int(report.get("sampled", 0))
    filtered_count = int(report.get("filtered_count", 0))
    split = report.get("split", "unknown")
    rgb_c = contracts["rgb"]
    x_c = contracts["x"]

    if rgb_c == HIGH_PRECISION and x_c == HIGH_PRECISION:
        mode_msg = "MultiModal 使用高精度计算"
    elif rgb_c == LEGACY and x_c == LEGACY:
        mode_msg = "MultiModal 使用正常精度计算"
    else:
        rgb_label = "高精度" if rgb_c == HIGH_PRECISION else "正常精度"
        x_label = "高精度" if x_c == HIGH_PRECISION else "正常精度"
        mode_msg = f"MultiModal 使用混合精度计算 (RGB={rgb_label} X模态={x_label})"

    detail_parts = [f"split={split}", f"抽样={sampled}"]
    if filtered_count > 0:
        detail_parts.append(f"过滤={filtered_count}")
    report_str = str(report_path) if report_path else "N/A"
    LOGGER.info("%s | %s | 报告=%s", mode_msg, " ".join(detail_parts), report_str)
