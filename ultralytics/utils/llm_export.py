# Ultralytics YOLO, AGPL-3.0 license
"""
LLM-friendly JSON export utilities for multimodal training final validation results.

This module provides functions to export final validation results in a structured,
LLM-friendly JSON format. Only multimodal models (YOLOMM/RTDETRMM) support this feature.

Example:
    >>> from ultralytics.utils.llm_export import export_final_val_llm_json
    >>> # Called automatically in multimodal trainer's final_eval()
    >>> export_final_val_llm_json(trainer)
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ultralytics.utils import LOGGER  # noqa: F401 - kept for potential future use

# Schema version for the JSON export format
SCHEMA_VERSION = "1.0.0"

# Task-specific primary metrics configuration
TASK_PRIMARY_METRICS = {
    "detect": {"name": "map50_95", "group": "B"},
    "obb": {"name": "map50_95", "group": "B"},
    "segment": {"name": "map50_95", "group": "M"},
    "pose": {"name": "map50_95", "group": "P"},
    "classify": {"name": "top1", "group": "C"},
}

# Per-class ranking keys for different tasks
PERCLASS_RANK_KEYS = {
    "detect": "mAP50-95",
    "obb": "mAP50-95",
    "segment": "Mask-F1",
    "pose": "Pose-F1",
}

# Maximum classes before truncation
MAX_PERCLASS_FULL = 100
MAX_PERCLASS_TRUNCATED = 20


def export_final_val_llm_json(trainer, filename: str = "final_validation.json") -> Path:
    """
    Export final validation results to LLM-friendly JSON format.

    Main entrypoint. Build report from trainer and write JSON to trainer.save_dir.

    Args:
        trainer: Ultralytics trainer instance (multimodal only).
        filename: Output filename.

    Returns:
        Path to the exported JSON file.

    Raises:
        ValueError: If required fields are missing or trainer is not multimodal.
    """
    # Validate multimodal trainer
    if not _is_multimodal_trainer(trainer):
        raise ValueError(
            "LLM JSON export is only supported for multimodal trainers (YOLOMM/RTDETRMM). "
            "Traditional YOLO/RTDETR single-modal trainers do not support this feature."
        )

    # Build the report
    report = build_llm_report(trainer)

    # Write to file
    output_path = Path(trainer.save_dir) / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return output_path


def build_llm_report(trainer) -> dict:
    """
    Build a LLM-friendly, reproducible JSON report from trainer.

    Args:
        trainer: Ultralytics multimodal trainer instance.

    Returns:
        Dictionary containing the structured report.

    Raises:
        ValueError: If required fields are missing.
    """
    integrity = {"missing_fields": [], "warnings": []}

    # 1. Experiment block
    experiment = _build_experiment_block(trainer, integrity)

    # 2. Model block
    model = _build_model_block(trainer, integrity)

    # 3. Multimodal block
    multimodal = _build_multimodal_block(trainer, integrity)

    # 4. Data block
    data = _build_data_block(trainer, integrity)

    # 5. Final validation block
    final_validation = _build_final_validation_block(trainer, integrity)

    # 6. Summary block
    task = experiment.get("task", "detect")
    metrics_normalized = final_validation.get("metrics_normalized", {})
    summary = {"text": _generate_summary(task, metrics_normalized, multimodal)}

    # Build complete report
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment": experiment,
        "model": model,
        "multimodal": multimodal,
        "data": data,
        "final_validation": final_validation,
        "summary": summary,
        "integrity": integrity,
    }

    return report


def _is_multimodal_trainer(trainer) -> bool:
    """Check if trainer is a multimodal trainer."""
    # Check for multimodal-specific attributes
    has_mm_config = hasattr(trainer, "multimodal_config") and trainer.multimodal_config is not None
    has_mm_flag = hasattr(trainer, "is_multimodal") and trainer.is_multimodal
    has_dual_modal = hasattr(trainer, "is_dual_modal")

    # Check trainer class name
    class_name = trainer.__class__.__name__
    is_mm_class = "MultiModal" in class_name or "RTDETRMM" in class_name

    return (has_mm_config or has_mm_flag or has_dual_modal) and is_mm_class


def _build_experiment_block(trainer, integrity: dict) -> dict:
    """Build experiment configuration block."""
    args = trainer.args
    task = getattr(args, "task", None) or getattr(trainer, "task", "detect")

    # Extract key training arguments
    args_dict = {}
    for key in ["epochs", "batch", "imgsz", "data", "lr0", "optimizer", "device"]:
        value = getattr(args, key, None)
        if value is not None:
            # Convert Path to string
            if isinstance(value, Path):
                value = str(value)
            args_dict[key] = value

    return {"task": task, "args": args_dict}


def _build_model_block(trainer, integrity: dict) -> dict:
    """Build model structure information block."""
    model_block = {}

    # Determine model family
    class_name = trainer.__class__.__name__
    if "RTDETRMM" in class_name:
        model_block["family"] = "RTDETRMM"
    else:
        model_block["family"] = "YOLOMM"

    # Task
    args = trainer.args
    model_block["task"] = getattr(args, "task", None) or getattr(trainer, "task", "detect")

    # Model class
    if hasattr(trainer, "model") and trainer.model is not None:
        model_block["model_class"] = trainer.model.__class__.__name__

    # Model YAML
    model_cfg = getattr(args, "model", None)
    if model_cfg:
        if isinstance(model_cfg, Path):
            model_cfg = str(model_cfg)
        model_block["yaml"] = Path(model_cfg).name if model_cfg else None
    else:
        integrity["missing_fields"].append("model.yaml")

    # Number of classes and names
    if hasattr(trainer, "data") and trainer.data:
        model_block["nc"] = trainer.data.get("nc")
        names = trainer.data.get("names")
        if names:
            if isinstance(names, dict):
                model_block["names"] = names
            elif isinstance(names, list):
                model_block["names"] = {str(i): n for i, n in enumerate(names)}

    # Channels
    if hasattr(trainer, "model") and hasattr(trainer.model, "yaml"):
        ch = trainer.model.yaml.get("ch", 3)
        model_block["channels"] = ch

    return model_block


def _build_multimodal_block(trainer, integrity: dict) -> dict:
    """Build multimodal configuration block."""
    mm_block = {"enabled": True}

    # Mode: dual or single
    modality = getattr(trainer, "modality", None)
    is_dual = getattr(trainer, "is_dual_modal", True)

    if modality:
        mm_block["mode"] = "single"
        mm_block["modality"] = modality
    else:
        mm_block["mode"] = "dual"
        mm_block["modality"] = None

    # Get multimodal config
    mm_config = getattr(trainer, "multimodal_config", None)
    if mm_config:
        models = mm_config.get("models", [])
        mm_block["models"] = models

        # Determine x_modality
        x_modality = _determine_x_modality(trainer, mm_config)
        mm_block["x_modality"] = x_modality

        # Get x_channels from data config
        if hasattr(trainer, "data") and trainer.data:
            x_ch = trainer.data.get("Xch", 3)
            mm_block["x_channels"] = x_ch

        # Modalities mapping
        modalities_map = mm_config.get("modalities", {})
        if modalities_map:
            mm_block["modalities"] = modalities_map
    else:
        integrity["warnings"].append("multimodal_config not found")

    return mm_block


def _determine_x_modality(trainer, mm_config: dict) -> Optional[str]:
    """Determine the X modality name from trainer or config."""
    # Try to get from multimodal_config
    models = mm_config.get("models", [])
    for m in models:
        if m != "rgb":
            return m

    # Try trainer's method if available
    if hasattr(trainer, "_determine_x_modality_from_data"):
        try:
            return trainer._determine_x_modality_from_data()
        except Exception:
            pass

    # Try to infer from data config
    if hasattr(trainer, "data") and trainer.data:
        modality_used = trainer.data.get("modality_used", [])
        for m in modality_used:
            if m != "rgb":
                return m

    return None


def _build_data_block(trainer, integrity: dict) -> dict:
    """Build dataset information block."""
    data_block = {}

    args = trainer.args
    data_cfg = getattr(args, "data", None)
    if data_cfg:
        data_block["data_cfg"] = str(data_cfg) if isinstance(data_cfg, Path) else data_cfg

    # Data dict
    if hasattr(trainer, "data") and trainer.data:
        data_dict = {}
        for key in ["path", "train", "val", "test", "nc", "Xch"]:
            if key in trainer.data:
                value = trainer.data[key]
                if isinstance(value, Path):
                    value = str(value)
                data_dict[key] = value
        data_block["data_dict"] = data_dict

    return data_block


def _build_final_validation_block(trainer, integrity: dict) -> dict:
    """Build final validation results block."""
    val_block = {}

    # Checkpoint paths
    checkpoint = {}
    if hasattr(trainer, "best") and trainer.best:
        checkpoint["best"] = str(trainer.best)
    if hasattr(trainer, "last") and trainer.last:
        checkpoint["last"] = str(trainer.last)
    val_block["checkpoint"] = checkpoint

    # Get raw metrics
    metrics_raw = _get_metrics_raw(trainer)
    val_block["metrics_raw"] = metrics_raw

    # Normalize metrics
    task = getattr(trainer.args, "task", None) or getattr(trainer, "task", "detect")
    metrics_normalized = _get_metrics_normalized(task, metrics_raw)
    val_block["metrics_normalized"] = metrics_normalized

    # Per-class metrics (skip for classify task)
    if task != "classify":
        per_class = _get_per_class_metrics(trainer, task, integrity)
        if per_class:
            val_block["per_class"] = per_class

    return val_block


def _get_metrics_raw(trainer) -> Dict[str, Any]:
    """Extract raw metrics from trainer."""
    metrics_raw = {}

    # Try to get from trainer.metrics (dict-like)
    if hasattr(trainer, "metrics") and trainer.metrics:
        metrics = trainer.metrics
        if isinstance(metrics, dict):
            metrics_raw = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        elif hasattr(metrics, "results_dict"):
            metrics_raw = metrics.results_dict
        elif hasattr(metrics, "keys") and hasattr(metrics, "mean_results"):
            # Build from keys and mean_results
            keys = metrics.keys
            values = metrics.mean_results()
            if keys and values:
                for k, v in zip(keys, values):
                    metrics_raw[k] = float(v) if v is not None else None

    return metrics_raw


def _get_metrics_normalized(task: str, metrics_raw: Dict[str, Any]) -> dict:
    """
    Normalize raw Ultralytics metrics dict into a stable, task-aware structure.

    Args:
        task: Task type (detect, obb, segment, pose, classify).
        metrics_raw: Raw metrics dictionary.

    Returns:
        Normalized metrics structure with groups and primary_metric.
    """
    normalized = {"task": task, "groups": []}

    # Parse metrics by group
    groups_data = _parse_metrics_by_group(task, metrics_raw)

    for group_name, group_metrics in groups_data.items():
        group_entry = {"group": group_name, "raw_keys": {}}

        for metric_name, (value, raw_key) in group_metrics.items():
            group_entry[metric_name] = value
            group_entry["raw_keys"][metric_name] = raw_key

        normalized["groups"].append(group_entry)

    # Determine primary metric
    primary_config = TASK_PRIMARY_METRICS.get(task, TASK_PRIMARY_METRICS["detect"])
    primary_metric = {
        "name": primary_config["name"],
        "group": primary_config["group"],
        "value": None,
    }

    # Find the primary metric value
    for group in normalized["groups"]:
        if group["group"] == primary_config["group"]:
            primary_metric["value"] = group.get(primary_config["name"])
            break

    normalized["primary_metric"] = primary_metric

    return normalized


def _parse_metrics_by_group(task: str, metrics_raw: Dict[str, Any]) -> Dict[str, Dict]:
    """Parse raw metrics into groups based on task type."""
    groups = {}

    # Mapping from raw key patterns to normalized names
    metric_patterns = {
        r"precision": "precision",
        r"recall": "recall",
        r"mAP50(?!-)": "map50",
        r"mAP50-95": "map50_95",
        r"accuracy_top1": "top1",
        r"accuracy_top5": "top5",
    }

    # Group patterns based on suffix in parentheses
    group_pattern = re.compile(r"\(([A-Z])\)$")

    for raw_key, value in metrics_raw.items():
        if not isinstance(value, (int, float)):
            continue

        # Determine group
        group_match = group_pattern.search(raw_key)
        if group_match:
            group_name = group_match.group(1)
        else:
            # Default group based on task
            if task == "classify":
                group_name = "C"
            else:
                group_name = "B"

        if group_name not in groups:
            groups[group_name] = {}

        # Determine normalized metric name
        for pattern, norm_name in metric_patterns.items():
            if re.search(pattern, raw_key):
                groups[group_name][norm_name] = (float(value), raw_key)
                break

    return groups


def _get_per_class_metrics(trainer, task: str, integrity: dict) -> Optional[dict]:
    """Extract per-class metrics if available."""
    if task == "classify":
        return None  # Classify doesn't export per_class

    per_class_result = {"classes": [], "selection": "full"}

    # Try to get per-class data from validator or metrics
    class_metrics = []

    if hasattr(trainer, "metrics"):
        metrics = trainer.metrics
        # Try to access summary() method which often contains per-class data
        if hasattr(metrics, "summary"):
            try:
                summary_df = metrics.summary()
                if summary_df is not None and len(summary_df) > 0:
                    class_metrics = _parse_summary_dataframe(summary_df, task)
            except Exception:
                pass

        # Alternative: try to access ap_class_index and class_result
        if not class_metrics and hasattr(metrics, "ap_class_index") and hasattr(metrics, "class_result"):
            try:
                ap_indices = metrics.ap_class_index
                names = getattr(metrics, "names", {})
                for i, idx in enumerate(ap_indices):
                    result = metrics.class_result(i)
                    class_name = names.get(idx, str(idx))
                    class_entry = _build_class_entry(idx, class_name, result, task)
                    if class_entry:
                        class_metrics.append(class_entry)
            except Exception:
                pass

    if not class_metrics:
        integrity["warnings"].append("per_class metrics not available")
        return None

    # Sort by ranking key (ascending = worst first for truncation)
    rank_key = PERCLASS_RANK_KEYS.get(task, "mAP50-95")
    class_metrics.sort(key=lambda x: x.get(rank_key, 0))

    # Truncate if too many classes
    if len(class_metrics) > MAX_PERCLASS_FULL:
        class_metrics = class_metrics[:MAX_PERCLASS_TRUNCATED]
        per_class_result["selection"] = f"worst_{MAX_PERCLASS_TRUNCATED}"

    per_class_result["classes"] = class_metrics
    per_class_result["rank_key"] = rank_key
    per_class_result["total_classes"] = len(class_metrics)

    return per_class_result


def _parse_summary_dataframe(summary_df, task: str) -> List[dict]:
    """Parse summary DataFrame to extract per-class metrics."""
    class_metrics = []

    try:
        import pandas as pd

        if not isinstance(summary_df, pd.DataFrame):
            return []

        for _, row in summary_df.iterrows():
            entry = {}
            if "class_id" in row:
                entry["class_id"] = int(row["class_id"])
            if "class_name" in row:
                entry["class_name"] = str(row["class_name"])
            if "P" in row:
                entry["precision"] = float(row["P"])
            if "R" in row:
                entry["recall"] = float(row["R"])
            if "mAP50" in row:
                entry["mAP50"] = float(row["mAP50"])
            if "mAP50-95" in row:
                entry["mAP50-95"] = float(row["mAP50-95"])

            # Task-specific metrics
            if task == "segment":
                if "Mask-F1" in row:
                    entry["Mask-F1"] = float(row["Mask-F1"])
            elif task == "pose":
                if "Pose-F1" in row:
                    entry["Pose-F1"] = float(row["Pose-F1"])

            if entry:
                class_metrics.append(entry)
    except Exception:
        pass

    return class_metrics


def _build_class_entry(idx: int, name: str, result: tuple, task: str) -> Optional[dict]:
    """Build a single class entry from class_result tuple."""
    if not result or len(result) < 4:
        return None

    entry = {
        "class_id": int(idx),
        "class_name": str(name),
        "precision": float(result[0]) if result[0] is not None else None,
        "recall": float(result[1]) if result[1] is not None else None,
        "mAP50": float(result[2]) if result[2] is not None else None,
        "mAP50-95": float(result[3]) if result[3] is not None else None,
    }

    return entry


def _generate_summary(task: str, metrics_normalized: dict, mm_info: dict) -> str:
    """
    Generate a concise natural-language summary (LLM-friendly).

    Args:
        task: Task type.
        metrics_normalized: Normalized metrics dictionary.
        mm_info: Multimodal configuration dictionary.

    Returns:
        Concise summary string.
    """
    parts = []

    # Task info
    parts.append(f"task={task}")

    # Multimodal info
    mode = mm_info.get("mode", "dual")
    x_modality = mm_info.get("x_modality", "x")
    x_channels = mm_info.get("x_channels", 3)
    parts.append(f"multimodal={mode}, x_modality={x_modality}, x_channels={x_channels}")

    # Primary metric
    primary = metrics_normalized.get("primary_metric", {})
    if primary and primary.get("value") is not None:
        parts.append(f"primary={primary['group']}.{primary['name']}={primary['value']:.4f}")

    # Group summaries
    groups = metrics_normalized.get("groups", [])
    group_strs = []
    for g in groups:
        group_name = g.get("group", "?")
        if task == "classify":
            top1 = g.get("top1")
            top5 = g.get("top5")
            if top1 is not None and top5 is not None:
                group_strs.append(f"{group_name}: Top1={top1:.4f}, Top5={top5:.4f}")
        else:
            p = g.get("precision")
            r = g.get("recall")
            map50 = g.get("map50")
            map50_95 = g.get("map50_95")
            metrics_str = []
            if p is not None:
                metrics_str.append(f"P={p:.4f}")
            if r is not None:
                metrics_str.append(f"R={r:.4f}")
            if map50 is not None:
                metrics_str.append(f"mAP50={map50:.4f}")
            if map50_95 is not None:
                metrics_str.append(f"mAP50-95={map50_95:.4f}")
            if metrics_str:
                group_strs.append(f"{group_name}: {', '.join(metrics_str)}")

    if group_strs:
        parts.append(f"groups=[{'; '.join(group_strs)}]")

    return "; ".join(parts)
