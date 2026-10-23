# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseAFSSTaskAdapter


@BaseAFSSTaskAdapter.register("obb")
class OBBAFSSTaskAdapter(BaseAFSSTaskAdapter):
    """AFSS adapter for OBB task using rotated-box validator primitives."""

    task_name = "obb"
    required_validator_methods = (
        "build_dataset",
        "preprocess",
        "postprocess",
        "_prepare_batch",
        "_prepare_pred",
        "_process_batch",
        "afss_score_batch",
    )
    _REQUIRED_METRIC_KEYS = (
        "obb_precision",
        "obb_recall",
        "matched_count",
        "pred_count",
        "gt_count",
    )

    @staticmethod
    def _resolve_task_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize task metrics while preserving strict OBB semantics."""
        metrics = row.get("task_metrics")
        if isinstance(metrics, dict):
            normalized = dict(metrics)
        else:
            normalized = {
            "obb_precision": float(row.get("precision_op", row.get("precision", 0.0))),
            "obb_recall": float(row.get("recall_op", row.get("recall", 0.0))),
            "matched_count": int(row.get("matched_count", 0)),
            "pred_count": int(row.get("pred_count", 0)),
            "gt_count": int(row.get("gt_count", 0)),
        }
        missing = [key for key in OBBAFSSTaskAdapter._REQUIRED_METRIC_KEYS if key not in normalized]
        if missing:
            raise RuntimeError(f"AFSS task=obb score row missing required task_metrics fields: {missing}")
        return normalized

    def score_batch(self, trainer: Any, validator: Any, preds: Any, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Delegate OBB per-sample scoring to validator helpers and normalize rows."""
        score_rows = validator.afss_score_batch(preds, batch)
        for row in score_rows:
            task_metrics = self._resolve_task_metrics(row)
            row["task_name"] = self.task_name
            row["task_metrics"] = task_metrics
            row["matched_count"] = int(task_metrics.get("matched_count", row.get("matched_count", 0)))
            row["pred_count"] = int(task_metrics.get("pred_count", row.get("pred_count", 0)))
            row["gt_count"] = int(task_metrics.get("gt_count", row.get("gt_count", 0)))
        return score_rows

    def summarize_task_metrics(self, score_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Summarize only meaningful OBB AFSS metrics for logging."""
        return self._summarize_selected_task_metrics(
            score_rows,
            ("obb_precision", "obb_recall", "matched_count", "pred_count", "gt_count"),
        )
