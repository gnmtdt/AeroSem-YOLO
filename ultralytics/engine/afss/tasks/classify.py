# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseAFSSTaskAdapter


@BaseAFSSTaskAdapter.register("classify")
class ClassificationAFSSTaskAdapter(BaseAFSSTaskAdapter):
    """AFSS adapter for classification task using top1_prob_if_correct sufficiency."""

    task_name = "classify"
    required_validator_methods = (
        "build_dataset",
        "preprocess",
        "postprocess",
        "afss_score_batch",
    )
    _REQUIRED_METRIC_KEYS = (
        "top1_correct",
        "top1_prob",
        "margin",
        "target_class",
        "pred_class",
    )

    @staticmethod
    def _resolve_task_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize classification task metrics with explicit top1 semantics."""
        metrics = row.get("task_metrics")
        if isinstance(metrics, dict):
            normalized = dict(metrics)
        else:
            normalized = {
            "top1_correct": int(float(row.get("precision_op", 0.0)) > 0.5),
            "top1_prob": float(row.get("recall_op", 0.0)),
            "margin": float(row.get("margin", 0.0)),
            "target_class": int(row.get("target_class", -1)),
            "pred_class": int(row.get("pred_class", -1)),
        }
        missing = [key for key in ClassificationAFSSTaskAdapter._REQUIRED_METRIC_KEYS if key not in normalized]
        if missing:
            raise RuntimeError(f"AFSS task=classify score row missing required task_metrics fields: {missing}")
        return normalized

    def score_batch(self, trainer: Any, validator: Any, preds: Any, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Delegate per-sample scoring to validator helpers and enforce shared AFSS row schema."""
        score_rows = validator.afss_score_batch(preds, batch)
        for row in score_rows:
            task_metrics = self._resolve_task_metrics(row)
            row["task_name"] = self.task_name
            row["task_metrics"] = task_metrics
            # Compatibility fields only. Classification true semantics live in task_metrics.
            row["precision_op"] = float(row.get("precision_op", task_metrics["top1_correct"]))
            row["recall_op"] = float(row.get("recall_op", task_metrics["top1_prob"]))
        return score_rows

    def summarize_task_metrics(self, score_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Summarize meaningful classification AFSS metrics and skip class-id fields."""
        return self._summarize_selected_task_metrics(
            score_rows,
            ("top1_correct", "top1_prob", "margin"),
        )
