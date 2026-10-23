# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import BaseAFSSTaskAdapter


def compute_detect_afss_pr(matched: int, pred_count: int, gt_count: int) -> tuple[float, float]:
    """Compute detect-family precision and recall for AFSS scoring."""
    if gt_count == 0 and pred_count == 0:
        return 1.0, 1.0
    if gt_count == 0:
        return 0.0, 1.0
    if pred_count == 0:
        return 0.0, 0.0
    return matched / pred_count, matched / gt_count


def build_detect_afss_score_row(
    *,
    im_file: str,
    matched: int,
    pred_count: int,
    gt_count: int,
    task_name: str = "detect",
) -> dict[str, object]:
    """Build a normalized AFSS score row for any detect-family validator."""
    precision_op, recall_op = compute_detect_afss_pr(matched, pred_count, gt_count)
    sufficiency_raw = min(precision_op, recall_op)
    return {
        "sample_key": str(Path(im_file).resolve()),
        "im_file": im_file,
        "task_name": task_name,
        "precision": float(precision_op),
        "recall": float(recall_op),
        "precision_op": float(precision_op),
        "recall_op": float(recall_op),
        "sufficiency_raw": float(sufficiency_raw),
        "valid_for_afss": True,
        "matched_count": int(matched),
        "pred_count": int(pred_count),
        "gt_count": int(gt_count),
        "task_metrics": {
            "matched_count": int(matched),
            "pred_count": int(pred_count),
            "gt_count": int(gt_count),
        },
    }


@BaseAFSSTaskAdapter.register("detect")
class DetectAFSSTaskAdapter(BaseAFSSTaskAdapter):
    """Default AFSS adapter that mirrors the existing detection behavior."""
    task_name = "detect"
    required_validator_methods = (
        "build_dataset",
        "preprocess",
        "postprocess",
        "_prepare_batch",
        "_prepare_pred",
        "_process_batch",
        "afss_score_batch",
    )

    @staticmethod
    def _resolve_task_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize task_metrics while preserving legacy detect counters."""
        metrics = row.get("task_metrics")
        if isinstance(metrics, dict):
            return dict(metrics)
        return {
            "matched_count": int(row.get("matched_count", 0)),
            "pred_count": int(row.get("pred_count", 0)),
            "gt_count": int(row.get("gt_count", 0)),
        }

    def score_batch(self, trainer: Any, validator: Any, preds: Any, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Delegate detect scoring to validator-native helpers and normalize legacy fields."""
        score_rows = validator.afss_score_batch(preds, batch)
        for row in score_rows:
            task_metrics = self._resolve_task_metrics(row)
            row["task_name"] = self.task_name
            row["task_metrics"] = task_metrics
            row["matched_count"] = int(task_metrics.get("matched_count", row.get("matched_count", 0)))
            row["pred_count"] = int(task_metrics.get("pred_count", row.get("pred_count", 0)))
            row["gt_count"] = int(task_metrics.get("gt_count", row.get("gt_count", 0)))
        return score_rows

    _SUMMARY_METRICS = ("precision", "recall")

    def summarize_task_metrics(self, score_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate detect-specific metrics for logging, excluding raw counters."""
        return self._summarize_selected_task_metrics(score_rows, self._SUMMARY_METRICS)
