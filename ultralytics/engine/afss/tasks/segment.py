# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseAFSSTaskAdapter


@BaseAFSSTaskAdapter.register("segment")
class SegmentAFSSTaskAdapter(BaseAFSSTaskAdapter):
    """AFSS adapter for segmentation with joint_min(box_sufficiency, mask_sufficiency)."""

    task_name = "segment"
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
        "box_precision",
        "box_recall",
        "mask_precision",
        "mask_recall",
        "box_sufficiency",
        "mask_sufficiency",
    )

    @staticmethod
    def _ensure_required_task_metrics(task_metrics: Dict[str, Any]) -> None:
        """Fail fast when segment task metric schema is incomplete."""
        missing = [key for key in SegmentAFSSTaskAdapter._REQUIRED_METRIC_KEYS if key not in task_metrics]
        if missing:
            raise RuntimeError(f"AFSS task=segment row missing required task_metrics keys: {missing}")

    def score_batch(self, trainer: Any, validator: Any, preds: Any, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Delegate segmentation sample scoring to validator helpers and normalize result rows."""
        score_rows = validator.afss_score_batch(preds, batch)
        for row in score_rows:
            task_metrics = dict(row.get("task_metrics", {}))
            self._ensure_required_task_metrics(task_metrics)
            row["task_name"] = self.task_name
            row["task_metrics"] = task_metrics
            # Keep the shared compatibility fields tied to box branch.
            row["precision_op"] = float(row.get("precision_op", task_metrics["box_precision"]))
            row["recall_op"] = float(row.get("recall_op", task_metrics["box_recall"]))
            if "sufficiency_raw" not in row:
                row["sufficiency_raw"] = min(
                    float(task_metrics["box_sufficiency"]), float(task_metrics["mask_sufficiency"])
                )
            row["matched_count"] = int(row.get("matched_count", task_metrics.get("matched_box_count", 0)))
            row["pred_count"] = int(row.get("pred_count", task_metrics.get("pred_count", 0)))
            row["gt_count"] = int(row.get("gt_count", task_metrics.get("gt_count", 0)))
            row["valid_for_afss"] = bool(row.get("valid_for_afss", True))
        return score_rows

    def summarize_task_metrics(self, score_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Summarize meaningful segment AFSS metrics and skip categorical metadata fields."""
        return self._summarize_selected_task_metrics(
            score_rows,
            (
                "box_precision",
                "box_recall",
                "mask_precision",
                "mask_recall",
                "box_sufficiency",
                "mask_sufficiency",
                "matched_box_count",
                "matched_mask_count",
                "pred_count",
                "gt_count",
                "pred_mask_count",
                "gt_mask_count",
            ),
        )
