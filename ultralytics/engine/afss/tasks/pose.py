# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ultralytics.utils.metrics import OKS_SIGMA

from .base import BaseAFSSTaskAdapter


@BaseAFSSTaskAdapter.register("pose")
class PoseAFSSTaskAdapter(BaseAFSSTaskAdapter):
    """AFSS adapter for YOLOMM pose task using joint box+keypoint sufficiency."""

    task_name = "pose"
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
    def _ensure_pose_sigma(validator: Any) -> None:
        """Ensure pose validator has keypoint sigma initialized for _process_batch()."""
        if getattr(validator, "sigma", None) is not None:
            return
        kpt_shape = None
        if getattr(validator, "data", None):
            kpt_shape = validator.data.get("kpt_shape")
        if not kpt_shape:
            kpt_shape = [17, 3]
        validator.kpt_shape = list(kpt_shape)
        is_pose_17 = validator.kpt_shape == [17, 3]
        nkpt = int(validator.kpt_shape[0]) if len(validator.kpt_shape) else 17
        validator.sigma = OKS_SIGMA if is_pose_17 else np.ones(nkpt) / nkpt

    @staticmethod
    def _resolve_task_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize required pose task metrics."""
        metrics = row.get("task_metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError("AFSS task=pose requires row['task_metrics'] to be a dict")
        required = (
            "box_precision",
            "box_recall",
            "pose_precision",
            "pose_recall",
            "box_sufficiency",
            "pose_sufficiency",
        )
        missing = [name for name in required if name not in metrics]
        if missing:
            raise RuntimeError(f"AFSS task=pose score row missing required task_metrics fields: {missing}")
        return dict(metrics)

    def score_batch(self, trainer: Any, validator: Any, preds: Any, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Delegate pose scoring to validator-native helpers and enforce joint_min metrics schema."""
        self._ensure_pose_sigma(validator)
        score_rows = validator.afss_score_batch(preds, batch)
        for row in score_rows:
            task_metrics = self._resolve_task_metrics(row)
            row["task_name"] = self.task_name
            row["task_metrics"] = task_metrics
            row["matched_count"] = int(task_metrics.get("matched_box_count", row.get("matched_count", 0)))
            row["pred_count"] = int(task_metrics.get("pred_count", row.get("pred_count", 0)))
            row["gt_count"] = int(task_metrics.get("gt_count", row.get("gt_count", 0)))
        return score_rows

    def summarize_task_metrics(self, score_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Summarize meaningful pose AFSS metrics and skip metadata-like fields."""
        return self._summarize_selected_task_metrics(
            score_rows,
            (
                "box_precision",
                "box_recall",
                "pose_precision",
                "pose_recall",
                "box_sufficiency",
                "pose_sufficiency",
                "matched_box_count",
                "matched_pose_count",
                "pred_count",
                "gt_count",
            ),
        )
