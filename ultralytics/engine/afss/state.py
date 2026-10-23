# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .schema import AFSSConfig, AFSSStateSnapshot, AFSS_STATE_SCHEMA_VERSION, SampleState, snapshot_state_dir


class AFSSStateStore:
    """State store for AFSS bootstrap and future per-sample updates."""

    def __init__(self, sample_states: List[SampleState], snapshot_meta: Optional[Dict[str, Any]] = None):
        self.sample_states = sample_states
        self._state_by_key = {state.sample_key: state for state in sample_states}
        self._snapshot_meta = snapshot_meta or {}

    @classmethod
    def from_dataset(cls, dataset: Any, task_name: str | None = None) -> "AFSSStateStore":
        """Build initial state store from dataset image file list."""
        im_files = getattr(dataset, "im_files", None)
        if not im_files:
            raise ValueError("AFSS requires dataset.im_files to bootstrap sample states")

        resolved_task_name = task_name
        if not resolved_task_name:
            dataset_task = getattr(dataset, "task", None)
            if isinstance(dataset_task, str) and dataset_task:
                resolved_task_name = dataset_task
            elif bool(getattr(dataset, "use_obb", False)):
                resolved_task_name = "obb"
            elif bool(getattr(dataset, "use_keypoints", False)):
                resolved_task_name = "pose"
            elif bool(getattr(dataset, "use_segments", False)):
                resolved_task_name = "segment"
            elif "classify" in dataset.__class__.__name__.lower():
                resolved_task_name = "classify"
            else:
                resolved_task_name = "detect"

        sample_states: List[SampleState] = []
        seen_keys = set()
        for index, im_file in enumerate(im_files):
            key = str(Path(im_file).resolve())
            if key in seen_keys:
                raise ValueError(f"Duplicate AFSS sample_key detected for dataset image: {im_file}")
            seen_keys.add(key)

            # Determine modality_flag from dataset capabilities.
            # Prefer public interface over private _find_corresponding_x_image.
            modality_flag = "paired"
            # 优先使用私有方法（1参数，内置 x_modality_dir）
            x_finder = getattr(dataset, "_find_corresponding_x_image", None)
            if callable(x_finder):
                x_path = x_finder(im_file)
            else:
                # 回退到公共方法（需要显式传入 x_modality_dir）
                x_finder = getattr(dataset, "find_corresponding_x_image", None)
                if callable(x_finder):
                    x_mod_dir = getattr(dataset, "x_modality_dir", None)
                    if x_mod_dir is not None:
                        x_mod_suffix = getattr(dataset, "x_modality_suffix", None)
                        x_path = x_finder(im_file, x_mod_dir, x_mod_suffix)
                    else:
                        x_path = None
                else:
                    x_path = None
            if x_path is not None and not Path(x_path).exists():
                modality_flag = (
                    "generated_x"
                    if getattr(dataset, "enable_self_modal_generation", False)
                    else "missing_x"
                )

            sample_states.append(
                SampleState(
                    sample_key=key,
                    dataset_index=index,
                    im_file=str(im_file),
                    task_name=str(resolved_task_name),
                    modality_flag=modality_flag,
                )
            )
        return cls(sample_states)

    @property
    def all_indices(self) -> List[int]:
        """Return current dataset indices in canonical order."""
        return [state.dataset_index for state in self.sample_states]

    @property
    def num_samples(self) -> int:
        """Return total tracked samples."""
        return len(self.sample_states)

    def snapshot(self, config: AFSSConfig, selection: Dict[str, Any], epoch: int) -> AFSSStateSnapshot:
        """Create serializable runtime snapshot."""
        task_metric_keys = sorted(
            set(config.task_metrics)
            | {str(key) for state in self.sample_states for key in state.task_metrics.keys()}
        )
        return AFSSStateSnapshot(
            schema_version=AFSS_STATE_SCHEMA_VERSION,
            config=asdict(config),
            sample_states=[asdict(state) for state in self.sample_states],
            selection=selection,
            epoch=epoch,
            task_name=config.task_name,
            task_metrics=task_metric_keys,
            afss_tasks=dict(config.afss_tasks),
            afss_task_overrides=dict(config.afss_task_overrides),
        )

    def save_latest(self, save_dir: Path, config: AFSSConfig, selection: Dict[str, Any], epoch: int) -> Path:
        """Persist latest AFSS snapshot for future resume support."""
        state_dir = snapshot_state_dir(save_dir, config.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        latest_path = state_dir / "state_latest.pt"
        self._atomic_torch_save(asdict(self.snapshot(config, selection, epoch)), latest_path)
        return latest_path

    def update_train_usage(
        self,
        active_indices: List[int],
        epoch: int,
        forced_review_indices: Optional[List[int]] = None,
        forced_coverage_indices: Optional[List[int]] = None,
    ) -> None:
        """Update per-sample train usage counters after an epoch."""
        active_set = set(active_indices)
        forced_review_set = set(forced_review_indices or [])
        forced_coverage_set = set(forced_coverage_indices or [])
        for state in self.sample_states:
            if state.dataset_index in active_set:
                state.last_train_epoch = epoch
                state.selected_count += 1
                if state.dataset_index in forced_review_set:
                    state.forced_review_count += 1
                if state.dataset_index in forced_coverage_set:
                    state.forced_coverage_count += 1
            else:
                state.skipped_count += 1

    @staticmethod
    def _classify_score(score: float, modality_flag: str, config: AFSSConfig) -> str:
        """Classify AFSS difficulty from a resolved score value."""
        if modality_flag != "paired" and not config.allow_generated_x_to_easy:
            return "hard" if score < config.hard_threshold else "moderate"
        if score > config.easy_threshold:
            return "easy"
        if score < config.hard_threshold:
            return "hard"
        return "moderate"

    def update_scores(self, score_rows: List[Dict[str, Any]], epoch: int, config: AFSSConfig) -> None:
        """Write per-sample scoring results back into state store."""
        alpha = float(config.state_ema_alpha)
        for row in score_rows:
            sample_key = row["sample_key"]
            if sample_key not in self._state_by_key:
                raise KeyError(f"AFSS score row references unknown sample_key: {sample_key}")
            state = self._state_by_key[sample_key]
            if "task_name" not in row:
                raise ValueError(f"AFSS score row missing required field 'task_name' for sample_key={sample_key}")
            if "task_metrics" not in row:
                raise ValueError(f"AFSS score row missing required field 'task_metrics' for sample_key={sample_key}")
            row_task_name = str(row["task_name"])
            if row_task_name != config.task_name:
                raise ValueError(
                    "AFSS score row task mismatch: "
                    f"sample_key={sample_key}, row.task_name={row_task_name!r}, config.task_name={config.task_name!r}"
                )
            precision_op = float(row.get("precision_op", row["precision"]))
            recall_op = float(row.get("recall_op", row["recall"]))
            sufficiency_raw = float(row.get("sufficiency_raw", min(precision_op, recall_op)))
            sufficiency_ema = (
                sufficiency_raw
                if state.last_score_epoch < 0
                else alpha * sufficiency_raw + (1.0 - alpha) * state.sufficiency_ema
            )
            state.precision_op = precision_op
            state.recall_op = recall_op
            state.sufficiency_raw = sufficiency_raw
            state.sufficiency_ema = sufficiency_ema
            state.precision = precision_op
            state.recall = recall_op
            state.sufficiency = sufficiency_ema
            state.last_score_epoch = epoch
            state.task_name = row_task_name
            state.modality_flag = str(row.get("modality_flag", state.modality_flag))
            state.valid_for_afss = bool(row.get("valid_for_afss", state.valid_for_afss))
            state.task_metrics = self._normalize_task_metrics(row["task_metrics"], sample_key=sample_key)
            state.difficulty = self._classify_score(state.sufficiency_ema, state.modality_flag, config)

    @staticmethod
    def _normalize_task_metrics(task_metrics: Dict[str, Any], sample_key: str) -> Dict[str, float | int | str]:
        """Normalize per-sample task metrics into JSON-safe scalar values."""
        if not isinstance(task_metrics, dict):
            raise ValueError(
                f"AFSS task_metrics must be dict for sample_key={sample_key}, got {type(task_metrics)!r}"
            )
        normalized: Dict[str, float | int | str] = {}
        for key, value in task_metrics.items():
            metric_key = str(key)
            if isinstance(value, bool):
                normalized[metric_key] = int(value)
            elif isinstance(value, (int, float, str)):
                normalized[metric_key] = value
            else:
                normalized[metric_key] = str(value)
        return normalized

    @staticmethod
    def _summarize_values(values: List[float]) -> Dict[str, float]:
        """Summarize a sorted score vector."""
        values = sorted(values)

        def q(p: float) -> float:
            if not values:
                return 0.0
            idx = min(len(values) - 1, max(0, int((len(values) - 1) * p)))
            return values[idx]

        return {
            "q50": q(0.50),
            "q90": q(0.90),
            "q99": q(0.99),
            "min": values[0] if values else 0.0,
            "max": values[-1] if values else 0.0,
        }

    def score_distribution_summary(self, config: AFSSConfig) -> Dict[str, Any]:
        """Return summary statistics for the latest AFSS raw and EMA scores."""
        scored_states = [state for state in self.sample_states if state.valid_for_afss and state.last_score_epoch >= 0]
        raw_values = [float(state.sufficiency_raw) for state in scored_states]
        ema_values = [float(state.sufficiency_ema) for state in scored_states]
        raw_counts = {"easy": 0, "moderate": 0, "hard": 0}
        ema_counts = {"easy": 0, "moderate": 0, "hard": 0}
        for state in scored_states:
            raw_counts[self._classify_score(state.sufficiency_raw, state.modality_flag, config)] += 1
            ema_counts[self._classify_score(state.sufficiency_ema, state.modality_flag, config)] += 1
        return {
            "valid_count": len(scored_states),
            "raw": {
                "counts": raw_counts,
                **self._summarize_values(raw_values),
            },
            "ema": {
                "counts": ema_counts,
                **self._summarize_values(ema_values),
            },
        }

    def ensure_non_degenerate_scores(self, config: AFSSConfig, summary: Dict[str, Any]) -> None:
        """Fail fast if post-score AFSS raw buckets collapse into all-hard valid samples."""
        if not config.fail_on_all_hard_after_scoring:
            return
        valid_count = int(summary.get("valid_count", 0))
        raw_summary = summary.get("raw", {})
        hard_count = int(raw_summary.get("counts", {}).get("hard", 0))
        if valid_count > 0 and hard_count == valid_count:
            raise RuntimeError(
                "AFSS scored all valid samples as hard after refresh (raw scores): "
                f"q50={raw_summary.get('q50', 0.0):.4f}, q90={raw_summary.get('q90', 0.0):.4f}, "
                f"q99={raw_summary.get('q99', 0.0):.4f}, min={raw_summary.get('min', 0.0):.4f}, "
                f"max={raw_summary.get('max', 0.0):.4f}"
            )

    @staticmethod
    def _atomic_torch_save(data: dict, target_path: Path) -> None:
        """Write torch data to temp file then atomically rename."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(target_path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                torch.save(data, f)
            os.replace(tmp_path, str(target_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_json_save(data: dict, target_path: Path) -> None:
        """Write JSON data to temp file then atomically rename."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(target_path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(target_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def save_epoch_artifacts(
        self, save_dir: Path, config: AFSSConfig, selection: Dict[str, Any], epoch: int
    ) -> None:
        """Persist per-epoch state snapshot and selection JSON with atomic writes."""
        state_dir = snapshot_state_dir(save_dir, config.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_torch_save(
            asdict(self.snapshot(config, selection, epoch)), state_dir / f"state_epoch_{epoch}.pt"
        )
        self._atomic_json_save(selection, state_dir / f"selection_epoch_{epoch}.json")

    @classmethod
    def load_latest(cls, save_dir: Path, config: AFSSConfig) -> "AFSSStateStore":
        """Load persisted AFSS state snapshot."""
        latest_path = snapshot_state_dir(save_dir, config.state_dir) / "state_latest.pt"
        if not latest_path.exists():
            raise FileNotFoundError(f"AFSS state file not found: {latest_path}")
        payload = torch.load(latest_path, map_location="cpu")
        if payload.get("schema_version") != AFSS_STATE_SCHEMA_VERSION:
            raise ValueError(
                "AFSS state snapshot schema mismatch: "
                f"expected {AFSS_STATE_SCHEMA_VERSION}, got {payload.get('schema_version')!r}. "
                f"Please clear {latest_path.parent} before resuming."
            )
        if "task_name" not in payload:
            raise ValueError(
                "AFSS state snapshot is incompatible with current runtime: missing top-level 'task_name'. "
                f"Please clear {latest_path.parent} before resuming."
            )
        if "task_metrics" not in payload:
            raise ValueError(
                "AFSS state snapshot is incompatible with current runtime: missing top-level 'task_metrics'. "
                f"Please clear {latest_path.parent} before resuming."
            )
        stored_task_name = str(payload.get("task_name"))
        if stored_task_name != config.task_name:
            raise ValueError(
                f"AFSS state snapshot was created for task '{stored_task_name}', "
                f"but current config targets task '{config.task_name}'"
            )
        raw_states = payload.get("sample_states", [])
        required_fields = {
            "precision_op",
            "recall_op",
            "sufficiency_raw",
            "sufficiency_ema",
            "task_name",
            "task_metrics",
        }
        for idx, state in enumerate(raw_states):
            missing = sorted(required_fields - set(state))
            if missing:
                raise ValueError(
                    "AFSS state snapshot is incompatible with current runtime: "
                    f"sample_states[{idx}] missing fields {missing}. "
                    f"Please clear {latest_path.parent} before resuming."
                )
            if str(state.get("task_name")) != config.task_name:
                raise ValueError(
                    "AFSS state snapshot contains cross-task sample state: "
                    f"sample_states[{idx}].task_name={state.get('task_name')!r}, "
                    f"config.task_name={config.task_name!r}. Please clear {latest_path.parent} before resuming."
                )
            if not isinstance(state.get("task_metrics"), dict):
                raise ValueError(
                    "AFSS state snapshot is incompatible with current runtime: "
                    f"sample_states[{idx}].task_metrics must be dict, got {type(state.get('task_metrics'))!r}. "
                    f"Please clear {latest_path.parent} before resuming."
                )
        sample_states = [SampleState(**state) for state in raw_states]
        keys = [s.sample_key for s in sample_states]
        if len(keys) != len(set(keys)):
            dup_count = len(keys) - len(set(keys))
            raise ValueError(f"AFSS state contains {dup_count} duplicate sample_key entries, refusing to load")
        if not sample_states:
            raise ValueError(f"AFSS state file is empty or invalid: {latest_path}")
        metadata = {
            "task_name": payload.get("task_name"),
            "task_metrics": payload.get("task_metrics", []),
            "afss_tasks": payload.get("afss_tasks", {}),
            "afss_task_overrides": payload.get("afss_task_overrides", {}),
        }
        return cls(sample_states, snapshot_meta=metadata)

    def validate_against_dataset(self, dataset: Any) -> None:
        """Ensure persisted state still matches current dataset via set-equivalence."""
        current_files = getattr(dataset, "im_files", None)
        if not current_files:
            raise ValueError("Cannot validate AFSS state against dataset without dataset.im_files")
        current_keys = set(str(Path(im_file).resolve()) for im_file in current_files)
        state_keys = set(state.sample_key for state in self.sample_states)
        if len(current_keys) != len(state_keys):
            raise ValueError(
                "AFSS state sample count does not match current dataset: "
                f"{len(state_keys)} != {len(current_keys)}"
            )
        if current_keys != state_keys:
            missing = state_keys - current_keys
            extra = current_keys - state_keys
            parts = ["AFSS state sample keys do not match current dataset."]
            if missing:
                parts.append(f"  State has {len(missing)} keys not in dataset (e.g. {list(missing)[:3]}).")
            if extra:
                parts.append(f"  Dataset has {len(extra)} keys not in state (e.g. {list(extra)[:3]}).")
            raise ValueError("\n".join(parts))

    def validate_against_config(self, config: AFSSConfig) -> None:
        """Ensure persisted state metadata still matches current AFSS task configuration."""
        stored_task = self._snapshot_meta.get("task_name")
        if stored_task is None:
            raise ValueError("AFSS state snapshot missing task_name metadata for resume validation")
        if stored_task != config.task_name:
            raise ValueError(
                f"AFSS state snapshot was created for task '{stored_task}', "
                f"but current config targets task '{config.task_name}'"
            )
        if "task_metrics" not in self._snapshot_meta:
            raise ValueError("AFSS state snapshot missing task_metrics metadata for resume validation")
