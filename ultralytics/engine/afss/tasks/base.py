# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import inspect
import io
import logging
from abc import ABC
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Protocol, Type

import torch
from torch.utils.data import DataLoader

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import de_parallel

from ..schema import AFSSConfig

_ADAPTER_REGISTRY: Dict[str, Type["BaseAFSSTaskAdapter"]] = {}


class AFSSTaskAdapter(Protocol):
    """Protocol for AFSS task adapters."""

    task_name: str

    def supports(self, trainer: Any, validator: Any) -> bool:
        """Return whether the adapter can operate with the given trainer/validator pair."""

    def build_scoring_dataloader(self, trainer: Any, validator: Any):
        """Build the scoring dataloader used by AFSS state refresh."""

    def score_epoch(self, trainer: Any, validator: Any, dataloader: Any) -> List[Dict[str, Any]]:
        """Run a full scoring epoch and return per-sample score rows."""

    def score_batch(self, trainer: Any, validator: Any, preds: Any, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert one preprocessed batch into task score rows."""

    def summarize_task_metrics(self, score_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Summarize task-specific metrics for logging."""


class BaseAFSSTaskAdapter(ABC):
    """Shared AFSS adapter base with common train-view scoring helpers."""

    task_name = ""
    required_validator_methods = ("build_dataset", "preprocess", "postprocess")

    def __init__(self, config: AFSSConfig):
        self.config = config
        self.task_overrides = config.get_task_overrides()

    @classmethod
    def register(cls, name: str):
        """Register an AFSS adapter class under a task name."""

        def decorator(adapter_cls: Type["BaseAFSSTaskAdapter"]) -> Type["BaseAFSSTaskAdapter"]:
            if name in _ADAPTER_REGISTRY:
                raise ValueError(f"AFSS task {name!r} is already registered")
            _ADAPTER_REGISTRY[name] = adapter_cls
            return adapter_cls

        return decorator

    def missing_validator_methods(self, validator: Any) -> List[str]:
        """Return the validator methods required by this adapter but missing from the validator."""
        return [name for name in self.required_validator_methods if not hasattr(validator, name)]

    def supports(self, trainer: Any, validator: Any) -> bool:
        """Return whether validator prerequisites are present."""
        return not self.missing_validator_methods(validator)

    @contextmanager
    def _quiet_dataset_build(self):
        """Reduce dataset construction chatter during AFSS scoring while preserving errors."""
        old_level = LOGGER.level
        sink = io.StringIO()
        try:
            LOGGER.setLevel(logging.WARNING)
            with redirect_stdout(sink):
                yield
        finally:
            LOGGER.setLevel(old_level)

    def build_scoring_dataloader(self, trainer: Any, validator: Any):
        """Build a train-view dataloader with val-mode transforms for AFSS scoring."""
        if getattr(validator, "data", None) is None:
            validator.data = trainer.data
        model = trainer.ema.ema or trainer.model
        model_unwrapped = de_parallel(model)
        if not getattr(validator, "stride", None):
            validator.stride = max(int(model_unwrapped.stride.max() if hasattr(model_unwrapped, "stride") else 0), 32)
        batch_size = getattr(trainer, "batch_size", None) or getattr(trainer.args, "batch", 16)
        orig_cache = getattr(validator.args, "cache", None)
        try:
            validator.args.cache = False
            with self._quiet_dataset_build():
                sig = inspect.signature(validator.build_dataset)
                kwargs: Dict[str, Any] = {"img_path": trainer.data["train"]}
                if "mode" in sig.parameters:
                    kwargs["mode"] = "val"
                if "batch" in sig.parameters:
                    kwargs["batch"] = batch_size
                dataset = validator.build_dataset(**kwargs)
        finally:
            validator.args.cache = orig_cache
        return DataLoader(
            dataset=dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=getattr(dataset, "collate_fn", None),
        )

    def _infer_modality_flag(self, dataloader: Any, im_file: str) -> str:
        """Infer AFSS modality flag from the scoring dataset."""
        dataset = getattr(dataloader, "dataset", None)
        if dataset is None or not hasattr(dataset, "_find_corresponding_x_image"):
            return "paired"
        try:
            x_path = Path(dataset._find_corresponding_x_image(im_file))
            if x_path.exists():
                return "paired"
            if bool(getattr(dataset, "enable_self_modal_generation", False)):
                return "generated_x"
            return "missing_x"
        except Exception:
            return "paired"

    def _finalize_score_row(self, trainer: Any, dataloader: Any, row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a task-specific row into the shared AFSS score schema."""
        if "im_file" not in row:
            raise KeyError(f"AFSS task={self.task_name} row is missing required field 'im_file'")
        if "task_metrics" not in row:
            raise KeyError(f"AFSS task={self.task_name} row is missing required field 'task_metrics'")
        row_task_name = str(row.get("task_name", self.task_name))
        if row_task_name != self.task_name:
            raise RuntimeError(
                f"AFSS task adapter mismatch: adapter={self.task_name!r}, row.task_name={row_task_name!r}"
            )

        im_file = str(row["im_file"])
        precision_op = float(row.get("precision_op", row.get("precision", 0.0)))
        recall_op = float(row.get("recall_op", row.get("recall", 0.0)))
        finalized = dict(row)
        finalized["im_file"] = im_file
        finalized["sample_key"] = str(finalized.get("sample_key", Path(im_file).resolve()))
        finalized["task_name"] = self.task_name
        finalized["modality_flag"] = str(finalized.get("modality_flag", self._infer_modality_flag(dataloader, im_file)))
        finalized["precision_op"] = precision_op
        finalized["recall_op"] = recall_op
        finalized["precision"] = float(finalized.get("precision", precision_op))
        finalized["recall"] = float(finalized.get("recall", recall_op))
        finalized["sufficiency_raw"] = float(finalized.get("sufficiency_raw", min(precision_op, recall_op)))
        finalized["task_metrics"] = dict(finalized["task_metrics"])
        finalized["valid_for_afss"] = bool(finalized.get("valid_for_afss", True))
        return finalized

    def score_batch(self, trainer: Any, validator: Any, preds: Any, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert one preprocessed batch into task score rows."""
        raise NotImplementedError(f"AFSS task={self.task_name} must implement score_batch()")

    def score_epoch(self, trainer: Any, validator: Any, dataloader: Any) -> List[Dict[str, Any]]:
        """Run a no-augment train-view scoring pass and return normalized rows."""
        missing = self.missing_validator_methods(validator)
        if missing:
            raise RuntimeError(
                f"AFSS task={self.task_name} requires validator methods {missing}, "
                f"got validator={validator.__class__.__name__}"
            )

        model = trainer.ema.ema or trainer.model
        validator.device = trainer.device
        validator.data = trainer.data
        orig_half = getattr(validator.args, "half", False)
        orig_conf = getattr(validator.args, "conf", None)
        orig_iou = getattr(validator.args, "iou", None)
        orig_iouv = validator.iouv.clone() if hasattr(validator, "iouv") else None
        orig_niou = getattr(validator, "niou", None)
        score_conf = float(getattr(trainer.args, "afss_score_conf", self.config.score_conf))
        score_iou = float(getattr(trainer.args, "afss_score_iou", self.config.score_iou))
        validator.args.half = trainer.device.type != "cpu" and trainer.amp
        if hasattr(validator.args, "conf"):
            validator.args.conf = score_conf
        if hasattr(validator.args, "iou"):
            validator.args.iou = score_iou
        if hasattr(validator, "iouv"):
            validator.iouv = torch.tensor([score_iou], device=validator.device)
            validator.niou = int(validator.iouv.numel())
        orig_training = model.training
        orig_dataloader = getattr(validator, "dataloader", None)
        model = model.half() if validator.args.half else model.float()
        validator.dataloader = dataloader

        rows: List[Dict[str, Any]] = []
        try:
            with torch.inference_mode():
                for batch in dataloader:
                    batch = validator.preprocess(batch)
                    preds = model(batch["img"])
                    preds = validator.postprocess(preds)
                    score_rows = self.score_batch(trainer, validator, preds, batch)
                    rows.extend(self._finalize_score_row(trainer, dataloader, row) for row in score_rows)
        finally:
            validator.dataloader = orig_dataloader
            validator.args.half = orig_half
            if orig_conf is not None:
                validator.args.conf = orig_conf
            if orig_iou is not None:
                validator.args.iou = orig_iou
            if orig_iouv is not None:
                validator.iouv = orig_iouv
            if orig_niou is not None:
                validator.niou = orig_niou
            model.float()
            if orig_training:
                model.train()
        return rows

    def summarize_task_metrics(self, score_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate numeric task_metrics for logging."""
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for row in score_rows:
            for key, value in row.get("task_metrics", {}).items():
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, (int, float)):
                    totals[key] = totals.get(key, 0.0) + float(value)
                    counts[key] = counts.get(key, 0) + 1
        return {key: totals[key] / counts[key] for key in totals if counts.get(key, 0) > 0}

    def _summarize_selected_task_metrics(
        self, score_rows: List[Dict[str, Any]], metric_keys: tuple[str, ...]
    ) -> Dict[str, float]:
        """Aggregate only selected numeric task metrics to avoid misleading summaries."""
        summary: Dict[str, float] = {}
        for key in metric_keys:
            values: List[float] = []
            for row in score_rows:
                value = row.get("task_metrics", {}).get(key)
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if values:
                summary[key] = sum(values) / len(values)
        return summary


def get_registered_afss_task_adapter(name: str) -> Type[BaseAFSSTaskAdapter]:
    """Return the registered adapter class for a task name."""
    try:
        return _ADAPTER_REGISTRY[name]
    except KeyError as exc:
        raise RuntimeError(f"AFSS does not support task={name} yet") from exc
