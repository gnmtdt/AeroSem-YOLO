# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

from ..schema import AFSSConfig, AFSS_SUPPORTED_TASKS
from .base import AFSSTaskAdapter, BaseAFSSTaskAdapter, get_registered_afss_task_adapter

_ADAPTER_IMPORTS: Dict[str, Tuple[str, str]] = {
    "detect": ("ultralytics.engine.afss.tasks.detect", "DetectAFSSTaskAdapter"),
    "obb": ("ultralytics.engine.afss.tasks.obb", "OBBAFSSTaskAdapter"),
    "pose": ("ultralytics.engine.afss.tasks.pose", "PoseAFSSTaskAdapter"),
    "segment": ("ultralytics.engine.afss.tasks.segment", "SegmentAFSSTaskAdapter"),
    "classify": ("ultralytics.engine.afss.tasks.classify", "ClassificationAFSSTaskAdapter"),
}


def _infer_task_name(task_name: str | None = None, trainer: Any = None, validator: Any = None) -> str:
    """Infer AFSS task name from explicit input or trainer/validator context."""
    if task_name:
        resolved = str(task_name)
    else:
        resolved = str(
            getattr(getattr(trainer, "args", None), "task", None)
            or getattr(getattr(validator, "args", None), "task", None)
            or ""
        )
    if not resolved:
        raise RuntimeError("AFSS could not infer task name from trainer/validator context")
    if resolved not in AFSS_SUPPORTED_TASKS:
        raise RuntimeError(f"AFSS does not support task={resolved} yet")
    return resolved


def _ensure_task_module_loaded(task_name: str) -> None:
    """Import the task adapter module so registry side effects are applied exactly once."""
    module_name, _ = _ADAPTER_IMPORTS[task_name]
    try:
        import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"AFSS does not support task={task_name} yet") from exc


def get_afss_task_adapter(
    task_name: str | None = None,
    config: AFSSConfig | None = None,
    trainer: Any = None,
    validator: Any = None,
) -> BaseAFSSTaskAdapter:
    """Resolve and instantiate an AFSS task adapter."""
    resolved_task = _infer_task_name(task_name=task_name, trainer=trainer, validator=validator)
    cfg = config or AFSSConfig(task_name=resolved_task)
    if cfg.task_name != resolved_task:
        raise RuntimeError(
            f"AFSS adapter resolution mismatch: requested task={resolved_task!r}, config.task_name={cfg.task_name!r}"
        )
    _ensure_task_module_loaded(resolved_task)
    adapter = get_registered_afss_task_adapter(resolved_task)(cfg)
    if adapter.task_name != resolved_task:
        raise RuntimeError(
            f"AFSS adapter task mismatch: resolved task={resolved_task!r}, adapter.task_name={adapter.task_name!r}"
        )
    if trainer is not None and validator is not None and not adapter.supports(trainer, validator):
        missing = []
        if hasattr(adapter, "missing_validator_methods"):
            missing = adapter.missing_validator_methods(validator)
        raise RuntimeError(
            f"AFSS task={resolved_task} is not compatible with validator={validator.__class__.__name__}. "
            f"Missing validator methods: {missing}"
        )
    return adapter


def resolve_afss_task_adapter(
    task_name: str | None = None,
    config: AFSSConfig | None = None,
    trainer: Any = None,
    validator: Any = None,
) -> BaseAFSSTaskAdapter:
    """Alias kept for clearer runtime call sites."""
    return get_afss_task_adapter(task_name=task_name, config=config, trainer=trainer, validator=validator)


__all__ = (
    "AFSSTaskAdapter",
    "BaseAFSSTaskAdapter",
    "get_afss_task_adapter",
    "resolve_afss_task_adapter",
)
