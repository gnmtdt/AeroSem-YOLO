# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

AFSS_SUPPORTED_TASKS = ("detect", "obb", "pose", "segment", "classify")  # semseg disabled, not yet released
DEFAULT_AFSS_TASK_FLAGS = {
    "detect": True,
    "obb": False,
    "pose": False,
    "segment": False,
    "classify": False,
}


@dataclass(slots=True)
class AFSSConfig:
    """Configuration parsed from training args for AFSS runtime bootstrap."""

    enabled: bool = False
    warmup_epochs: int = 10
    state_update_interval: int = 5
    easy_threshold: float = 0.85
    hard_threshold: float = 0.55
    easy_ratio: float = 0.02
    easy_review_gap: int = 10
    easy_forced_review_cap_ratio: float = 0.5
    moderate_ratio: float = 0.40
    moderate_cover_gap: int = 3
    score_on_train_eval: bool = True
    score_conf: float = 0.25
    score_iou: float = 0.7
    state_ema_alpha: float = 0.3
    fail_on_all_hard_after_scoring: bool = True
    allow_generated_x_to_easy: bool = False
    state_dir: str = "afss"
    seed: int = 0
    task_name: str = "detect"
    task_metrics: List[str] = field(default_factory=lambda: ["precision", "recall"])
    afss_tasks: Dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_AFSS_TASK_FLAGS))
    afss_task_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_args(cls, args: Any) -> "AFSSConfig":
        """Create AFSS config from trainer args / cfg namespace."""
        normalized_tasks = cls._normalize_task_flags(getattr(args, "afss_tasks", None))
        trainer_task = getattr(args, "task", None)
        raw_task_name = trainer_task if trainer_task is not None else getattr(args, "afss_task_name", "detect")
        task_name = str(raw_task_name) if raw_task_name is not None else "detect"
        if task_name not in AFSS_SUPPORTED_TASKS:
            raise ValueError(f"afss task '{task_name}' is not supported, expected one of {AFSS_SUPPORTED_TASKS}")

        explicit_afss_task_name = getattr(args, "afss_task_name", None)
        if (
            trainer_task is not None
            and explicit_afss_task_name is not None
            and str(explicit_afss_task_name) != str(trainer_task)
        ):
            raise ValueError(
                "afss_task_name must stay consistent with trainer.args.task. "
                f"Got afss_task_name={explicit_afss_task_name!r}, task={trainer_task!r}"
            )
        if trainer_task is None and explicit_afss_task_name is not None:
            task_name = str(explicit_afss_task_name)

        cfg = cls(
            enabled=bool(getattr(args, "afss_enabled", False)),
            warmup_epochs=int(getattr(args, "afss_warmup_epochs", 10)),
            state_update_interval=int(getattr(args, "afss_state_update_interval", 5)),
            easy_threshold=float(getattr(args, "afss_easy_threshold", 0.85)),
            hard_threshold=float(getattr(args, "afss_hard_threshold", 0.55)),
            easy_ratio=float(getattr(args, "afss_easy_ratio", 0.02)),
            easy_review_gap=int(getattr(args, "afss_easy_review_gap", 10)),
            easy_forced_review_cap_ratio=float(getattr(args, "afss_easy_forced_review_cap_ratio", 0.5)),
            moderate_ratio=float(getattr(args, "afss_moderate_ratio", 0.40)),
            moderate_cover_gap=int(getattr(args, "afss_moderate_cover_gap", 3)),
            score_on_train_eval=bool(getattr(args, "afss_score_on_train_eval", True)),
            score_conf=float(getattr(args, "afss_score_conf", 0.25)),
            score_iou=float(getattr(args, "afss_score_iou", 0.7)),
            state_ema_alpha=float(getattr(args, "afss_state_ema_alpha", 0.3)),
            fail_on_all_hard_after_scoring=bool(getattr(args, "afss_fail_on_all_hard_after_scoring", True)),
            allow_generated_x_to_easy=bool(getattr(args, "afss_allow_generated_x_to_easy", False)),
            state_dir=str(getattr(args, "afss_state_dir", "afss")),
            seed=int(getattr(args, "afss_seed", 0)),
            task_name=task_name,
            task_metrics=cls._normalize_task_metrics(getattr(args, "afss_task_metrics", None)),
            afss_tasks=normalized_tasks,
            afss_task_overrides=cls._normalize_task_overrides(getattr(args, "afss_task_overrides", None)),
        )
        cfg.validate()
        cfg.apply_task_overrides()
        return cfg

    @staticmethod
    def _normalize_task_metrics(value: Any) -> List[str]:
        if value is None:
            return ["precision", "recall"]
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return [str(item) for item in value if item is not None]
        return [str(value)]

    @staticmethod
    def _normalize_task_flags(raw: Any) -> Dict[str, bool]:
        task_flags = dict(DEFAULT_AFSS_TASK_FLAGS)
        if raw is None:
            return task_flags
        raw_dict = namespace_to_dict(raw)
        unknown = sorted(set(raw_dict.keys()) - set(AFSS_SUPPORTED_TASKS))
        if unknown:
            raise ValueError(f"afss_tasks contains unknown task keys: {unknown}")
        for task_name, enabled in raw_dict.items():
            task_flags[str(task_name)] = bool(enabled)
        return task_flags

    @staticmethod
    def _normalize_task_overrides(raw: Any) -> Dict[str, Dict[str, Any]]:
        if raw is None:
            return {}
        raw_dict = namespace_to_dict(raw)
        unknown = sorted(set(raw_dict.keys()) - set(AFSS_SUPPORTED_TASKS))
        if unknown:
            raise ValueError(f"afss_task_overrides contains unknown task keys: {unknown}")
        overrides: Dict[str, Dict[str, Any]] = {}
        for task_name, override in raw_dict.items():
            if override is None:
                overrides[str(task_name)] = {}
                continue
            if isinstance(override, SimpleNamespace):
                overrides[str(task_name)] = vars(override).copy()
                continue
            if isinstance(override, dict):
                overrides[str(task_name)] = dict(override)
                continue
            raise ValueError(
                "afss_task_overrides entries must be dict-like objects, "
                f"got task={task_name!r}, type={type(override)!r}"
            )
        return overrides

    @property
    def tasks(self) -> Dict[str, bool]:
        """Alias for multi-task flags."""
        return self.afss_tasks

    @property
    def task_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Alias for multi-task overrides."""
        return self.afss_task_overrides

    def is_task_enabled(self, task_name: str | None = None) -> bool:
        """Return whether AFSS is enabled for a specific task."""
        resolved_task = task_name or self.task_name
        return bool(self.afss_tasks.get(resolved_task, False))

    @property
    def enabled_effective(self) -> bool:
        """Effective AFSS enable state considering both global switch and per-task switching."""
        return self.enabled and self.is_task_enabled(self.task_name)

    def get_task_overrides(self, task_name: str | None = None) -> Dict[str, Any]:
        """Return a defensive copy of task-specific overrides."""
        resolved_task = task_name or self.task_name
        return dict(self.afss_task_overrides.get(resolved_task, {}))

    def validate(self) -> None:
        """Validate user-facing AFSS config and fail fast on invalid settings."""
        if self.warmup_epochs < 0:
            raise ValueError(f"afss_warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if self.state_update_interval <= 0:
            raise ValueError(f"afss_state_update_interval must be > 0, got {self.state_update_interval}")
        if not 0.0 <= self.hard_threshold <= 1.0:
            raise ValueError(f"afss_hard_threshold must be in [0, 1], got {self.hard_threshold}")
        if not 0.0 <= self.easy_threshold <= 1.0:
            raise ValueError(f"afss_easy_threshold must be in [0, 1], got {self.easy_threshold}")
        if self.hard_threshold > self.easy_threshold:
            raise ValueError(
                "afss_hard_threshold must be <= afss_easy_threshold, "
                f"got {self.hard_threshold} > {self.easy_threshold}"
            )
        if not 0.0 <= self.easy_ratio <= 1.0:
            raise ValueError(f"afss_easy_ratio must be in [0, 1], got {self.easy_ratio}")
        if not 0.0 <= self.moderate_ratio <= 1.0:
            raise ValueError(f"afss_moderate_ratio must be in [0, 1], got {self.moderate_ratio}")
        if not 0.0 <= self.easy_forced_review_cap_ratio <= 1.0:
            raise ValueError(
                "afss_easy_forced_review_cap_ratio must be in [0, 1], "
                f"got {self.easy_forced_review_cap_ratio}"
            )
        if not 0.0 <= self.score_conf <= 1.0:
            raise ValueError(f"afss_score_conf must be in [0, 1], got {self.score_conf}")
        if not 0.0 <= self.score_iou <= 1.0:
            raise ValueError(f"afss_score_iou must be in [0, 1], got {self.score_iou}")
        if not 0.0 < self.state_ema_alpha <= 1.0:
            raise ValueError(f"afss_state_ema_alpha must be in (0, 1], got {self.state_ema_alpha}")
        if self.easy_review_gap < 0:
            raise ValueError(f"afss_easy_review_gap must be >= 0, got {self.easy_review_gap}")
        if self.moderate_cover_gap < 0:
            raise ValueError(f"afss_moderate_cover_gap must be >= 0, got {self.moderate_cover_gap}")
        if not self.state_dir:
            raise ValueError("afss_state_dir must not be empty")
        if not self.task_name:
            raise ValueError("afss_task_name must not be empty")
        if self.task_name not in AFSS_SUPPORTED_TASKS:
            raise ValueError(f"afss_task_name must be one of {AFSS_SUPPORTED_TASKS}, got {self.task_name!r}")
        if not self.task_metrics:
            raise ValueError("afss_task_metrics must specify at least one metric")
        if not all(isinstance(metric, str) and metric for metric in self.task_metrics):
            raise ValueError("afss_task_metrics must be a list of non-empty strings")
        if not isinstance(self.afss_tasks, dict):
            raise ValueError("afss_tasks must be a dict mapping task names to configs")
        if not isinstance(self.afss_task_overrides, dict):
            raise ValueError("afss_task_overrides must be a dict mapping task names to overrides")
        unknown_tasks = sorted(set(self.afss_tasks.keys()) - set(AFSS_SUPPORTED_TASKS))
        if unknown_tasks:
            raise ValueError(f"afss_tasks contains unknown task keys: {unknown_tasks}")
        unknown_override_tasks = sorted(set(self.afss_task_overrides.keys()) - set(AFSS_SUPPORTED_TASKS))
        if unknown_override_tasks:
            raise ValueError(f"afss_task_overrides contains unknown task keys: {unknown_override_tasks}")
        for task_name, enabled in self.afss_tasks.items():
            if not isinstance(enabled, bool):
                raise ValueError(
                    "afss_tasks values must be bool, "
                    f"got task={task_name!r}, value={enabled!r}, type={type(enabled)!r}"
                )
        for task_name, override in self.afss_task_overrides.items():
            if not isinstance(override, dict):
                raise ValueError(
                    "afss_task_overrides values must be dict-like, "
                    f"got task={task_name!r}, type={type(override)!r}"
                )

    def apply_task_overrides(self) -> None:
        """Apply active task overrides to top-level config fields when names match."""
        overrideable_fields = {
            "warmup_epochs",
            "state_update_interval",
            "easy_threshold",
            "hard_threshold",
            "easy_ratio",
            "easy_review_gap",
            "easy_forced_review_cap_ratio",
            "moderate_ratio",
            "moderate_cover_gap",
            "score_on_train_eval",
            "score_conf",
            "score_iou",
            "state_ema_alpha",
            "fail_on_all_hard_after_scoring",
            "allow_generated_x_to_easy",
            "state_dir",
            "seed",
        }
        for key, value in self.get_task_overrides().items():
            if key in overrideable_fields:
                expected_type = type(getattr(self, key, None))
                if expected_type is not None and not isinstance(value, expected_type):
                    try:
                        value = expected_type(value)
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"AFSS task override '{key}' expects {expected_type.__name__}, "
                            f"got {type(value).__name__}: {value!r}"
                        ) from e
                setattr(self, key, value)
        self.validate()


@dataclass(slots=True)
class SampleState:
    """Canonical AFSS state for a single training sample."""

    sample_key: str
    dataset_index: int
    im_file: str
    task_name: str = "detect"
    last_train_epoch: int = -1
    last_score_epoch: int = -1
    precision: float = 0.0
    recall: float = 0.0
    sufficiency: float = 0.0
    precision_op: float = 0.0
    recall_op: float = 0.0
    sufficiency_raw: float = 0.0
    sufficiency_ema: float = 0.0
    difficulty: str = "unknown"
    selected_count: int = 0
    skipped_count: int = 0
    forced_review_count: int = 0
    forced_coverage_count: int = 0
    modality_flag: str = "paired"
    valid_for_afss: bool = True
    task_metrics: Dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True)
class EpochSelectionPlan:
    """Structured epoch selection result used by runtime and samplers."""

    epoch: int
    active_indices: List[int]
    total_samples: int
    reason: str
    counts: Dict[str, int] = field(default_factory=dict)
    forced_review_indices: List[int] = field(default_factory=list)
    forced_coverage_indices: List[int] = field(default_factory=list)

    @property
    def active_ratio(self) -> float:
        """Return active sample ratio for logging."""
        if self.total_samples <= 0:
            return 0.0
        return len(self.active_indices) / float(self.total_samples)


@dataclass(slots=True)
class AFSSStateSnapshot:
    """Serializable snapshot of AFSS runtime state."""

    schema_version: str
    config: Dict[str, Any]
    sample_states: List[Dict[str, Any]]
    selection: Dict[str, Any]
    epoch: int = -1
    task_name: str = ""
    task_metrics: List[str] = field(default_factory=list)
    afss_tasks: Dict[str, Any] = field(default_factory=dict)
    afss_task_overrides: Dict[str, Any] = field(default_factory=dict)


AFSS_STATE_SCHEMA_VERSION = "4.0.0"


def namespace_to_dict(ns: Any) -> Dict[str, Any]:
    """Return a stable dict view for a namespace-like object."""
    if ns is None:
        return {}
    if isinstance(ns, Mapping):
        return {str(k): _normalize_namespace_value(v) for k, v in ns.items()}
    if isinstance(ns, SimpleNamespace):
        return {str(k): _normalize_namespace_value(v) for k, v in vars(ns).items()}
    if hasattr(ns, "__dict__"):
        return {str(k): _normalize_namespace_value(v) for k, v in vars(ns).items()}
    raise TypeError(f"Unsupported namespace type for serialization: {type(ns)!r}")


def _normalize_namespace_value(value: Any) -> Any:
    """Recursively convert namespace-like values to plain Python containers."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _normalize_namespace_value(v) for k, v in value.items()}
    if isinstance(value, SimpleNamespace) or hasattr(value, "__dict__"):
        return namespace_to_dict(value)
    if isinstance(value, list):
        return [_normalize_namespace_value(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_namespace_value(v) for v in value]
    return value


def sample_states_to_payload(states: Iterable[SampleState]) -> List[Dict[str, Any]]:
    """Serialize sample states to plain dictionaries."""
    return [asdict(state) for state in states]


def snapshot_state_dir(save_dir: Path, relative_dir: str) -> Path:
    """Resolve AFSS state directory under trainer save_dir."""
    return save_dir / relative_dir
