from __future__ import annotations

import os
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Iterator, Mapping

import torch


@dataclass(slots=True)
class MetricStat:
    """单指标聚合统计 (total/count/avg/max/min)。"""

    total: float = 0.0
    count: int = 0
    max: float = 0.0
    min: float = float("inf")

    def add(self, elapsed: float) -> None:
        v = float(elapsed)
        self.total += v
        self.count += 1
        if v > self.max:
            self.max = v
        if v < self.min:
            self.min = v

    def merge(self, other: "MetricStat") -> None:
        if other.count == 0:
            return
        self.total += other.total
        self.count += other.count
        if other.max > self.max:
            self.max = other.max
        if other.min < self.min:
            self.min = other.min

    def summarize(self, key: str) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {
            f"{key}.total": self.total,
            f"{key}.count": float(self.count),
            f"{key}.avg": self.total / self.count,
            f"{key}.max": self.max,
            f"{key}.min": self.min,
        }


@dataclass(slots=True)
class EpochAccumulator:
    """聚合单个 epoch 的阶段耗时。metrics 统一使用 MetricStat 存储。"""

    epoch: int
    metrics: dict[str, MetricStat] = field(default_factory=dict)

    def _get_or_create(self, key: str) -> MetricStat:
        stat = self.metrics.get(key)
        if stat is None:
            stat = MetricStat()
            self.metrics[key] = stat
        return stat

    def add(self, key: str, elapsed: float) -> None:
        self._get_or_create(key).add(float(elapsed))

    def add_aggregate(self, key: str, stat: MetricStat) -> None:
        self._get_or_create(key).merge(stat)

    def add_many(self, values: Mapping[str, float], *, prefix: str = "") -> None:
        for key, value in values.items():
            self.add(f"{prefix}{key}" if prefix else key, value)

    def summarize(self) -> dict[str, float]:
        summary: dict[str, float] = {}
        for key, stat in self.metrics.items():
            summary.update(stat.summarize(key))
        return summary


class TimeDebug:
    """训练计时内核。

    设计要点：
    1. `timedebug=False` 时所有公开方法立即返回，避免无意义文件 IO。
    2. GPU 阶段使用 `torch.cuda.synchronize(device)` + `time.perf_counter()`，保证 wall-clock 口径。
    3. 日志格式使用稳定的 `key=value` 单行事件，便于人工阅读和脚本解析。
    4. 只负责写 `timedebug.log`，不接管任何 CSV 或 metrics 输出。
    5. IO 异常时单次告警后熔断，不中断训练主流程。
    """

    __slots__ = ("enabled", "device", "_fh", "_t0", "_current_epoch", "_write_failed")

    def __init__(self, enabled: bool = False, device: torch.device | None = None) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self._fh = None
        self._t0 = 0.0
        self._current_epoch: EpochAccumulator | None = None
        self._write_failed = False

    def open(self, save_dir: str | os.PathLike[str], *, meta: Mapping[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._fh = open(path / "timedebug.log", "w", encoding="utf-8", buffering=1)
        self._t0 = time.perf_counter()
        self.log_event("session_start", **(dict(meta or {}) | {"ts": time.strftime("%Y-%m-%d %H:%M:%S")}))

    def close(self, **meta: Any) -> None:
        if not self._fh:
            return
        self.log_event("session_end", total=self.now(sync_cuda=False), **meta)
        self._fh.close()
        self._fh = None

    def now(self, *, sync_cuda: bool) -> float:
        if sync_cuda and self.device is not None and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - self._t0

    @contextmanager
    def span(self, event: str, *, sync_cuda: bool, **meta: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = self.now(sync_cuda=sync_cuda)
        try:
            yield
        finally:
            self.log_event(event, elapsed=self.now(sync_cuda=sync_cuda) - t0, **meta)

    def log_event(self, event: str, **meta: Any) -> None:
        if not self.enabled or not self._fh:
            return
        ordered = " ".join(f"{key}={self._format_value(value)}" for key, value in meta.items())
        self._write_line(f"[{self.now(sync_cuda=False):.6f}] event={event} {ordered}")

    def _write_line(self, line: str) -> None:
        """写入一行日志，IO 异常时单次告警后熔断，不中断训练主流程。"""
        if self._write_failed or not self._fh:
            return
        try:
            self._fh.write(line + "\n")
        except OSError:
            self._write_failed = True
            self._fh = None
            import sys
            print("[timedebug] log write failed, timing disabled for remainder of session", file=sys.stderr)

    def begin_epoch(self, epoch: int) -> None:
        if not self.enabled:
            return
        self._current_epoch = EpochAccumulator(epoch=epoch)
        self.log_event("epoch_start", epoch=epoch)

    def add_epoch_metric(self, key: str, elapsed: float) -> None:
        if self._current_epoch is not None:
            self._current_epoch.add(key, elapsed)

    def add_epoch_metrics(self, values: Mapping[str, float], *, prefix: str = "") -> None:
        if self._current_epoch is not None:
            self._current_epoch.add_many(values, prefix=prefix)

    def add_epoch_aggregate(self, key: str, stat: MetricStat) -> None:
        """注入预聚合的 MetricStat 到当前 epoch 累计器。"""
        if self._current_epoch is not None:
            self._current_epoch.add_aggregate(key, stat)

    def end_epoch(self, *, total: float, **meta: Any) -> None:
        if not self.enabled or self._current_epoch is None:
            return
        payload = {"epoch": self._current_epoch.epoch, "total": total}
        payload.update(self._current_epoch.summarize())
        payload.update(meta)
        self.log_event("epoch_summary", **payload)
        self._current_epoch = None

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)


# ── worker-side timing helpers (imported by ultralytics.data.*) ──────────────


def sample_timedebug_enabled(sample: Mapping[str, Any]) -> bool:
    return bool(sample.get("_timedebug_enabled", False))


def add_sample_timing(sample: dict[str, Any], key: str, elapsed: float) -> None:
    """向 sample 的 _timedebug_worker_stats 写入一次 timing 观测。"""
    if not sample_timedebug_enabled(sample):
        return
    stats = sample.setdefault("_timedebug_worker_stats", {})
    stat = stats.get(key)
    if stat is None:
        stat = MetricStat()
        stats[key] = stat
    stat.add(float(elapsed))


def add_worker_timing(stats: dict[str, Any] | None, key: str, elapsed: float) -> None:
    """向 worker stats dict 写入一次 timing 观测。stats 为 None 时静默跳过。"""
    if stats is None:
        return
    stat = stats.get(key)
    if stat is None:
        stat = MetricStat()
        stats[key] = stat
    stat.add(float(elapsed))


def merge_worker_stats(items: list[dict[str, MetricStat]]) -> dict[str, MetricStat]:
    """合并多个 worker stats dict，每个 key 的 MetricStat 做 merge。"""
    merged: dict[str, MetricStat] = {}
    for item in items:
        for key, stat in item.items():
            if key not in merged:
                merged[key] = MetricStat()
            merged[key].merge(stat)
    return merged


class YOLOVODebugLogger:
    """YOLOVO 管线诊断日志器。

    仅由 ``yolovo_debug`` 控制，将诊断输出写入 ``{save_dir}/yolovo_debug.log``，
    不污染控制台输出。设计要点：

    1. ``yolovo_debug=False`` 时所有公开方法立即返回，零开销。
    2. 行缓冲（buffering=1），实时可读。
    3. IO 异常时单次告警后熔断，不中断训练主流程。
    4. 格式：``[epoch,batch] event key=value ...``。
    """

    __slots__ = ("enabled", "_fh", "_write_failed")

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._fh = None
        self._write_failed = False

    def __bool__(self) -> bool:
        return bool(self.enabled and self._fh is not None)

    def open(self, save_dir: str | os.PathLike[str]) -> None:
        if not self.enabled or self._fh is not None:
            return
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._fh = open(path / "yolovo_debug.log", "w", encoding="utf-8", buffering=1)
        self.log(
            "",
            "session_start",
            ts=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def close(self) -> None:
        if not self._fh:
            return
        self.log("", "session_end")
        self._fh.close()
        self._fh = None

    def __del__(self):
        """Last-resort cleanup if training crashes before on_train_end."""
        fh = getattr(self, "_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            self._fh = None

    def log(self, tag: str, event: str, **meta: Any) -> None:
        """写一行诊断日志。

        Args:
            tag: epoch/batch 标签，如 ``"3,42"`` 表示 epoch=3 batch=42。
            event: 事件名，如 ``"preprocess_batch"``。
            **meta: key=value 键值对。
        """
        if not self.enabled or not self._fh:
            return
        prefix = f"[{tag}] " if tag else ""
        ordered = " ".join(
            f"{key}={self._format_value(value)}" for key, value in meta.items()
        )
        ts = time.strftime("%H:%M:%S")
        self._write_line(f"{ts} {prefix}event={event} {ordered}")

    def _write_line(self, line: str) -> None:
        if self._write_failed or not self._fh:
            return
        try:
            self._fh.write(line + "\n")
        except OSError:
            self._write_failed = True
            self._fh = None
            import sys

            print(
                "[yolovo-debug] log write failed, debug logging disabled for remainder of session",
                file=sys.stderr,
            )

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return f"{float(value.detach().item()):.6f}"
            return str(tuple(value.shape))
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        if isinstance(value, str) and any(ch.isspace() for ch in value):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
