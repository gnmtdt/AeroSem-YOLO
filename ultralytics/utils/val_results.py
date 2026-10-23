from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _extract_prefixed(mapping: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Extract key/value pairs from mapping whose key starts with prefix."""
    return {k: v for k, v in mapping.items() if isinstance(k, str) and k.startswith(prefix)}


@dataclass(frozen=True)
class COCOValResults:
    """COCO 扩展验证结果（扁平字典 + 可选保存目录）。"""

    metrics: Dict[str, Any]
    save_dir: Optional[Path] = None

    @classmethod
    def from_stats_dict(cls, stats: Dict[str, Any], *, save_dir: Optional[Path] = None) -> "COCOValResults":
        # 兼容多种前缀：
        # - metrics/coco/*（检测 COCO）
        # - metrics/coco_mask/*、metrics/coco_box/*（分割 COCO）
        return cls(metrics=_extract_prefixed(stats, "metrics/coco"), save_dir=save_dir)


@dataclass(frozen=True)
class MMValResults:
    """
    统一的验证结果容器：
    - standard: 标准验证器 metrics 对象（DetMetrics/SegmentMetrics/...）
    - coco: 可选 COCO 扩展结果
    - results_dict: 扁平 dict（用于日志/导出）
    """

    standard: Any
    coco: Optional[COCOValResults]
    results_dict: Dict[str, Any]
    save_dir: Optional[Path] = None

    def __getattr__(self, name: str) -> Any:
        # 代理到标准 metrics，保持用户旧代码兼容性；不吞异常（Fail-Fast）
        return getattr(self.standard, name)
