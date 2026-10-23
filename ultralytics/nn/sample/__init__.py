"""上下采样模块出口（延迟导入以避免循环依赖）。

本包汇集了自定义的下采样（Downsample）和上采样（Upsample）模块，
替代标准 Conv stride=2 下采样和 nn.Upsample 上采样。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    # NOTE: 这里使用延迟导入（PEP 562 __getattr__）以避免与 `ultralytics.nn.modules.*` 的循环依赖。
    # 下采样模块 (Downsample)
    "LAWDS": (".lawds", "LAWDS"),
    "EdgeLAWDS": (".edge_lawds", "EdgeLAWDS"),
    "FreqLAWDS": (".freq_lawds", "FreqLAWDS"),
    "HWD": (".hwd", "HWD"),
    "RouterLAWDS": (".router_lawds", "RouterLAWDS"),
    "V7DownSampling": (".v7down", "V7DownSampling"),
    # 上采样模块 (Upsample)
    "CARAFE": (".carafe", "CARAFE"),
    "DySample": (".dysample", "DySample"),
    "DSUB": (".dsub", "DSUB"),
    "Converse2D_Up": (".converse2d_up", "Converse2D_Up"),
    "EUCB_SC": (".eucb_sc", "EUCB_SC"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value  # 缓存，避免重复导入
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
