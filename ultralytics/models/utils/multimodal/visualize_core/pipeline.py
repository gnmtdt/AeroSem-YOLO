"""Core visualization pipeline (family-agnostic, Fail-Fast)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .exceptions import (
    DeviceMismatchError,
    MethodNotRegisteredError,
    VisualizationError,
)
from .input_resolver import InputResolver
from .layer_resolver import LayerResolver
from .registry import REGISTRY
from .router_adapter import RouterAdapter
from .cache import Cache
from .saver import Saver
from ultralytics.utils import LOGGER
from .types import PipelineContext, CoreVisualizationResult


class Pipeline:
    """
    Family-agnostic visualization pipeline.

    Responsibilities:
    - 解析并校验输入/模态（InputResolver）
    - 校验设备一致性（不自动迁移）
    - 层索引校验与命名映射（LayerResolver）
    - 方法插件查找与执行（Registry）
    - 输出目录解析（按 family 分域），不负责保存实现（由插件决定）
    """

    def __init__(self, model: Any, family: str) -> None:
        self.model = model
        self.family = str(family)
        self._cache = Cache()

    def _ensure_out_dir(
        self,
        *,
        project: Optional[str],
        name: Optional[str],
        out_dir: Optional[str],  # deprecated
    ) -> Path:
        """
        解析输出目录：优先使用 project/name，保留 out_dir 兼容（已废弃）。

        规则：
        - 若提供 out_dir（废弃），直接使用，并打印一次警告。
        - 若提供 project 与 name：输出目录为 project/name（不做自动编号）。
        - 若仅提供 project：在 project 下自动编号 exp, exp2, ...。
        - 若仅提供 name：在默认 runs/visualize/<family>/name 下，不编号。
        - 若均未提供：在默认 runs/visualize/<family> 下自动编号。
        """
        # 兼容性：优先处理已废弃的 out_dir
        if out_dir:
            try:
                LOGGER.warning("[visualize_core] 参数 out_dir 已废弃，请使用 project/name 代替。")
            except Exception:
                pass
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            return p

        # 新语义：project/name 显式输出路径
        if project and name:
            p = Path(project) / str(name)
            p.mkdir(parents=True, exist_ok=True)
            return p

        # 部分提供：仅 project → 在 project 下自动编号
        if project and not name:
            base = Path(project)
            base.mkdir(parents=True, exist_ok=True)
            i = 1
            while True:
                nm = "exp" if i == 1 else f"exp{i}"
                p = base / nm
                if not p.exists():
                    p.mkdir(parents=True, exist_ok=True)
                    return p
                i += 1

        # 部分提供：仅 name → 放在默认 family 域下，使用该 name
        if name and not project:
            base = Path("runs") / "visualize" / self.family
            base.mkdir(parents=True, exist_ok=True)
            p = base / str(name)
            p.mkdir(parents=True, exist_ok=True)
            return p

        # 默认：自动编号到 runs/visualize/<family>/exp*
        base = Path("runs") / "visualize" / self.family
        base.mkdir(parents=True, exist_ok=True)
        i = 1
        while True:
            nm = "exp" if i == 1 else f"exp{i}"
            p = base / nm
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                return p
            i += 1

    def run(
        self,
        *,
        rgb_source: Optional[Any],
        x_source: Optional[Any],
        method: str,
        layers: List[int],
        modality: Optional[str],
        save: bool,
        overlay: Optional[str] = None,
        # 新：使用 project/name 控制保存目录
        project: Optional[str] = None,
        name: Optional[str] = None,
        # 旧：out_dir 已废弃，仅保留兼容
        out_dir: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> List[CoreVisualizationResult]:
        # 设备一致性（Fail‑Fast）
        try:
            current_device = next(self.model.parameters()).device.type
        except Exception:
            current_device = "cuda" if hasattr(self.model, "to") else "cpu"
        if device is not None and str(device) != current_device:
            raise DeviceMismatchError(current=current_device, requested=str(device))

        # 输入与模态解析（Fail‑Fast）
        inputs = InputResolver.resolve(rgb_source, x_source, modality)

        # 层解析（Fail‑Fast）
        valid_layers = LayerResolver.validate_indices(self.model, layers)
        layer_names = LayerResolver.to_names(valid_layers)

        # 输出目录
        out_path = self._ensure_out_dir(project=project, name=name, out_dir=out_dir)

        # 方法插件查找（无降级）
        method_key = {
            "heat": "heat",
            "heatmap": "heat",
            "feature": "feature",
            "feature_map": "feature",
        }.get(str(method).lower(), None)

        if not method_key:
            raise MethodNotRegisteredError(str(method), REGISTRY.list())

        plugin = REGISTRY.get(method_key)

        # Router coordination (best-effort; refined in Step 8)
        router = RouterAdapter(self.model)
        if router.available:
            router.set_runtime_params(modality)
            try:
                router.log_summary(prefix=f"vis[{self.family}]")
            except Exception:
                pass
        x_modality_name = router.get_x_modality_name() if router.available else "unknown"

        # 目录/批量模式：当 inputs 为列表时逐样本处理，不使用全局缓存，结果落在带 img_key 的子目录
        if isinstance(inputs, list):
            all_results: list[CoreVisualizationResult] = []
            for sample in inputs:
                try:
                    ikey = sample.get('img_key', None) if isinstance(sample, dict) else None
                    res = plugin.run(
                        model=self.model,
                        inputs=sample,
                        layers=valid_layers,
                        layer_names=layer_names,
                        save=save,
                        out_dir=out_path,
                        modality=modality,
                        x_modality_name=x_modality_name,
                        family=self.family,
                        overlay=overlay,
                        img_key=ikey,
                        **kwargs,
                    )
                    if save and res:
                        saved = Saver.save(results=res, out_dir=out_path, method=method_key)
                        try:
                            LOGGER.info(f"可视化结果已保存到: {str(out_path)}（{len(saved)}个文件）")
                        except Exception:
                            pass
                    all_results.extend(res)
                finally:
                    router.restore()
            return all_results

        # 单样本模式：保留缓存逻辑
        cache_key = self._cache.make_key(
            family=self.family,
            method=method_key,
            layers=valid_layers,
            modality=modality,
            extra={**{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}, **({"overlay": overlay} if overlay is not None else {})},
        )

        try:
            cached = self._cache.get(cache_key)
            if cached is not None:
                if save:
                    saved = Saver.save(results=cached, out_dir=out_path, method=method_key)
                    try:
                        LOGGER.info(f"可视化结果已保存到: {str(out_path)}（{len(saved)}个文件）")
                    except Exception:
                        pass
                return cached

            results = plugin.run(
                model=self.model,
                inputs=inputs,
                layers=valid_layers,
                layer_names=layer_names,
                save=save,
                out_dir=out_path,
                modality=modality,
                x_modality_name=x_modality_name,
                family=self.family,
                overlay=overlay,
                **kwargs,
            )

            if save and results:
                saved = Saver.save(results=results, out_dir=out_path, method=method_key)
                try:
                    LOGGER.info(f"可视化结果已保存到: {str(out_path)}（{len(saved)}个文件）")
                except Exception:
                    pass
            self._cache.set(cache_key, results)
            return results
        finally:
            router.restore()
