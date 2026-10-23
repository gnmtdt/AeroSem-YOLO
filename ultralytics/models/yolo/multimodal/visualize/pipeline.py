"""
Visualization Pipeline (Refactor Skeleton)

This module introduces a componentized pipeline for multimodal visualization that
orchestrates input resolution, router coordination, preprocessing, layer resolution,
and method execution via existing VisualizationManager. It is designed to be extended
with plugin-style visualizers in future iterations while preserving current behavior.

Current scope:
- Provide a drop-in `VisualizationPipeline.run(...)` entry that standardizes inputs
  and delegates to `VisualizationManager` with correct layer naming conventions.
- Introduce RouterAdapter, InputResolver, Preprocessor (minimal), LayerResolver, and
  a lightweight logger to prepare for progressive migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ultralytics.utils import LOGGER


# -----------------------------
# Adapters & Resolvers (skeletons)
# -----------------------------


class RouterAdapter:
    """Adapter for accessing model multimodal router in a robust way."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._router = self._resolve_router(model)

    @staticmethod
    def _resolve_router(model: Any):
        # Probe common attachment points
        for attr_chain in (
            ("mm_router",),
            ("multimodal_router",),
            ("model", "mm_router"),
            ("model", "multimodal_router"),
        ):
            m = model
            ok = True
            for a in attr_chain:
                if hasattr(m, a):
                    m = getattr(m, a)
                else:
                    ok = False
                    break
            if ok and m is not None:
                return m
        return None

    @property
    def available(self) -> bool:
        return self._router is not None

    def set_runtime_params(self, modality: Optional[str] = None, strategy: Optional[str] = None) -> None:
        if not self.available:
            return
        try:
            if strategy is None:
                self._router.set_runtime_params(modality)
            else:
                self._router.set_runtime_params(modality, strategy=strategy)
        except Exception as e:
            LOGGER.warning(f"RouterAdapter: set_runtime_params failed: {e}")

    def update_dataset_config(self, data: Optional[dict]) -> None:
        if not self.available or not data:
            return
        try:
            self._router.update_dataset_config(data)
        except Exception:
            # Optional in many paths; do not fail hard
            pass

    def restore(self) -> None:
        if not self.available:
            return
        try:
            # Not all routers expose restore; ignore silently if absent
            if hasattr(self._router, "restore"):
                self._router.restore()
        except Exception as e:
            LOGGER.warning(f"RouterAdapter: restore failed: {e}")


class InputResolver:
    """Standardize input into a dict with fixed modality keys and predictable order."""

    VALID_KEYS = ("rgb", "x", "thermal", "depth", "ir", "infrared")

    @staticmethod
    def resolve(
        source: Union[str, Path, np.ndarray, Dict[str, Any], List[str], List[np.ndarray]],
        modality: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Accepts multiple source formats and returns a standardized dict.

        Rules:
        - Dual-list [rgb_path, x_path] → {'rgb': np.ndarray, 'x': np.ndarray}
        - Single path/array + modality ('rgb'|'x') → dict with single key
        - Dict of paths/arrays → subset to known keys
        - Direct np.ndarray (3/6 ch) is not handled here (left to ChannelComposer in future);
          for current delegation flow, VisualizationManager accepts dict/array, so we pass through array.
        """
        from .utils import load_image  # local import to avoid cycles

        # Dual list of two paths
        if isinstance(source, list) and len(source) == 2 and all(isinstance(s, (str, Path)) for s in source):
            return {"rgb": load_image(source[0]), "x": load_image(source[1])}

        # Single path + modality
        if isinstance(source, (str, Path)) and modality in {"rgb", "x"}:
            return {modality: load_image(source)}

        # Dict of paths/arrays
        if isinstance(source, dict):
            out: Dict[str, np.ndarray] = {}
            for k, v in source.items():
                k_low = str(k).lower()
                if k_low not in InputResolver.VALID_KEYS:
                    continue
                if isinstance(v, (str, Path)):
                    out["rgb" if k_low == "rgb" else ("x" if k_low in {"x", "thermal", "ir", "infrared", "depth"} else k_low)] = load_image(v)
                elif isinstance(v, np.ndarray):
                    out["rgb" if k_low == "rgb" else ("x" if k_low in {"x", "thermal", "ir", "infrared", "depth"} else k_low)] = v
            if out:
                return out

        # Pass-through: numpy array or list of numpy arrays handled by downstream
        if isinstance(source, np.ndarray) or (
            isinstance(source, list) and source and isinstance(source[0], np.ndarray)
        ):
            return {"__passthrough__": source}  # special marker

        raise ValueError(
            "Unsupported source format. Provide two paths [rgb,x], a single path with modality='rgb'|'x',\n"
            "a dict with keys among {'rgb','x','thermal','ir','infrared','depth'}, or a numpy array."
        )


class Preprocessor:
    """Minimal stub for unified preprocessing (reserved for future use)."""

    @staticmethod
    def ensure_numpy_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        # passthrough for now; future: dtype/bit-depth normalization
        return data


class LayerResolver:
    """Resolve external layer indices to method-specific internal layer identifiers."""

    @staticmethod
    def to_manager_layers(layers: List[int], method: str) -> List[str]:
        if method == "feature_map":
            # FeatureMapVisualizer expects 'model.{idx}' or Module; pass names here
            return [f"model.{i}" for i in layers]
        # HeatmapVisualizer expects indices as string relative to model.model
        return [str(i) for i in layers]


@dataclass
class PipelineContext:
    router: RouterAdapter
    input_dict: Dict[str, Any]
    method: str
    layers: List[int]
    alg: Optional[str]
    save: bool
    project: str
    name: str
    modality: Optional[str]
    data_cfg: Optional[dict]


class PipelineLogger:
    @staticmethod
    def summary(ctx: PipelineContext, out_dir: Optional[Path] = None) -> None:
        try:
            msg = (
                f"vis: method={ctx.method}, layers={ctx.layers}, alg={ctx.alg}, save={ctx.save}, "
                f"project={ctx.project}/{ctx.name}, modality={ctx.modality}, router={'on' if ctx.router.available else 'off'}"
            )
            if out_dir is not None:
                msg += f", out={out_dir}"
            LOGGER.info(msg)
        except Exception:
            pass


# -----------------------------
# Pipeline (delegating to VisualizationManager)
# -----------------------------


class VisualizationPipeline:
    """Refactor skeleton that delegates actual rendering to VisualizationManager."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def run(
        self,
        source: Union[str, Path, np.ndarray, Dict[str, Any], List[str], List[np.ndarray]],
        method: str = "heatmap",
        layers: Optional[List[int]] = None,
        alg: Optional[str] = None,
        modality: Optional[str] = None,
        save: bool = True,
        project: str = "runs/visualize",
        name: str = "exp",
        data_cfg: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Any]:
        # Resolve input dict or passthrough
        resolved = InputResolver.resolve(source, modality=modality)

        # Router coordination
        router = RouterAdapter(self.model)
        if router.available:
            router.update_dataset_config(data_cfg)
            router.set_runtime_params(modality)

        # Minimal preprocess placeholder
        resolved = Preprocessor.ensure_numpy_dict(resolved)

        # Translate layers for current visualizer
        if not layers or not isinstance(layers, list) or not all(isinstance(i, int) for i in layers):
            raise ValueError("layers must be a non-empty List[int] for pipeline visualization")
        manager_layers = LayerResolver.to_manager_layers(layers, method=method)

        # Build context & log
        ctx = PipelineContext(
            router=router,
            input_dict=resolved,
            method=method,
            layers=layers,
            alg=alg,
            save=save,
            project=project,
            name=name,
            modality=modality,
            data_cfg=data_cfg,
        )

        # Delegate to VisualizationManager
        from .manager import VisualizationManager  # local import to avoid cycles

        vm = VisualizationManager(model=self.model, project=project, name=name)

        # Determine source param for manager (dict or passthrough array)
        if "__passthrough__" in resolved:
            vm_source = resolved["__passthrough__"]
        else:
            vm_source = resolved

        try:
            results = vm.visualize(
                source=vm_source,
                method=method,
                layers=[m for m in manager_layers],
                save=save,
                alg=alg or "gradcam",
                **kwargs,
            )
            # Log summary with output dir
            PipelineLogger.summary(ctx, out_dir=vm.output_dir)
            return results
        finally:
            # Always try to restore router runtime state
            router.restore()

