"""Layer resolution utilities for visualize_core (Fail-Fast)."""

from __future__ import annotations

from typing import Any, List, Tuple, Dict

from .exceptions import LayerResolutionError


class LayerResolver:
    """Resolve external integer indices to model layer names and validate bounds."""

    @staticmethod
    def validate_indices(model: Any, layers: List[int]) -> List[int]:
        total = len(getattr(model, 'model', [])) if hasattr(model, 'model') else 0
        if not isinstance(layers, list) or not layers or not all(isinstance(i, int) for i in layers):
            suggestions = LayerResolver.suggest_layers(model, topn=10)
            overview = LayerResolver.enumerate_layers(model, topn=20)
            raise LayerResolutionError(invalid=[-1], valid_range=(0, total - 1), suggestions=suggestions, overview=overview)
        max_idx = len(model.model) - 1 if hasattr(model, 'model') else -1
        invalid = [i for i in layers if i < 0 or i > max_idx]
        if invalid:
            suggestions = LayerResolver.suggest_layers(model, topn=10)
            overview = LayerResolver.enumerate_layers(model, topn=20)
            raise LayerResolutionError(invalid=invalid, valid_range=(0, max_idx), suggestions=suggestions, overview=overview)
        # 去重保持顺序
        seen = set()
        ordered = []
        for i in layers:
            if i not in seen:
                seen.add(i)
                ordered.append(i)
        return ordered

    @staticmethod
    def to_names(layers: List[int]) -> List[str]:
        return [f"model.{i}" for i in layers]

    @staticmethod
    def enumerate_layers(model: Any, topn: int = 20) -> List[Tuple[int, str]]:
        names: List[Tuple[int, str]] = []
        if hasattr(model, 'model'):
            for i, m in enumerate(model.model):
                names.append((i, m.__class__.__name__))
        return names[:topn]

    @staticmethod
    def is_visualizable_module(m: Any) -> bool:
        name = m.__class__.__name__
        # prioritize typical feature-producing blocks
        return any(k in name for k in ("Conv", "C2", "C3", "Bottleneck", "SPP", "SPPF", "Down", "Stem", "Stage"))

    @staticmethod
    def suggest_layers(model: Any, topn: int = 10) -> List[int]:
        if not hasattr(model, 'model'):
            return []
        cand: List[int] = []
        for i, m in enumerate(model.model):
            if LayerResolver.is_visualizable_module(m):
                cand.append(i)
        # fallback: choose evenly spaced indices
        if not cand:
            n = len(model.model)
            if n > 0:
                step = max(1, n // max(1, topn))
                cand = list(range(0, n, step))[:topn]
        return cand[:topn]
