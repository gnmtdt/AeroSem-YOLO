"""
Core visualization framework (visualize_core).

This package provides the fail-fast, family-agnostic visualization pipeline and
core components shared by YOLOMM and RTDETRMM. It focuses on orchestrating inputs,
layer resolution, plugin discovery, rendering, saving and caching, without embedding
any model-family specific logic.

Note:
- Methods (e.g., heat/feature) are intentionally not implemented here yet. The
  aim is to establish the end-to-end framework first; plugins will be added later.
- Any attempt to execute an unregistered method must raise a clear, actionable error
  (Fail‑Fast). No graceful fallback or placeholder rendering is performed.
"""

from .exceptions import (
    VisualizationError,
    MethodNotRegisteredError,
    InputValidationError,
    ModalityConflictError,
    LayerResolutionError,
    DeviceMismatchError,
)
from .input_resolver import InputResolver
from .layer_resolver import LayerResolver
from .pipeline import Pipeline
from .types import PipelineContext, CoreVisualizationResult
from .registry import MethodRegistry, REGISTRY
from . import plugins  # noqa: F401  # ensure default plugins are registered
from .router_adapter import RouterAdapter
from .cache import Cache
from .saver import Saver
from .renderer import Renderer
from .utils import infer_modality
from .preprocessor import Preprocessor

__all__ = (
    "Pipeline",
    "PipelineContext",
    "CoreVisualizationResult",
    "InputResolver",
    "LayerResolver",
    "MethodRegistry",
    "REGISTRY",
    "RouterAdapter",
    "Cache",
    "Saver",
    "Renderer",
    "infer_modality",
    "Preprocessor",
    "VisualizationError",
    "MethodNotRegisteredError",
    "InputValidationError",
    "ModalityConflictError",
    "LayerResolutionError",
    "DeviceMismatchError",
)
