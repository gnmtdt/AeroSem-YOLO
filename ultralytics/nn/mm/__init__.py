# Ultralytics Multimodal Router Module
# Universal Dynamic Modality Routing System for YOLO and RTDETR
# Version: v2.0

"""
Ultralytics Multimodal Router Module

This module provides a comprehensive multimodal routing system supporting
arbitrary dual-modality architectures, with backward-compatible RGB+X support.

Core Components:
- MultiModalRouter: Universal dynamic modality data router
- MultiModalConfigParser: YAML configuration parsing for multimodal architectures
- Utility functions: System status, model validation, and configuration helpers

Supported Input Aliases (5th field):
- Legacy: RGB (PRIMARY), X (SECONDARY), Dual (FUSED)
- New: A (PRIMARY), B (SECONDARY), Fuse (FUSED)

Features:
- Zero-copy tensor view routing
- Configuration-driven data flow
- Arbitrary dual-modality support (Phase 1)
- Runtime checkpoint migration (no disk modification)
"""

# Core multimodal router
from .router import MultiModalRouter

# Configuration parser
from .parser import MultiModalConfigParser

# Utility functions
from .utils import (
    validate_mm_config_format,
    mm_system_status,
    check_mm_model_attributes,
    get_mm_system_info,
    INPUT_SOURCE_ALIASES,
    is_rgb_modality,
    resolve_modality_pair,
    get_runtime_primary_modality_name,
)
from .generators import DepthGen, DEMGen, EdgeGen

# Dtype utilities (moved from dtype_policy.py)
from .utils import check_tensor_dtype_for_half

# Multimodal cfg_dict preparation (shared across all trainers)
from .utils import prepare_multimodal_cfg_dict

# Source matching utilities
from .source_matcher import MultiModalSourceMatcher

# Distillation configuration
from .distill import (
    DistillConfig,
    TeacherSpec,
    FeatureMappingSpec,
    MappingSpec,  # backward-compatible alias for FeatureMappingSpec
    OutputTeacherSpec,
    load_distill_config,
)

# Version — single source: ultralytics/cfg/default.yaml → mm_project_version
try:
    from ultralytics.utils import DEFAULT_CFG_DICT
    PROJECT_VERSION = DEFAULT_CFG_DICT.get("mm_project_version", "v1.0")
except Exception:
    PROJECT_VERSION = "v1.0"

# Export all components
__all__ = [
    # Core classes
    "MultiModalRouter",
    "MultiModalConfigParser",

    # Utility functions
    "validate_mm_config_format",
    "mm_system_status",
    "check_mm_model_attributes",
    "get_mm_system_info",
    "INPUT_SOURCE_ALIASES",
    "is_rgb_modality",
    "resolve_modality_pair",
    "get_runtime_primary_modality_name",
    # Generators
    "DepthGen",
    "DEMGen",
    "EdgeGen",

    # Source matching utilities
    "MultiModalSourceMatcher",

    # Dtype utilities
    "check_tensor_dtype_for_half",

    # Distillation
    "DistillConfig",
    "TeacherSpec",
    "FeatureMappingSpec",
    "MappingSpec",
    "OutputTeacherSpec",
    "load_distill_config",

    # Version
    "PROJECT_VERSION",
]

# Module metadata
__author__ = "YOLOMM Team"
__description__ = "Universal Multimodal Routing System"
__supported_modalities__ = ["RGB", "X", "Dual", "A", "B", "Fuse", "Text"]
__supported_architectures__ = ["YOLO", "RTDETR", "YOLOVO"]
