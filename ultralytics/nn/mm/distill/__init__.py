# Ultralytics Multimodal Distillation Module
# Knowledge distillation support for YOLOMM and RTDETRMM detection models.

"""
Distillation sub-package for multimodal detection models.

Provides YAML-driven configuration, teacher runtime coordination,
family-level adapters and loss computation for knowledge distillation.

Two mapping groups with distinct semantics:
- ``feature``: Layer-index mappings (``FeatureMappingSpec``).
- ``output``: Teacher-level declarations (``OutputTeacherSpec``), no layer indices.
"""

from .schema import (
    TeacherSpec,
    FeatureMappingSpec,
    MappingSpec,  # backward-compatible alias for FeatureMappingSpec
    OutputTeacherSpec,
    DistillConfig,
    load_distill_config,
)

__all__ = [
    "TeacherSpec",
    "FeatureMappingSpec",
    "MappingSpec",
    "OutputTeacherSpec",
    "DistillConfig",
    "load_distill_config",
]
