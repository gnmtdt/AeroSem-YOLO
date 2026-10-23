"""Shared data types for visualize_core.

This module holds lightweight dataclasses used across the core visualization
pipeline and helpers, to avoid circular imports between components such as
pipeline and saver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PipelineContext:
    family: str
    model: Any
    method: str
    layers: List[int]
    save: bool
    out_dir: Optional[Path]
    device: Optional[str]
    modality: Optional[str]
    extra: Dict[str, Any]


@dataclass
class CoreVisualizationResult:
    type: str
    data: Any
    meta: Dict[str, Any]

