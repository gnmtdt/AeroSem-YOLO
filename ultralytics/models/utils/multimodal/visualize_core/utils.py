"""Misc utilities for visualize_core."""

from __future__ import annotations

from typing import Dict


def infer_modality(inputs: Dict[str, object]) -> str:
    if 'rgb' in inputs and 'x' in inputs:
        return 'dual'
    if 'rgb' in inputs:
        return 'rgb'
    return 'x'

