# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Trainability normalization helpers for pruned YOLOMM models.

This module provides utilities to restore floating-point parameters from frozen
state (requires_grad=False) back to trainable state (requires_grad=True).

The key design principle is that these helpers do NOT encode runtime freeze
policies such as DFL freezing. Runtime freezes remain owned by the trainer
(BaseTrainer._setup_train). This module only ensures that checkpoints do not
carry residual frozen states from upstream operations (e.g., EMA models used
as pruning inputs).
"""

from __future__ import annotations

from typing import List

import torch.nn as nn


def restore_parameter_trainability(model: nn.Module) -> List[str]:
    """Restore all floating-point parameters in `model` to trainable state.

    This helper intentionally does not encode runtime freeze policy.
    Runtime freezes (for example DFL) remain owned by the trainer.

    Args:
        model: The PyTorch module whose parameters should be restored.

    Returns:
        List of parameter names that were restored from frozen to trainable.
    """
    restored: List[str] = []
    for name, param in model.named_parameters():
        if not param.dtype.is_floating_point:
            continue
        if not param.requires_grad:
            param.requires_grad_(True)
            restored.append(name)
    return restored


def find_frozen_floating_parameters(model: nn.Module) -> List[str]:
    """Return names of floating-point parameters that still have requires_grad=False.

    This is a fail-fast utility for checking whether a model has been
    properly normalized. Used at key boundaries to assert that no
    unexpected frozen parameters remain.

    Args:
        model: The PyTorch module to inspect.

    Returns:
        List of parameter names that are floating-point but frozen.
    """
    frozen: List[str] = []
    for name, param in model.named_parameters():
        if param.dtype.is_floating_point and not param.requires_grad:
            frozen.append(name)
    return frozen
