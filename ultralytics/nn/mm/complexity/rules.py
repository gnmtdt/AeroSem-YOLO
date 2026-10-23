# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""[DEPRECATED] Per-module-type FLOPs rules.

This module's manual rule registry has been replaced by automatic FLOPs
calculation in auto_flops.py. The RULES dict is kept as an empty stub
for backward compatibility. New modules do NOT need rules here.

The primitive helpers (conv2d_flops, linear_flops) are retained as they
may be used by other modules.
"""

from __future__ import annotations

import torch.nn as nn


# ------------------------------------------------------------------
# Primitive helpers (retained for potential external use)
# ------------------------------------------------------------------

def conv2d_flops(conv, in_shape, out_shape) -> float:
    """Calculate FLOPs for a Conv2d operation.

    Uses arithmetic FLOPs: 2 * output_h * output_w * out_channels * (in_channels / groups) * kernel_h * kernel_w
    """
    kernel_h, kernel_w = conv.kernel_size
    groups = max(int(conv.groups), 1)
    return float(
        2
        * out_shape.height
        * out_shape.width
        * conv.out_channels
        * (conv.in_channels // groups)
        * kernel_h
        * kernel_w
    )


def linear_flops(linear, batch_tokens: int) -> float:
    """Calculate FLOPs for a Linear operation."""
    return float(2 * batch_tokens * linear.in_features * linear.out_features)


# ------------------------------------------------------------------
# Empty rule registry (deprecated)
# ------------------------------------------------------------------

RULES = {}
