# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Automatic FLOPs calculation for YOLOMM complexity engine.

Replaces the manual RULES registry with per-node automatic FLOPs counting.
Two strategies:
1. thop per-node profiling (when thop is available)
2. Recursive Conv2d + Linear sub-module counting (always-available fallback)

Any module registered in tasks.py parse_model is automatically supported.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .schema import TensorShapeSpec

# ------------------------------------------------------------------
# Constants (mirrored from graph.py to avoid circular imports)
# ------------------------------------------------------------------

_ROUTE_ONLY_TYPES = frozenset({"Concat", "Upsample", "Index"})
_HEAD_TYPES = frozenset({"Detect", "Segment", "Pose", "OBB", "Classification"})

# Check thop availability
_THOP_AVAILABLE = False
try:
    import thop
    _THOP_AVAILABLE = True
except ImportError:
    pass


# ------------------------------------------------------------------
# Core: automatic FLOPs computation
# ------------------------------------------------------------------

def auto_compute_flops(node, input_shapes, output_shapes, device=None):
    """Compute FLOPs for a single node automatically.

    Strategy (in order of priority):
    1. route_only / head -> 0.0
    2. thop available -> per-node thop profiling
    3. Fallback -> recursive Conv2d + Linear sub-module counting

    Args:
        node: PruneNode instance.
        input_shapes: Tuple of input TensorShapeSpec.
        output_shapes: Tuple of output TensorShapeSpec.
        device: Optional torch.device override.

    Returns:
        FLOPs as float.
    """
    ltype = node.type_name

    # 1. Zero-FLOPs modules
    if node.is_route_only or ltype in _ROUTE_ONLY_TYPES:
        return 0.0
    if node.is_head or ltype in _HEAD_TYPES:
        return 0.0

    if not input_shapes or not output_shapes:
        return 0.0

    if device is None:
        try:
            device = next(node.module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # 2. Try thop per-node profiling
    if _THOP_AVAILABLE:
        result = _thop_node_flops(node, input_shapes, device)
        if result is not None:
            return result

    # 3. Fallback: recursive sub-module counting
    return _recursive_flops(node.module, output_shapes[0])


# ------------------------------------------------------------------
# thop per-node profiling
# ------------------------------------------------------------------

def _thop_node_flops(node, input_shapes, device):
    """Use thop to profile a single node's FLOPs.

    Returns:
        FLOPs as float on success, None on failure.
    """
    src = input_shapes[0]
    c, h, w = src.channels, src.height, src.width

    dummy = torch.zeros(1, c, h, w, device=device)

    try:
        m_copy = copy.deepcopy(node.module)
        m_copy.eval()

        # Single-input modules
        if len(input_shapes) == 1:
            flops = thop.profile(m_copy, inputs=(dummy,), verbose=False)[0]
        else:
            # Multi-input: wrap as list input
            dummies = [
                torch.zeros(1, s.channels, s.height, s.width, device=device)
                for s in input_shapes
            ]
            wrapper = _ListInputWrapper(m_copy)
            flops = thop.profile(wrapper, inputs=tuple(dummies), verbose=False)[0]

        # thop returns MACs (multiply-accumulate), convert to arithmetic FLOPs
        return float(flops * 2)

    except Exception:
        return None


class _ListInputWrapper(nn.Module):
    """Wrapper to convert multi-tensor args to list input for modules like Concat."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, *inputs):
        return self.base(list(inputs))


# ------------------------------------------------------------------
# Recursive sub-module FLOPs counting (always-available fallback)
# ------------------------------------------------------------------

def _recursive_flops(module, out_shape):
    """Count FLOPs by traversing all Conv2d and Linear sub-modules.

    This is a generalization of the per-type rules in the old rules.py.
    Instead of matching module type names, it directly inspects sub-modules.

    Args:
        module: The nn.Module to analyze.
        out_shape: TensorShapeSpec of the module's output.

    Returns:
        Total FLOPs as float.
    """
    total = 0.0
    h_out, w_out = out_shape.height, out_shape.width

    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            groups = max(int(m.groups), 1)
            total += float(
                2
                * h_out
                * w_out
                * m.out_channels
                * (m.in_channels // groups)
                * m.kernel_size[0]
                * m.kernel_size[1]
            )
        elif isinstance(m, nn.Linear):
            # For Linear layers in conv models, spatial dims = number of tokens
            tokens = h_out * w_out
            total += float(2 * tokens * m.in_features * m.out_features)

    return total
