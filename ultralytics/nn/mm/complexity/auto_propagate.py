# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Automatic shape propagation for YOLOMM complexity engine.

Replaces the manual SHAPE_RULES registry with per-node dummy forward passes.
Any module registered in tasks.py parse_model is automatically supported
without additional shape rules.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER

from .schema import TensorShapeSpec

# ------------------------------------------------------------------
# Constants (mirrored from graph.py to avoid circular imports)
# ------------------------------------------------------------------

_ROUTE_ONLY_TYPES = frozenset({"Concat", "Upsample", "Index"})
_HEAD_TYPES = frozenset({"Detect", "Segment", "Pose", "OBB", "Classification"})
_MULTI_OUTPUT_TYPES = frozenset({"FCM", "MultiHeadCrossAttention", "ChannelTransformer", "SpatialPriorModuleLite"})


# ------------------------------------------------------------------
# Device helper
# ------------------------------------------------------------------

def _get_device(module: nn.Module) -> torch.device:
    """Get the device of a module's parameters."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_floating_dtype(module: nn.Module) -> torch.dtype:
    """Get a representative floating dtype for dummy-forward inputs."""
    for tensor in module.parameters():
        if torch.is_floating_point(tensor):
            return tensor.dtype
    for tensor in module.buffers():
        if torch.is_floating_point(tensor):
            return tensor.dtype
    return torch.float32


# ------------------------------------------------------------------
# Core: automatic shape inference
# ------------------------------------------------------------------

def auto_infer_output_shapes(node, input_shapes, device=None):
    """Infer output tensor shapes for a node using dummy forward pass.

    Strategy (in order of priority):
    1. Route-only modules (Concat/Upsample/Index): hardcoded rules
    2. Detection heads: pass through all input shapes
    3. Multi-output modules (FCM/MHCA): special dispatch with reshaped inputs
    4. General modules: dummy tensor forward pass
    5. Fallback: reflection (find last Conv2d out_channels, spatial invariant)

    Args:
        node: PruneNode instance (has .module, .type_name, .primary_out_channels).
        input_shapes: Tuple of TensorShapeSpec for each input.
        device: Optional torch.device override.

    Returns:
        Tuple of TensorShapeSpec.
    """
    if not input_shapes:
        return (TensorShapeSpec(channels=node.primary_out_channels, height=1, width=1),)

    if device is None:
        device = _get_device(node.module)

    ltype = node.type_name

    # 1. Route-only modules: hardcoded rules (trivial, no forward needed)
    if ltype == "Concat":
        return _shape_concat(input_shapes)
    if ltype == "Upsample":
        return _shape_upsample(node, input_shapes)
    if ltype == "Index":
        return (input_shapes[0],)

    # YOLOVO text path uses non-spatial tensors represented as (D, 1, 1).
    if ltype == "VCTETextInput":
        dim = int(getattr(node.module, "dim", node.primary_out_channels) or node.primary_out_channels)
        return (TensorShapeSpec(channels=dim, height=1, width=1),)
    if ltype == "VCTETextAdapter":
        return (TensorShapeSpec(channels=input_shapes[0].channels, height=1, width=1),)
    if ltype == "VCTEVisualAdapter":
        out_dim = int(getattr(getattr(node.module, "proj", None), "out_features", node.primary_out_channels) or node.primary_out_channels)
        return (TensorShapeSpec(channels=out_dim, height=1, width=1),)
    if ltype == "VCTETextEnhancer":
        text_shape = input_shapes[0]
        return (
            TensorShapeSpec(channels=text_shape.channels, height=1, width=1),
            TensorShapeSpec(channels=text_shape.channels, height=1, width=1),
        )
    if ltype == "VCTENeckModulator":
        return (input_shapes[0],)
    if ltype == "YOLOVODetect":
        return tuple(
            TensorShapeSpec(channels=s.channels, height=s.height, width=s.width)
            for s in input_shapes[:3]
        )

    # 2. Detection heads: pass through all input shapes
    if ltype in _HEAD_TYPES:
        return tuple(
            TensorShapeSpec(channels=s.channels, height=s.height, width=s.width)
            for s in input_shapes
        )

    # 3. Multi-output modules: special forward dispatch
    if ltype == "FCM":
        result = _shape_fcm(node, input_shapes, device)
        if result is not None:
            return result
    elif ltype == "MultiHeadCrossAttention":
        result = _shape_mhca(node, input_shapes, device)
        if result is not None:
            return result
    elif ltype == "ChannelTransformer":
        result = _shape_channel_transformer(node, input_shapes, device)
        if result is not None:
            return result

    # 4. General modules: dummy forward pass
    result = _try_dummy_forward(node.module, input_shapes, device)
    if result is not None:
        return result

    # 5. Fallback: reflection
    return _shape_fallback(node, input_shapes)


# ------------------------------------------------------------------
# Route-only shape rules (hardcoded, ~15 lines total)
# ------------------------------------------------------------------

def _shape_concat(input_shapes):
    """Concat: channels sum, spatial dims must match."""
    first_h, first_w = input_shapes[0].height, input_shapes[0].width
    for s in input_shapes[1:]:
        if s.height != first_h or s.width != first_w:
            raise RuntimeError(
                f"Concat received spatially inconsistent inputs: "
                f"expected ({first_h}, {first_w}), got ({s.height}, {s.width})"
            )
    return (
        TensorShapeSpec(
            channels=sum(s.channels for s in input_shapes),
            height=first_h,
            width=first_w,
        ),
    )


def _shape_upsample(node, input_shapes):
    """Upsample: spatial dims scaled, channels unchanged."""
    src = input_shapes[0]
    scale = int(getattr(node.module, "scale_factor", 2) or 2)
    return (
        TensorShapeSpec(
            channels=src.channels,
            height=src.height * scale,
            width=src.width * scale,
        ),
    )


# ------------------------------------------------------------------
# Multi-output module shape rules
# ------------------------------------------------------------------

def _shape_fcm(node, input_shapes, device):
    """FCM: forward(x1, x2, H, W) with [B, N, C] input format."""
    src = input_shapes[0]
    c, h, w = src.channels, src.height, src.width
    n = h * w  # sequence length

    dtype = _get_floating_dtype(node.module)
    dummy = torch.zeros(1, n, c, device=device, dtype=dtype)
    try:
        with torch.no_grad():
            out1, out2 = node.module(dummy, dummy, h, w)

        return (
            TensorShapeSpec(channels=out1.shape[-1], height=h, width=w),
            TensorShapeSpec(channels=out2.shape[-1], height=h, width=w),
        )
    except Exception as e:
        LOGGER.debug(
            f"Complexity FCM shape inference failed for {node.module.__class__.__name__} "
            f"with input shape={(1, n, c)} dtype={dtype}: {e}"
        )
        return None


def _shape_mhca(node, input_shapes, device):
    """MultiHeadCrossAttention: forward(vis, inf) with [B, N, C] input format."""
    src = input_shapes[0]
    c, h, w = src.channels, src.height, src.width
    n = h * w

    dtype = _get_floating_dtype(node.module)
    dummy = torch.zeros(1, n, c, device=device, dtype=dtype)
    try:
        with torch.no_grad():
            out1, out2 = node.module(dummy, dummy)

        return (
            TensorShapeSpec(channels=out1.shape[-1], height=h, width=w),
            TensorShapeSpec(channels=out2.shape[-1], height=h, width=w),
        )
    except Exception as e:
        LOGGER.debug(
            f"Complexity MHCA shape inference failed for {node.module.__class__.__name__} "
            f"with input shape={(1, n, c)} dtype={dtype}: {e}"
        )
        return None


def _shape_channel_transformer(node, input_shapes, device):
    """ChannelTransformer: forward([p3, p4, p5[, p6]]) with 4D feature maps."""
    dtype = _get_floating_dtype(node.module)
    dummy_inputs = [
        torch.zeros(1, s.channels, s.height, s.width, device=device, dtype=dtype)
        for s in input_shapes
    ]
    try:
        with torch.no_grad():
            outputs = node.module(dummy_inputs)

        if not isinstance(outputs, (tuple, list)):
            return None

        shapes = []
        for out in outputs:
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                shapes.append(
                    TensorShapeSpec(channels=out.shape[1], height=out.shape[2], width=out.shape[3])
                )
        return tuple(shapes) if shapes else None
    except Exception as e:
        LOGGER.debug(
            f"Complexity ChannelTransformer shape inference failed for {node.module.__class__.__name__}: {e}"
        )
        return None


# ------------------------------------------------------------------
# General dummy forward
# ------------------------------------------------------------------

def _try_dummy_forward(module, input_shapes, device):
    """Attempt a dummy forward pass to infer output shapes.

    Returns:
        Tuple of TensorShapeSpec on success, None on failure.
    """
    src = input_shapes[0]
    c, h, w = src.channels, src.height, src.width

    dummy = torch.zeros(1, c, h, w, device=device, dtype=_get_floating_dtype(module))

    try:
        with torch.no_grad():
            out = module(dummy)

        if isinstance(out, torch.Tensor):
            if out.dim() == 4:
                return (
                    TensorShapeSpec(
                        channels=out.shape[1],
                        height=out.shape[2],
                        width=out.shape[3],
                    ),
                )
            # 2D output (e.g. global pooled) — treat as (channels, 1, 1)
            if out.dim() == 2:
                return (TensorShapeSpec(channels=out.shape[1], height=1, width=1),)

        if isinstance(out, (tuple, list)):
            shapes = []
            for o in out:
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    shapes.append(
                        TensorShapeSpec(
                            channels=o.shape[1],
                            height=o.shape[2],
                            width=o.shape[3],
                        )
                    )
            if shapes:
                return tuple(shapes)

        return None

    except Exception as e:
        LOGGER.debug(
            f"Complexity dummy forward failed for {module.__class__.__name__} "
            f"with input shape={(c, h, w)} dtype={dummy.dtype}: {e}"
        )
        return None


# ------------------------------------------------------------------
# Fallback: reflection
# ------------------------------------------------------------------

def _shape_fallback(node, input_shapes):
    """Fallback shape inference via module reflection.

    Assumes spatial dimensions are preserved (true for ~95% of modules).
    Channels are read from node.primary_out_channels (set during graph build).
    """
    src = input_shapes[0]
    return (
        TensorShapeSpec(
            channels=node.primary_out_channels,
            height=src.height,
            width=src.width,
        ),
    )
