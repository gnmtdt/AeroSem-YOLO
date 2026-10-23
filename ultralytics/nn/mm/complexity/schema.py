# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Complexity analysis data structures for YOLOMM pruning stage.

This module defines the input specification, tensor shape, node complexity,
and report objects used by the graph-driven complexity engine. The design
ensures that complexity calculations start from explicit input semantics
rather than relying on implicit dummy input inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


RouteMode = Literal["dual", "rgb", "x"]
StageKind = Literal["backbone", "neck", "head"]


@dataclass(frozen=True)
class ComplexityInputSpec:
    """Explicit input specification for complexity analysis.

    This captures the true external input semantics, preventing the engine
    from inferring input channels from first-layer parameters (which is
    unreliable for mid-fusion multimodal models).

    Attributes:
        imgsz: (height, width) of input images.
        route_mode: One of 'dual' (RGB+X), 'rgb', or 'x'.
        rgb_channels: Number of RGB channels (always 3).
        x_channels: Number of X modality channels (configurable).
    """
    imgsz: tuple[int, int]
    route_mode: RouteMode
    rgb_channels: int
    x_channels: int


@dataclass(frozen=True)
class TensorShapeSpec:
    """Shape specification for a single tensor.

    Only spatial dimensions and channels are tracked; batch dimension
    is omitted since FLOPs are linear in batch size.

    Attributes:
        channels: Number of channels.
        height: Spatial height.
        width: Spatial width.
    """
    channels: int
    height: int
    width: int


@dataclass
class NodeComplexity:
    """Complexity result for a single graph node.

    Attributes:
        node_idx: Layer index within the model.
        type_name: Canonical type string (e.g., 'Conv', 'C2f').
        stage: Which multimodal stage this node belongs to.
        input_shapes: Tuple of input tensor shapes.
        output_shapes: Tuple of output tensor shapes.
        flops: Arithmetic FLOPs for this node (multiply + add each count as 1).
    """
    node_idx: int
    type_name: str
    stage: StageKind
    input_shapes: tuple[TensorShapeSpec, ...]
    output_shapes: tuple[TensorShapeSpec, ...]
    flops: float


@dataclass
class ComplexityReport:
    """Complete complexity report for a model.

    Attributes:
        input_spec: The input specification used for this analysis.
        nodes: List of per-node complexity results.
    """
    input_spec: ComplexityInputSpec
    nodes: list[NodeComplexity] = field(default_factory=list)

    @property
    def total_flops(self) -> float:
        """Total arithmetic FLOPs across all nodes."""
        return sum(node.flops for node in self.nodes)

    def stage_flops(self) -> dict[StageKind, float]:
        """Return FLOPs broken down by multimodal stage."""
        totals: dict[StageKind, float] = {}
        for node in self.nodes:
            totals[node.stage] = totals.get(node.stage, 0.0) + node.flops
        return totals
