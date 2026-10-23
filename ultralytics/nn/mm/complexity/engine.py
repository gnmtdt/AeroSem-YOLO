# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Graph-driven complexity engine for YOLOMM multimodal models.

This module provides a unified complexity analysis that:
1. Starts from explicit input semantics (not inferred from parameters)
2. Propagates tensor shapes through the prune graph via per-node dummy forward
3. Computes FLOPs automatically via thop or recursive sub-module counting
4. Returns stage-breakdown results

No per-module-type manual registration is needed. Any module registered in
tasks.py parse_model is automatically supported.
"""

from __future__ import annotations

from ultralytics.nn.mm.pruning.graph import PruneGraph, build_prune_graph
from ultralytics.utils import LOGGER

from .auto_flops import auto_compute_flops
from .auto_propagate import auto_infer_output_shapes
from .schema import (
    ComplexityInputSpec,
    ComplexityReport,
    NodeComplexity,
    RouteMode,
    StageKind,
    TensorShapeSpec,
)


# ------------------------------------------------------------------
# Debug mode — reads from global default.yaml 'debug' parameter
# ------------------------------------------------------------------
def _is_debug() -> bool:
    try:
        from ultralytics.utils import DEFAULT_CFG_DICT
        return DEFAULT_CFG_DICT.get("debug", False)
    except Exception:
        return False


def _debug_log(msg: str):
    """Print debug message if global debug mode is enabled."""
    if _is_debug():
        LOGGER.info(f"[ComplexityDebug] {msg}")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def compute_multimodal_complexity_report(
    model,
    imgsz: int = 640,
    route_mode: RouteMode = "dual",
    modality: str | None = None,
) -> ComplexityReport:
    """Compute graph-driven complexity report with explicit multimodal input semantics."""
    graph = build_prune_graph(model)
    input_spec = build_complexity_input_spec(
        model=model,
        graph=graph,
        imgsz=imgsz,
        route_mode=route_mode,
        modality=modality,
    )
    backbone_len = _get_backbone_len(model)
    return _run_complexity_engine(model=model, graph=graph, input_spec=input_spec, backbone_len=backbone_len)


def compute_default_multimodal_complexity_report(model, imgsz: int = 640) -> ComplexityReport:
    """Compute the default-structure complexity report for YOLOMM mainline flows.

    This intentionally ignores runtime modality switches and reports the model's
    structural complexity truth source using the shared graph-driven engine.
    """
    return compute_multimodal_complexity_report(
        model=model,
        imgsz=imgsz,
        route_mode="dual",
        modality=None,
    )


def compute_pruning_complexity_report(model, imgsz: int = 640) -> ComplexityReport:
    """Compute pruning-stage complexity report using the shared multimodal engine."""
    return compute_default_multimodal_complexity_report(model=model, imgsz=imgsz)


# ------------------------------------------------------------------
# Input spec construction
# ------------------------------------------------------------------

def build_complexity_input_spec(
    model,
    graph,
    imgsz: int,
    route_mode: RouteMode = "dual",
    modality: str | None = None,
) -> ComplexityInputSpec:
    """Build explicit input spec from router or graph metadata.

    Priority order:
    1. Read from multimodal_router.INPUT_SOURCES (most reliable)
    2. Fall back to graph entry node analysis
    3. Default to single-modal RGB if no multimodal info
    """
    _ = modality  # Reserved for future routing-sensitive input-spec branching.

    # Try to read from the real router
    router = getattr(model, "multimodal_router", None) or getattr(model, "mm_router", None)
    if router is not None and hasattr(router, "INPUT_SOURCES"):
        rgb_channels = int(router.INPUT_SOURCES.get("PRIMARY", 3))
        x_channels = int(router.INPUT_SOURCES.get("SECONDARY", 3))
        return ComplexityInputSpec(
            imgsz=(imgsz, imgsz),
            route_mode=route_mode,
            rgb_channels=rgb_channels,
            x_channels=x_channels,
        )

    # Fall back to graph entry analysis
    x_entries = [
        node for node in graph.nodes
        if getattr(node.module, "_mm_input_source", None) == "X"
    ]
    if x_entries:
        x_channels = int(getattr(x_entries[0].module, "in_channels", 3) or 3)
        return ComplexityInputSpec(
            imgsz=(imgsz, imgsz),
            route_mode=route_mode,
            rgb_channels=3,
            x_channels=x_channels,
        )

    # No X modality detected
    return ComplexityInputSpec(
        imgsz=(imgsz, imgsz),
        route_mode="rgb" if route_mode != "dual" else "dual",
        rgb_channels=3,
        x_channels=0,
    )


# ------------------------------------------------------------------
# Shape propagation
# ------------------------------------------------------------------

def _seed_entry_shapes(graph, input_spec):
    """Seed input shapes for all entry nodes."""
    h, w = input_spec.imgsz
    seeded = {}

    for node in graph.nodes:
        if not node.is_entry:
            continue

        source = getattr(node.module, "_mm_input_source", None)

        if source in ("RGB", "PRIMARY"):
            seeded[node.idx] = (
                TensorShapeSpec(channels=input_spec.rgb_channels, height=h, width=w),
            )
        elif source in ("X", "SECONDARY"):
            seeded[node.idx] = (
                TensorShapeSpec(channels=input_spec.x_channels, height=h, width=w),
            )
        elif source in ("Dual", "FUSED"):
            seeded[node.idx] = (
                TensorShapeSpec(
                    channels=input_spec.rgb_channels + input_spec.x_channels,
                    height=h,
                    width=w,
                ),
            )
        else:
            # Default to RGB for unspecified entries
            seeded[node.idx] = (
                TensorShapeSpec(channels=input_spec.rgb_channels, height=h, width=w),
            )

    return seeded


def _propagate_shapes(graph, input_spec):
    """Propagate tensor shapes through the entire graph using automatic inference."""
    node_outputs = {}
    entry_inputs = _seed_entry_shapes(graph, input_spec)

    for node in graph.nodes:
        # Get input shapes
        if node.idx in entry_inputs:
            input_shapes = entry_inputs[node.idx]
        else:
            # Collect from upstream edges
            shapes = []
            for edge in node.input_edges:
                if edge.node_idx in node_outputs:
                    output_slot_shapes = node_outputs[edge.node_idx]
                    if edge.output_slot < len(output_slot_shapes):
                        shapes.append(output_slot_shapes[edge.output_slot])
            if not shapes:
                upstream_info = [f"layer_{edge.node_idx}[slot_{edge.output_slot}]" for edge in node.input_edges]
                raise RuntimeError(
                    f"Failed to resolve input shapes for node {node.idx} ({node.type_name}). "
                    f"upstream={upstream_info}"
                )
            input_shapes = tuple(shapes)

        # Infer output shapes via automatic propagation
        output_shapes = auto_infer_output_shapes(node, input_shapes)
        node_outputs[node.idx] = output_shapes

        # Debug logging
        if _is_debug():
            _debug_log(
                f"Node {node.idx} ({node.type_name}): "
                f"input={[(s.channels, s.height, s.width) for s in input_shapes]} -> "
                f"output={[(s.channels, s.height, s.width) for s in output_shapes]}"
            )

    return entry_inputs, node_outputs


# ------------------------------------------------------------------
# Stage classification
# ------------------------------------------------------------------

def _get_backbone_len(model) -> int:
    """Return the number of layers in the YAML backbone section.

    Uses model.yaml which preserves the original backbone/head split.
    Returns 0 with a warning if yaml is unavailable — Stages output
    will be unreliable in that case.
    """
    yaml_cfg = getattr(model, "yaml", None)
    if yaml_cfg and isinstance(yaml_cfg, dict) and "backbone" in yaml_cfg:
        return len(yaml_cfg["backbone"])
    LOGGER.warning("model.yaml missing or has no 'backbone' key — Stages breakdown skipped")
    return 0


def _classify_stage(node, backbone_len: int) -> StageKind:
    """Classify a node into its structural stage: backbone, neck, or head.

    Classification is based on the YAML backbone/head section boundary:
    - backbone: all layers in the backbone section (RGB/X backbones, fusion, etc.)
    - neck: non-head layers in the head section (FPN/PAN: Upsample, Concat, C3k2, etc.)
    - head: detection/segmentation/pose heads (Detect, Segment, Pose, OBB, Classification)
    """
    if node.is_head:
        return "head"
    if node.idx < backbone_len:
        return "backbone"
    return "neck"


# ------------------------------------------------------------------
# Main engine
# ------------------------------------------------------------------

def _run_complexity_engine(model, graph, input_spec, backbone_len: int) -> ComplexityReport:
    """Run the full complexity analysis pipeline.

    Shape propagation and FLOPs calculation are fully automatic.
    No per-module-type registration is required.
    """
    entry_inputs, node_outputs = _propagate_shapes(graph, input_spec)

    nodes = []
    for node in graph.nodes:
        # Get input shapes
        if node.idx in entry_inputs:
            input_shapes = entry_inputs[node.idx]
        else:
            shapes = []
            for edge in node.input_edges:
                if edge.node_idx in node_outputs:
                    output_slot_shapes = node_outputs[edge.node_idx]
                    if edge.output_slot < len(output_slot_shapes):
                        shapes.append(output_slot_shapes[edge.output_slot])
            input_shapes = tuple(shapes) if shapes else ()

        # Get output shapes
        output_shapes = node_outputs.get(node.idx, ())

        # Classify stage
        stage = _classify_stage(node, backbone_len)

        # Compute FLOPs automatically
        flops = auto_compute_flops(node, input_shapes, output_shapes)

        # Debug logging
        if _is_debug():
            _debug_log(
                f"Node {node.idx} ({node.type_name}, stage={stage}): "
                f"FLOPs={flops / 1e6:.2f}M"
            )

        nodes.append(
            NodeComplexity(
                node_idx=node.idx,
                type_name=node.type_name,
                stage=stage,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                flops=flops,
            )
        )

    return ComplexityReport(input_spec=input_spec, nodes=nodes)
