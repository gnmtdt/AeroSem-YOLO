# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""YAML-driven structured pruning engine for YOLOMM multimodal models.

This engine directly operates on weight tensors without any third-party pruning
library. It leverages the known YAML-defined layer topology to propagate channel
changes through the network.

Key design:
- Edge-independent proposals: each prunable node generates its own KeepProposal
- Consumer-local coordination: multi-input consumers resolve via contract patterns
- Per-node type dispatch: route-only / single-input / multi-input handlers
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER

from .importance import OUTPUT_IMPORTANCE_SCORERS, compute_importance
from .consumer_ops import adapt_consumer, rebuild_consumer_if_supported
from .contracts import UnsupportedMultiInputConsumerError, require_contract, require_output_spec
from .graph import PruneGraph, build_prune_graph, _MULTI_OUTPUT_TYPES
from .report import LayerExecutionTrace, build_prune_report, save_prune_report
from .trainability import find_frozen_floating_parameters, restore_parameter_trainability


# ------------------------------------------------------------------
# KeepProposal data structure
# ------------------------------------------------------------------


@dataclass
class KeepProposal:
    """Proposal for which output channels to keep in a node.

    Attributes:
        scores: Raw importance scores tensor [out_channels].
        ordered_idx: Channel indices sorted by score (descending) [out_channels].
        n_keep: Number of channels to keep after pruning.
        final_idx: Final keep indices after rounding and min-channels [n_keep].
    """

    scores: torch.Tensor
    ordered_idx: torch.Tensor
    n_keep: int
    final_idx: torch.Tensor


from .ops import (
    _ssa_group_input_indices,
    prune_adown_in,
    prune_adown_out,
    prune_a2c2f_in,
    prune_a2c2f_out,
    prune_bottleneck_csp_in,
    prune_bottleneck_csp_out,
    prune_c2fattn_in,
    prune_c2fattn_out,
    prune_c2f_in,
    prune_c2f_internal,
    prune_c2f_out,
    prune_c2psa_in,
    prune_c2psa_internal,
    prune_c2psa_out,
    prune_c3_in,
    prune_c3_out,
    prune_conv_in,
    prune_conv_out,
    prune_detect_in,
    prune_fcmfeaturefusion_out,
    prune_featurefusion_in,
    prune_featurefusion_out,
    prune_ghostconv_in,
    prune_ghostconv_out,
    prune_mcfgatedfusion_in,
    prune_mcfgatedfusion_out,
    prune_scdown_in,
    prune_scdown_out,
    prune_ssa_in,
    prune_spp_in,
    prune_spp_out,
    prune_sppelan_in,
    prune_sppelan_out,
    prune_sppf_in,
    prune_sppf_internal,
    prune_sppf_out,
)
from .utils import print_prune_summary

# NOTE: Phase 1 replaces pruning complexity with graph-driven engine.
# Training/val/logger complexity helpers in ultralytics.utils.torch_utils
# and multimodal trainers remain intentionally untouched and are tracked
# as Phase 2 integration points.
from ultralytics.nn.mm.complexity import compute_pruning_complexity_report


def _round_to(n: int, divisor: int) -> int:
    """Round n to nearest multiple of divisor, minimum divisor."""
    if divisor <= 1:
        return max(n, 1)
    return max(divisor, round(n / divisor) * divisor)


def _get_layer_type(layer: nn.Module) -> str:
    """Return a canonical type string for a model layer."""
    name = type(layer).__name__
    # Use string comparison to handle modules that share the same class name
    # but live in different namespaces (e.g., C2PSA in block.py vs extraction/)
    if name == "C3k2":
        return "C3k2"
    elif name == "C2f":
        return "C2f"
    elif name == "C2PSA":
        return "C2PSA"
    elif name == "SPPF":
        return "SPPF"
    elif name == "SPP":
        return "SPP"
    elif name == "GhostConv":
        return "GhostConv"
    elif name == "C2fAttn":
        return "C2fAttn"
    elif name == "A2C2f":
        return "A2C2f"
    elif name == "SCDown":
        return "SCDown"
    elif name == "AConv":
        return "AConv"
    elif name == "Conv":
        return "Conv"
    elif name == "Detect":
        return "Detect"
    elif isinstance(layer, nn.Upsample):
        return "Upsample"
    elif name == "Concat":
        return "Concat"
    elif name in ("C3", "C3x", "C3Ghost", "RepC3"):
        return "C3"
    elif name == "SequenceShuffleAttention":
        return "SequenceShuffleAttention"
    elif name == "BottleneckCSP":
        return "BottleneckCSP"
    elif name == "ADown":
        return "ADown"
    elif name == "SPPELAN":
        return "SPPELAN"
    elif name in ("FeatureFusion", "FCMFeatureFusion", "MCFGatedFusion"):
        return name  # Return type name as-is
    LOGGER.warning(f"[Prune] Unrecognized layer type '{name}' at this layer, skipping pruning")
    return name


def _get_out_channels(layer: nn.Module, ltype: str) -> int:
    """Get the output channel count of a layer."""
    if ltype == "Conv":
        return layer.conv.out_channels
    elif ltype in ("C3k2", "C2f", "C3"):
        return layer.cv2.conv.out_channels
    elif ltype == "C2PSA":
        return layer.cv2.conv.out_channels
    elif ltype == "SPPF":
        return layer.cv2.conv.out_channels
    elif ltype == "SPP":
        return layer.cv2.conv.out_channels
    elif ltype == "GhostConv":
        # GhostConv output is cat(cv1, cv2), both output c_ = c2//2
        # So output channels = 2 * cv1.out_channels = c2
        return layer.cv2.conv.out_channels * 2
    elif ltype == "C2fAttn":
        return layer.cv2.conv.out_channels
    elif ltype == "A2C2f":
        return layer.cv2.conv.out_channels
    elif ltype == "SCDown":
        return layer.cv2.conv.out_channels
    elif ltype == "AConv":
        # AConv is just a Conv inside
        return layer.cv1.conv.out_channels
    elif ltype == "Detect":
        return -1  # Not applicable
    elif ltype == "Upsample":
        return -1  # Pass-through
    elif ltype == "Concat":
        return -1  # Computed from sources
    elif ltype == "BottleneckCSP":
        return layer.cv4.conv.out_channels
    elif ltype == "ADown":
        return layer.cv1.conv.out_channels + layer.cv2.conv.out_channels
    elif ltype == "SPPELAN":
        return layer.cv5.conv.out_channels
    elif ltype == "FeatureFusion":
        # FeatureFusion output channels come from channel_emb (dim)
        channel_emb = getattr(layer, "channel_emb", None)
        if channel_emb is not None:
            return getattr(channel_emb, "out_channels", 0) or 0
        return 0
    elif ltype == "FCMFeatureFusion":
        # FCMFeatureFusion delegates to inner FeatureFusion
        ffm = getattr(layer, "ffm", None)
        if ffm is not None:
            return _get_out_channels(ffm, "FeatureFusion")
        return 0
    elif ltype == "MCFGatedFusion":
        # MCFGatedFusion: output = main channels (gate output in add mode, post output in concat mode)
        post = getattr(layer, "post", None)
        if post is not None:
            return getattr(post, "conv", None).out_channels if hasattr(post, "conv") else 0
        # add mode: output = main input channels
        return getattr(layer, "gate", None).out_channels if hasattr(layer, "gate") else 0
    elif ltype == "SequenceShuffleAttention":
        gating = getattr(layer, "gating", None)
        if gating is not None:
            for l in gating:
                if isinstance(l, nn.Conv2d):
                    return l.out_channels
        return 0
    return -1


class YAMLPruneEngine:
    """YAML-driven structured pruning engine for YOLOMM.

    Traverses the model layer-by-layer, computes importance scores, selects
    channels to keep, prunes weights in-place, and propagates channel changes
    to downstream layers.

    Args:
        model: The nn.Module to prune (typically model.model from YOLOMM).
        method: Importance scoring method ('l1', 'l2', 'lamp', 'bn', 'random').
        ratio: Target pruning ratio in (0.0, 1.0).
        imgsz: Input image size for verification and GFLOPs.
        round_to: Round kept channels to multiples of this value.
        min_ch: Minimum channels to keep per layer.
        prune_types: List of layer types to prune. If None, prune all supported.
        skip_types: List of layer types to skip. Takes precedence over prune_types.
        save_report: Structured pruning report output control.
    """

    SUPPORTED_METHODS = ("l1", "l2", "lamp", "bn", "random")

    def __init__(
        self,
        model: nn.Module,
        method: str = "l1",
        ratio: float = 0.3,
        imgsz: int = 640,
        round_to: int = 8,
        min_ch: int = 8,
        prune_types: Optional[List[str]] = None,
        skip_types: Optional[List[str]] = None,
        save_report: Union[bool, str, Path] = False,
    ):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method '{method}'. Choose from: {self.SUPPORTED_METHODS}")
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"Pruning ratio must be in (0.0, 1.0), got {ratio}")

        self.model = model
        self.method = method
        self.ratio = ratio
        self.imgsz = imgsz
        self.round_to = round_to
        self.min_ch = min_ch
        self.prune_types = prune_types
        self.skip_types = skip_types if skip_types else []
        self.save_report = save_report

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, save_dir: Optional[str] = None) -> nn.Module:
        """Execute the full pruning pipeline.

        1. Build the pruning graph from the deserialized model object.
        2. Compute edge-independent keep proposals.
        3. Resolve multi-input consumers via explicit contracts.
        4. Apply pruning layer-by-layer with graph-aware propagation.
        5. Verify forward pass.
        6. Normalize trainability (restore frozen floating parameters).
        7. Optionally save.

        The returned model is guaranteed to have all floating-point parameters
        in trainable state (requires_grad=True). Runtime freeze policies
        (e.g., DFL) remain owned by the trainer, not encoded in the checkpoint.

        Returns:
            The pruned model (same object, modified in-place).
        """
        model = self.model
        params_before = sum(p.numel() for p in model.parameters())

        if self.save_report is True and save_dir is None:
            raise ValueError("save_report=True 时必须同时提供 save_dir。")

        graph_before = build_prune_graph(model)
        report_before = compute_pruning_complexity_report(model, imgsz=self.imgsz)
        LOGGER.info(f"[Prune] {len(graph_before.nodes)} graph nodes, method={self.method}, ratio={self.ratio}")

        proposals = self._compute_keep_proposals(graph_before)
        LOGGER.info(f"[Prune] {len(proposals)} keep proposals")

        _, trace_map = self._apply_pruning_new(graph_before, proposals)

        # 5. Verify
        self._verify_forward(model)

        # 6. Normalize trainability: restore any frozen floating parameters.
        #    This removes residual EMA frozen states from the source model.
        restored = restore_parameter_trainability(model)
        remaining = find_frozen_floating_parameters(model)
        if remaining:
            raise RuntimeError(
                "Pruned model still contains frozen floating parameters after normalization: "
                + ", ".join(remaining[:20])
            )
        if restored:
            LOGGER.info(
                f"[Prune] Restored trainability for {len(restored)} floating parameters "
                "before returning/saving the pruned model."
            )

        # 7. Summary
        graph_after = build_prune_graph(model)
        report_after = compute_pruning_complexity_report(model, imgsz=self.imgsz)
        params_after = sum(p.numel() for p in model.parameters())
        print_prune_summary(
            params_before=params_before,
            pruned_model=model,
            report_before=report_before,
            report_after=report_after,
        )

        report_path = self._resolve_report_output_path(save_dir)
        has_report = report_path is not None
        report_relpath = self._relative_to_save_dir(report_path, save_dir) if report_path is not None else None
        if report_path is not None:
            prune_meta = self._collect_prune_metadata(model, graph_after)
            report_payload = build_prune_report(
                graph_before=graph_before,
                graph_after=graph_after,
                proposals=proposals,
                trace_map=trace_map,
                report_before=report_before,
                report_after=report_after,
                params_before=params_before,
                params_after=params_after,
                meta={
                    **prune_meta,
                    "prune_method": self.method,
                    "prune_ratio": float(self.ratio),
                    "imgsz": int(self.imgsz),
                    "round_to": int(self.round_to),
                    "min_ch": int(self.min_ch),
                    "prune_types": list(self.prune_types) if self.prune_types is not None else None,
                    "skip_types": list(self.skip_types),
                },
            )
            saved_report = save_prune_report(report_payload, report_path)
            LOGGER.info(f"[Prune] Structured report saved to {saved_report}")

        # 8. Save
        if save_dir is not None:
            self._save(model, save_dir, has_report=has_report, report_relpath=report_relpath)

        return model

    # ------------------------------------------------------------------
    # Topology analysis
    # ------------------------------------------------------------------

    def _build_topology(self) -> List[dict]:
        """Build topology info for each layer in model.model.

        Returns:
            List of dicts with keys: idx, layer, type, from_, out_ch.
        """
        topo = []
        for i, layer in enumerate(self.model.model):
            f = layer.f if hasattr(layer, "f") else -1
            ltype = _get_layer_type(layer)

            # Apply prune_types/skip_types filtering
            if self.skip_types and ltype in self.skip_types:
                continue
            if self.prune_types is not None and ltype not in self.prune_types:
                continue

            out_ch = _get_out_channels(layer, ltype)
            topo.append({
                "idx": i,
                "layer": layer,
                "type": ltype,
                "from_": f,
                "out_ch": out_ch,
            })
        return topo

    def _resolve_out_channels(self, topo: List[dict]) -> List[int]:
        """Resolve actual output channels for every layer, handling Concat/Upsample."""
        n = len(topo)
        ch = [0] * n
        for i, info in enumerate(topo):
            ltype = info["type"]
            if ltype == "Concat":
                sources = info["from_"]
                if isinstance(sources, (list, tuple)):
                    ch[i] = sum(ch[self._abs_idx(s, i)] for s in sources)
                else:
                    ch[i] = ch[self._abs_idx(sources, i)]
            elif ltype == "Upsample":
                src = info["from_"]
                src_idx = self._abs_idx(src if not isinstance(src, (list, tuple)) else src[0], i)
                ch[i] = ch[src_idx]
            elif info["out_ch"] > 0:
                ch[i] = info["out_ch"]
            else:
                # Detect or unknown - use previous
                src = info["from_"]
                if isinstance(src, (list, tuple)):
                    ch[i] = sum(ch[self._abs_idx(s, i)] for s in src)
                else:
                    src_idx = self._abs_idx(src, i)
                    ch[i] = ch[src_idx]
        return ch

    @staticmethod
    def _abs_idx(rel: int, cur: int) -> int:
        """Convert relative index to absolute."""
        return rel if rel >= 0 else cur + rel

    # ------------------------------------------------------------------
    # Prunable layers and symmetric pairs
    # ------------------------------------------------------------------

    def _prunable_layers(self, topo: List[dict]) -> Set[int]:
        """Return indices of layers that can be pruned.

        Excludes: Detect, Upsample, Concat.
        """
        prunable = set()
        for info in topo:
            ltype = info["type"]
            if ltype in ("Conv", "C3k2", "C2f", "SPPF", "SPP", "GhostConv", "C2fAttn", "A2C2f", "SCDown", "AConv", "C2PSA", "C3", "BottleneckCSP", "ADown", "SPPELAN", "FeatureFusion", "FCMFeatureFusion", "MCFGatedFusion"):
                prunable.add(info["idx"])
        return prunable

    def _symmetric_pairs(self, topo: List[dict]) -> List[Tuple[int, int]]:
        """Identify RGB/X symmetric layer pairs from multimodal input markers.

        In the standard mid-fusion YAML, layers 0-10 are RGB backbone and
        layers 11-21 are X backbone with identical structure.
        """
        pairs = []
        # Find layers with 'RGB' and 'X' input markers
        rgb_start = None
        x_start = None

        for info in topo:
            layer = info["layer"]
            idx = info["idx"]
            mm_input = getattr(layer, "_mm_input_source", None)
            if mm_input == "RGB" and rgb_start is None:
                rgb_start = idx
            elif mm_input == "X" and x_start is None:
                x_start = idx

        # If no explicit markers, detect from entry layers
        if rgb_start is None or x_start is None:
            entry = self._find_entry_layers(topo)
            sorted_entry = sorted(entry)
            if len(sorted_entry) >= 2:
                rgb_start = sorted_entry[0]
                x_start = sorted_entry[1]
            else:
                return []

        # Determine backbone length: from x_start to the first Concat that
        # references a layer before x_start
        backbone_len = x_start - rgb_start
        for offset in range(backbone_len):
            rgb_idx = rgb_start + offset
            x_idx = x_start + offset
            if rgb_idx < len(topo) and x_idx < len(topo):
                rt = topo[rgb_idx]["type"]
                xt = topo[x_idx]["type"]
                if rt == xt and rt in ("Conv", "C3k2", "C2f", "SPPF", "SPP", "GhostConv", "C2fAttn", "A2C2f", "SCDown", "AConv", "C2PSA", "C3", "BottleneckCSP", "ADown", "SPPELAN"):
                    pairs.append((rgb_idx, x_idx))
        return pairs

    # ------------------------------------------------------------------
    # Importance & keep indices
    # ------------------------------------------------------------------

    def _compute_keep_indices(
        self, topo: List[dict], prunable: Set[int], sym_pairs: List[Tuple[int, int]]
    ) -> Dict[int, torch.Tensor]:
        """Compute which output channels to keep for each prunable layer.

        For symmetric pairs (RGB/X), importance scores are merged and the
        same indices are used for both layers.

        Returns:
            Dict mapping layer_idx -> 1-D tensor of sorted keep indices.
        """
        keep = {}
        sym_map = {}  # idx -> partner_idx
        for r, x in sym_pairs:
            sym_map[r] = x
            sym_map[x] = r

        processed = set()

        for idx in sorted(prunable):
            if idx in processed:
                continue

            layer = topo[idx]["layer"]
            ltype = topo[idx]["type"]
            out_ch = _get_out_channels(layer, ltype)
            if out_ch <= 0:
                continue

            # C2PSA constraint: c1 == c2, skip output pruning
            # (will be handled via input propagation)
            if ltype == "C2PSA":
                continue

            if idx in sym_map:
                # Symmetric pair: merge importance
                partner = sym_map[idx]
                scores_a = compute_importance(layer, self.method)
                scores_b = compute_importance(topo[partner]["layer"], self.method)
                # Ensure same size
                min_len = min(len(scores_a), len(scores_b))
                scores = scores_a[:min_len] + scores_b[:min_len]
                n_keep = self._calc_n_keep(min_len)
                _, indices = scores.topk(n_keep)
                keep_idx = indices.sort().values
                keep[idx] = keep_idx
                keep[partner] = keep_idx
                processed.add(idx)
                processed.add(partner)
            else:
                scores = compute_importance(layer, self.method)
                n_keep = self._calc_n_keep(len(scores))
                _, indices = scores.topk(n_keep)
                keep[idx] = indices.sort().values
                processed.add(idx)

        return keep

    def _calc_n_keep(self, total: int) -> int:
        """Calculate number of channels to keep."""
        n = max(self.min_ch, round(total * (1 - self.ratio)))
        n = _round_to(n, self.round_to)
        return min(n, total)

    # ------------------------------------------------------------------
    # Edge-independent keep proposals
    # ------------------------------------------------------------------

    def _compute_keep_proposals(self, graph: PruneGraph) -> Dict[int, KeepProposal]:
        """Compute one KeepProposal per prunable single-output node."""
        proposals: Dict[int, KeepProposal] = {}

        for node in graph.prunable_nodes():
            if self.skip_types and node.type_name in self.skip_types:
                continue
            if self.prune_types is not None and node.type_name not in self.prune_types:
                continue

            if node.type_name in _MULTI_OUTPUT_TYPES:
                require_output_spec(node)
                continue

            out_ch = node.primary_out_channels
            if out_ch <= 0 or node.type_name in {"C2PSA", "SequenceShuffleAttention"}:
                continue

            scorer = OUTPUT_IMPORTANCE_SCORERS.get(node.type_name)
            if scorer is not None:
                scores = scorer(node.module, self.method)
            else:
                scores = compute_importance(node.module, self.method)

            if len(scores) != out_ch:
                raise ValueError(
                    f"[Prune] Score length mismatch at idx={node.idx} ({node.type_name}): "
                    f"scores={len(scores)} vs out_ch={out_ch}"
                )

            n_keep = self._calc_n_keep(len(scores))
            ordered_idx = scores.argsort(descending=True)
            final_idx = ordered_idx[:n_keep].sort().values
            proposals[node.idx] = KeepProposal(scores=scores, ordered_idx=ordered_idx, n_keep=n_keep, final_idx=final_idx)

        return self._apply_ssa_consumer_constraints(graph, proposals)

    @staticmethod
    def _trace_single_source_producer(graph: PruneGraph, edge) -> tuple:
        """Trace through route-only nodes to locate the single concrete producer for SSA."""
        visited: set[tuple[int, int]] = set()
        current_edge = edge
        while True:
            key = (current_edge.node_idx, current_edge.output_slot)
            if key in visited:
                raise ValueError(f"[Prune] Detected route cycle while resolving SSA source from edge {key}")
            visited.add(key)

            node = graph.node(current_edge.node_idx)
            if not node.is_route_only:
                return current_edge, node

            if node.type_name == "Concat" or len(node.input_edges) != 1:
                raise ValueError(
                    f"[Prune] SSA requires a single concrete producer, but source node {node.idx} ({node.type_name}) "
                    "is a multi-input route-only node"
                )
            current_edge = node.input_edges[0]

    @staticmethod
    def _build_constrained_ordered_idx(
        scores: torch.Tensor,
        original_ordered_idx: torch.Tensor,
        selected_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Move constrained-selected channels to the front while preserving remaining order."""
        keep_mask = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
        keep_mask[selected_idx] = True
        selected_ordered = selected_idx[scores[selected_idx].argsort(descending=True)]
        remaining = original_ordered_idx[~keep_mask[original_ordered_idx]]
        return torch.cat([selected_ordered, remaining], dim=0)

    def _constrain_proposal_for_ssa(
        self,
        proposal: KeepProposal,
        producer_node,
        ssa_node,
    ) -> KeepProposal:
        """Constrain an upstream proposal so it remains legal for SSA shuffled groups."""
        group = int(getattr(ssa_node.module, "group", 0) or 0)
        total_channels = int(producer_node.primary_out_channels)
        if group <= 0:
            raise ValueError(f"[Prune] SSA node {ssa_node.idx} has invalid group={group}")
        if total_channels % group != 0:
            raise ValueError(
                f"[Prune] SSA node {ssa_node.idx} expects input channels divisible by group: "
                f"producer={producer_node.idx}, channels={total_channels}, group={group}"
            )
        if proposal.n_keep % group != 0:
            raise ValueError(
                f"[Prune] SSA consumer constraint cannot be satisfied for producer {producer_node.idx} -> "
                f"SSA {ssa_node.idx}: n_keep={proposal.n_keep} is not divisible by group={group}. "
                "Adjust ratio/round_to so kept channels align with SSA groups."
            )

        groups = _ssa_group_input_indices(total_channels, group, proposal.scores.device)
        keep_per_group = proposal.n_keep // group
        selected_parts = []
        for original_group in groups:
            group_scores = proposal.scores[original_group]
            local_rank = group_scores.argsort(descending=True)[:keep_per_group]
            selected_parts.append(original_group[local_rank])

        selected_idx = torch.cat(selected_parts, dim=0)
        final_idx = selected_idx.sort().values
        ordered_idx = self._build_constrained_ordered_idx(proposal.scores, proposal.ordered_idx, final_idx)
        return KeepProposal(scores=proposal.scores, ordered_idx=ordered_idx, n_keep=proposal.n_keep, final_idx=final_idx)

    def _apply_ssa_consumer_constraints(
        self,
        graph: PruneGraph,
        proposals: Dict[int, KeepProposal],
    ) -> Dict[int, KeepProposal]:
        """Re-shape upstream proposals so SSA inputs remain shuffle/group legal."""
        constrained = dict(proposals)
        seen_constraints: dict[int, tuple[int, int]] = {}

        for node in graph.nodes:
            if node.type_name != "SequenceShuffleAttention" or not node.input_edges:
                continue

            resolved_edge, producer_node = self._trace_single_source_producer(graph, node.input_edges[0])
            if resolved_edge.output_slot != 0:
                raise ValueError(
                    f"[Prune] SSA node {node.idx} is fed from output_slot={resolved_edge.output_slot} of "
                    f"producer {producer_node.idx}; only primary-slot SSA inputs are currently supported"
                )

            proposal = constrained.get(producer_node.idx)
            if proposal is None:
                continue

            group = int(getattr(node.module, "group", 0) or 0)
            constraint_key = (group, proposal.n_keep)
            previous = seen_constraints.get(producer_node.idx)
            if previous is not None and previous != constraint_key:
                raise ValueError(
                    f"[Prune] Producer {producer_node.idx} is constrained by multiple SSA nodes with "
                    f"incompatible requirements: previous={previous}, current={constraint_key}"
                )

            new_proposal = self._constrain_proposal_for_ssa(proposal, producer_node, node)
            if not torch.equal(new_proposal.final_idx, proposal.final_idx):
                LOGGER.info(
                    "[Prune] Applied SSA group-aware keep constraint: producer %d -> SSA %d, keep=%d, group=%d",
                    producer_node.idx,
                    node.idx,
                    proposal.n_keep,
                    group,
                )
            constrained[producer_node.idx] = new_proposal
            seen_constraints[producer_node.idx] = constraint_key

        return constrained

    def _graph_input_channels(self, graph: PruneGraph) -> int:
        """Detect input channels from graph entry nodes."""
        total = 0
        for node in graph.nodes:
            if not node.is_entry:
                continue
            if node.branch_kind == "rgb":
                total += 3
            elif node.branch_kind == "x":
                total += max(node.in_channels, 0) or 3
            elif node.branch_kind == "dual":
                total += max(node.in_channels, 0) or 6
            elif node.in_channels > 0:
                total += node.in_channels
        return total or 6

    @staticmethod
    def _full_keep(width: int, device: torch.device) -> torch.Tensor:
        return torch.arange(width, device=device)

    def _resolve_node_slot_keeps(
        self,
        graph: PruneGraph,
        node_idx: int,
        keep_map: Dict[int, Tuple[torch.Tensor, ...]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, ...]:
        slot_keeps = keep_map.get(node_idx)
        if slot_keeps is not None:
            return slot_keeps
        node = graph.node(node_idx)
        return tuple(self._full_keep(width, device) for width in node.out_channels)

    def _resolve_edge_keep(
        self,
        graph: PruneGraph,
        edge,
        keep_map: Dict[int, Tuple[torch.Tensor, ...]],
        device: torch.device,
    ) -> torch.Tensor:
        slot_keeps = self._resolve_node_slot_keeps(graph, edge.node_idx, keep_map, device)
        if edge.output_slot >= len(slot_keeps):
            raise ValueError(
                f"Edge from node {edge.node_idx} requests slot {edge.output_slot}, "
                f"but only {len(slot_keeps)} slot(s) are available"
            )
        return slot_keeps[edge.output_slot]

    def _shrink_edge_keep(
        self,
        graph: PruneGraph,
        edge,
        target: int,
        proposals: Dict[int, KeepProposal],
        keep_map: Dict[int, Tuple[torch.Tensor, ...]],
        device: torch.device,
    ) -> torch.Tensor:
        node = graph.node(edge.node_idx)
        if node.type_name in _MULTI_OUTPUT_TYPES or node.is_route_only:
            return self._resolve_edge_keep(graph, edge, keep_map, device)[:target]

        proposal = proposals.get(edge.node_idx)
        if proposal is None:
            return self._resolve_edge_keep(graph, edge, keep_map, device)[:target]
        return proposal.ordered_idx[:target].sort().values.to(device)

    @staticmethod
    def _concat_output_keep(input_keeps: List[torch.Tensor]) -> torch.Tensor:
        parts = []
        offset = 0
        for keep in input_keeps:
            parts.append(keep + offset)
            offset += len(keep)
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)

    def _resolve_consumer_inputs(
        self,
        graph: PruneGraph,
        node,
        proposals: Dict[int, KeepProposal],
        keep_map: Dict[int, Tuple[torch.Tensor, ...]],
        output_keep: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        contract = require_contract(node)
        input_keeps = [self._resolve_edge_keep(graph, edge, keep_map, device) for edge in node.input_edges]

        if contract.mode == "concat_like":
            return input_keeps, None

        if contract.mode == "detect_head":
            return input_keeps, None

        if contract.mode == "equal_width_left_output":
            target = min(len(k) for k in input_keeps)
            resolved = [
                self._shrink_edge_keep(graph, edge, target, proposals, keep_map, device)
                for edge in node.input_edges
            ]
            return resolved, resolved[0]

        if contract.mode == "left_output":
            return input_keeps, input_keeps[0]

        if contract.mode == "declared_output":
            if output_keep is None:
                raise ValueError(f"{node.type_name} requires an explicit output keep proposal")
            return input_keeps, output_keep

        raise RuntimeError(f"Unhandled contract mode: {contract.mode}")

    def _slot_keeps_from_output_spec(
        self,
        node,
        input_keeps: List[torch.Tensor],
        output_keep: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        spec = require_output_spec(node)
        slot_count = spec.output_slots or len(node.out_channels)
        if spec.layout == "tuple_same_width":
            if len(input_keeps) >= slot_count:
                return tuple(input_keeps[i].clone() for i in range(slot_count))
            return tuple(output_keep.clone() for _ in range(slot_count))
        if spec.layout == "tuple_declared_width":
            if len(input_keeps) >= slot_count and all(len(input_keeps[i]) == node.out_channels[i] for i in range(slot_count)):
                return tuple(input_keeps[i].clone() for i in range(slot_count))
            device = output_keep.device
            return tuple(self._full_keep(width, device) for width in node.out_channels[:slot_count])
        return (output_keep,)

    @staticmethod
    def _tensor_keep_to_list(keep: torch.Tensor) -> List[int]:
        return [int(x) for x in keep.detach().cpu().tolist()]

    def _slot_keeps_to_lists(self, slot_keeps: Tuple[torch.Tensor, ...]) -> List[List[int]]:
        return [self._tensor_keep_to_list(keep) for keep in slot_keeps]

    def _input_keep_records(self, node, input_keeps: List[torch.Tensor]) -> List[dict]:
        records = []
        for edge, keep in zip(node.input_edges, input_keeps):
            records.append(
                {
                    "node_idx": int(edge.node_idx),
                    "output_slot": int(edge.output_slot),
                    "keep_idx": self._tensor_keep_to_list(keep),
                }
            )
        return records

    @staticmethod
    def _consumer_decision_reason(mode: str) -> str:
        mapping = {
            "equal_width_left_output": "consumer_equal_width_contract",
            "left_output": "consumer_left_output_contract",
            "declared_output": "consumer_declared_output_contract",
        }
        return mapping.get(mode, "consumer_contract_applied")

    def _collect_prune_metadata(self, model: nn.Module, graph: Optional[PruneGraph] = None) -> dict:
        router = getattr(model, "multimodal_router", None) or getattr(model, "mm_router", None)
        graph = graph or build_prune_graph(model)
        yaml_meta = getattr(model, "yaml", None)
        yaml_file = yaml_meta.get("yaml_file", "") if isinstance(yaml_meta, dict) else ""
        source_weights = getattr(model, "pt_path", "") or getattr(model, "yaml_file", "") or yaml_file
        x_modality = getattr(router, "x_modality_type", "unknown") if router is not None else "unknown"
        if router is not None and hasattr(router, "INPUT_SOURCES"):
            x_channels = int(router.INPUT_SOURCES.get("SECONDARY", 3))
        else:
            x_channels = max(self._graph_input_channels(graph) - 3, 0)
        return {
            "source_weights": str(source_weights),
            "x_modality": str(x_modality),
            "x_channels": int(x_channels),
        }

    def _resolve_report_output_path(self, save_dir: Optional[str]) -> Optional[Path]:
        if self.save_report is False:
            return None
        if self.save_report is True:
            if save_dir is None:
                raise ValueError("save_report=True 时必须同时提供 save_dir。")
            return Path(save_dir) / "prune_report.json"
        if isinstance(self.save_report, (str, Path)):
            report_path = Path(self.save_report)
            if not str(report_path).strip():
                raise ValueError("save_report 显式路径不能为空。")
            return report_path
        raise TypeError("save_report 仅支持 False、True、str 或 pathlib.Path。")

    @staticmethod
    def _relative_to_save_dir(report_path: Optional[Path], save_dir: Optional[str]) -> Optional[str]:
        if report_path is None or save_dir is None:
            return None
        try:
            return str(report_path.resolve().relative_to(Path(save_dir).resolve()))
        except ValueError:
            return None

    def _apply_consumer_update(self, node, input_keeps: List[torch.Tensor], output_keep: Optional[torch.Tensor]):
        input_channels = [len(k) for k in input_keeps]
        if rebuild_consumer_if_supported(node.type_name, node.module, input_channels):
            return

        if len(input_keeps) != 2:
            raise UnsupportedMultiInputConsumerError(
                f"{node.type_name} currently expects 2 input keeps, got {len(input_keeps)} at node {node.idx}"
            )

        if not adapt_consumer(node.type_name, node.module, input_keeps[0], input_keeps[1], output_keep):
            raise UnsupportedMultiInputConsumerError(
                f"No safe consumer adapter registered for '{node.type_name}' at node {node.idx}"
            )

    def _prune_detect_inputs_graph(
        self,
        graph: PruneGraph,
        node,
        keep_map: Dict[int, Tuple[torch.Tensor, ...]],
        device: torch.device,
    ):
        for scale_idx, edge in enumerate(node.input_edges):
            keep_idx = self._resolve_edge_keep(graph, edge, keep_map, device)
            prune_detect_in(node.module, scale_idx, keep_idx)

    def _apply_pruning_new(self, graph: PruneGraph, proposals: Dict[int, KeepProposal]):
        """Apply pruning using the real graph and explicit consumer contracts."""
        keep_map: Dict[int, Tuple[torch.Tensor, ...]] = {}
        trace_map: Dict[int, LayerExecutionTrace] = {}
        device = next(self.model.parameters()).device

        for node in graph.nodes:
            ltype = node.type_name

            if node.is_route_only:
                if ltype == "Concat":
                    input_keeps = [self._resolve_edge_keep(graph, edge, keep_map, device) for edge in node.input_edges]
                    keep_map[node.idx] = (self._concat_output_keep(input_keeps),)
                elif node.input_edges:
                    input_keeps = [self._resolve_edge_keep(graph, node.input_edges[0], keep_map, device)]
                    keep_map[node.idx] = (input_keeps[0].clone(),)
                else:
                    input_keeps = []
                    keep_map[node.idx] = (self._full_keep(node.primary_out_channels, device),)
                trace_map[node.idx] = LayerExecutionTrace(
                    node_idx=node.idx,
                    decision_reason="route_passthrough",
                    input_keep_by_edge=self._input_keep_records(node, input_keeps),
                    resolved_output_slots=self._slot_keeps_to_lists(keep_map[node.idx]),
                )
                continue

            if node.is_head:
                input_keeps = [self._resolve_edge_keep(graph, edge, keep_map, device) for edge in node.input_edges]
                self._prune_detect_inputs_graph(graph, node, keep_map, device)
                keep_map[node.idx] = tuple(self._full_keep(width, device) for width in node.out_channels)
                head_input_updated = any(
                    len(keep) != graph.node(edge.node_idx).out_channels[edge.output_slot]
                    for edge, keep in zip(node.input_edges, input_keeps)
                )
                trace_map[node.idx] = LayerExecutionTrace(
                    node_idx=node.idx,
                    decision_reason="head_input_passthrough",
                    input_keep_by_edge=self._input_keep_records(node, input_keeps),
                    resolved_output_slots=self._slot_keeps_to_lists(keep_map[node.idx]),
                    head_input_updated=head_input_updated,
                )
                continue

            proposal = proposals.get(node.idx)
            output_keep = proposal.final_idx.to(device) if proposal is not None else self._full_keep(node.primary_out_channels, device)

            if node.is_multi_input:
                input_keeps, resolved_output_keep = self._resolve_consumer_inputs(
                    graph, node, proposals, keep_map, output_keep, device
                )
                self._apply_consumer_update(node, input_keeps, resolved_output_keep)
                if node.type_name in _MULTI_OUTPUT_TYPES:
                    keep_map[node.idx] = self._slot_keeps_from_output_spec(node, input_keeps, resolved_output_keep)
                else:
                    keep_map[node.idx] = (resolved_output_keep,)
                trace_map[node.idx] = LayerExecutionTrace(
                    node_idx=node.idx,
                    decision_reason=self._consumer_decision_reason(require_contract(node).mode),
                    input_keep_by_edge=self._input_keep_records(node, input_keeps),
                    resolved_output_slots=self._slot_keeps_to_lists(keep_map[node.idx]),
                    consumer_coordinated=True,
                )
                continue

            source_keep = None
            input_keeps: List[torch.Tensor] = []
            if node.input_edges:
                source_keep = self._resolve_edge_keep(graph, node.input_edges[0], keep_map, device)
                input_keeps = [source_keep]

            if proposal is not None:
                self._prune_layer_output(node.module, ltype, output_keep)

            if (
                not node.is_entry
                and source_keep is not None
                and len(source_keep) != graph.node(node.input_edges[0].node_idx).out_channels[node.input_edges[0].output_slot]
            ):
                self._prune_layer_input(node.module, ltype, source_keep)

            if ltype == "C2PSA" and source_keep is not None and len(source_keep) != node.primary_out_channels:
                output_keep = self._full_keep(len(source_keep), device)
                prune_c2psa_out(node.module, output_keep)

            if ltype == "SequenceShuffleAttention" and source_keep is not None and len(source_keep) != node.primary_out_channels:
                output_keep = self._full_keep(len(source_keep), device)

            input_constrained_reset = (
                proposal is None
                and source_keep is not None
                and len(source_keep) != node.primary_out_channels
                and ltype in {"C2PSA", "SequenceShuffleAttention"}
            )

            keep_map[node.idx] = (output_keep,)
            trace_map[node.idx] = LayerExecutionTrace(
                node_idx=node.idx,
                decision_reason=(
                    "input_constrained_output_reset"
                    if input_constrained_reset
                    else "self_proposal_applied" if proposal is not None else "non_prunable_passthrough"
                ),
                input_keep_by_edge=self._input_keep_records(node, input_keeps),
                resolved_output_slots=self._slot_keeps_to_lists(keep_map[node.idx]),
            )

        return keep_map, trace_map

    def _apply_pruning(self, topo: List[dict], keep: Dict[int, torch.Tensor]):
        """Apply pruning layer-by-layer, propagating channel changes.

        Maintains `out_ch_map[i]` = actual output channel count after pruning
        for each layer, and `keep_map[i]` = the keep indices (or full range
        if not pruned).

        NOTE: This is the legacy method kept for backward compatibility.
        The new implementation uses _apply_pruning_new with edge-independent
        proposals and consumer-local coordination.
        """
        # Current output channels per layer (before pruning)
        ch = self._resolve_out_channels(topo)

        # Identify modal entry layers (input from image, not from other layers)
        entry_layers = self._find_entry_layers(topo)

        # Track output channel info after pruning
        out_ch_after = list(ch)  # will be updated
        keep_map: Dict[int, torch.Tensor] = {}

        device = next(self.model.parameters()).device

        for i, info in enumerate(topo):
            layer = info["layer"]
            ltype = info["type"]
            f = info["from_"]

            # --- Step 1: Prune output channels if this layer is in keep ---
            if i in keep:
                keep_idx = keep[i].to(device)
                self._prune_layer_output(layer, ltype, keep_idx)
                out_ch_after[i] = len(keep_idx)
                keep_map[i] = keep_idx
            else:
                keep_map[i] = torch.arange(out_ch_after[i], device=device)

            # --- Step 2: Prune input channels based on upstream changes ---
            if ltype in ("Upsample", "Concat", "Detect"):
                if ltype == "Concat":
                    sources = f if isinstance(f, (list, tuple)) else [f]
                    total = sum(out_ch_after[self._abs_idx(s, i)] for s in sources)
                    out_ch_after[i] = total
                    keep_parts = []
                    offset = 0
                    for s in sources:
                        src_idx = self._abs_idx(s, i)
                        keep_parts.append(torch.arange(out_ch_after[src_idx], device=device) + offset)
                        offset += out_ch_after[src_idx]
                    keep_map[i] = torch.cat(keep_parts)
                elif ltype == "Upsample":
                    src_idx = self._abs_idx(f if not isinstance(f, (list, tuple)) else f[0], i)
                    out_ch_after[i] = out_ch_after[src_idx]
                    keep_map[i] = torch.arange(out_ch_after[i], device=device)
                elif ltype == "Detect":
                    self._prune_detect_inputs(layer, topo, i, out_ch_after)
                continue

            # Skip input pruning for modal entry layers (input from image)
            if i in entry_layers:
                continue

            # Check if upstream channels changed
            src_indices = self._get_input_indices(i, f, topo, out_ch_after, ch, keep_map, device)
            if src_indices is not None:
                self._prune_layer_input(layer, ltype, src_indices)

            # For C2PSA: if input changed, output must match (c1==c2 constraint)
            # Strategy: only adjust cv1 input and cv2 output, keep internal
            # hidden channels (c) and PSABlock/Attention unchanged.
            # cv1: new_in -> 2*c (input changes, output stays)
            # m: PSABlock(c) unchanged
            # cv2: 2*c -> new_in (input stays, output changes)
            if ltype == "C2PSA":
                new_in = self._get_input_ch_count(i, f, out_ch_after, topo)
                if new_in != ch[i]:
                    keep_out = torch.arange(new_in, device=device)
                    prune_c2psa_out(layer, keep_out)
                    out_ch_after[i] = new_in
                    keep_map[i] = torch.arange(new_in, device=device)

    def _prune_layer_output(self, layer: nn.Module, ltype: str, keep_idx: torch.Tensor):
        """Prune output channels of a layer."""
        if ltype == "Conv":
            prune_conv_out(layer, keep_idx)
        elif ltype in ("C3k2", "C2f"):
            prune_c2f_out(layer, keep_idx)
        elif ltype == "SPPF":
            prune_sppf_out(layer, keep_idx)
        elif ltype == "SPP":
            prune_spp_out(layer, keep_idx)
        elif ltype == "GhostConv":
            prune_ghostconv_out(layer, keep_idx)
        elif ltype == "C2fAttn":
            prune_c2fattn_out(layer, keep_idx)
        elif ltype == "A2C2f":
            prune_a2c2f_out(layer, keep_idx)
        elif ltype == "SCDown":
            prune_scdown_out(layer, keep_idx)
        elif ltype == "AConv":
            prune_conv_out(layer.cv1, keep_idx)
        elif ltype == "C2PSA":
            prune_c2psa_out(layer, keep_idx)
        elif ltype == "C3":
            prune_c3_out(layer, keep_idx)
        elif ltype == "BottleneckCSP":
            prune_bottleneck_csp_out(layer, keep_idx)
        elif ltype == "ADown":
            prune_adown_out(layer, keep_idx)
        elif ltype == "SPPELAN":
            prune_sppelan_out(layer, keep_idx)
        elif ltype == "FeatureFusion":
            prune_featurefusion_out(layer, keep_idx)
        elif ltype == "FCMFeatureFusion":
            prune_fcmfeaturefusion_out(layer, keep_idx)
        elif ltype == "MCFGatedFusion":
            prune_mcfgatedfusion_out(layer, keep_idx)

    def _prune_layer_input(self, layer: nn.Module, ltype: str, keep_idx: torch.Tensor):
        """Prune input channels of a layer."""
        if ltype == "Conv":
            prune_conv_in(layer, keep_idx)
        elif ltype in ("C3k2", "C2f"):
            prune_c2f_in(layer, keep_idx)
        elif ltype == "SPPF":
            prune_sppf_in(layer, keep_idx)
        elif ltype == "SPP":
            prune_spp_in(layer, keep_idx)
        elif ltype == "GhostConv":
            prune_ghostconv_in(layer, keep_idx)
        elif ltype == "C2fAttn":
            prune_c2fattn_in(layer, keep_idx)
        elif ltype == "A2C2f":
            prune_a2c2f_in(layer, keep_idx)
        elif ltype == "SCDown":
            prune_scdown_in(layer, keep_idx)
        elif ltype == "AConv":
            prune_conv_in(layer.cv1, keep_idx)
        elif ltype == "C2PSA":
            prune_c2psa_in(layer, keep_idx)
        elif ltype == "C3":
            prune_c3_in(layer, keep_idx)
        elif ltype == "BottleneckCSP":
            prune_bottleneck_csp_in(layer, keep_idx)
        elif ltype == "ADown":
            prune_adown_in(layer, keep_idx)
        elif ltype == "SPPELAN":
            prune_sppelan_in(layer, keep_idx)
        elif ltype == "FeatureFusion":
            prune_featurefusion_in(layer, keep_idx)
        elif ltype == "MCFGatedFusion":
            prune_mcfgatedfusion_in(layer, keep_idx)
        elif ltype == "SequenceShuffleAttention":
            prune_ssa_in(layer, keep_idx)
        # Note: FCMFeatureFusion doesn't need separate input pruning

    def _prune_detect_inputs(self, detect: nn.Module, topo: List[dict], det_idx: int, out_ch_after: List[int]):
        """Update Detect head input channels to match pruned upstream."""
        device = next(detect.parameters()).device
        f = topo[det_idx]["from_"]
        sources = f if isinstance(f, (list, tuple)) else [f]
        for scale_idx, s in enumerate(sources):
            src_idx = self._abs_idx(s, det_idx)
            new_ch = out_ch_after[src_idx]
            keep_idx = torch.arange(new_ch, device=device)
            prune_detect_in(detect, scale_idx, keep_idx)

    def _get_input_indices(
        self, cur_idx: int, f, topo: List[dict],
        out_ch_after: List[int], ch_before: List[int],
        keep_map: Dict[int, torch.Tensor], device
    ) -> Optional[torch.Tensor]:
        """Compute input channel indices for a layer based on upstream changes.

        Returns None if no change is needed.
        """
        if isinstance(f, (list, tuple)):
            # Multiple inputs (shouldn't happen for non-Concat, but handle)
            parts = []
            changed = False
            offset = 0
            for s in f:
                src_idx = self._abs_idx(s, cur_idx)
                old_ch = ch_before[src_idx]
                new_ch = out_ch_after[src_idx]
                if new_ch != old_ch:
                    changed = True
                parts.append(torch.arange(new_ch, device=device) + offset)
                offset += new_ch
            return torch.cat(parts) if changed else None
        else:
            src_idx = self._abs_idx(f, cur_idx)
            old_ch = ch_before[src_idx]
            new_ch = out_ch_after[src_idx]
            if new_ch != old_ch:
                return torch.arange(new_ch, device=device)
            return None

    def _get_input_ch_count(self, cur_idx: int, f, out_ch_after: List[int], topo: List[dict]) -> int:
        """Get total input channel count for a layer."""
        if isinstance(f, (list, tuple)):
            return sum(out_ch_after[self._abs_idx(s, cur_idx)] for s in f)
        return out_ch_after[self._abs_idx(f, cur_idx)]

    # ------------------------------------------------------------------
    # Entry layer detection
    # ------------------------------------------------------------------

    def _find_entry_layers(self, topo: List[dict]) -> Set[int]:
        """Find modal entry layers whose input comes from the image, not other layers.

        Entry layers are identified by:
        1. Explicit mm_input markers ('RGB', 'X', 'Dual')
        2. Conv layers with in_channels matching image channels (3) that are
           the start of a backbone branch
        """
        entry = set()

        # Method 1: Check explicit _mm_input_source markers (set by MultiModalRouter)
        for info in topo:
            layer = info["layer"]
            mm_input = getattr(layer, "_mm_input_source", None)
            if mm_input in ("RGB", "X", "Dual"):
                entry.add(info["idx"])

        # Always include layer 0
        entry.add(0)

        if len(entry) > 1:
            return entry

        # Method 2: Find Conv layers with in_channels == 3 (image input)
        # These are backbone entry points
        for info in topo:
            if info["idx"] == 0:
                continue
            if info["type"] == "Conv":
                layer = info["layer"]
                if hasattr(layer, "conv") and layer.conv.in_channels == 3:
                    entry.add(info["idx"])

        return entry

    # ------------------------------------------------------------------
    # Detect input channels
    # ------------------------------------------------------------------

    def _detect_input_channels(self, topo: List[dict]) -> int:
        """Detect the model's total input channel count.

        For mid-fusion models (separate RGB/X backbones each with in_channels=3),
        the total input is 6 (dual-modal). For early-fusion (Dual, in_channels=6),
        the first layer already has 6. For single-modal, it is 3.
        """
        entry_layers = self._find_entry_layers(topo)
        total_ch = 0
        for idx in entry_layers:
            layer = topo[idx]["layer"]
            if hasattr(layer, "conv"):
                total_ch += layer.conv.in_channels
        return total_ch if total_ch > 0 else 6

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_forward(self, model: nn.Module):
        """Run a dummy forward pass to verify the pruned model."""
        graph = build_prune_graph(model)
        in_ch = self._graph_input_channels(graph)

        model.eval()
        device = next(model.parameters()).device
        dummy = torch.randn(1, in_ch, self.imgsz, self.imgsz, device=device)
        try:
            with torch.no_grad():
                model(dummy)
            LOGGER.info("[Prune] Forward pass verification OK")
        except Exception as e:
            LOGGER.warning(f"[Prune] Forward pass verification FAILED: {e}")
            raise

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self, model: nn.Module, save_dir: str, has_report: bool = False, report_relpath: Optional[str] = None):
        """Save pruned model weights and YAML config.

        The saved checkpoint is finetrain-ready: all floating-point parameters
        are trainable (requires_grad=True). Runtime freeze policies such as
        DFL freezing are NOT encoded here; they remain owned by the trainer.
        If a structured report was written during pruning, the checkpoint
        carries ``has_report`` and ``report_file`` metadata in ``prune_info``.
        """
        from datetime import datetime

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. Generate pruned YAML (reference document)
        self._save_yaml(model, save_path)

        # 2. Save checkpoint compatible with ultralytics loading pipeline
        pt_path = save_path / "pruned.pt"
        pruned_model = deepcopy(model).half()

        graph = build_prune_graph(model)
        prune_meta = self._collect_prune_metadata(model, graph)

        # Minimal display-only prune_info (runtime truth comes from model structure)
        ckpt = {
            "model": pruned_model,
            "prune_info": {
                "is_pruned": True,
                "source_weights": prune_meta["source_weights"],
                "prune_method": self.method,
                "prune_ratio": float(self.ratio),
                "x_modality": prune_meta["x_modality"],
                "x_channels": prune_meta["x_channels"],
                "has_report": bool(has_report),
                "report_file": report_relpath,
            },
            "train_args": getattr(model, "args", {}),
            "date": datetime.now().isoformat(),
        }
        torch.save(ckpt, pt_path)
        LOGGER.info(f"[Prune] Saved to {pt_path}")

    def _save_yaml(self, model: nn.Module, save_path: Path) -> Path:
        """Generate pruned YAML config reflecting post-prune output channel counts.

        Note: Internal hidden channels (e.g. C3k2 bottleneck, C2PSA attention)
        are NOT pruned, so this YAML is a reference document showing the pruned
        output channels. To reload the pruned model, use the .pt checkpoint
        which contains the full pickled model object.
        """
        import yaml as _yaml

        src_yaml = getattr(model, "yaml", None)
        if src_yaml is None:
            LOGGER.warning("[Prune] Model has no .yaml attribute, skipping YAML output")
            return save_path / "pruned.yaml"

        pruned_yaml = deepcopy(src_yaml)

        # Remove parser metadata — this YAML is for reference only
        pruned_yaml.pop("scales", None)
        pruned_yaml.pop("scale", None)
        pruned_yaml.pop("yaml_file", None)

        # Build actual output channels per layer index
        topo = self._build_topology()
        ch_map = {}  # layer_idx -> actual out_channels after pruning
        for i, layer in enumerate(model.model):
            ltype = _get_layer_type(layer)
            out_ch = _get_out_channels(layer, ltype)
            if out_ch > 0:
                ch_map[i] = out_ch

        # Update backbone and head args
        layer_idx = 0
        for section in ("backbone", "head"):
            if section not in pruned_yaml:
                continue
            for row in pruned_yaml[section]:
                # row format: [from, repeats, module, args, *optional_mm_input]
                module_name = row[2]
                args = row[3]

                if module_name in ("Conv", "C3k2", "C2f", "SPPF", "SPP", "GhostConv", "C2fAttn", "A2C2f", "SCDown", "AConv", "C2PSA", "C3", "BottleneckCSP", "ADown", "SPPELAN") and layer_idx in ch_map:
                    # args[0] is the output channel count
                    args[0] = ch_map[layer_idx]
                elif module_name == "Detect":
                    # Detect args stay as-is (nc)
                    pass

                layer_idx += 1

        yaml_path = save_path / "pruned.yaml"
        with open(yaml_path, "w") as f:
            # Header
            f.write("# Pruned YAML (reference only) - generated by YAMLPruneEngine\n")
            f.write(f"# method={self.method}, ratio={self.ratio}\n")
            f.write("# NOTE: Internal hidden channels are unchanged. Load model from pruned.pt\n\n")

            # Write each section with one-liner layer format matching standard YAML style
            for section in ("backbone", "head"):
                if section not in pruned_yaml:
                    continue
                f.write(f"{section}:\n")
                for row in pruned_yaml[section]:
                    # Format: [from, repeats, module, args, *mm_input]
                    # Build YAML line manually to avoid single-quoting strings
                    parts = []
                    for item in row:
                        if item is None:
                            parts.append("None")
                        elif isinstance(item, bool):
                            parts.append("True" if item else "False")
                        elif isinstance(item, str):
                            # Quote module names (Conv, C3k2, etc.) and markers (RGB/X/Dual)
                            # Do NOT quote bare identifiers like nc, nearest, nn.Upsample
                            if item in ("RGB", "X", "Dual"):
                                parts.append(f"'{item}'")
                            else:
                                parts.append(item)
                        elif isinstance(item, (int, float)):
                            parts.append(str(item))
                        elif isinstance(item, list):
                            inner = ", ".join(str(x) for x in item)
                            parts.append(f"[{inner}]")
                        else:
                            parts.append(str(item))
                    line = f"  - [{', '.join(parts)}]\n"
                    f.write(line)

        LOGGER.info(f"[Prune] YAML saved to {yaml_path}")
        return yaml_path
