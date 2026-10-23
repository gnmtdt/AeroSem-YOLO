# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Structured report helpers for YAML-driven multimodal pruning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class LayerExecutionTrace:
    """Runtime pruning trace for a single graph node."""

    node_idx: int
    decision_reason: str
    input_keep_by_edge: list[dict[str, Any]] = field(default_factory=list)
    resolved_output_slots: list[list[int]] = field(default_factory=list)
    consumer_coordinated: bool = False
    head_input_updated: bool = False

    @property
    def resolved_output_keep(self) -> list[int]:
        """Return the primary output keep indices."""
        return self.resolved_output_slots[0] if self.resolved_output_slots else []


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _full_keep(width: int) -> list[int]:
    return list(range(int(width)))


def _default_input_keep_records(graph_before, node) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for edge in node.input_edges:
        source = graph_before.node(edge.node_idx)
        source_width = source.out_channels[edge.output_slot]
        records.append(
            {
                "node_idx": int(edge.node_idx),
                "output_slot": int(edge.output_slot),
                "keep_idx": _full_keep(source_width),
            }
        )
    return records


def _default_output_slots(node) -> list[list[int]]:
    return [_full_keep(width) for width in node.out_channels]


def _build_scoring_payload(proposal) -> dict[str, Any] | None:
    if proposal is None:
        return None

    scores_tensor = proposal.scores.detach().float().cpu()
    scores = [float(x) for x in scores_tensor.tolist()]
    ordered_idx = [int(x) for x in proposal.ordered_idx.detach().cpu().tolist()]
    proposal_keep_idx = [int(x) for x in proposal.final_idx.detach().cpu().tolist()]
    keep_set = set(proposal_keep_idx)
    proposal_drop_idx = [idx for idx in range(len(scores)) if idx not in keep_set]

    if scores:
        score_stats = {
            "min": float(min(scores)),
            "max": float(max(scores)),
            "mean": float(sum(scores) / len(scores)),
        }
    else:
        score_stats = {"min": 0.0, "max": 0.0, "mean": 0.0}

    return {
        "scores": scores,
        "score_stats": score_stats,
        "ordered_idx": ordered_idx,
        "proposal_n_keep": int(proposal.n_keep),
        "proposal_keep_idx": proposal_keep_idx,
        "proposal_drop_idx": proposal_drop_idx,
    }


def build_prune_report(
    *,
    graph_before,
    graph_after,
    proposals: Mapping[int, Any],
    trace_map: Mapping[int, LayerExecutionTrace],
    report_before,
    report_after,
    params_before: int,
    params_after: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured JSON-serializable pruning report."""

    stage_before = {key: value / 1e9 for key, value in report_before.stage_flops().items()}
    stage_after = {key: value / 1e9 for key, value in report_after.stage_flops().items()}
    total_before = report_before.total_flops / 1e9
    total_after = report_after.total_flops / 1e9

    layers: list[dict[str, Any]] = []
    for before_node in graph_before.nodes:
        after_node = graph_after.node(before_node.idx)
        proposal = proposals.get(before_node.idx)
        trace = trace_map.get(before_node.idx)

        input_keep_by_edge = (
            trace.input_keep_by_edge if trace is not None else _default_input_keep_records(graph_before, before_node)
        )
        output_slots = trace.resolved_output_slots if trace is not None else _default_output_slots(after_node)
        if not output_slots:
            output_slots = _default_output_slots(after_node)

        slot_records: list[dict[str, Any]] = []
        for slot_idx, before_width in enumerate(before_node.out_channels):
            fallback_width = (
                after_node.out_channels[slot_idx] if slot_idx < len(after_node.out_channels) else before_width
            )
            keep_idx = output_slots[slot_idx] if slot_idx < len(output_slots) else _full_keep(fallback_width)
            keep_set = set(keep_idx)
            drop_idx = [idx for idx in range(int(before_width)) if idx not in keep_set]
            slot_records.append(
                {
                    "slot": int(slot_idx),
                    "keep_idx": [int(x) for x in keep_idx],
                    "drop_idx": drop_idx,
                }
            )

        primary_keep = slot_records[0]["keep_idx"] if slot_records else []
        primary_drop = slot_records[0]["drop_idx"] if slot_records else []

        total_out_before = sum(int(width) for width in before_node.out_channels)
        total_out_after = sum(int(width) for width in after_node.out_channels)
        output_pruned_count = max(total_out_before - total_out_after, 0)
        input_pruned_count = max(int(before_node.in_channels) - int(after_node.in_channels), 0)
        output_pruned = (not before_node.is_route_only and not before_node.is_head) and output_pruned_count > 0
        input_adapted = input_pruned_count > 0

        layers.append(
            {
                "idx": int(before_node.idx),
                "qualified_name": f"model.{before_node.idx}",
                "type_name": before_node.type_name,
                "branch_kind": before_node.branch_kind,
                "flags": {
                    "is_entry": bool(before_node.is_entry),
                    "is_head": bool(before_node.is_head),
                    "is_route_only": bool(before_node.is_route_only),
                    "is_multi_input": bool(before_node.is_multi_input),
                },
                "input_edges": [
                    {"node_idx": int(edge.node_idx), "output_slot": int(edge.output_slot)}
                    for edge in before_node.input_edges
                ],
                "channels_before": {
                    "in_channels": int(before_node.in_channels),
                    "out_channels": [int(width) for width in before_node.out_channels],
                },
                "channels_after": {
                    "in_channels": int(after_node.in_channels),
                    "out_channels": [int(width) for width in after_node.out_channels],
                },
                "delta": {
                    "input_pruned_count": input_pruned_count,
                    "input_pruned_ratio": _safe_ratio(input_pruned_count, int(before_node.in_channels)),
                    "output_pruned_count": output_pruned_count,
                    "output_pruned_ratio": _safe_ratio(output_pruned_count, total_out_before),
                    "primary_output_pruned_count": max(int(before_node.primary_out_channels) - len(primary_keep), 0),
                    "primary_output_pruned_ratio": _safe_ratio(
                        max(int(before_node.primary_out_channels) - len(primary_keep), 0),
                        int(before_node.primary_out_channels),
                    ),
                },
                "scoring": None
                if proposal is None
                else {
                    "method": meta.get("prune_method"),
                    **_build_scoring_payload(proposal),
                },
                "actual": {
                    "output_keep_idx": primary_keep,
                    "output_drop_idx": primary_drop,
                    "output_slots": slot_records,
                    "input_keep_by_edge": input_keep_by_edge,
                },
                "change_flags": {
                    "output_pruned": output_pruned,
                    "input_adapted": input_adapted,
                    "consumer_coordinated": bool(trace.consumer_coordinated) if trace is not None else False,
                    "head_input_updated": bool(trace.head_input_updated) if trace is not None else False,
                },
                "decision_reason": trace.decision_reason if trace is not None else "non_prunable_passthrough",
            }
        )

    return {
        "schema_version": 1,
        "meta": meta,
        "summary": {
            "params_before": int(params_before),
            "params_after": int(params_after),
            "params_pruned": int(max(params_before - params_after, 0)),
            "params_pruned_ratio": _safe_ratio(max(params_before - params_after, 0), params_before),
            "gflops_before": float(total_before),
            "gflops_after": float(total_after),
            "gflops_pruned": float(max(total_before - total_after, 0.0)),
            "gflops_pruned_ratio": _safe_ratio(max(total_before - total_after, 0.0), total_before),
            "stage_gflops_before": stage_before,
            "stage_gflops_after": stage_after,
        },
        "layers": layers,
    }


def save_prune_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Save the structured pruning report as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
