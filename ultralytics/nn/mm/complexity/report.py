# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Shared formatting helpers for graph-driven multimodal complexity reports."""

from __future__ import annotations


def get_model_param_count(model) -> int:
    """Return total parameter count for a PyTorch model-like object."""
    if hasattr(model, "parameters"):
        return sum(p.numel() for p in model.parameters())
    if hasattr(model, "model") and hasattr(model.model, "parameters"):
        return sum(p.numel() for p in model.model.parameters())
    return 0


def build_default_complexity_summary(model, report) -> dict:
    """Convert a complexity report into a reusable summary payload."""
    params = get_model_param_count(model)
    stage_raw = report.stage_flops()
    stage_gflops = {key: value / 1e9 for key, value in stage_raw.items()}
    return {
        "params": params,
        "gflops_total": report.total_flops / 1e9,
        "stage_gflops": stage_gflops,
    }


def format_default_complexity_lines(model, report) -> list[str]:
    """Return user-facing log lines for default-structure complexity."""
    summary = build_default_complexity_summary(model, report)
    params = summary["params"]
    stage = summary["stage_gflops"]
    lines = [
        f"Params: {params / 1e6:.2f}M ({params:,}) | GFLOPs(total[default]): {summary['gflops_total']:.2f}",
    ]
    if "backbone" in stage:
        lines.append(
            "Stages: "
            f"backbone={stage.get('backbone', 0.0):.2f} "
            f"neck={stage.get('neck', 0.0):.2f} "
            f"head={stage.get('head', 0.0):.2f}"
        )
    return lines


def log_default_complexity(model, report, logger) -> None:
    """Log default-structure complexity lines using the provided logger."""
    for line in format_default_complexity_lines(model, report):
        logger.info(line)
