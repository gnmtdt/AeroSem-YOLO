# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Pruning utilities and summary formatting.

NOTE: Phase 1 replaces the old compute_model_gflops() with the graph-driven
complexity engine. This module now only handles summary formatting and logging.

Phase 2 integration points (untouched in Phase 1):
- ultralytics/utils/torch_utils.py::compute_model_gflops
- ultralytics/utils/torch_utils.py::log_multimodal_model_complexity
- ultralytics/models/yolo/multimodal/train.py and related trainers
- val/cocoval complexity output
- logger callbacks (WandB/ClearML/DVCLive/Neptune)
"""

from ultralytics.utils import LOGGER


def print_prune_summary(
    params_before: int,
    pruned_model,
    report_before,
    report_after,
):
    """Print before/after comparison of model params and complexity.

    This is now a pure formatting layer. All FLOPs computation is handled
    by the graph-driven complexity engine in ultralytics/nn/mm/complexity/.

    Args:
        params_before: Total parameter count before pruning.
        pruned_model: Model after pruning (used for param count).
        report_before: ComplexityReport before pruning.
        report_after: ComplexityReport after pruning.
    """
    params_after = sum(p.numel() for p in pruned_model.parameters())
    param_reduction = (1 - params_after / params_before) * 100 if params_before > 0 else 0

    def _fmt(n):
        return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.2f}K"

    before_total = report_before.total_flops / 1e9
    after_total = report_after.total_flops / 1e9
    stage_after = report_after.stage_flops()

    if before_total > 0:
        gflops_reduction = (1 - after_total / before_total) * 100
        gflops_str = f"GFLOPs(total[dual]): {before_total:.2f} → {after_total:.2f} (-{gflops_reduction:.1f}%)"
    else:
        gflops_str = f"GFLOPs(total[dual]): N/A → {after_total:.2f}"

    LOGGER.info(
        f"[Prune] Params: {_fmt(params_before)} → {_fmt(params_after)} "
        f"(-{param_reduction:.1f}%)  {gflops_str}"
    )

    # Stage breakdown
    LOGGER.info(
        "[Prune] Stages(after): "
        f"backbone={stage_after.get('backbone', 0.0) / 1e9:.2f} "
        f"neck={stage_after.get('neck', 0.0) / 1e9:.2f} "
        f"head={stage_after.get('head', 0.0) / 1e9:.2f}"
    )


def get_prunable_layers(model):
    """Get list of Conv2d layers that can be pruned.

    Args:
        model: The model to analyze.

    Returns:
        List of (name, module) tuples for prunable Conv2d layers.
    """
    prunable = []
    for name, module in model.named_modules():
        if hasattr(module, 'conv') and hasattr(module.conv, 'out_channels'):
            # Standard Conv wrapper
            prunable.append((name, module))
    return prunable
