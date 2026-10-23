# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Channel importance scoring for structured pruning."""

import torch
import torch.nn as nn


def _get_output_conv(layer: nn.Module) -> nn.Module:
    """Get the output-defining convolution of a composite layer.

    For Conv: the layer itself.
    For C2f/C3k2: cv2 (the output projection).
    For C2PSA: cv2.
    For SPPF: cv2.
    Otherwise: return None (fallback to generic).
    """
    from ultralytics.nn.modules.block import C2PSA, C2f, SPPF
    from ultralytics.nn.modules.conv import Conv

    if isinstance(layer, (C2f, C2PSA)):
        return layer.cv2
    elif isinstance(layer, SPPF):
        return layer.cv2
    elif isinstance(layer, Conv):
        return layer
    return None


def compute_importance(layer: nn.Module, method: str = "l1") -> torch.Tensor:
    """Compute per-output-channel importance scores for a layer.

    For composite modules (C3k2, C2PSA, SPPF), uses the output convolution's
    weights to determine importance, ensuring consistent channel dimensions.

    Args:
        layer: Module whose output channels are being scored.
        method: Scoring method - 'l1', 'l2', 'lamp', 'bn', or 'random'.

    Returns:
        Tensor of shape [out_channels] with importance scores.
    """
    # Try to get the output-defining conv for consistent channel count
    out_conv = _get_output_conv(layer)

    if out_conv is not None and hasattr(out_conv, "conv"):
        # Use the output conv's weight as primary importance signal
        w = out_conv.conv.weight.data.flatten(1)  # [out_ch, ...]
        out_ch = w.shape[0]

        if method == "l1":
            return w.abs().sum(dim=1)
        elif method == "l2":
            return (w**2).sum(dim=1).sqrt()
        elif method == "lamp":
            scores = w.abs().sum(dim=1)
            return _lamp_transform(scores)
        elif method == "random":
            return torch.rand(out_ch, device=w.device)
        elif method == "bn":
            if hasattr(out_conv, "bn") and out_conv.bn.weight is not None:
                return out_conv.bn.weight.data.abs()
            return w.abs().sum(dim=1)
        else:
            raise ValueError(f"Unknown importance method: {method}")

    # Fallback: collect all Conv2d with matching out_channels
    weights = []
    for m in layer.modules():
        if isinstance(m, nn.Conv2d) and m.weight.shape[0] > 0:
            weights.append(m.weight.data.flatten(1))

    if not weights:
        raise ValueError(f"No Conv2d found in {type(layer).__name__}")

    # Use the first Conv2d's out_channels as reference
    out_ch = weights[0].shape[0]
    # Only use weights with matching out_channels
    matched = [w for w in weights if w.shape[0] == out_ch]

    if method == "l1":
        return sum(w.abs().sum(dim=1) for w in matched)
    elif method == "l2":
        return sum((w**2).sum(dim=1).sqrt() for w in matched)
    elif method == "lamp":
        scores = sum(w.abs().sum(dim=1) for w in matched)
        return _lamp_transform(scores)
    elif method == "random":
        return torch.rand(out_ch, device=matched[0].device)
    elif method == "bn":
        bn_scores = torch.zeros(out_ch, device=matched[0].device)
        found = False
        for m in layer.modules():
            if isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                if m.weight.shape[0] == out_ch:
                    bn_scores += m.weight.data.abs()
                    found = True
        if not found:
            return sum(w.abs().sum(dim=1) for w in matched)
        return bn_scores
    else:
        raise ValueError(f"Unknown importance method: {method}")


def _lamp_transform(scores: torch.Tensor) -> torch.Tensor:
    """Apply LAMP importance transformation."""
    sorted_scores, sorted_idx = scores.sort(descending=True)
    cumsum = sorted_scores.flip(0).cumsum(0).flip(0)
    rank = sorted_idx.argsort()
    return scores / cumsum[rank]


# ------------------------------------------------------------------
# Output importance scorers for fusion modules
# ------------------------------------------------------------------


def score_featurefusion_output(layer: nn.Module, method: str = "l1") -> torch.Tensor:
    """Score output channels of FeatureFusion based on channel_emb sub-modules.

    FeatureFusion uses ChannelEmbed which has:
    - channel_emb.residual: 1x1 conv (identity path)
    - channel_emb.channel_embed[3]: final 1x1 conv after depthwise

    The output channels are controlled by channel_emb.channel_embed[3].
    Must NOT fall back to proj_out assumption.
    """
    channel_emb = getattr(layer, "channel_emb", None)
    if channel_emb is None:
        raise ValueError(f"FeatureFusion: channel_emb not found on {type(layer).__name__}")

    # channel_embed is a Sequential, index 3 is the final 1x1 conv
    channel_embed_seq = getattr(channel_emb, "channel_embed", None)
    if channel_embed_seq is None or len(channel_embed_seq) < 4:
        raise ValueError(f"FeatureFusion: channel_embed seq not found or too short on {type(layer).__name__}")

    final_conv = channel_embed_seq[3]  # 1x1 conv after depthwise
    residual = getattr(channel_emb, "residual", None)
    if not hasattr(final_conv, "weight") or residual is None or not hasattr(residual, "weight"):
        raise ValueError(f"FeatureFusion: channel_embed[3] has no weight on {type(layer).__name__}")

    final_w = final_conv.weight.data.flatten(1)
    residual_w = residual.weight.data.flatten(1)
    if final_w.shape[0] != residual_w.shape[0]:
        raise ValueError(
            f"FeatureFusion: residual/channel_embed[3] output mismatch: "
            f"{residual_w.shape[0]} vs {final_w.shape[0]}"
        )

    w = final_w
    out_ch = final_w.shape[0]

    if method == "l1":
        return final_w.abs().sum(dim=1) + residual_w.abs().sum(dim=1)
    elif method == "l2":
        return (final_w**2).sum(dim=1).sqrt() + (residual_w**2).sum(dim=1).sqrt()
    elif method == "lamp":
        scores = final_w.abs().sum(dim=1) + residual_w.abs().sum(dim=1)
        return _lamp_transform(scores)
    elif method == "random":
        return torch.rand(out_ch, device=w.device)
    elif method == "bn":
        raise ValueError("FeatureFusion: bn method not supported, use l1/l2/lamp/random")
    else:
        raise ValueError(f"Unknown importance method: {method}")


def score_fcmfeaturefusion_output(layer: nn.Module, method: str = "l1") -> torch.Tensor:
    """Score output channels of FCMFeatureFusion.

    FCMFeatureFusion chains FCM -> FeatureFusion.
    The output channels are controlled by the inner FeatureFusion's channel_emb.
    """
    ffm = getattr(layer, "ffm", None)
    if ffm is None:
        raise ValueError(f"FCMFeatureFusion: ffm (FeatureFusion) not found on {type(layer).__name__}")

    return score_featurefusion_output(ffm, method)


def score_mcfgatedfusion_output(layer: nn.Module, method: str = "l1") -> torch.Tensor:
    """Score output channels of MCFGatedFusion.

    MCFGatedFusion has:
    - gate: Conv2d for gating the aux modality
    - post: Conv (optional, only in concat mode)

    The output channels are determined by the post module in concat mode,
    or the input main channels in add mode. We score based on gate/post,
    NOT main_proj/aux_proj/out_proj (those don't exist on this module).
    """
    post = getattr(layer, "post", None)

    if post is not None and hasattr(post, "conv") and hasattr(post.conv, "weight"):
        # concat mode: output is controlled by post conv
        w = post.conv.weight.data.flatten(1)
    elif hasattr(layer, "gate") and hasattr(layer.gate, "weight"):
        # add mode: output follows main input channels (gate output = main channels)
        w = layer.gate.weight.data.flatten(1)
    else:
        raise ValueError(f"MCFGatedFusion: no gate or post with weight found on {type(layer).__name__}")

    out_ch = w.shape[0]

    if method == "l1":
        return w.abs().sum(dim=1)
    elif method == "l2":
        return (w**2).sum(dim=1).sqrt()
    elif method == "lamp":
        scores = w.abs().sum(dim=1)
        return _lamp_transform(scores)
    elif method == "random":
        return torch.rand(out_ch, device=w.device)
    elif method == "bn":
        raise ValueError("MCFGatedFusion: bn method not supported, use l1/l2/lamp/random")
    else:
        raise ValueError(f"Unknown importance method: {method}")


def score_crosstransformer_output(layer: nn.Module, method: str = "l1") -> torch.Tensor:
    """Score output channels of CrossTransformerFusion.

    The module outputs ``concat(vis_out, inf_out)`` where each half is projected by
    the last encoder layer's ``fc_out_vis`` / ``fc_out_inf``.
    """
    encoder = getattr(layer, "encoder", None)
    layers = getattr(encoder, "layers", None) if encoder is not None else None
    if not layers:
        raise ValueError(f"CrossTransformerFusion: encoder layers not found on {type(layer).__name__}")

    last_layer = layers[-1]
    cross_attention = getattr(last_layer, "cross_attention", None)
    if cross_attention is None:
        raise ValueError(f"CrossTransformerFusion: cross_attention not found on {type(layer).__name__}")

    vis_linear = getattr(cross_attention, "fc_out_vis", None)
    inf_linear = getattr(cross_attention, "fc_out_inf", None)
    if vis_linear is None or inf_linear is None:
        raise ValueError(f"CrossTransformerFusion: fc_out_vis/fc_out_inf not found on {type(layer).__name__}")

    vis_w = vis_linear.weight.data.flatten(1)
    inf_w = inf_linear.weight.data.flatten(1)

    if method == "l1":
        vis_scores = vis_w.abs().sum(dim=1)
        inf_scores = inf_w.abs().sum(dim=1)
    elif method == "l2":
        vis_scores = (vis_w**2).sum(dim=1).sqrt()
        inf_scores = (inf_w**2).sum(dim=1).sqrt()
    elif method == "lamp":
        vis_scores = _lamp_transform(vis_w.abs().sum(dim=1))
        inf_scores = _lamp_transform(inf_w.abs().sum(dim=1))
    elif method == "random":
        vis_scores = torch.rand(vis_w.shape[0], device=vis_w.device)
        inf_scores = torch.rand(inf_w.shape[0], device=inf_w.device)
    elif method == "bn":
        raise ValueError("CrossTransformerFusion: bn method not supported, use l1/l2/lamp/random")
    else:
        raise ValueError(f"Unknown importance method: {method}")

    return torch.cat([vis_scores, inf_scores], dim=0)


# Registry of output-specific scorers for fusion modules
OUTPUT_IMPORTANCE_SCORERS = {
    "FeatureFusion": score_featurefusion_output,
    "FCMFeatureFusion": score_fcmfeaturefusion_output,
    "MCFGatedFusion": score_mcfgatedfusion_output,
    "CrossTransformerFusion": score_crosstransformer_output,
}
