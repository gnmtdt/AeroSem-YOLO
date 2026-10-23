# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Multi-input consumer parameter adapters for structured channel pruning.

This module provides:
1. `rebuild_consumer_if_supported()` - rebuilds lazy-initialized fusion modules
   after input channel counts change.
2. Fixed-dimension consumer adapters that perform parameter-level pruning
   without rebuilding new modules (FeatureFusion, FCM, MCFGatedFusion,
   CrossTransformerFusion, etc.).

All functions modify modules in-place.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.mm.pruning.ops import (
    prune_batchnorm2d_out,
    prune_conv_in,
    prune_conv_out,
    prune_raw_conv2d_in,
    prune_raw_conv2d_out,
)


# ---------------------------------------------------------------------------
# rebuild_consumer_if_supported - lazy rebuild dispatch
# ---------------------------------------------------------------------------


def _rebuild_equal_width_single_arg(module, ch: list[int]) -> None:
    """Rebuild single-arg lazy modules with equal left/right width = ch[0].

    Used for FCMFeatureFusion, SEFN, FusionConvMSAA, SpatialDependencyPerception.
    """
    c = ch[0]
    module._build_if_needed(c)


REBUILD_DISPATCH: dict[str, callable] = {
    # Two-arg lazy modules: left channel, right channel
    "CAM": lambda module, ch: module._build_if_needed(ch[0], ch[1]),
    "RFF": lambda module, ch: module._build(ch[0], ch[1]),
    # Single-arg lazy modules: both inputs share the same channel count
    "SEFN": _rebuild_equal_width_single_arg,
    "FusionConvMSAA": _rebuild_equal_width_single_arg,
    "SpatialDependencyPerception": _rebuild_equal_width_single_arg,
    # CrossTransformerFusion uses a simple __init__ (no lazy _build);
    # handle it separately via prune_cross_transformer_fusion_consumer.
}


def rebuild_consumer_if_supported(node_type: str, module, input_channels: list[int]) -> bool:
    """Rebuild a lazy-initialized multi-input consumer after input channels change.

    Args:
        node_type: Class name of the module (e.g. "CAM", "RFF", "FCMFeatureFusion").
        module: The actual module instance.
        input_channels: List of input channel counts, one per edge.
                         For two-input equal-width modules this is [ch, ch].

    Returns:
        True if the module type is supported and rebuild was called.
        False if the module type has no registered rebuild handler.
    """
    rebuild = REBUILD_DISPATCH.get(node_type)
    if rebuild is None:
        return False
    rebuild(module, input_channels)
    return True


# ---------------------------------------------------------------------------
# FeatureFusion consumer adapters
# ---------------------------------------------------------------------------


def _take_prefix_keep(width: int, new_width: int, device: torch.device) -> torch.Tensor:
    return torch.arange(min(width, new_width), device=device)


def _prune_linear_in(linear: nn.Linear, keep_idx: torch.Tensor) -> None:
    linear.weight = nn.Parameter(linear.weight.data[:, keep_idx])
    linear.in_features = len(keep_idx)


def _prune_linear_out(linear: nn.Linear, keep_idx: torch.Tensor) -> None:
    linear.weight = nn.Parameter(linear.weight.data[keep_idx])
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias.data[keep_idx])
    linear.out_features = len(keep_idx)


def _prune_layernorm(layernorm: nn.LayerNorm, keep_idx: torch.Tensor) -> None:
    if layernorm.elementwise_affine:
        layernorm.weight = nn.Parameter(layernorm.weight.data[keep_idx])
        layernorm.bias = nn.Parameter(layernorm.bias.data[keep_idx])
    layernorm.normalized_shape = (len(keep_idx),)


def prune_feature_interaction_inputs(module, left_keep: torch.Tensor, right_keep: torch.Tensor) -> None:
    """Prune the FeatureInteraction (FeatureFusion.cross) inputs for equal-width co-pruning.

    FeatureInteraction structure:
        channel_proj1: Linear(c, c//reduction * 2)   -> input side
        channel_proj2: Linear(c, c//reduction * 2)   -> input side
        end_proj1:     Linear(c//reduction * 2, c)  -> output side
        end_proj2:     Linear(c//reduction * 2, c)  -> output side

    When left and right are co-pruned (left_keep == right_keep, both indices from
    the same original channel space), we rebuild all four projection layers
    with the new channel count.

    Args:
        module: FeatureInteraction instance (FeatureFusion.cross).
        left_keep: Indices to keep from left input (unused here, c derived from dim).
        right_keep: Indices to keep from right input (unused here).
    """
    new_dim = len(left_keep)
    hidden_old = module.channel_proj1.out_features // 2
    reduction = max(module.channel_proj1.in_features // max(hidden_old, 1), 1)
    hidden_keep = _take_prefix_keep(hidden_old, max(new_dim // reduction, 1), left_keep.device)
    proj_keep = torch.cat([hidden_keep, hidden_keep + hidden_old])

    _prune_linear_in(module.channel_proj1, left_keep)
    _prune_linear_out(module.channel_proj1, proj_keep)
    _prune_linear_in(module.channel_proj2, right_keep)
    _prune_linear_out(module.channel_proj2, proj_keep)

    cross_attn = module.cross_attn
    hidden_kv_keep = torch.cat([hidden_keep, hidden_keep + hidden_old])
    for attr in ("q1", "q2"):
        linear = getattr(cross_attn, attr)
        _prune_linear_in(linear, hidden_keep)
        _prune_linear_out(linear, hidden_keep)
    for attr in ("kv1", "kv2"):
        linear = getattr(cross_attn, attr)
        _prune_linear_in(linear, hidden_keep)
        _prune_linear_out(linear, hidden_kv_keep)

    if getattr(cross_attn, "sr_ratio", 1) > 1:
        for attr in ("sr1", "sr2"):
            conv = getattr(cross_attn, attr)
            prune_raw_conv2d_in(conv, hidden_keep)
            prune_raw_conv2d_out(conv, hidden_keep)
            conv.groups = len(hidden_keep)
        for attr in ("norm1", "norm2"):
            _prune_layernorm(getattr(cross_attn, attr), hidden_keep)

    _prune_linear_in(module.end_proj1, proj_keep)
    _prune_linear_out(module.end_proj1, left_keep)
    _prune_linear_in(module.end_proj2, proj_keep)
    _prune_linear_out(module.end_proj2, right_keep)
    _prune_layernorm(module.norm1, left_keep)
    _prune_layernorm(module.norm2, right_keep)


def prune_channel_embed_inputs(module, total_in: int, left_width: int) -> None:
    """Prune ChannelEmbed input channels.

    ChannelEmbed forward: receives concatenated [left, right] features.
    Input channels = total_in (= 2 * original_dim after pruning).

    Args:
        module: ChannelEmbed instance (FeatureFusion.channel_emb).
        total_in: Total input channels (left + right after pruning).
        left_width: Width of left branch (unused here, kept for API consistency).
    """
    del total_in, left_width


def prune_channel_embed_output(module, keep_idx: torch.Tensor) -> None:
    """Prune ChannelEmbed output channels.

    Args:
        module: ChannelEmbed instance.
        keep_idx: Output channel indices to keep.
    """
    new_out = len(keep_idx)
    module.out_channels = new_out


def prune_featurefusion_consumer(
    module,
    left_keep: torch.Tensor,
    right_keep: torch.Tensor,
    out_keep: torch.Tensor | None = None,
) -> None:
    """Prune FeatureFusion consumer for co-pruned inputs.

    FeatureFusion structure:
        cross:      FeatureInteraction (takes [N, dim] sequences)
        channel_emb: ChannelEmbed (takes [N, 2*dim] -> outputs [dim])

    Args:
        module: FeatureFusion instance.
        left_keep: Indices to keep from left input branch.
        right_keep: Indices to keep from right input branch.
        out_keep: Output channel indices to keep. If None, output follows left width.
    """
    prune_feature_interaction_inputs(module.cross, left_keep, right_keep)
    orig_dim = module.channel_emb.out_channels
    final_keep = out_keep if out_keep is not None else torch.arange(len(left_keep), device=left_keep.device)
    hidden_old = module.channel_emb.channel_embed[0].out_channels
    reduction = max(orig_dim // max(hidden_old, 1), 1)
    hidden_keep = _take_prefix_keep(hidden_old, max(len(final_keep) // reduction, 1), left_keep.device)
    input_keep = torch.cat([left_keep, right_keep + orig_dim])

    prune_raw_conv2d_in(module.channel_emb.residual, input_keep)
    prune_raw_conv2d_out(module.channel_emb.residual, final_keep)

    prune_raw_conv2d_in(module.channel_emb.channel_embed[0], input_keep)
    prune_raw_conv2d_out(module.channel_emb.channel_embed[0], hidden_keep)

    dw = module.channel_emb.channel_embed[1]
    prune_raw_conv2d_in(dw, hidden_keep)
    prune_raw_conv2d_out(dw, hidden_keep)
    dw.groups = len(hidden_keep)

    prune_raw_conv2d_in(module.channel_emb.channel_embed[3], hidden_keep)
    prune_raw_conv2d_out(module.channel_emb.channel_embed[3], final_keep)

    if isinstance(module.channel_emb.channel_embed[4], nn.BatchNorm2d):
        prune_batchnorm2d_out(module.channel_emb.channel_embed[4], final_keep)
    if isinstance(module.channel_emb.norm, nn.BatchNorm2d):
        prune_batchnorm2d_out(module.channel_emb.norm, final_keep)
    module.channel_emb.out_channels = len(final_keep)


def prune_featurefusion_output(module, keep_idx: torch.Tensor) -> None:
    """Prune FeatureFusion output channels.

    Args:
        module: FeatureFusion instance.
        keep_idx: Output channel indices to keep.
    """
    prune_channel_embed_output(module.channel_emb, keep_idx)


# ---------------------------------------------------------------------------
# FCM consumer adapters (dual-output producer)
# ---------------------------------------------------------------------------


def prune_fcm_consumer(module, left_keep: torch.Tensor, right_keep: torch.Tensor) -> None:
    """Prune FCM consumer for co-pruned inputs (dual-output producer).

    FCM structure:
        spatial_weights:  (takes concat [2*dim, H, W])
        channel_weights:  (takes concat [2*dim, H, W])
        weights:          nn.Parameter [2] (fuse_weights)

    Both outputs (main, aux) share the same channel count = original dim.
    When co-pruning, both left and right are reduced to the same width.

    Args:
        module: FCM instance.
        left_keep: Indices to keep from left input (used for both branches).
        right_keep: Indices to keep from right input (same as left_keep for co-pruning).
    """
    new_dim = len(left_keep)
    orig_dim = module.spatial_weights.dim

    spatial_conv1 = module.spatial_weights.mlp[0]
    spatial_conv2 = module.spatial_weights.mlp[2]
    spatial_hidden_old = spatial_conv1.out_channels
    spatial_reduction = max(orig_dim // max(spatial_hidden_old, 1), 1)
    spatial_hidden_keep = _take_prefix_keep(
        spatial_hidden_old, max(new_dim // spatial_reduction, 1), left_keep.device
    )
    fused_keep = torch.cat([left_keep, right_keep + orig_dim])

    prune_raw_conv2d_in(spatial_conv1, fused_keep)
    prune_raw_conv2d_out(spatial_conv1, spatial_hidden_keep)
    prune_raw_conv2d_in(spatial_conv2, spatial_hidden_keep)
    module.spatial_weights.dim = new_dim

    fc1 = module.channel_weights.mlp[0]
    fc2 = module.channel_weights.mlp[2]
    hidden_old = fc1.out_features
    reduction = max((orig_dim * 6) // max(hidden_old, 1), 1)
    hidden_keep = _take_prefix_keep(hidden_old, max((new_dim * 6) // reduction, 1), left_keep.device)
    stats_keep = torch.cat([
        left_keep,
        right_keep + orig_dim,
        left_keep + orig_dim * 2,
        right_keep + orig_dim * 3,
        left_keep + orig_dim * 4,
        right_keep + orig_dim * 5,
    ])
    out_keep = torch.cat([left_keep, right_keep + orig_dim])

    _prune_linear_in(fc1, stats_keep)
    _prune_linear_out(fc1, hidden_keep)
    _prune_linear_in(fc2, hidden_keep)
    _prune_linear_out(fc2, out_keep)
    module.channel_weights.dim = new_dim


def prune_fcm_output(module, keep_idx: torch.Tensor) -> None:
    """Prune FCM output channels (both main and aux outputs are pruned together).

    Args:
        module: FCM instance.
        keep_idx: Output channel indices to keep.
    """
    # FCM outputs (main, aux) have the same channel count as input dim.
    # No internal rebuild needed for output-only pruning since outputs
    # are element-wise operations, not channel operations.
    pass


# ---------------------------------------------------------------------------
# MCFGatedFusion consumer adapters
# ---------------------------------------------------------------------------


def prune_mcfgatedfusion_consumer(
    module,
    main_keep: torch.Tensor,
    aux_keep: torch.Tensor,
    out_keep: torch.Tensor,
) -> None:
    """Prune MCFGatedFusion consumer for independently-pruned main and aux inputs.

    MCFGatedFusion structure:
        gate:  Conv2d(c_aux, c_out, k)   -- aux branch gating
        bn:    BatchNorm2d or Identity
        act:   SiLU or Identity
        post:  Conv(c_out + c_main, c_main) -- only in concat mode

    Args:
        module: MCFGatedFusion instance.
        main_keep: Channel indices to keep from main input.
        aux_keep: Channel indices to keep from aux input.
        out_keep: Output channel indices to keep.
    """
    prune_raw_conv2d_in(module.gate, aux_keep)

    if module.mode == "concat" and module.post is not None:
        gate_width = module.gate.out_channels
        old_main_width = module.post.conv.in_channels - gate_width
        post_keep = torch.cat([main_keep, torch.arange(gate_width, device=main_keep.device) + old_main_width])
        prune_conv_in(module.post, post_keep)
        prune_conv_out(module.post, out_keep)
    else:
        final_keep = out_keep if out_keep is not None else main_keep
        prune_raw_conv2d_out(module.gate, final_keep)
        if isinstance(module.bn, nn.BatchNorm2d):
            prune_batchnorm2d_out(module.bn, final_keep)


def prune_mcfgatedfusion_output(module, keep_idx: torch.Tensor) -> None:
    """Prune MCFGatedFusion output channels.

    Args:
        module: MCFGatedFusion instance.
        keep_idx: Output channel indices to keep.
    """
    mode = module.mode

    if mode == "concat":
        prune_conv_out(module.post, keep_idx)
    else:
        prune_raw_conv2d_out(module.gate, keep_idx)
        if isinstance(module.bn, nn.BatchNorm2d):
            prune_batchnorm2d_out(module.bn, keep_idx)


# ---------------------------------------------------------------------------
# CrossTransformerFusion consumer adapters
# ---------------------------------------------------------------------------


def prune_cross_transformer_fusion_consumer(
    module,
    left_keep: torch.Tensor,
    right_keep: torch.Tensor,
    out_keep: torch.Tensor | None = None,
) -> None:
    """Prune CrossTransformerFusion consumer for co-pruned inputs.

    CrossTransformerFusion structure:
        encoder: TransformerEncoder
            embedding:  Linear(input_dim, model_dim)
            positional_encoding: PositionalEncoding
            layers:     ModuleList of TransformerEncoderLayer

    TransformerEncoderLayer:
        cross_attention: MultiHeadCrossAttention
            query_vis, key_vis, value_vis: Linear(model_dim, model_dim)
            query_inf, key_inf, value_inf: Linear(model_dim, model_dim)
            fc_out_vis, fc_out_inf: Linear(model_dim, model_dim)
        norm1: LayerNorm(model_dim)
        ff: FeedForward
            fc1: Linear(model_dim, hidden_dim)
            fc2: Linear(hidden_dim, model_dim)
        norm2: LayerNorm(model_dim)

    The module processes [B,C,H,W] -> [B,HW,C] internally, so the actual
    channel dimension is the model's model_dim (== input_dim).

    Args:
        module: CrossTransformerFusion instance.
        left_keep: Indices to keep from left input (unused here, derived from dim).
        right_keep: Indices to keep from right input (same as left for co-pruning).
        out_keep: Output channel indices to keep. If None, output = 2 * dim.
    """
    # Get current model_dim from the embedding layer
    new_dim = len(left_keep)

    device = module.encoder.embedding.weight.device
    dtype = module.encoder.embedding.weight.dtype
    hidden_dim = module.hidden_dim  # input_dim * 2

    # Rebuild embedding: Linear(input_dim, model_dim) -> Linear(new_dim, new_dim)
    module.encoder.embedding = nn.Linear(new_dim, new_dim).to(device=device, dtype=dtype)

    # Rebuild positional encoding
    module.encoder.positional_encoding = type(module.encoder.positional_encoding)(
        new_dim, module.encoder.positional_encoding.dropout.p
    ).to(device=device, dtype=dtype)

    # Rebuild each TransformerEncoderLayer
    num_layers = len(module.encoder.layers)
    num_heads = module.encoder.layers[0].cross_attention.num_heads
    dropout = module.encoder.layers[0].ff.dropout.p

    new_hidden = new_dim * 2  # hidden_dim scales with model_dim

    module.encoder.layers = nn.ModuleList()
    for _ in range(num_layers):
        from ultralytics.nn.modules.fusion.ctf import (
            FeedForward,
            MultiHeadCrossAttention,
            TransformerEncoderLayer,
        )

        cross_attn = MultiHeadCrossAttention(new_dim, num_heads)
        norm1 = nn.LayerNorm(new_dim)
        ff = FeedForward(new_dim, new_hidden, dropout)
        norm2 = nn.LayerNorm(new_dim)
        layer = TransformerEncoderLayer(new_dim, num_heads, new_hidden, dropout)
        layer.cross_attention = cross_attn
        layer.norm1 = norm1
        layer.ff = ff
        layer.norm2 = norm2
        module.encoder.layers.append(layer)

    # Update stored dimensions
    module.model_dim = new_dim
    module.hidden_dim = new_hidden


def prune_cross_transformer_fusion_output(module, keep_idx: torch.Tensor) -> None:
    """Prune CrossTransformerFusion output channels.

    CrossTransformerFusion output = concat(vis_out, inf_out), each of width model_dim.
    When output is pruned, we need to adjust the encoder to output the new dimension.

    Args:
        module: CrossTransformerFusion instance.
        keep_idx: Output channel indices to keep. Since output is concat of two
                  equal-width tensors, this should cover both halves.
    """
    # The output is 2*model_dim (concat of vis and inf).
    # Pruning the output means rebuilding with new model_dim.
    new_total = len(keep_idx)
    if new_total % 2 != 0:
        raise ValueError(f"CrossTransformerFusion output channels must be even, got {new_total}")
    new_dim = new_total // 2

    device = module.encoder.embedding.weight.device
    dtype = module.encoder.embedding.weight.dtype
    num_layers = len(module.encoder.layers)
    num_heads = module.encoder.layers[0].cross_attention.num_heads
    dropout = module.encoder.layers[0].ff.dropout.p
    new_hidden = new_dim * 2

    # Rebuild embedding
    module.encoder.embedding = nn.Linear(new_dim, new_dim).to(device=device, dtype=dtype)

    # Rebuild positional encoding
    module.encoder.positional_encoding = type(module.encoder.positional_encoding)(
        new_dim, dropout
    ).to(device=device, dtype=dtype)

    # Rebuild layers
    from ultralytics.nn.modules.fusion.ctf import (
        FeedForward,
        MultiHeadCrossAttention,
        TransformerEncoderLayer,
    )

    module.encoder.layers = nn.ModuleList()
    for _ in range(num_layers):
        cross_attn = MultiHeadCrossAttention(new_dim, num_heads)
        norm1 = nn.LayerNorm(new_dim)
        ff = FeedForward(new_dim, new_hidden, dropout)
        norm2 = nn.LayerNorm(new_dim)
        layer = TransformerEncoderLayer(new_dim, num_heads, new_hidden, dropout)
        layer.cross_attention = cross_attn
        layer.norm1 = norm1
        layer.ff = ff
        layer.norm2 = norm2
        module.encoder.layers.append(layer)

    module.model_dim = new_dim
    module.hidden_dim = new_hidden


# ---------------------------------------------------------------------------
# FCMFeatureFusion consumer adapters (chains FCM -> FeatureFusion)
# ---------------------------------------------------------------------------


def prune_fcmfeaturefusion_consumer(
    module,
    left_keep: torch.Tensor,
    right_keep: torch.Tensor,
    out_keep: torch.Tensor | None = None,
) -> None:
    """Prune FCMFeatureFusion consumer.

    FCMFeatureFusion chains: FCM(dim) -> FeatureFusion(dim)

    Args:
        module: FCMFeatureFusion instance.
        left_keep: Indices to keep from left input.
        right_keep: Indices to keep from right input.
        out_keep: Output channel indices to keep.
    """
    prune_fcm_consumer(module.fcm, left_keep, right_keep)
    prune_featurefusion_consumer(module.ffm, left_keep, right_keep, out_keep)


def prune_fcmfeaturefusion_output(module, keep_idx: torch.Tensor) -> None:
    """Prune FCMFeatureFusion output channels."""
    prune_featurefusion_output(module.ffm, keep_idx)


# ---------------------------------------------------------------------------
# High-level consumer adapter dispatcher
# ---------------------------------------------------------------------------

CONSUMER_ADAPTERS: dict[str, callable] = {
    "FeatureFusion": prune_featurefusion_consumer,
    "FCM": prune_fcm_consumer,
    "FCMFeatureFusion": prune_fcmfeaturefusion_consumer,
    "MCFGatedFusion": prune_mcfgatedfusion_consumer,
    "CrossTransformerFusion": prune_cross_transformer_fusion_consumer,
}


def adapt_consumer(
    node_type: str,
    module,
    left_keep: torch.Tensor,
    right_keep: torch.Tensor,
    out_keep: torch.Tensor | None = None,
) -> bool:
    """High-level dispatcher for fixed-dimension consumer adapters.

    Args:
        node_type: Class name of the consumer module.
        module: The module instance.
        left_keep: Indices to keep from left input.
        right_keep: Indices to keep from right input.
        out_keep: Output channel indices to keep. Optional.

    Returns:
        True if the module type has a registered adapter.
        False otherwise.
    """
    adapter = CONSUMER_ADAPTERS.get(node_type)
    if adapter is None:
        return False
    adapter(module, left_keep, right_keep, out_keep)
    return True
