# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Per-module pruning operations for structured channel pruning.

Each function directly modifies module weights in-place.
"""

import torch
import torch.nn as nn


def _ssa_group_input_indices(channel_count: int, groups: int, device: torch.device) -> torch.Tensor:
    """Return original-channel indices grouped by SSA's grouped conv after channel_shuffle.

    SSA applies ``channel_shuffle()`` before its grouped 1x1 gating conv. The grouped
    conv therefore consumes contiguous chunks in the shuffled space, which correspond
    to interleaved channel sets in the original space.
    """
    if groups <= 0:
        raise ValueError(f"SSA groups must be positive, got {groups}")
    if channel_count % groups != 0:
        raise ValueError(f"SSA channels {channel_count} must be divisible by groups {groups}")

    channels_per_group = channel_count // groups
    order = torch.arange(channel_count, device=device).reshape(channels_per_group, groups).permute(1, 0)
    return order.contiguous()


# ---------------------------------------------------------------------------
# Conv (Conv2d + BN + Act)
# ---------------------------------------------------------------------------

def prune_conv_out(module, keep_idx: torch.Tensor):
    """Prune output channels of a Conv module (conv + bn).

    Args:
        module: Conv module with .conv (Conv2d) and .bn (BatchNorm2d).
        keep_idx: 1-D tensor of channel indices to keep.
    """
    conv = module.conv
    conv.weight = nn.Parameter(conv.weight.data[keep_idx])
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[keep_idx])
    conv.out_channels = len(keep_idx)

    if hasattr(module, "bn") and isinstance(module.bn, nn.BatchNorm2d):
        bn = module.bn
        bn.weight = nn.Parameter(bn.weight.data[keep_idx])
        bn.bias = nn.Parameter(bn.bias.data[keep_idx])
        bn.running_mean = bn.running_mean[keep_idx]
        bn.running_var = bn.running_var[keep_idx]
        bn.num_features = len(keep_idx)


def prune_conv_in(module, keep_idx: torch.Tensor):
    """Prune input channels of a Conv module.

    Args:
        module: Conv module with .conv (Conv2d).
        keep_idx: 1-D tensor of input channel indices to keep.
    """
    conv = module.conv
    if conv.groups == 1:
        conv.weight = nn.Parameter(conv.weight.data[:, keep_idx])
    else:
        # DWConv: groups == in_channels == out_channels
        conv.weight = nn.Parameter(conv.weight.data[keep_idx])
        conv.groups = len(keep_idx)
        conv.out_channels = len(keep_idx)
        # Also update BN for DWConv
        if hasattr(module, "bn"):
            bn = module.bn
            bn.weight = nn.Parameter(bn.weight.data[keep_idx])
            bn.bias = nn.Parameter(bn.bias.data[keep_idx])
            bn.running_mean = bn.running_mean[keep_idx]
            bn.running_var = bn.running_var[keep_idx]
            bn.num_features = len(keep_idx)
    conv.in_channels = len(keep_idx)


def prune_raw_conv2d_in(conv2d: nn.Conv2d, keep_idx: torch.Tensor):
    """Prune input channels of a raw nn.Conv2d (no BN wrapper)."""
    conv2d.weight = nn.Parameter(conv2d.weight.data[:, keep_idx])
    conv2d.in_channels = len(keep_idx)


def prune_raw_conv2d_out(conv2d: nn.Conv2d, keep_idx: torch.Tensor):
    """Prune output channels of a raw nn.Conv2d (no BN wrapper)."""
    conv2d.weight = nn.Parameter(conv2d.weight.data[keep_idx])
    if conv2d.bias is not None:
        conv2d.bias = nn.Parameter(conv2d.bias.data[keep_idx])
    conv2d.out_channels = len(keep_idx)


def prune_batchnorm2d_out(bn: nn.BatchNorm2d, keep_idx: torch.Tensor):
    """Prune BatchNorm2d channels."""
    bn.weight = nn.Parameter(bn.weight.data[keep_idx])
    bn.bias = nn.Parameter(bn.bias.data[keep_idx])
    bn.running_mean = bn.running_mean[keep_idx]
    bn.running_var = bn.running_var[keep_idx]
    bn.num_features = len(keep_idx)


# ---------------------------------------------------------------------------
# Bottleneck
# ---------------------------------------------------------------------------

def prune_bottleneck_hidden(bn_module, keep_idx: torch.Tensor):
    """Prune the hidden channels of a Bottleneck.

    Adjusts cv1 output and cv2 input to match the new hidden channel count.

    Args:
        bn_module: Bottleneck module with .cv1 and .cv2.
        keep_idx: 1-D tensor of hidden channel indices to keep.
    """
    prune_conv_out(bn_module.cv1, keep_idx)
    prune_conv_in(bn_module.cv2, keep_idx)


def prune_bottleneck_inout(bn_module, keep_idx: torch.Tensor):
    """Prune input/output channels of a Bottleneck (when shortcut add=True, c1==c2).

    Args:
        bn_module: Bottleneck module.
        keep_idx: 1-D tensor of channel indices to keep.
    """
    # Input side: cv1 input
    prune_conv_in(bn_module.cv1, keep_idx)
    # Output side: cv2 output
    prune_conv_out(bn_module.cv2, keep_idx)


# ---------------------------------------------------------------------------
# C2f / C3k2 (C3k2 inherits C2f)
# ---------------------------------------------------------------------------

def prune_c2f_out(module, keep_idx: torch.Tensor):
    """Prune C2f/C3k2 output channels = prune cv2 output."""
    prune_conv_out(module.cv2, keep_idx)


def prune_c2f_in(module, keep_idx: torch.Tensor):
    """Prune C2f/C3k2 input channels = prune cv1 input."""
    prune_conv_in(module.cv1, keep_idx)


def prune_c2f_internal(module, keep_hidden: torch.Tensor):
    """Prune C2f/C3k2 internal hidden channels.

    C2f structure:
        cv1: in -> 2*c (split into c, c)
        m[i]: Bottleneck(c, c)  (each takes c, outputs c)
        cv2: (2+n)*c -> out

    When we prune hidden channels c -> len(keep_hidden):
        - cv1 output: 2*c -> 2*len(keep_hidden)
        - Each Bottleneck: c -> len(keep_hidden)
        - cv2 input: (2+n)*c -> (2+n)*len(keep_hidden)

    Args:
        module: C2f or C3k2 module.
        keep_hidden: 1-D tensor of hidden channel indices to keep.
    """
    n = len(module.m)
    new_c = len(keep_hidden)

    # cv1 output: 2*c channels, prune both halves with same indices
    keep_cv1_out = torch.cat([keep_hidden, keep_hidden + module.c])
    prune_conv_out(module.cv1, keep_cv1_out)

    # Each Bottleneck: prune in/out to new_c
    for bottleneck in module.m:
        prune_bottleneck_inout(bottleneck, keep_hidden)
        # Also prune hidden channels inside bottleneck
        c_ = bottleneck.cv1.conv.out_channels
        if c_ > 0:
            keep_bn_hidden = _compute_sub_keep(bottleneck.cv1, c_)
            prune_bottleneck_hidden(bottleneck, keep_bn_hidden)

    # cv2 input: (2+n)*c channels
    # Build the concatenated keep indices with offsets
    chunks = [keep_hidden + i * module.c for i in range(2 + n)]
    keep_cv2_in = torch.cat(chunks)
    prune_conv_in(module.cv2, keep_cv2_in)

    # Update stored hidden channel count
    module.c = new_c


def _compute_sub_keep(conv_module, n_channels: int) -> torch.Tensor:
    """Return all indices (no pruning) for a sub-module's channels."""
    device = conv_module.conv.weight.device
    return torch.arange(n_channels, device=device)


# ---------------------------------------------------------------------------
# C3k (inherits C3) - used inside C3k2 when c3k=True
# ---------------------------------------------------------------------------

def prune_c3k_internal(module, keep_hidden: torch.Tensor):
    """Prune C3k internal hidden channels.

    C3k structure (inherits C3):
        cv1: in -> c_  (c_ = c2*e)
        cv2: in -> c_
        cv3: 2*c_ -> out
        m: Sequential of Bottleneck(c_, c_)

    Args:
        module: C3k module.
        keep_hidden: 1-D tensor of hidden channel indices to keep.
    """
    # cv1 output
    prune_conv_out(module.cv1, keep_hidden)
    # cv2 output
    prune_conv_out(module.cv2, keep_hidden)

    # Each Bottleneck in m
    for bottleneck in module.m:
        prune_bottleneck_inout(bottleneck, keep_hidden)
        c_ = bottleneck.cv1.conv.out_channels
        if c_ > 0:
            keep_bn_hidden = torch.arange(c_, device=keep_hidden.device)
            prune_bottleneck_hidden(bottleneck, keep_bn_hidden)

    # cv3 input = cat(m(cv1(x)), cv2(x)) = 2 * new_c after pruning
    new_c = len(keep_hidden)
    keep_cv3_in = torch.arange(2 * new_c, device=keep_hidden.device)
    prune_conv_in(module.cv3, keep_cv3_in)


# ---------------------------------------------------------------------------
# SPPF
# ---------------------------------------------------------------------------

def prune_sppf_out(module, keep_idx: torch.Tensor):
    """Prune SPPF output channels = prune cv2 output."""
    prune_conv_out(module.cv2, keep_idx)


def prune_sppf_in(module, keep_idx: torch.Tensor):
    """Prune SPPF input channels = prune cv1 input."""
    prune_conv_in(module.cv1, keep_idx)


def prune_sppf_internal(module, keep_hidden: torch.Tensor):
    """Prune SPPF internal hidden channels.

    SPPF structure:
        cv1: in -> c_ (c_ = c1//2)
        MaxPool2d x n (no params)
        cv2: c_*(n+1) -> out

    Args:
        module: SPPF module.
        keep_hidden: 1-D tensor of hidden channel indices to keep.
    """
    n = module.n
    prune_conv_out(module.cv1, keep_hidden)
    # cv2 input = (n+1) * c_ channels
    keep_cv2_in = torch.cat([keep_hidden + i * len(keep_hidden) for i in range(n + 1)])
    prune_conv_in(module.cv2, keep_cv2_in)


# ---------------------------------------------------------------------------
# C2PSA
# ---------------------------------------------------------------------------

def prune_c2psa_out(module, keep_idx: torch.Tensor):
    """Prune C2PSA output channels.

    C2PSA requires c1 == c2, so this also constrains input.
    Only prune cv2 output here; input pruning is separate.
    """
    prune_conv_out(module.cv2, keep_idx)


def prune_c2psa_in(module, keep_idx: torch.Tensor):
    """Prune C2PSA input channels = prune cv1 input."""
    prune_conv_in(module.cv1, keep_idx)


def prune_c2psa_internal(module, keep_hidden: torch.Tensor):
    """Prune C2PSA internal hidden channels.

    C2PSA structure:
        cv1: in -> 2*c (split into a=c, b=c)
        m: Sequential of PSABlock(c) applied to b
        cv2: 2*c -> out

    Args:
        module: C2PSA module.
        keep_hidden: 1-D tensor of hidden channel indices to keep.
    """
    old_c = module.c
    new_c = len(keep_hidden)

    # cv1 output: 2*c
    keep_cv1_out = torch.cat([keep_hidden, keep_hidden + old_c])
    prune_conv_out(module.cv1, keep_cv1_out)

    # PSABlock internals - each PSABlock has attn and ffn
    for psa_block in module.m:
        _prune_psa_block(psa_block, keep_hidden)

    # cv2 input: 2*c
    keep_cv2_in = torch.cat([keep_hidden, keep_hidden + old_c])
    prune_conv_in(module.cv2, keep_cv2_in)

    module.c = new_c


def _prune_psa_block(psa_block, keep_idx: torch.Tensor):
    """Prune a PSABlock's internal channels.

    PSABlock structure:
        attn: Attention(dim=c, num_heads, attn_ratio)
        ffn: nn.Sequential(Conv(c, c*2), Conv(c*2, c))

    Args:
        psa_block: PSABlock module.
        keep_idx: 1-D tensor of channel indices to keep.
    """
    new_c = len(keep_idx)

    # Prune Attention module
    attn = psa_block.attn
    if hasattr(attn, "qkv"):
        # qkv: Conv(c, c + 2*key_dim)
        # We need to prune input channels
        prune_conv_in(attn.qkv, keep_idx)
        # Output: c + 2*key_dim - keep q part pruned, k/v unchanged
        # Actually for simplicity, keep attention key_dim unchanged
        # Only prune the input dimension
    if hasattr(attn, "proj"):
        # proj: Conv(c, c)
        prune_conv_in(attn.proj, keep_idx)
        prune_conv_out(attn.proj, keep_idx)
    if hasattr(attn, "pe"):
        # pe: Conv(c, c, groups=c) - DWConv
        if attn.pe is not None:
            prune_conv_in(attn.pe, keep_idx)
            prune_conv_out(attn.pe, keep_idx)

    # Prune FFN: Sequential(Conv(c, c*2), Conv(c*2, c))
    ffn = psa_block.ffn
    if isinstance(ffn, nn.Sequential) and len(ffn) >= 2:
        # First conv: input c -> output c*2
        prune_conv_in(ffn[0], keep_idx)
        # Keep all expanded channels of ffn[0] output, prune ffn[1] output
        prune_conv_out(ffn[1], keep_idx)


# ---------------------------------------------------------------------------
# C3 / C3x / C3Ghost / RepC3
# ---------------------------------------------------------------------------

def prune_c3_out(module, keep_idx: torch.Tensor):
    """Prune C3 output channels = prune cv3 output.

    C3 structure:
        cv1: Conv(c1, c_, 1, 1)
        cv2: Conv(c1, c_, 1, 1)
        m: Sequential(Bottleneck(c_, c_) for _ in range(n))
        cv3: Conv(2*c_, c2, 1)
        output = cat(m(cv1(x)), cv2(x)) -> cv3 -> out

    Args:
        module: C3 (or C3x, C3Ghost, RepC3) module.
        keep_idx: 1-D tensor of output channel indices to keep.
    """
    prune_conv_out(module.cv3, keep_idx)


def prune_c3_in(module, keep_idx: torch.Tensor):
    """Prune C3 input channels = prune cv1 and cv2 input.

    Both cv1 and cv2 receive the same input, so prune both with same indices.

    Args:
        module: C3 (or C3x, C3Ghost, RepC3) module.
        keep_idx: 1-D tensor of input channel indices to keep.
    """
    prune_conv_in(module.cv1, keep_idx)
    prune_conv_in(module.cv2, keep_idx)


def prune_c3_internal(module, keep_hidden: torch.Tensor):
    """Prune C3 internal hidden channels.

    C3 structure:
        cv1: c1 -> c_ (c_ = c2*e)
        cv2: c1 -> c_
        m: Sequential of Bottleneck(c_, c_)
        cv3: 2*c_ -> c2

    When pruning hidden channels c_ -> len(keep_hidden):
        - cv1 output: c_ -> len(keep_hidden)
        - cv2 output: c_ -> len(keep_hidden)
        - Each Bottleneck: prune its in/out to keep_hidden
        - cv3 input: 2*c_ -> 2*new_c

    Args:
        module: C3 (or C3x, C3Ghost, RepC3) module.
        keep_hidden: 1-D tensor of hidden channel indices to keep.
    """
    new_c = len(keep_hidden)

    # cv1 and cv2 output: c_ channels each
    prune_conv_out(module.cv1, keep_hidden)
    prune_conv_out(module.cv2, keep_hidden)

    # Each Bottleneck in m: prune in/out to new_c
    for bottleneck in module.m:
        prune_bottleneck_inout(bottleneck, keep_hidden)
        # Also prune hidden channels inside bottleneck
        c_ = bottleneck.cv1.conv.out_channels
        if c_ > 0:
            keep_bn_hidden = torch.arange(c_, device=keep_hidden.device)
            prune_bottleneck_hidden(bottleneck, keep_bn_hidden)

    # cv3 input = cat(m(cv1(x)), cv2(x)) = 2 * new_c after pruning
    keep_cv3_in = torch.arange(2 * new_c, device=keep_hidden.device)
    prune_conv_in(module.cv3, keep_cv3_in)


# ---------------------------------------------------------------------------
# BottleneckCSP
# ---------------------------------------------------------------------------

def prune_bottleneck_csp_out(module, keep_idx: torch.Tensor):
    """Prune BottleneckCSP output channels = prune cv4 output.

    BottleneckCSP structure:
        cv1: Conv(c1, c_, 1, 1)
        cv2: Conv(c1, c_, 1, 1)
        m: n x Bottleneck(c_, c_)
        cv3: Conv(c_, c_, 1, 1) applied inside m
        bn: BatchNorm2d(2*c_) applied to cat(cv2, cv3)
        act: SiLU
        cv4: Conv(2*c_, c2, 1)
        output = cat(cv1(x), m(cv2(x))) -> bn -> act -> cv4 -> out

    Args:
        module: BottleneckCSP module.
        keep_idx: 1-D tensor of output channel indices to keep.
    """
    prune_conv_out(module.cv4, keep_idx)


def prune_bottleneck_csp_in(module, keep_idx: torch.Tensor):
    """Prune BottleneckCSP input channels = prune cv1 and cv2 input.

    Args:
        module: BottleneckCSP module.
        keep_idx: 1-D tensor of input channel indices to keep.
    """
    prune_conv_in(module.cv1, keep_idx)
    prune_conv_in(module.cv2, keep_idx)


def prune_bottleneck_csp_internal(module, keep_hidden: torch.Tensor):
    """Prune BottleneckCSP internal hidden channels.

    The internal hidden channels are c_ (the expansion channels).
    Both cv1/cv2 output c_ channels, and each Bottleneck takes c_ -> c_.

    Args:
        module: BottleneckCSP module.
        keep_hidden: 1-D tensor of hidden channel indices to keep.
    """
    new_c = len(keep_hidden)

    # cv1 and cv2 output: c_ channels each
    prune_conv_out(module.cv1, keep_hidden)
    prune_conv_out(module.cv2, keep_hidden)

    # Each Bottleneck in m: prune in/out to new_c
    for bottleneck in module.m:
        prune_bottleneck_inout(bottleneck, keep_hidden)
        c_ = bottleneck.cv1.conv.out_channels
        if c_ > 0:
            keep_bn_hidden = torch.arange(c_, device=keep_hidden.device)
            prune_bottleneck_hidden(bottleneck, keep_bn_hidden)

    # cv3 inside each bottleneck - handled by bottleneck pruning above
    # bn takes 2*c_ input (unchanged, it's the concat of two c_ streams)
    # cv4 input: 2*c_ -> 2*new_c
    keep_cv4_in = torch.cat([keep_hidden, keep_hidden + new_c])
    prune_conv_in(module.cv4, keep_cv4_in)


# ---------------------------------------------------------------------------
# ADown
# ---------------------------------------------------------------------------

def prune_adown_out(module, keep_idx: torch.Tensor):
    """Prune ADown output channels.

    ADown structure:
        cv1: Conv(c1//2, c2//2, 3, 2, 1)  -- downsample branch
        cv2: Conv(c1//2, c2//2, 1, 1, 0)  -- pool branch
        input chunk(2) -> cv1 + cv2 -> cat -> out

    Output channels = c2, prune cv1 output (c2//2) and adjust cv2 to match.

    Args:
        module: ADown module.
        keep_idx: 1-D tensor of output channel indices to keep.
    """
    old_out = module.cv1.conv.out_channels
    new_out = len(keep_idx)
    half = new_out

    # Prune cv1 output: its output is the first half of concat
    # keep_idx[0:half] -> indices for cv1 output
    # But since they are concatenated equally, we just prune cv1 output
    prune_conv_out(module.cv1, torch.arange(half, device=keep_idx.device))

    # Prune cv2 output to match: cv2 output must equal cv1 output for concat
    prune_conv_out(module.cv2, torch.arange(half, device=keep_idx.device))

    # Update stored channel count
    module.c = half


def prune_adown_in(module, keep_idx: torch.Tensor):
    """Prune ADown input channels.

    ADown takes input, chunks it into two halves. The first half goes through
    avg_pool -> cv1, the second half goes through max_pool -> cv2.

    Args:
        module: ADown module.
        keep_idx: 1-D tensor of input channel indices to keep.
    """
    prune_conv_in(module.cv1, keep_idx)
    prune_conv_in(module.cv2, keep_idx)


def prune_adown_internal(module, keep_hidden: torch.Tensor):
    """Prune ADown internal hidden channels (not applicable).

    ADown does not have internal hidden channels - both branches are simple
    Conv -> out with no intermediate channel changes.
    """
    pass


# ---------------------------------------------------------------------------
# SPPELAN
# ---------------------------------------------------------------------------

def prune_sppelan_out(module, keep_idx: torch.Tensor):
    """Prune SPPELAN output channels = prune cv5 output.

    SPPELAN structure:
        cv1: Conv(c1, c3, 1, 1)
        cv2/cv3/cv4: MaxPool2d(k) x 3 (no params)
        cv5: Conv(4*c3, c2, 1)
        output = cat(cv1, cv2, cv3, cv4) -> cv5 -> out

    Args:
        module: SPPELAN module.
        keep_idx: 1-D tensor of output channel indices to keep.
    """
    prune_conv_out(module.cv5, keep_idx)


def prune_sppelan_in(module, keep_idx: torch.Tensor):
    """Prune SPPELAN input channels = prune cv1 input.

    Args:
        module: SPPELAN module.
        keep_idx: 1-D tensor of input channel indices to keep.
    """
    prune_conv_in(module.cv1, keep_idx)


def prune_sppelan_internal(module, keep_hidden: torch.Tensor):
    """Prune SPPELAN internal hidden channels.

    SPPELAN structure:
        cv1: c1 -> c3
        cv2/cv3/cv4: MaxPool2d (no params, channels unchanged)
        cv5: 4*c3 -> c2

    When pruning hidden channels c3 -> len(keep_hidden):
        - cv1 output: c3 -> len(keep_hidden)
        - cv2/cv3/cv4: pass through with same channels (no params)
        - cv5 input: 4*c3 -> 4*new_c

    Args:
        module: SPPELAN module.
        keep_hidden: 1-D tensor of hidden channel indices to keep.
    """
    new_c = len(keep_hidden)

    # cv1 output: c3 -> new_c
    prune_conv_out(module.cv1, keep_hidden)

    # cv2/cv3/cv4 are MaxPool2d with no params, no pruning needed

    # cv5 input: 4*c3 -> 4*new_c
    # Build indices: [0, new_c, 2*new_c, 3*new_c] for 4 branches
    keep_cv5_in = torch.cat([keep_hidden + i * new_c for i in range(4)])
    prune_conv_in(module.cv5, keep_cv5_in)


# ---------------------------------------------------------------------------
# SPP (Spatial Pyramid Pooling)
# ---------------------------------------------------------------------------

def prune_spp_out(module, keep_idx: torch.Tensor):
    """Prune SPP output channels = prune cv2 output.

    SPP structure:
        cv1: Conv(c1, c_, 1, 1)
        m: 3 x MaxPool (no params)
        cv2: Conv(c_ * 4, c2, 1, 1)  [c_ = c1 // 2, 4 = len(k)+1]
        output = cat(cv1(x), m[0](x), m[1](x), m[2](x)) -> cv2 -> out

    Args:
        module: SPP module.
        keep_idx: 1-D tensor of output channel indices to keep.
    """
    prune_conv_out(module.cv2, keep_idx)


def prune_spp_in(module, keep_idx: torch.Tensor):
    """Prune SPP input channels = prune cv1 input.

    Args:
        module: SPP module.
        keep_idx: 1-D tensor of input channel indices to keep.
    """
    prune_conv_in(module.cv1, keep_idx)


def prune_spp_internal(module, keep_hidden: torch.Tensor):
    """Prune SPP internal hidden channels.

    SPP structure:
        cv1: c1 -> c_ (c_ = c1 // 2)
        m: 3 x MaxPool (no params)
        cv2: 4*c_ -> c2

    When pruning hidden channels c_ -> len(keep_hidden):
        - cv1 output: c_ -> len(keep_hidden)
        - MaxPool: no params, channels unchanged
        - cv2 input: 4*c_ -> 4*new_c

    Args:
        module: SPP module.
        keep_hidden: 1-D tensor of hidden channel indices to keep.
    """
    new_c = len(keep_hidden)

    # cv1 output: c_ -> new_c
    prune_conv_out(module.cv1, keep_hidden)

    # MaxPool layers have no params, no pruning needed

    # cv2 input: 4*c_ -> 4*new_c
    # Build indices: [0, new_c, 2*new_c, 3*new_c] for 4 branches
    keep_cv2_in = torch.cat([keep_hidden + i * new_c for i in range(4)])
    prune_conv_in(module.cv2, keep_cv2_in)


# ---------------------------------------------------------------------------
# GhostConv
# ---------------------------------------------------------------------------

def prune_ghostconv_out(module, keep_idx: torch.Tensor):
    """Prune GhostConv output channels.

    GhostConv structure:
        cv1: Conv(c1, c_, k, s)       # primary branch
        cv2: Conv(c_, c_, 5, 1, g=c_)  # cheap/dw branch
        output = cat(y, cv2(y)) = 2 * c_ = c2  (when c2 is even)

    cv2 outputs c_ channels (the cheap operation), and the final output
    is cat(cv1_out, cv2_out) = cat(c_, c_) = 2*c_ = c2.

    Pruning output means we keep 'keep_idx' channels from the final concat.
    Since cv1 and cv2 each output c_ channels which are concatenated equally,
    we keep the first len(keep_idx)//2 from each.

    Args:
        module: GhostConv module.
        keep_idx: 1-D tensor of output channel indices to keep.
    """
    half = len(keep_idx) // 2
    new_c = half  # each branch outputs new_c channels

    # Prune cv1 output: keep first new_c channels (primary branch)
    prune_conv_out(module.cv1, torch.arange(new_c, device=keep_idx.device))

    # Prune cv2 output: keep first new_c channels (cheap branch)
    prune_conv_out(module.cv2, torch.arange(new_c, device=keep_idx.device))


def prune_ghostconv_in(module, keep_idx: torch.Tensor):
    """Prune GhostConv input channels = prune cv1 input.

    GhostConv takes a single input that feeds into cv1.

    Args:
        module: GhostConv module.
        keep_idx: 1-D tensor of input channel indices to keep.
    """
    prune_conv_in(module.cv1, keep_idx)


# ---------------------------------------------------------------------------
# C2fAttn
# ---------------------------------------------------------------------------

def prune_c2fattn_out(module, keep_idx: torch.Tensor):
    """Prune C2fAttn output channels = prune cv2 output.

    C2fAttn structure:
        cv1: Conv(c1, 2*self.c, 1, 1)
        m: n x Bottleneck(self.c, self.c)
        attn: MaxSigmoidAttnBlock(self.c, self.c, ...)
        cv2: Conv((3+n)*self.c, c2, 1)
        output = cat(bottleneck_out, attn_out) -> cv2 -> out

    Args:
        module: C2fAttn module.
        keep_idx: 1-D tensor of output channel indices to keep.
    """
    prune_c2f_out(module, keep_idx)


def prune_c2fattn_in(module, keep_idx: torch.Tensor):
    """Prune C2fAttn input channels = prune cv1 input.

    Args:
        module: C2fAttn module.
        keep_idx: 1-D tensor of input channel indices to keep.
    """
    prune_c2f_in(module, keep_idx)


# ---------------------------------------------------------------------------
# A2C2f
# ---------------------------------------------------------------------------

def prune_a2c2f_out(module, keep_idx: torch.Tensor):
    """Prune A2C2f output channels = prune cv2 output.

    A2C2f structure:
        cv1: Conv(c1, c_, 1, 1)  where c_ = int(c2 * e)
        m: n x (ABlock or C3k)
        gamma: learnable residual scaling (when a2=True)
        cv2: Conv((1+n)*c_, c2, 1)
        output = cat(m(cv1(x)), x) -> cv2 -> out

    Args:
        module: A2C2f module.
        keep_idx: 1-D tensor of output channel indices to keep.
    """
    prune_conv_out(module.cv2, keep_idx)


def prune_a2c2f_in(module, keep_idx: torch.Tensor):
    """Prune A2C2f input channels = prune cv1 input.

    Args:
        module: A2C2f module.
        keep_idx: 1-D tensor of input channel indices to keep.
    """
    prune_conv_in(module.cv1, keep_idx)


# ---------------------------------------------------------------------------
# SCDown
# ---------------------------------------------------------------------------

def prune_scdown_out(module, keep_idx: torch.Tensor):
    """Prune SCDown output channels.

    SCDown structure:
        cv1: Conv(c1, c2, 1, 1)      # reduce channels
        cv2: Conv(c2, c2, k, s, g=c2)  # DWConv downsampling
        output = cv2(cv1(x))  # serial

    Args:
        module: SCDown module.
        keep_idx: 1-D tensor of output channel indices to keep.
    """
    prune_conv_out(module.cv2, keep_idx)


def prune_scdown_in(module, keep_idx: torch.Tensor):
    """Prune SCDown input channels = prune cv1 input.

    Args:
        module: SCDown module.
        keep_idx: 1-D tensor of input channel indices to keep.
    """
    prune_conv_in(module.cv1, keep_idx)


# ---------------------------------------------------------------------------
# Detect head (input-only pruning)
# ---------------------------------------------------------------------------

def prune_detect_in(detect, scale_idx: int, keep_idx: torch.Tensor):
    """Update Detect head input channels for a given scale.

    Args:
        detect: Detect module.
        scale_idx: Which scale (0=P3, 1=P4, 2=P5).
        keep_idx: 1-D tensor of input channel indices to keep.
    """
    # cv2[i]: Sequential(Conv, Conv, Conv2d)
    cv2_seq = detect.cv2[scale_idx]
    prune_conv_in(cv2_seq[0], keep_idx)

    # cv3[i]: Sequential(Sequential(DWConv, Conv), Sequential(DWConv, Conv), Conv2d)
    # or legacy: Sequential(Conv, Conv, Conv2d)
    cv3_seq = detect.cv3[scale_idx]
    first = cv3_seq[0]
    if isinstance(first, nn.Sequential):
        # Non-legacy: first = Sequential(DWConv, Conv)
        # DWConv input
        _prune_dwconv_module(first[0], keep_idx)
        # Conv after DWConv: its input is DWConv output (same channels)
        prune_conv_in(first[1], keep_idx)
    else:
        # Legacy: first = Conv
        prune_conv_in(first, keep_idx)


def _prune_dwconv_module(dwconv_module, keep_idx: torch.Tensor):
    """Prune a DWConv module (groups=channels)."""
    conv = dwconv_module.conv
    conv.weight = nn.Parameter(conv.weight.data[keep_idx])
    if conv.bias is not None:
        conv.bias = nn.Parameter(conv.bias.data[keep_idx])
    conv.in_channels = len(keep_idx)
    conv.out_channels = len(keep_idx)
    conv.groups = len(keep_idx)

    if hasattr(dwconv_module, "bn"):
        bn = dwconv_module.bn
        bn.weight = nn.Parameter(bn.weight.data[keep_idx])
        bn.bias = nn.Parameter(bn.bias.data[keep_idx])
        bn.running_mean = bn.running_mean[keep_idx]
        bn.running_var = bn.running_var[keep_idx]
        bn.num_features = len(keep_idx)


# ---------------------------------------------------------------------------
# Fusion modules (FeatureFusion, FCMFeatureFusion, MCFGatedFusion)
# ---------------------------------------------------------------------------


def prune_featurefusion_out(module, keep_idx: torch.Tensor):
    """Prune output channels of FeatureFusion.

    FeatureFusion uses ChannelEmbed which has:
    - channel_emb.residual: 1x1 conv (identity path, out_channels = dim)
    - channel_emb.channel_embed[3]: final 1x1 conv after depthwise (out_channels = dim)

    The output channels are controlled by channel_emb.channel_embed[3].
    """
    channel_emb = getattr(module, "channel_emb", None)
    if channel_emb is None:
        raise ValueError(f"FeatureFusion: channel_emb not found on {type(module).__name__}")

    channel_embed_seq = getattr(channel_emb, "channel_embed", None)
    if channel_embed_seq is None or len(channel_embed_seq) < 4:
        raise ValueError(f"FeatureFusion: channel_embed seq not found or too short")

    prune_raw_conv2d_out(channel_emb.residual, keep_idx)
    prune_raw_conv2d_out(channel_embed_seq[3], keep_idx)
    if len(channel_embed_seq) >= 5 and isinstance(channel_embed_seq[4], nn.BatchNorm2d):
        prune_batchnorm2d_out(channel_embed_seq[4], keep_idx)
    if isinstance(channel_emb.norm, nn.BatchNorm2d):
        prune_batchnorm2d_out(channel_emb.norm, keep_idx)
    channel_emb.out_channels = len(keep_idx)


def prune_fcmfeaturefusion_out(module, keep_idx: torch.Tensor):
    """Prune output channels of FCMFeatureFusion.

    FCMFeatureFusion chains FCM -> FeatureFusion.
    Prune the inner FeatureFusion's output channels.
    """
    ffm = getattr(module, "ffm", None)
    if ffm is None:
        raise ValueError(f"FCMFeatureFusion: ffm (FeatureFusion) not found")
    prune_featurefusion_out(ffm, keep_idx)


def prune_mcfgatedfusion_out(module, keep_idx: torch.Tensor):
    """Prune output channels of MCFGatedFusion.

    MCFGatedFusion has:
    - gate: Conv2d for gating aux modality
    - post: Conv (optional, only in concat mode)

    In concat mode, prune post conv output.
    In add mode, the output follows main input channels (gate output = main channels).
    """
    post = getattr(module, "post", None)
    if post is not None:
        # concat mode: prune post conv output
        prune_conv_out(post, keep_idx)
    elif hasattr(module, "gate"):
        # add mode: gate output = main channels, prune gate output dimension
        gate = module.gate
        gate.out_channels = len(keep_idx)
        gate.weight = nn.Parameter(gate.weight.data[keep_idx])
        if gate.bias is not None:
            gate.bias = nn.Parameter(gate.bias.data[keep_idx])
        if hasattr(module, "bn") and hasattr(module.bn, "weight"):
            bn = module.bn
            bn.num_features = len(keep_idx)
            bn.weight = nn.Parameter(bn.weight.data[keep_idx])
            if bn.bias is not None:
                bn.bias = nn.Parameter(bn.bias.data[keep_idx])
            bn.running_mean = bn.running_mean[keep_idx]
            bn.running_var = bn.running_var[keep_idx]


def prune_featurefusion_in(module, keep_idx: torch.Tensor):
    """Prune input channels of FeatureFusion.

    FeatureFusion input channels = dim * 2 (concatenation of two modalities).
    """
    channel_emb = getattr(module, "channel_emb", None)
    if channel_emb is None:
        raise ValueError(f"FeatureFusion: channel_emb not found")

    prune_raw_conv2d_in(channel_emb.channel_embed[0], keep_idx)
    prune_raw_conv2d_in(channel_emb.residual, keep_idx)


def prune_mcfgatedfusion_in(module, keep_idx: torch.Tensor):
    """Prune input channels of MCFGatedFusion.

    MCFGatedFusion has two inputs:
    - main (index 0): pass-through, no pruning needed
    - aux (index 1): gated by gate conv
    """
    # Only prune aux input through gate
    gate = getattr(module, "gate", None)
    if gate is not None:
        gate.in_channels = len(keep_idx)
        gate.weight = nn.Parameter(gate.weight.data[:, keep_idx, :, :])


# ---------------------------------------------------------------------------
# SequenceShuffleAttention (SSA)
# ---------------------------------------------------------------------------


def prune_ssa_in(module, keep_idx: torch.Tensor):
    """Prune SequenceShuffleAttention input (and output) channels.

    SSA is a shape-preserving module: output channels == input channels.
    Its internal Conv2d is grouped with in_channels == out_channels == c. Because
    SSA applies ``channel_shuffle()`` before the grouped conv, ``keep_idx`` lives in
    the original input-channel space and must first be projected into shuffled groups.

    Args:
        module: SequenceShuffleAttention instance.
        keep_idx: 1-D tensor of channel indices to keep.
    """
    gating = getattr(module, "gating", None)
    if gating is None:
        return

    for layer in gating:
        if isinstance(layer, nn.Conv2d):
            c = layer.in_channels
            g = layer.groups
            group_map = _ssa_group_input_indices(c, g, keep_idx.device)
            kept_total = int(keep_idx.numel())
            if kept_total == 0:
                raise ValueError("SSA prune received empty keep_idx")
            if kept_total % g != 0:
                raise ValueError(
                    f"SSA prune requires keep_idx count divisible by groups: keep={kept_total}, groups={g}"
                )

            new_weight_parts = []
            bias_keep_parts = []
            kept_per_group = None
            for gi in range(g):
                original_group = group_map[gi]
                local_keep = torch.nonzero(torch.isin(original_group, keep_idx), as_tuple=False).flatten()
                if kept_per_group is None:
                    kept_per_group = int(local_keep.numel())
                    if kept_per_group <= 0:
                        raise ValueError(
                            f"SSA prune produced empty local group keep at group {gi}; keep={kept_total}, groups={g}"
                        )
                elif int(local_keep.numel()) != kept_per_group:
                    raise ValueError(
                        "SSA prune requires equal kept channels per shuffled group, "
                        f"but group {gi} has {int(local_keep.numel())} vs expected {kept_per_group}"
                    )

                start = gi * (c // g)
                end = start + (c // g)
                group_weight = layer.weight.data[start:end]
                new_weight_parts.append(group_weight[local_keep][:, local_keep])
                bias_keep_parts.append(local_keep + start)

            layer.weight = nn.Parameter(torch.cat(new_weight_parts, dim=0))
            if layer.bias is not None:
                layer.bias = nn.Parameter(layer.bias.data[torch.cat(bias_keep_parts)])

            new_c = len(keep_idx)
            if new_c % g != 0:
                raise ValueError(f"SSA pruned channels {new_c} must remain divisible by groups {g}")
            layer.in_channels = new_c
            layer.out_channels = new_c

    # Update cached channel count
    module._c = len(keep_idx)
