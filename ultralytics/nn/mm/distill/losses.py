# Ultralytics Multimodal Distillation - Loss Functions
# Distillation loss computation for feature-level knowledge transfer.

"""
Distillation loss functions.

This module provides shared loss utilities used by the family-level adapters.

**Output distillation** is NO LONGER implemented here.  Family-specific output
distillation has been moved to:
- ``output_yolomm.py`` -- foreground-guided + cls/loc decoupled (YOLOMM)
- ``output_rtdetrmm.py`` -- query-aware + cls/box decoupled (RTDETRMM)

The old ``compute_output_distill_loss`` (global flatten-MSE) has been retired
from the main path and is kept ONLY as a deprecated helper.  It is NOT called
by any adapter.

Feature distillation:
- Detection-friendly 4-term formulation with foreground-guided masking:
  L = L_fg + bg_weight * L_bg + cwd_weight * L_cwd + ctx_weight * L_ctx
- Automatic spatial alignment (shared across families).
- Channel adaptation handled upstream by ``ChannelAdapter`` in adapters.py.

Key constraints:
- Dual single-modal teachers: losses computed separately, then averaged.
- Spatial size mismatch: automatically aligned via interpolation.
- Channel mismatch: handled by ``ChannelAdapter`` before reaching loss functions.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER


# ---------------------------------------------------------------------------
# Feature distillation default hyperparameters (internal constants)
# ---------------------------------------------------------------------------
_FEATURE_FG_WEIGHT = 1.0
_FEATURE_BG_WEIGHT = 0.25
_FEATURE_CWD_WEIGHT = 0.5
_FEATURE_CTX_WEIGHT = 0.25
_FEATURE_CWD_TAU = 1.0


# ---------------------------------------------------------------------------
# Feature distillation helper functions
# ---------------------------------------------------------------------------

def normalize_feature_map(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize feature map using spatial mean and std.

    Args:
        feat: Feature map (B, C, H, W).
        eps: Minimum std value to prevent division by zero.

    Returns:
        Normalized feature map, same shape as input.
    """
    mean = feat.mean(dim=(2, 3), keepdim=True)
    std = feat.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (feat - mean) / std


def compute_masked_smooth_l1(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked SmoothL1 loss on normalised features.

    Normalises both feature maps first, then computes element-wise SmoothL1
    weighted by the given mask (fg_mask or bg_mask).

    Args:
        student_feat: Student feature map (B, C, H, W).
        teacher_feat: Teacher feature map (B, C, H, W), will be detached.
        mask: Spatial mask (B, 1, H, W), values in [0, 1].
        eps: Small constant for numerical stability.

    Returns:
        Scalar masked SmoothL1 loss.
    """
    mask_sum = mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, device=student_feat.device, dtype=student_feat.dtype)

    s_norm = normalize_feature_map(student_feat)
    t_norm = normalize_feature_map(teacher_feat)

    # Element-wise SmoothL1: (B, C, H, W)
    point_loss = F.smooth_l1_loss(s_norm, t_norm.detach(), reduction="none")

    # mask (B, 1, H, W) broadcasts to (B, C, H, W)
    C = student_feat.shape[1]
    loss = (point_loss * mask).sum() / (mask_sum * C + eps)
    return loss


def compute_masked_cwd(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    mask: torch.Tensor,
    tau: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Channel-Wise Distribution Distillation loss on foreground region.

    For each channel, computes a spatial probability distribution (via softmax)
    over the masked region, then minimises the KL divergence between student
    and teacher distributions.

    Args:
        student_feat: Student feature map (B, C, H, W).
        teacher_feat: Teacher feature map (B, C, H, W), will be detached.
        mask: Foreground mask (B, 1, H, W), values in [0, 1].
        tau: Temperature for softmax smoothing.
        eps: Small constant for numerical stability.

    Returns:
        Scalar CWD loss.
    """
    # Binarize soft mask: any non-zero position counts as foreground for CWD
    binary_mask = (mask > 1e-6).float()  # (B, 1, H, W)

    fg_count = binary_mask.sum()
    if fg_count == 0:
        return torch.tensor(0.0, device=student_feat.device, dtype=student_feat.dtype)

    B, C, H, W = student_feat.shape

    # Flatten spatial dims: (B, C, H*W)
    s_flat = student_feat.reshape(B, C, H * W)
    t_flat = teacher_feat.detach().reshape(B, C, H * W)

    # Flatten binary mask: (B, 1, H*W) -> broadcast to (B, C, H*W)
    mask_flat = binary_mask.reshape(B, 1, H * W)

    # Apply mask: non-foreground positions get very small value to suppress in softmax
    large_neg = torch.tensor(-1e9, device=student_feat.device, dtype=student_feat.dtype)
    s_masked = torch.where(mask_flat > 0.5, s_flat / tau, large_neg)
    t_masked = torch.where(mask_flat > 0.5, t_flat / tau, large_neg)

    # Stabilize softmax: subtract spatial max per channel (only over fg positions)
    s_max = s_masked.max(dim=-1, keepdim=True).values.clamp(min=-1e8)
    t_max = t_masked.max(dim=-1, keepdim=True).values.clamp(min=-1e8)
    s_masked = s_masked - s_max
    t_masked = t_masked - t_max

    # Compute spatial distributions per channel
    s_dist = F.softmax(s_masked, dim=-1)
    t_dist = F.softmax(t_masked, dim=-1)

    # KL divergence: input is log(p), target is q
    # Clamp s_dist to avoid log(0) -- bg positions will have near-zero prob after softmax
    kl = F.kl_div(
        s_dist.clamp(min=eps).log(),
        t_dist,
        reduction="batchmean",
    )
    # Guard against residual NaN from edge cases
    if torch.isnan(kl) or torch.isinf(kl):
        return torch.tensor(0.0, device=student_feat.device, dtype=student_feat.dtype)
    return kl


def compute_masked_context_cosine(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Context cosine loss: foreground-pooled semantic vector similarity.

    Pools student and teacher features over the foreground region to produce
    (B, C) semantic vectors, then computes 1 - cosine_similarity.

    Args:
        student_feat: Student feature map (B, C, H, W).
        teacher_feat: Teacher feature map (B, C, H, W), will be detached.
        mask: Foreground mask (B, 1, H, W), values in [0, 1].
        eps: Small constant for numerical stability.

    Returns:
        Scalar context cosine loss (mean over batch).
    """
    # mask_sum per sample: (B, 1)
    mask_sum = mask.sum(dim=(2, 3))  # (B, 1)

    if mask_sum.sum() == 0:
        return torch.tensor(0.0, device=student_feat.device, dtype=student_feat.dtype)

    # Foreground-weighted pooling: (B, C)
    s_vec = (student_feat * mask).sum(dim=(2, 3)) / (mask_sum + eps)  # (B, C)
    t_vec = (teacher_feat.detach() * mask).sum(dim=(2, 3)) / (mask_sum + eps)  # (B, C)

    # Cosine similarity: (B,)
    cos_sim = F.cosine_similarity(s_vec, t_vec, dim=-1, eps=eps)

    loss = (1.0 - cos_sim).mean()
    return loss


# ---------------------------------------------------------------------------
# Feature-level distillation (shared across families)
# ---------------------------------------------------------------------------

def compute_feature_distill_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    fg_mask: torch.Tensor,
    bg_weight: float = _FEATURE_BG_WEIGHT,
    cwd_weight: float = _FEATURE_CWD_WEIGHT,
    ctx_weight: float = _FEATURE_CTX_WEIGHT,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute detection-friendly feature distillation loss.

    Replaces the previous global MSE with a 4-term formulation:
        L_total = L_fg + bg_weight * L_bg + cwd_weight * L_cwd + ctx_weight * L_ctx

    Args:
        student_feat: Student feature map (B, C, H_s, W_s).
        teacher_feat: Teacher feature map (B, C, H_t, W_t).
        fg_mask: Foreground soft mask (B, 1, H_m, W_m), values in [0, 1].
            From Story 2-1 (YOLOMM) or Story 2-2 (RTDETRMM) mask generators.
        bg_weight: Weight for background SmoothL1 term (default 0.25).
        cwd_weight: Weight for CWD term (default 0.5).
        ctx_weight: Weight for context cosine term (default 0.25).

    Returns:
        (total_loss, items_dict) where items_dict contains:
            - "fg": foreground SmoothL1 loss (detached)
            - "bg": background SmoothL1 loss (detached)
            - "cwd": channel-wise distribution loss (detached)
            - "ctx": context cosine loss (detached)
            - "total": total weighted loss (detached)

    Raises:
        AssertionError: If channel dimensions do not match (indicates upstream bug).
    """
    zero = torch.tensor(0.0, device=student_feat.device, dtype=student_feat.dtype)

    # --- Non-4D fallback: simple MSE, cannot use mask ---
    if student_feat.dim() != 4 or teacher_feat.dim() != 4:
        mse_loss = F.mse_loss(student_feat, teacher_feat.detach())
        items = {
            "fg": mse_loss.detach(),
            "bg": zero.detach(),
            "cwd": zero.detach(),
            "ctx": zero.detach(),
            "total": mse_loss.detach(),
        }
        return mse_loss, items

    # --- Channel consistency check (defensive) ---
    # Channel adaptation is handled upstream by ChannelAdapter in adapters.py.
    # By the time features reach this function, channels MUST already match.
    s_c = student_feat.shape[1]
    t_c = teacher_feat.shape[1]
    assert s_c == t_c, (
        f"BUG: channel mismatch reached loss function "
        f"(student={s_c}, teacher={t_c}). "
        f"This should have been handled by ChannelAdapter in adapters.py."
    )

    # --- Spatial alignment: interpolate teacher to student spatial size ---
    s_h, s_w = student_feat.shape[2], student_feat.shape[3]
    t_h, t_w = teacher_feat.shape[2], teacher_feat.shape[3]
    if (s_h, s_w) != (t_h, t_w):
        teacher_feat = F.interpolate(
            teacher_feat, size=(s_h, s_w), mode="bilinear", align_corners=False
        )

    # --- Spatial alignment: interpolate fg_mask to student spatial size ---
    m_h, m_w = fg_mask.shape[2], fg_mask.shape[3]
    if (m_h, m_w) != (s_h, s_w):
        fg_mask = F.interpolate(
            fg_mask, size=(s_h, s_w), mode="bilinear", align_corners=False
        )
        fg_mask = fg_mask.clamp(0.0, 1.0)

    # --- Derive bg_mask ---
    bg_mask = 1.0 - fg_mask

    # --- Compute 4 loss terms ---
    l_fg = compute_masked_smooth_l1(student_feat, teacher_feat, fg_mask)
    l_bg = compute_masked_smooth_l1(student_feat, teacher_feat, bg_mask)
    l_cwd = compute_masked_cwd(student_feat, teacher_feat, fg_mask, tau=_FEATURE_CWD_TAU)
    l_ctx = compute_masked_context_cosine(student_feat, teacher_feat, fg_mask)

    # --- Weighted sum ---
    total = (
        _FEATURE_FG_WEIGHT * l_fg
        + bg_weight * l_bg
        + cwd_weight * l_cwd
        + ctx_weight * l_ctx
    )

    # --- Build items dict (all detached) ---
    items = {
        "fg": l_fg.detach(),
        "bg": l_bg.detach(),
        "cwd": l_cwd.detach(),
        "ctx": l_ctx.detach(),
        "total": total.detach(),
    }

    return total, items


# ---------------------------------------------------------------------------
# DEPRECATED: Old global flatten-MSE output distillation
# ---------------------------------------------------------------------------

def compute_output_distill_loss(
    student_preds: Union[torch.Tensor, tuple],
    teacher_preds: Union[torch.Tensor, tuple],
    temperature: float = 4.0,
) -> torch.Tensor:
    """[DEPRECATED] Global flatten-MSE output distillation.

    .. deprecated::
        This function is NO LONGER used by the main distillation path.
        Family-specific output distillation has replaced it:
        - YOLOMM: ``output_yolomm.compute_yolomm_output_kd``
        - RTDETRMM: ``output_rtdetrmm.compute_rtdetr_output_kd``

        Kept only as an internal helper for potential debugging.  Do NOT call
        from production training code.

    Args:
        student_preds: Student raw detection-head output (tensor or tuple).
        teacher_preds: Teacher raw detection-head output (tensor or tuple).
        temperature: Unused (retained for API compatibility).

    Returns:
        Scalar loss tensor.
    """
    s_flat = _flatten_preds(student_preds)
    t_flat = _flatten_preds(teacher_preds)

    if s_flat is None or t_flat is None:
        return torch.tensor(0.0, device=_any_device(student_preds))

    # Align shapes if possible
    if s_flat.shape != t_flat.shape:
        min_len = min(s_flat.numel(), t_flat.numel())
        s_flat = s_flat.reshape(-1)[:min_len]
        t_flat = t_flat.reshape(-1)[:min_len]

    loss = F.mse_loss(s_flat, t_flat.detach())
    return loss


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_preds(preds) -> torch.Tensor | None:
    """Flatten prediction tensors into a single 1-D tensor for loss computation."""
    if isinstance(preds, torch.Tensor):
        return preds.contiguous().reshape(-1)
    if isinstance(preds, (tuple, list)):
        tensors = [p.contiguous().reshape(-1) for p in preds if isinstance(p, torch.Tensor)]
        if tensors:
            return torch.cat(tensors)
    return None


def _any_device(preds) -> torch.device:
    if isinstance(preds, torch.Tensor):
        return preds.device
    if isinstance(preds, (tuple, list)):
        for p in preds:
            if isinstance(p, torch.Tensor):
                return p.device
    return torch.device("cpu")
