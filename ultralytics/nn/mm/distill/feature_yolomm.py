# Ultralytics Multimodal Distillation - YOLOMM Feature Guidance
# Foreground-guided feature-level distillation target generation.

"""
YOLOMM family feature-level distillation guidance.

Provides foreground-guided feature mask generation using teacher detection
output: high-confidence boxes are projected onto feature grids to produce
soft foreground masks for detection-friendly feature distillation.

Design:
- Reuses YOLOMMOutputBundle from output_yolomm.py (no duplicate decoding).
- Batch-vectorised mask generation (no Python per-image-per-box loops).
- Mask semantics: B x 1 x H x W, score/sqrt(area) accumulation, clamped [0,1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import torch

from .output_yolomm import YOLOMMOutputBundle, build_yolomm_output_bundle


# ---------------------------------------------------------------------------
# Internal constants (not exposed to YAML)
# ---------------------------------------------------------------------------

_YOLOMM_FEATURE_TOPK = 100
_FEATURE_CONF_THR = 0.05


# ---------------------------------------------------------------------------
# Feature guidance dataclass
# ---------------------------------------------------------------------------


@dataclass
class YOLOMMFeatureGuidance:
    """Teacher foreground guidance for YOLOMM feature distillation.

    Fields:
        boxes_xyxy: list of (N_i, 4) tensors, one per image, xyxy format.
                    Each tensor is detached (no gradient).
        scores: list of (N_i,) tensors, one per image, max class confidence.
                Each tensor is detached (no gradient).
    """

    boxes_xyxy: List[torch.Tensor]
    scores: List[torch.Tensor]


# ---------------------------------------------------------------------------
# Feature guidance construction
# ---------------------------------------------------------------------------


def build_yolomm_feature_guidance(
    teacher_preds,
    teacher_model,
    conf_thr: float = _FEATURE_CONF_THR,
    topk: int = _YOLOMM_FEATURE_TOPK,
) -> YOLOMMFeatureGuidance:
    """Build foreground guidance from YOLOMM teacher detection output.

    Reuses ``build_yolomm_output_bundle()`` to decode dense predictions, then
    selects high-confidence boxes per image.

    Args:
        teacher_preds: Raw teacher detection-head output.
        teacher_model: Teacher model (for head access).
        conf_thr: Minimum score threshold for foreground boxes.
        topk: Maximum number of boxes per image.

    Returns:
        YOLOMMFeatureGuidance with per-image boxes and scores.
    """
    # Reuse existing output bundle -- no duplicate decoding
    bundle: YOLOMMOutputBundle = build_yolomm_output_bundle(
        teacher_preds, teacher_model
    )

    # cls_logits: (B, A, nc), box_decoded: (B, A, 4) xyxy
    cls_logits = bundle.cls_logits
    box_decoded = bundle.box_decoded

    # Per-anchor max class confidence: (B, A)
    max_scores = cls_logits.sigmoid().max(dim=-1).values

    B = max_scores.shape[0]
    device = max_scores.device

    boxes_list: List[torch.Tensor] = []
    scores_list: List[torch.Tensor] = []

    # Per-image foreground selection (single-layer loop over batch is acceptable
    # here because it only does index selection, not spatial mask computation)
    for b in range(B):
        img_scores = max_scores[b]  # (A,)
        img_boxes = box_decoded[b]  # (A, 4)

        # Threshold filtering
        mask = img_scores >= conf_thr  # (A,)
        n_valid = mask.sum().item()

        if n_valid == 0:
            # No foreground boxes for this image
            boxes_list.append(
                torch.zeros(0, 4, device=device, dtype=img_boxes.dtype)
            )
            scores_list.append(
                torch.zeros(0, device=device, dtype=img_scores.dtype)
            )
            continue

        valid_scores = img_scores[mask]  # (n_valid,)
        valid_boxes = img_boxes[mask]  # (n_valid, 4)

        # Top-k selection by descending score
        if n_valid > topk:
            topk_vals, topk_idx = valid_scores.topk(topk)
            valid_scores = topk_vals
            valid_boxes = valid_boxes[topk_idx]

        # Detach -- guidance does not participate in gradient computation
        boxes_list.append(valid_boxes.detach())
        scores_list.append(valid_scores.detach())

    return YOLOMMFeatureGuidance(boxes_xyxy=boxes_list, scores=scores_list)


# ---------------------------------------------------------------------------
# Batch-vectorised soft foreground mask generation
# ---------------------------------------------------------------------------


def build_yolomm_feature_mask(
    guidance: YOLOMMFeatureGuidance,
    feature_shape: Union[Tuple[int, int, int, int], torch.Size],
    input_size: Union[Tuple[int, int], None] = None,
) -> torch.Tensor:
    """Generate soft foreground mask from teacher guidance.

    Batch-vectorised implementation: no Python per-image-per-box double loop.
    Uses padding alignment + torch.arange grid broadcast + batched weight
    accumulation for efficient mask generation.

    Args:
        guidance: Teacher foreground guidance (boxes + scores).
        feature_shape: (B, C, H, W) of the target feature map.
        input_size: (img_h, img_w) of the original input image. If None,
            assumes boxes are already in coordinates relative to the feature
            grid (i.e. no scaling is applied).

    Returns:
        Soft foreground mask of shape (B, 1, H, W), values in [0, 1].
    """
    B, _, H, W = feature_shape
    device = guidance.boxes_xyxy[0].device if guidance.boxes_xyxy else torch.device("cpu")

    # Determine dtype from guidance tensors (fall back to float32)
    dtype = torch.float32
    if guidance.boxes_xyxy and guidance.boxes_xyxy[0].numel() > 0:
        dtype = guidance.boxes_xyxy[0].dtype

    # ------------------------------------------------------------------
    # Step 1: Padding alignment -- find N_max, pad boxes and scores
    # ------------------------------------------------------------------
    n_boxes_per_image = [boxes.shape[0] for boxes in guidance.boxes_xyxy]
    N_max = max(n_boxes_per_image) if n_boxes_per_image else 0

    # Early exit: no foreground boxes in entire batch
    if N_max == 0:
        return torch.zeros(B, 1, H, W, device=device, dtype=dtype)

    # Padded tensors: (B, N_max, 4) and (B, N_max)
    boxes_padded = torch.zeros(B, N_max, 4, device=device, dtype=dtype)
    scores_padded = torch.zeros(B, N_max, device=device, dtype=dtype)
    valid_mask = torch.zeros(B, N_max, device=device, dtype=torch.bool)

    for b in range(B):
        n_b = n_boxes_per_image[b]
        if n_b > 0:
            boxes_padded[b, :n_b] = guidance.boxes_xyxy[b]
            scores_padded[b, :n_b] = guidance.scores[b]
            valid_mask[b, :n_b] = True

    # ------------------------------------------------------------------
    # Step 2: Coordinate batch scaling to feature grid (H, W)
    # ------------------------------------------------------------------
    if input_size is not None:
        img_h, img_w = input_size
        scale_w = W / img_w
        scale_h = H / img_h
        scale = torch.tensor(
            [scale_w, scale_h, scale_w, scale_h],
            device=device, dtype=dtype,
        )
        boxes_scaled = boxes_padded * scale.view(1, 1, 4)
    else:
        # Boxes are already in feature grid coordinates
        boxes_scaled = boxes_padded

    # ------------------------------------------------------------------
    # Step 3: Grid broadcast comparison -- build (B, N_max, H, W) coverage
    # ------------------------------------------------------------------
    # grid_x: (1, 1, 1, W), grid_y: (1, 1, H, 1)
    grid_x = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)
    grid_y = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)

    # Extract box boundaries: each (B, N_max, 1, 1)
    x1 = boxes_scaled[:, :, 0].view(B, N_max, 1, 1)
    y1 = boxes_scaled[:, :, 1].view(B, N_max, 1, 1)
    x2 = boxes_scaled[:, :, 2].view(B, N_max, 1, 1)
    y2 = boxes_scaled[:, :, 3].view(B, N_max, 1, 1)

    # Coverage matrix: (B, N_max, H, W)
    in_box = (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y < y2)

    # ------------------------------------------------------------------
    # Step 4: Weight computation and accumulation
    # ------------------------------------------------------------------
    # Area per box: (B, N_max)
    area = (boxes_scaled[:, :, 2] - boxes_scaled[:, :, 0]) * (
        boxes_scaled[:, :, 3] - boxes_scaled[:, :, 1]
    )
    # Weight: score / sqrt(area + eps), (B, N_max)
    weights = scores_padded / (area + 1e-6).sqrt()

    # Zero out invalid (padded) box weights
    weights = weights * valid_mask.float()

    # Broadcast weights into coverage and sum over box dimension
    # weights: (B, N_max, 1, 1), in_box: (B, N_max, H, W)
    weighted_coverage = in_box.float() * weights.view(B, N_max, 1, 1)
    mask = weighted_coverage.sum(dim=1, keepdim=False)  # (B, H, W)

    # Final shape: (B, 1, H, W), clamped to [0, 1]
    mask = mask.unsqueeze(1).clamp(0.0, 1.0)

    return mask
