# Ultralytics Multimodal Distillation - RTDETRMM Feature Guidance
# Query-aware foreground-guided feature mask generation for RTDETR family.

"""
RTDETRMM family feature-level distillation guidance.

Provides query-aware foreground-guided feature mask generation using teacher
DETR decoder output: high-confidence queries are projected onto feature grids
to produce soft foreground masks.

Design:
- Reuses RTDETROutputBundle from output_rtdetrmm.py (no duplicate decoding).
- Batch-vectorised mask generation (no Python per-image-per-query loops).
- Mask semantics unified with YOLOMM: B x 1 x H x W, score/sqrt(area),
  clamped [0,1], so shared feature loss can be used directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .output_rtdetrmm import (
    RTDETROutputBundle,
    build_rtdetr_output_bundle,
    select_rtdetr_kd_queries,
)


# ---------------------------------------------------------------------------
# Internal constants (not exposed to YAML)
# ---------------------------------------------------------------------------

_RTDETR_FEATURE_TOPK = 100
_FEATURE_CONF_THR = 0.05


# ---------------------------------------------------------------------------
# Feature guidance dataclass
# ---------------------------------------------------------------------------


@dataclass
class RTDETRMMFeatureGuidance:
    """Teacher foreground guidance for RTDETRMM feature distillation.

    Fields:
        boxes_xyxy: list of (N_i, 4) tensors, one per image, xyxy format
            in input image pixel coordinates.
        scores: list of (N_i,) tensors, one per image, max class confidence.

    The list length equals batch size B. Semantics are unified with
    YOLOMMFeatureGuidance so that shared feature loss can be applied
    identically.
    """

    boxes_xyxy: List[torch.Tensor]
    scores: List[torch.Tensor]


# ---------------------------------------------------------------------------
# Coordinate conversion helper
# ---------------------------------------------------------------------------


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        boxes: (..., 4) tensor in cxcywh format.

    Returns:
        (..., 4) tensor in xyxy format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ---------------------------------------------------------------------------
# Feature guidance construction
# ---------------------------------------------------------------------------


def build_rtdetr_feature_guidance(
    teacher_preds,
    conf_thr: float = _FEATURE_CONF_THR,
    topk: int = _RTDETR_FEATURE_TOPK,
    input_size: tuple[int, int] | None = None,
) -> RTDETRMMFeatureGuidance:
    """Build foreground guidance from RTDETRMM teacher decoder output.

    Reuses build_rtdetr_output_bundle() to extract final decoder layer output,
    then selects high-confidence queries per image using
    select_rtdetr_kd_queries().

    Args:
        teacher_preds: Raw teacher RTDETR detection-head output (5-tuple).
        conf_thr: Minimum score threshold for query selection.
        topk: Maximum number of queries per image.
        input_size: (img_h, img_w) of the original input image. Required when
            RTDETR decoder outputs normalised coordinates (range [0, 1]) to
            convert them to pixel coordinates.

    Returns:
        RTDETRMMFeatureGuidance with per-image boxes (xyxy, pixel coords)
        and scores, all detached from the computation graph.
    """
    # Step 1: build standardised output bundle (handles dn query stripping)
    bundle: RTDETROutputBundle = build_rtdetr_output_bundle(teacher_preds)

    # Step 2: select effective queries using existing query selection logic
    query_indices = select_rtdetr_kd_queries(
        bundle, conf_thr=conf_thr, topk_queries=topk
    )

    # Step 3: extract boxes and scores for selected queries per image
    raw_boxes = bundle.boxes   # (B, Q, 4) -- possibly cxcywh, possibly normalised
    raw_scores = bundle.scores  # (B, Q, nc)
    B = raw_boxes.shape[0]

    # Compute per-query max class confidence: (B, Q)
    max_scores = raw_scores.sigmoid().max(dim=-1).values

    # Convert all boxes from cxcywh to xyxy first (batch-level, no loop)
    boxes_xyxy_all = _cxcywh_to_xyxy(raw_boxes)  # (B, Q, 4)

    # If coordinates are normalised ([0, 1] range), scale to pixel coordinates
    if input_size is not None:
        img_h, img_w = input_size
        scale = torch.tensor(
            [img_w, img_h, img_w, img_h],
            dtype=boxes_xyxy_all.dtype,
            device=boxes_xyxy_all.device,
        )
        boxes_xyxy_all = boxes_xyxy_all * scale

    # Step 4: gather per-image results using query indices
    boxes_list: List[torch.Tensor] = []
    scores_list: List[torch.Tensor] = []

    for b in range(B):
        idx = query_indices[b]  # (N_i,) or (0,)
        if len(idx) == 0:
            boxes_list.append(
                torch.empty(0, 4, dtype=raw_boxes.dtype, device=raw_boxes.device)
            )
            scores_list.append(
                torch.empty(0, dtype=raw_scores.dtype, device=raw_scores.device)
            )
        else:
            boxes_list.append(boxes_xyxy_all[b, idx].detach())
            scores_list.append(max_scores[b, idx].detach())

    return RTDETRMMFeatureGuidance(boxes_xyxy=boxes_list, scores=scores_list)


# ---------------------------------------------------------------------------
# Batch-vectorised soft foreground mask generation
# ---------------------------------------------------------------------------


def build_rtdetr_feature_mask(
    guidance: RTDETRMMFeatureGuidance,
    feature_shape: tuple[int, int, int, int],
    input_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Generate soft foreground mask from RTDETR teacher guidance.

    Mask semantics are unified with YOLOMM: B x 1 x H x W, score/sqrt(area)
    accumulation, clamped [0,1]. This ensures the shared feature distillation
    loss can be applied identically.

    Batch-vectorised implementation: no Python per-image-per-query double loop.

    Implementation strategy:
    1. Pad per-image boxes/scores to (B, N_max, 4) and (B, N_max) with a
       validity mask.
    2. Scale xyxy coordinates from input image space to (H, W) feature space.
    3. Use torch.arange grid + box boundary broadcast comparison to build
       (B, N_max, H, W) coverage matrices.
    4. Compute score / sqrt(area + eps) weights, broadcast-multiply into
       coverage, sum along box dimension, clamp to [0, 1].

    Args:
        guidance: Teacher foreground guidance (boxes + scores).
        feature_shape: (B, C, H, W) of the target feature map.
        input_size: (img_h, img_w) of the original input image. If None,
            boxes are assumed to already be in feature-map coordinates.

    Returns:
        Soft foreground mask of shape (B, 1, H, W), values in [0, 1].
    """
    B, _, H, W = feature_shape

    # Determine device and dtype from guidance tensors
    if len(guidance.boxes_xyxy) > 0 and len(guidance.boxes_xyxy[0]) > 0:
        device = guidance.boxes_xyxy[0].device
        dtype = guidance.boxes_xyxy[0].dtype
    else:
        # Fallback: find first non-empty tensor or use cpu
        device = torch.device("cpu")
        dtype = torch.float32
        for boxes_t in guidance.boxes_xyxy:
            if boxes_t.numel() > 0:
                device = boxes_t.device
                dtype = boxes_t.dtype
                break

    # Step 1: Compute N_max across batch for padding alignment
    counts = [boxes.shape[0] for boxes in guidance.boxes_xyxy]
    N_max = max(counts) if counts else 0

    if N_max == 0:
        # No foreground boxes in any image -> return zero mask
        return torch.zeros(B, 1, H, W, dtype=dtype, device=device)

    # Step 2: Pad boxes and scores to (B, N_max, ...) with validity mask
    padded_boxes = torch.zeros(B, N_max, 4, dtype=dtype, device=device)
    padded_scores = torch.zeros(B, N_max, dtype=dtype, device=device)
    valid_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

    for b in range(B):
        n = counts[b]
        if n > 0:
            padded_boxes[b, :n] = guidance.boxes_xyxy[b]
            padded_scores[b, :n] = guidance.scores[b]
            valid_mask[b, :n] = True

    # Step 3: Scale box coordinates from input image space to feature space
    if input_size is not None:
        img_h, img_w = input_size
        scale_x = W / img_w
        scale_y = H / img_h
    else:
        # Assume boxes are already in feature-map pixel coordinates
        scale_x = 1.0
        scale_y = 1.0

    # Apply scaling: x1, x2 by scale_x; y1, y2 by scale_y
    scaled_boxes = padded_boxes.clone()
    scaled_boxes[..., 0] *= scale_x  # x1
    scaled_boxes[..., 1] *= scale_y  # y1
    scaled_boxes[..., 2] *= scale_x  # x2
    scaled_boxes[..., 3] *= scale_y  # y2

    # Clamp to feature map boundaries
    scaled_boxes[..., 0].clamp_(min=0.0, max=float(W))
    scaled_boxes[..., 1].clamp_(min=0.0, max=float(H))
    scaled_boxes[..., 2].clamp_(min=0.0, max=float(W))
    scaled_boxes[..., 3].clamp_(min=0.0, max=float(H))

    # Step 4: Build spatial grids via torch.arange for broadcast comparison
    # grid_x: (1, 1, 1, W), grid_y: (1, 1, H, 1) -- pixel center coordinates
    grid_x = torch.arange(W, dtype=dtype, device=device).view(1, 1, 1, W) + 0.5
    grid_y = torch.arange(H, dtype=dtype, device=device).view(1, 1, H, 1) + 0.5

    # Extract box boundaries: (B, N_max) -> (B, N_max, 1, 1) for broadcasting
    x1 = scaled_boxes[..., 0].unsqueeze(-1).unsqueeze(-1)  # (B, N_max, 1, 1)
    y1 = scaled_boxes[..., 1].unsqueeze(-1).unsqueeze(-1)
    x2 = scaled_boxes[..., 2].unsqueeze(-1).unsqueeze(-1)
    y2 = scaled_boxes[..., 3].unsqueeze(-1).unsqueeze(-1)

    # Coverage matrix: (B, N_max, H, W) -- True where pixel center is inside box
    inside = (grid_x >= x1) & (grid_x < x2) & (grid_y >= y1) & (grid_y < y2)
    inside = inside.float()  # (B, N_max, H, W)

    # Step 5: Compute per-box weights: score / sqrt(area + eps)
    box_w = (scaled_boxes[..., 2] - scaled_boxes[..., 0]).clamp(min=0.0)
    box_h = (scaled_boxes[..., 3] - scaled_boxes[..., 1]).clamp(min=0.0)
    area = box_w * box_h  # (B, N_max)
    weights = padded_scores / (area + 1e-6).sqrt()  # (B, N_max)

    # Zero out invalid (padded) entries
    weights = weights * valid_mask.float()

    # Reshape weights for broadcasting: (B, N_max, 1, 1)
    weights = weights.unsqueeze(-1).unsqueeze(-1)

    # Step 6: Weighted accumulation along box dimension
    # (B, N_max, H, W) * (B, N_max, 1, 1) -> sum over dim=1 -> (B, H, W)
    mask = (inside * weights).sum(dim=1)  # (B, H, W)

    # Step 7: Clamp to [0, 1] and reshape to (B, 1, H, W)
    mask = mask.clamp(0.0, 1.0).unsqueeze(1)  # (B, 1, H, W)

    return mask
