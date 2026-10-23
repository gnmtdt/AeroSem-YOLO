# Ultralytics Multimodal Distillation - RTDETRMM Output Distillation
# Matching-aware / query-aware output-level knowledge distillation for DETR family.

"""
RTDETRMM family output distillation.

Implements query-aware output distillation designed for DETR-style detection
heads, where the output is a set of learned queries rather than dense anchors.

Key design:
- **Query selection**: only distill "effective queries" -- those with high
  teacher confidence -- rather than all 300 queries uniformly.
- **Final decoder layer only**: distillation operates on the last decoder layer
  output (no multi-layer decoder guidance in this version).
- **Classification distillation**: soft classification on selected queries.
- **Box distillation**: L1 + GIoU on selected query boxes.

References:
- Chang et al., DETRDistill, ICCV 2023
- Wang et al., KD-DETR, CVPR 2024
- D3ETR, IJCAI 2024
- OD-DETR, IJCAI 2024
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER


# ---------------------------------------------------------------------------
# Output bundle
# ---------------------------------------------------------------------------


@dataclass
class RTDETROutputBundle:
    """Standardised detection-head output for RTDETRMM output distillation.

    Fields:
        scores: (B, num_queries, nc) classification scores/logits from the
            **final decoder layer**.
        boxes: (B, num_queries, 4) predicted boxes from the final decoder layer,
            in normalised cxcywh or xyxy format depending on decoder output.
    """

    scores: torch.Tensor
    boxes: torch.Tensor


def build_rtdetr_output_bundle(preds) -> RTDETROutputBundle:
    """Build a standardised output bundle from RTDETRMM detection-head predictions.

    RTDETR ``distill_forward`` returns preds as the 5-tuple:
    ``(dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)``

    For output distillation we only use the **final decoder layer** outputs:
    - ``dec_bboxes[-1]`` shape ``(B, num_queries, 4)``
    - ``dec_scores[-1]`` shape ``(B, num_queries, nc)``

    For a teacher in eval mode (no dn queries), preds may also be the 5-tuple
    but with ``dn_meta=None``.

    Args:
        preds: Raw RTDETR detection-head output (5-tuple from train-mode semantics).

    Returns:
        RTDETROutputBundle with final decoder layer scores and boxes.
    """
    if isinstance(preds, (tuple, list)) and len(preds) == 5:
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds

        # dec_bboxes: (num_decoder_layers, B, num_queries, 4)
        # dec_scores: (num_decoder_layers, B, num_queries, nc)
        # Take the final decoder layer
        if dec_bboxes.dim() == 4:
            final_boxes = dec_bboxes[-1]   # (B, num_queries, 4)
            final_scores = dec_scores[-1]  # (B, num_queries, nc)
        elif dec_bboxes.dim() == 3:
            final_boxes = dec_bboxes
            final_scores = dec_scores
        else:
            raise ValueError(
                f"Unexpected dec_bboxes shape: {dec_bboxes.shape}. "
                f"Expected 3D or 4D tensor."
            )

        # Handle dn queries: if dn_meta exists, strip dn queries
        if dn_meta is not None and "dn_num_split" in dn_meta:
            dn_split = dn_meta["dn_num_split"]
            # dn queries are prepended; real queries follow
            _, final_boxes = torch.split(final_boxes, dn_split, dim=1)
            _, final_scores = torch.split(final_scores, dn_split, dim=1)

        return RTDETROutputBundle(scores=final_scores, boxes=final_boxes)
    else:
        raise ValueError(
            f"Expected 5-tuple (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta), "
            f"got {type(preds)} with length {len(preds) if isinstance(preds, (tuple, list)) else 'N/A'}"
        )


# ---------------------------------------------------------------------------
# Query selection
# ---------------------------------------------------------------------------


def select_rtdetr_kd_queries(
    teacher_bundle: RTDETROutputBundle,
    conf_thr: float = 0.05,
    topk_queries: int = 100,
) -> List[torch.Tensor]:
    """Select effective queries for distillation based on teacher confidence.

    For each image in the batch:
    1. Compute ``max(sigmoid(scores))`` per query.
    2. Keep queries with max score >= ``conf_thr``.
    3. If candidates exceed ``topk_queries``, keep top-k by score.

    Args:
        teacher_bundle: Teacher's output bundle.
        conf_thr: Confidence threshold.
        topk_queries: Maximum number of queries to select per image.

    Returns:
        List of index tensors, one per image in the batch, containing the
        selected query indices.
    """
    scores = teacher_bundle.scores  # (B, Q, nc)
    B, Q, _ = scores.shape

    max_scores = scores.sigmoid().max(dim=-1).values  # (B, Q)

    query_indices = []
    for b in range(B):
        above_thr = max_scores[b] >= conf_thr  # (Q,)
        n_valid = above_thr.sum().item()

        if n_valid == 0:
            # No valid queries -- return empty index tensor
            query_indices.append(torch.empty(0, dtype=torch.long, device=scores.device))
        elif n_valid > topk_queries:
            _, topk_idx = max_scores[b].topk(topk_queries)
            query_indices.append(topk_idx)
        else:
            query_indices.append(above_thr.nonzero(as_tuple=False).squeeze(-1))

    return query_indices


# ---------------------------------------------------------------------------
# Classification distillation
# ---------------------------------------------------------------------------


def compute_rtdetr_cls_kd(
    student_bundle: RTDETROutputBundle,
    teacher_bundle: RTDETROutputBundle,
    query_indices: List[torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute classification distillation loss on selected queries.

    Args:
        student_bundle: Student output bundle.
        teacher_bundle: Teacher output bundle.
        query_indices: Per-image selected query indices.
        temperature: Temperature for soft classification.

    Returns:
        Scalar classification distillation loss.
    """
    s_scores = student_bundle.scores  # (B, Q, nc)
    t_scores = teacher_bundle.scores  # (B, Q, nc)
    B = s_scores.shape[0]
    device = s_scores.device

    losses = []
    for b in range(B):
        idx = query_indices[b]
        if len(idx) == 0:
            continue
        s_q = s_scores[b, idx]  # (n_sel, nc)
        t_q = t_scores[b, idx]  # (n_sel, nc)

        # Soft target from teacher sigmoid
        t_soft = (t_q / temperature).sigmoid().detach()
        loss = F.binary_cross_entropy_with_logits(
            s_q / temperature, t_soft, reduction='mean'
        )
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=device)
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Box distillation
# ---------------------------------------------------------------------------


def compute_rtdetr_box_kd(
    student_bundle: RTDETROutputBundle,
    teacher_bundle: RTDETROutputBundle,
    query_indices: List[torch.Tensor],
) -> torch.Tensor:
    """Compute box distillation loss on selected queries.

    Uses L1 + GIoU combination.

    Args:
        student_bundle: Student output bundle.
        teacher_bundle: Teacher output bundle.
        query_indices: Per-image selected query indices.

    Returns:
        Scalar box distillation loss.
    """
    s_boxes = student_bundle.boxes  # (B, Q, 4)
    t_boxes = teacher_bundle.boxes  # (B, Q, 4)
    B = s_boxes.shape[0]
    device = s_boxes.device

    all_s = []
    all_t = []
    for b in range(B):
        idx = query_indices[b]
        if len(idx) == 0:
            continue
        all_s.append(s_boxes[b, idx])  # (n_sel, 4)
        all_t.append(t_boxes[b, idx])  # (n_sel, 4)

    if not all_s:
        return torch.tensor(0.0, device=device)

    s_cat = torch.cat(all_s, dim=0)  # (N_total, 4)
    t_cat = torch.cat(all_t, dim=0).detach()  # (N_total, 4)

    # L1 loss
    l1_loss = F.l1_loss(s_cat, t_cat, reduction='mean')

    # GIoU loss (boxes may be in cxcywh; convert to xyxy for GIoU)
    s_xyxy = _cxcywh_to_xyxy(s_cat)
    t_xyxy = _cxcywh_to_xyxy(t_cat)
    giou_loss = _giou_loss(s_xyxy, t_xyxy)

    return l1_loss + giou_loss


# ---------------------------------------------------------------------------
# Combined output distillation entry point
# ---------------------------------------------------------------------------


# Internal hyperparameters (not exposed to YAML -- plan requirement)
_OUTPUT_DISTILL_WEIGHT = 0.5       # Overall output distillation weight
_OUTPUT_CLS_WEIGHT = 1.0           # Classification sub-weight
_OUTPUT_BOX_WEIGHT = 1.0           # Box sub-weight
_OUTPUT_WARMUP_EPOCHS = 5          # Linear warmup epochs
_QUERY_CONF_THR = 0.05            # Query confidence threshold
_QUERY_TOPK = 100                  # Max queries per image
_CLS_TEMPERATURE = 1.0            # Classification distillation temperature


def compute_rtdetr_output_kd(
    student_preds,
    teacher_preds,
    current_epoch: int = 0,
    total_epochs: int = 100,
    cls_weight: float = _OUTPUT_CLS_WEIGHT,
    box_weight: float = _OUTPUT_BOX_WEIGHT,
    overall_weight: float = _OUTPUT_DISTILL_WEIGHT,
    warmup_epochs: int = _OUTPUT_WARMUP_EPOCHS,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute full RTDETRMM output distillation loss.

    Orchestrates query selection, cls/box decoupled distillation, and warmup.

    Args:
        student_preds: Raw student RTDETR detection-head output (5-tuple).
        teacher_preds: Raw teacher RTDETR detection-head output (5-tuple).
        current_epoch: Current training epoch (for warmup).
        total_epochs: Total training epochs.
        cls_weight: Classification loss sub-weight.
        box_weight: Box regression loss sub-weight.
        overall_weight: Overall output distillation multiplier.
        warmup_epochs: Epochs to linearly ramp from 0 to overall_weight.

    Returns:
        (total_output_kd, items_dict)
    """
    device = _get_device(student_preds)

    # Build bundles
    s_bundle = build_rtdetr_output_bundle(student_preds)
    t_bundle = build_rtdetr_output_bundle(teacher_preds)

    # Select effective queries from teacher
    query_indices = select_rtdetr_kd_queries(
        t_bundle,
        conf_thr=_QUERY_CONF_THR,
        topk_queries=_QUERY_TOPK,
    )

    # Classification distillation
    cls_kd = compute_rtdetr_cls_kd(s_bundle, t_bundle, query_indices, temperature=_CLS_TEMPERATURE)

    # Box distillation
    box_kd = compute_rtdetr_box_kd(s_bundle, t_bundle, query_indices)

    # Weighted combination
    raw_total = cls_weight * cls_kd + box_weight * box_kd

    # Warmup scheduling
    warmup_factor = _compute_warmup_factor(current_epoch, warmup_epochs)
    total_output_kd = overall_weight * warmup_factor * raw_total

    items = {
        "distill_output_cls": cls_kd.detach(),
        "distill_output_box": box_kd.detach(),
        "distill_output_total": total_output_kd.detach(),
    }

    return total_output_kd, items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_warmup_factor(current_epoch: int, warmup_epochs: int) -> float:
    """Linear warmup from 0 to 1 over warmup_epochs."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, current_epoch / warmup_epochs)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    If boxes already appear to be in xyxy (x2 > x1, y2 > y1 typically), this
    will still work -- the GIoU computation handles both cases.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute GIoU loss between predicted and target boxes (xyxy format).

    Args:
        pred: (N, 4) predicted boxes.
        target: (N, 4) target boxes.

    Returns:
        Scalar mean GIoU loss.
    """
    # Intersection
    inter_x1 = torch.max(pred[:, 0], target[:, 0])
    inter_y1 = torch.max(pred[:, 1], target[:, 1])
    inter_x2 = torch.min(pred[:, 2], target[:, 2])
    inter_y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    area_pred = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_target = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = area_pred + area_target - inter + 1e-7

    iou = inter / union

    # Enclosing box
    enc_x1 = torch.min(pred[:, 0], target[:, 0])
    enc_y1 = torch.min(pred[:, 1], target[:, 1])
    enc_x2 = torch.max(pred[:, 2], target[:, 2])
    enc_y2 = torch.max(pred[:, 3], target[:, 3])
    area_enc = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + 1e-7

    giou = iou - (area_enc - union) / area_enc
    return (1.0 - giou).mean()


def _get_device(preds) -> torch.device:
    if isinstance(preds, torch.Tensor):
        return preds.device
    if isinstance(preds, (tuple, list)):
        for p in preds:
            if isinstance(p, torch.Tensor):
                return p.device
    return torch.device("cpu")
