# Ultralytics Multimodal Distillation - YOLOMM Output Distillation
# Foreground-guided, cls/loc decoupled output-level knowledge distillation.

"""
YOLOMM family output distillation.

Implements foreground-guided + classification/localization decoupled output
distillation specifically designed for anchor-free YOLO detection heads.

Key design:
- **Foreground selection**: teacher's max class score per anchor selects
  informative positions; per-scale threshold + top-k dual constraint.
- **Classification distillation**: BCE soft-target distillation on selected
  foreground positions (NOT full-vector MSE).
- **Localization distillation**: L1 + GIoU on decoded boxes plus DFL
  distribution KL divergence (LD, CVPR 2022) at foreground positions.
- **Warmup**: linear ramp-up from 0 to target weight over configurable epochs.

References:
- Hinton et al., Distilling the Knowledge in a Neural Network, 2015
- Zheng et al., Localization Distillation for Dense Object Detection, CVPR 2022
- Yang et al., BCKD, ICCV 2023
- Wang et al., CrossKD, CVPR 2024
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import make_anchors


# ---------------------------------------------------------------------------
# Internal hyperparameters
# ---------------------------------------------------------------------------

# Overall output distillation weight
_OUTPUT_DISTILL_WEIGHT = 0.5
_OUTPUT_CLS_WEIGHT = 1.0           # Classification sub-weight
_OUTPUT_LOC_WEIGHT = 1.0           # Localization sub-weight
_OUTPUT_WARMUP_EPOCHS = 5          # Linear warmup epochs
_FG_CONF_THR = 0.05               # Foreground confidence threshold
_FG_TOPK_PER_LEVEL = 256          # Max foreground per scale per image
_CLS_TEMPERATURE = 1.0            # Classification distillation temperature

# DFL distribution distillation (LD) hyperparameters
_LOC_DFL_WEIGHT = 0.5             # DFL KL sub-weight within localization loss
_DFL_TEMPERATURE = 10.0           # Temperature for DFL softmax (LD paper default)


# ---------------------------------------------------------------------------
# Output bundle
# ---------------------------------------------------------------------------


@dataclass
class YOLOMMOutputBundle:
    """Standardised detection-head output for YOLOMM output distillation.

    Fields:
        cls_logits: (B, total_anchors, nc) classification logits (pre-sigmoid).
        box_decoded: (B, total_anchors, 4) decoded boxes in xyxy format.
        dfl_logits: (B, total_anchors, 4*reg_max) raw DFL logits, or ``None``
            if not available (e.g. no head access or reg_max=1).
        reg_max: DFL distribution bins per direction (typically 16).
        anchors: (total_anchors, 2) anchor centres.
        strides: (total_anchors, 1) per-anchor strides.
        splits: list of int, number of anchors per scale level.
    """

    cls_logits: torch.Tensor
    box_decoded: torch.Tensor
    dfl_logits: Optional[torch.Tensor]
    reg_max: int
    anchors: torch.Tensor
    strides: torch.Tensor
    splits: List[int]


def build_yolomm_output_bundle(preds, model) -> YOLOMMOutputBundle:
    """Build a standardised output bundle from YOLOMM detection-head predictions.

    Args:
        preds: Raw detection-head output. This function supports two actual
            runtime protocols:

            1. **Training-time student output**: ``list[Tensor]`` where each
               tensor has shape ``(B, no, H, W)`` for one detection scale.
            2. **Eval-time teacher output via `predict()`**:
               ``(y_decoded, list_of_scale_preds)`` where the 2nd element is the
               same raw per-scale list described above.

            Historical 3-D concatenated tensors are still accepted as a
            fallback, but they are not the primary path anymore.
        model: The model (student or teacher) for accessing head attributes
            (``stride``, ``nc``, ``anchors``, etc.).

    Returns:
        YOLOMMOutputBundle with all fields populated.
    """
    # Extract the detection head
    head = None
    if hasattr(model, 'model'):
        head = model.model[-1]

    # Determine nc from model
    nc = model.yaml.get('nc', 80) if hasattr(model, 'yaml') else 80
    if head is not None and hasattr(head, 'nc'):
        nc = head.nc

    concat_pred, scale_preds, splits = _normalize_yolomm_preds(preds)

    # concat_pred shape: (B, no, total_anchors)
    B = concat_pred.shape[0]
    total_anchors = concat_pred.shape[2]

    # Split into box and cls components
    reg_channels = concat_pred.shape[1] - nc
    cls_logits = concat_pred[:, reg_channels:, :].permute(0, 2, 1)  # (B, A, nc)

    # Determine reg_max from head
    reg_max = getattr(head, 'reg_max', 16) if head is not None else 16
    dfl_logits: Optional[torch.Tensor] = None

    # Get decoded boxes from the head if available
    if head is not None and hasattr(head, 'decode_bboxes') and scale_preds is not None:
        try:
            # Rebuild anchors/strides from the raw per-scale outputs using the
            # same logic as Detect._inference(), so train-time student outputs
            # and eval-time teacher outputs share the same decoding semantics.
            shape = scale_preds[0].shape
            if (
                getattr(head, 'anchors', torch.empty(0)).numel() == 0
                or getattr(head, 'strides', torch.empty(0)).numel() == 0
                or getattr(head, 'shape', None) != shape
            ):
                head.anchors, head.strides = (t.transpose(0, 1) for t in make_anchors(scale_preds, head.stride, 0.5))
                head.shape = shape

            anchors = head.anchors
            strides = head.strides
            box_pred = concat_pred[:, :reg_channels, :]  # (B, reg_ch, A)

            # Preserve raw DFL logits for distribution distillation (LD)
            if reg_max > 1 and reg_channels == 4 * reg_max:
                dfl_logits = box_pred.permute(0, 2, 1)  # (B, A, 4*reg_max)

            decoded = head.decode_bboxes(head.dfl(box_pred), anchors.unsqueeze(0)) * strides
            box_decoded = xywh2xyxy(decoded.permute(0, 2, 1))  # (B, A, 4) xyxy
        except Exception:
            # Fallback: treat first 4 channels as decoded boxes
            box_decoded = concat_pred[:, :4, :].permute(0, 2, 1)  # (B, A, 4)
            box_decoded = xywh2xyxy(box_decoded)
            anchors = torch.zeros(total_anchors, 2, device=concat_pred.device)
            strides = torch.ones(total_anchors, 1, device=concat_pred.device)
            dfl_logits = None
    else:
        # No head available -- treat first 4 channels as raw box output
        box_decoded = concat_pred[:, :4, :].permute(0, 2, 1)
        box_decoded = xywh2xyxy(box_decoded)
        anchors = torch.zeros(total_anchors, 2, device=concat_pred.device)
        strides = torch.ones(total_anchors, 1, device=concat_pred.device)

    return YOLOMMOutputBundle(
        cls_logits=cls_logits,
        box_decoded=box_decoded,
        dfl_logits=dfl_logits,
        reg_max=reg_max,
        anchors=anchors.detach() if isinstance(anchors, torch.Tensor) else anchors,
        strides=strides.detach() if isinstance(strides, torch.Tensor) else strides,
        splits=splits,
    )


def _normalize_yolomm_preds(preds) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], List[int]]:
    """Normalize YOLOMM detection-head outputs to a concatenated 3-D tensor.

    Returns:
        concat_pred: ``(B, no, total_anchors)``
        scale_preds: original per-scale raw outputs if available
        splits: anchors per scale
    """
    scale_preds: Optional[List[torch.Tensor]] = None
    concat_pred: Optional[torch.Tensor] = None

    if isinstance(preds, tuple):
        # Inference/eval path from Detect.forward(): (y_decoded, x_raw_list)
        if len(preds) >= 2 and isinstance(preds[1], list) and preds[1] and all(isinstance(p, torch.Tensor) for p in preds[1]):
            scale_preds = preds[1]
        elif len(preds) >= 1 and isinstance(preds[0], torch.Tensor):
            concat_pred = preds[0]
    elif isinstance(preds, list) and preds and all(isinstance(p, torch.Tensor) for p in preds):
        # Training path from Detect.forward(): list of per-scale tensors (B, no, H, W)
        if preds[0].dim() == 4:
            scale_preds = preds
        elif preds[0].dim() == 3 and len(preds) == 1:
            concat_pred = preds[0]
    elif isinstance(preds, torch.Tensor):
        concat_pred = preds

    if scale_preds is not None:
        B = scale_preds[0].shape[0]
        concat_pred = torch.cat([sp.view(B, sp.shape[1], -1) for sp in scale_preds], dim=2)
        splits = [sp.shape[2] * sp.shape[3] for sp in scale_preds]
        return concat_pred, scale_preds, splits

    if concat_pred is None:
        raise TypeError(
            "Unsupported YOLOMM prediction format for output distillation. "
            f"Got type={type(preds).__name__}."
        )

    if concat_pred.dim() == 4:
        B = concat_pred.shape[0]
        concat_pred = concat_pred.view(B, concat_pred.shape[1], -1)
    elif concat_pred.dim() != 3:
        raise ValueError(
            f"Expected YOLOMM concatenated prediction tensor with 3 dims, got shape={tuple(concat_pred.shape)}"
        )

    return concat_pred, None, [concat_pred.shape[2]]


# ---------------------------------------------------------------------------
# Foreground mask construction
# ---------------------------------------------------------------------------


def build_yolomm_fg_mask(
    teacher_bundle: YOLOMMOutputBundle,
    conf_thr: float = 0.05,
    topk_per_level: int = 256,
) -> torch.Tensor:
    """Build foreground mask from teacher classification scores.

    For each scale level and each image in the batch:
    1. Compute ``max(sigmoid(cls_logits))`` per anchor.
    2. Keep anchors with score >= ``conf_thr``.
    3. If candidates exceed ``topk_per_level``, keep top-k by score.

    Args:
        teacher_bundle: Teacher's standardised output bundle.
        conf_thr: Confidence threshold for foreground selection.
        topk_per_level: Maximum foreground anchors per scale per image.

    Returns:
        Boolean mask of shape ``(B, total_anchors)`` marking foreground positions.
    """
    cls_logits = teacher_bundle.cls_logits  # (B, A, nc)
    B, A, _ = cls_logits.shape
    device = cls_logits.device

    max_scores = cls_logits.sigmoid().max(dim=-1).values  # (B, A)

    # Split by scale
    splits = teacher_bundle.splits
    fg_mask = torch.zeros(B, A, dtype=torch.bool, device=device)
    offset = 0
    for n_anchors in splits:
        scale_scores = max_scores[:, offset:offset + n_anchors]  # (B, n_anchors)
        # Threshold
        above_thr = scale_scores >= conf_thr  # (B, n_anchors)

        for b in range(B):
            valid = above_thr[b]  # (n_anchors,)
            n_valid = valid.sum().item()
            if n_valid == 0:
                # No foreground at this scale for this image -- skip
                pass
            elif n_valid > topk_per_level:
                # Take top-k
                _, topk_idx = scale_scores[b].topk(topk_per_level)
                fg_mask[b, offset + topk_idx] = True
            else:
                fg_mask[b, offset:offset + n_anchors] = valid
        offset += n_anchors

    return fg_mask


# ---------------------------------------------------------------------------
# Classification distillation
# ---------------------------------------------------------------------------


def compute_yolomm_cls_kd(
    student_bundle: YOLOMMOutputBundle,
    teacher_bundle: YOLOMMOutputBundle,
    fg_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute classification distillation loss on foreground positions.

    Uses teacher sigmoid outputs as soft targets with BCE loss on the student
    side (per-class binary distillation, NOT full-vector MSE).

    Args:
        student_bundle: Student output bundle.
        teacher_bundle: Teacher output bundle.
        fg_mask: Boolean mask ``(B, A)`` from ``build_yolomm_fg_mask``.
        temperature: Temperature for soft targets (applied to logits before sigmoid).

    Returns:
        Scalar classification distillation loss.
    """
    s_cls = student_bundle.cls_logits  # (B, A, nc)
    t_cls = teacher_bundle.cls_logits  # (B, A, nc)

    # Select foreground positions
    # fg_mask: (B, A) -> expand to (B, A, nc) is handled by indexing
    n_fg = fg_mask.sum().item()
    if n_fg == 0:
        return torch.tensor(0.0, device=s_cls.device)

    s_fg = s_cls[fg_mask]  # (N_fg, nc)
    t_fg = t_cls[fg_mask]  # (N_fg, nc)

    # Soft targets: sigmoid(teacher_logits / temperature)
    t_soft = (t_fg / temperature).sigmoid().detach()

    # BCE loss on student logits (scaled by temperature)
    loss = F.binary_cross_entropy_with_logits(
        s_fg / temperature, t_soft, reduction='mean'
    )
    return loss


# ---------------------------------------------------------------------------
# Localization distillation
# ---------------------------------------------------------------------------


def compute_yolomm_loc_kd(
    student_bundle: YOLOMMOutputBundle,
    teacher_bundle: YOLOMMOutputBundle,
    fg_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute localization distillation loss on foreground positions.

    Uses L1 + GIoU on decoded boxes, plus DFL distribution KL divergence
    on raw DFL logits (when available and reg_max matches).

    Args:
        student_bundle: Student output bundle.
        teacher_bundle: Teacher output bundle.
        fg_mask: Boolean mask ``(B, A)``.

    Returns:
        (total_loc_loss, dfl_loss) tuple. ``dfl_loss`` is detached for logging.
    """
    s_box = student_bundle.box_decoded  # (B, A, 4)
    t_box = teacher_bundle.box_decoded  # (B, A, 4)

    n_fg = fg_mask.sum().item()
    if n_fg == 0:
        zero = torch.tensor(0.0, device=s_box.device)
        return zero, zero.detach()

    s_fg = s_box[fg_mask]  # (N_fg, 4) xyxy
    t_fg = t_box[fg_mask].detach()  # (N_fg, 4) xyxy

    # L1 loss
    l1_loss = F.l1_loss(s_fg, t_fg, reduction='mean')

    # GIoU loss
    giou_loss = _giou_loss(s_fg, t_fg)

    # DFL distribution KL divergence (LD)
    dfl_loss = _compute_dfl_kl(student_bundle, teacher_bundle, fg_mask)

    total = l1_loss + giou_loss + _LOC_DFL_WEIGHT * dfl_loss
    return total, dfl_loss.detach()


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


# ---------------------------------------------------------------------------
# DFL distribution distillation (LD)
# ---------------------------------------------------------------------------


def _compute_dfl_kl(
    s_bundle: YOLOMMOutputBundle,
    t_bundle: YOLOMMOutputBundle,
    fg_mask: torch.Tensor,
    temperature: float = _DFL_TEMPERATURE,
) -> torch.Tensor:
    """DFL logit distribution KL distillation on foreground positions.

    Implements Localization Distillation (LD, CVPR 2022) by minimising the KL
    divergence between teacher and student DFL probability distributions at
    foreground anchor positions.

    Gracefully returns zero when:
    - Either bundle lacks DFL logits (e.g. no head access).
    - Student and teacher have different ``reg_max`` values.
    - No foreground positions exist.

    Args:
        s_bundle: Student output bundle (with ``dfl_logits`` and ``reg_max``).
        t_bundle: Teacher output bundle (with ``dfl_logits`` and ``reg_max``).
        fg_mask: Boolean foreground mask ``(B, A)``.
        temperature: Softmax temperature for DFL distributions (default 10.0).

    Returns:
        Scalar DFL KL divergence loss.
    """
    device = s_bundle.box_decoded.device
    zero = torch.tensor(0.0, device=device)

    # Guard: DFL logits not available
    if s_bundle.dfl_logits is None or t_bundle.dfl_logits is None:
        return zero

    # Guard: reg_max mismatch -> skip DFL, fall back to L1+GIoU only
    if s_bundle.reg_max != t_bundle.reg_max:
        LOGGER.warning(
            f"DFL distillation skipped: student reg_max={s_bundle.reg_max} != "
            f"teacher reg_max={t_bundle.reg_max}. Falling back to L1+GIoU only."
        )
        return zero

    n_fg = fg_mask.sum().item()
    if n_fg == 0:
        return zero

    reg_max = s_bundle.reg_max

    # Extract foreground DFL logits
    s_dfl = s_bundle.dfl_logits[fg_mask]  # (N_fg, 4*reg_max)
    t_dfl = t_bundle.dfl_logits[fg_mask]  # (N_fg, 4*reg_max)

    # Reshape to (N_fg*4, reg_max) -- each direction independently
    s_dfl = s_dfl.reshape(-1, reg_max)
    t_dfl = t_dfl.reshape(-1, reg_max)

    # KL divergence with temperature scaling
    s_log_prob = F.log_softmax(s_dfl / temperature, dim=-1)
    t_prob = F.softmax(t_dfl.detach() / temperature, dim=-1)

    kl = F.kl_div(s_log_prob, t_prob, reduction='batchmean') * (temperature ** 2)

    # Guard against NaN/Inf from edge cases
    if torch.isnan(kl) or torch.isinf(kl):
        return zero

    return kl


# ---------------------------------------------------------------------------
# Combined output distillation entry point
# ---------------------------------------------------------------------------


def compute_yolomm_output_kd(
    student_preds,
    teacher_preds,
    student_model,
    teacher_model,
    current_epoch: int = 0,
    total_epochs: int = 100,
    cls_weight: float = _OUTPUT_CLS_WEIGHT,
    loc_weight: float = _OUTPUT_LOC_WEIGHT,
    overall_weight: float = _OUTPUT_DISTILL_WEIGHT,
    warmup_epochs: int = _OUTPUT_WARMUP_EPOCHS,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute full YOLOMM output distillation loss.

    Orchestrates foreground mask building, cls/loc decoupled distillation,
    and warmup weight scheduling.

    Args:
        student_preds: Raw student detection-head output.
        teacher_preds: Raw teacher detection-head output (train-mode semantics).
        student_model: Student model (for head access).
        teacher_model: Teacher model (for head access).
        current_epoch: Current training epoch (for warmup).
        total_epochs: Total training epochs.
        cls_weight: Classification loss sub-weight.
        loc_weight: Localization loss sub-weight.
        overall_weight: Overall output distillation multiplier.
        warmup_epochs: Epochs to linearly ramp from 0 to overall_weight.

    Returns:
        (total_output_kd, items_dict)
    """
    device = _get_device(student_preds)

    # Build bundles
    s_bundle = build_yolomm_output_bundle(student_preds, student_model)
    t_bundle = build_yolomm_output_bundle(teacher_preds, teacher_model)

    # Build foreground mask from teacher
    fg_mask = build_yolomm_fg_mask(
        t_bundle,
        conf_thr=_FG_CONF_THR,
        topk_per_level=_FG_TOPK_PER_LEVEL,
    )

    # Classification distillation
    cls_kd = compute_yolomm_cls_kd(s_bundle, t_bundle, fg_mask, temperature=_CLS_TEMPERATURE)

    # Localization distillation (L1 + GIoU + DFL KL)
    loc_kd, dfl_kd = compute_yolomm_loc_kd(s_bundle, t_bundle, fg_mask)

    # Weighted combination
    raw_total = cls_weight * cls_kd + loc_weight * loc_kd

    # Warmup scheduling
    warmup_factor = _compute_warmup_factor(current_epoch, warmup_epochs)
    total_output_kd = overall_weight * warmup_factor * raw_total

    items = {
        "distill_output_cls": cls_kd.detach(),
        "distill_output_loc": loc_kd.detach(),
        "distill_output_dfl": dfl_kd,
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


def _get_device(preds) -> torch.device:
    if isinstance(preds, torch.Tensor):
        return preds.device
    if isinstance(preds, (tuple, list)):
        for p in preds:
            if isinstance(p, torch.Tensor):
                return p.device
    return torch.device("cpu")
