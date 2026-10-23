# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any, Dict, List, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist

# Import extended loss functions
from .iou_losses import (
    compute_extended_iou,
    SIoULoss,
    EIoULoss,
    WIoULoss,
    AlphaIoULoss,
    NWDLoss,
    MPDIoULoss,
    compute_mask_loss,
    DiceLoss,
    FocalTverskyLoss,
    ComboLoss,
)
from .cls_losses import EFClass, QualityFocalLoss, get_extended_cls_loss


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class LabelSmoothingBCELoss(nn.Module):
    """
    BCE Loss with Label Smoothing.

    Label smoothing helps prevent over-confidence in predictions by softening the target labels.
    Instead of using hard 0/1 labels, it uses values like 0.05/0.95 when smoothing is applied.

    Args:
        smoothing (float): Label smoothing factor in range [0, 1). Default: 0.0 (no smoothing)
        reduction (str): Reduction method. Options: 'none', 'mean', 'sum'. Default: 'none'
    """

    def __init__(self, smoothing: float = 0.0, reduction: str = "none"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        assert 0 <= smoothing < 1, f"smoothing must be in [0, 1), got {smoothing}"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate label smoothing BCE loss.

        Args:
            pred (torch.Tensor): Predicted logits, shape (B, N, C) or (B, N)
            target (torch.Tensor): Ground truth labels, shape (B, N, C) or (B, N)

        Returns:
            torch.Tensor: Loss value
        """
        if self.smoothing > 0:
            target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction=self.reduction)
        return loss


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing
    on hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    # Standard IoU types (using bbox_iou from metrics)
    STANDARD_IOU_TYPES = {
        "iou": {},
        "giou": {"GIoU": True},
        "diou": {"DIoU": True},
        "ciou": {"CIoU": True},
    }

    # Extended IoU types (using extended loss modules)
    EXTENDED_IOU_TYPES = {
        "siou": "SIoU",      # Sketch-IoU
        "eiou": "EIoU",      # Efficient-IoU
        "wiou": "WIoU",      # Wise-IoU
        "alphaiou": "AlphaIoU",  # Alpha-IoU (also accepts alpha_iou)
        "alpha_iou": "AlphaIoU",
        "nwd": "NWD",        # Normalized Wasserstein Distance
        "mpdiou": "MPDIoU",  # Minimum Point Distance IoU
    }

    IOU_TYPE_MAPPING = {**STANDARD_IOU_TYPES, **EXTENDED_IOU_TYPES}

    def __init__(
        self,
        reg_max: int = 16,
        iou_type: str = "ciou",
        # Extended IoU parameters
        siou_angle_alpha: float = 3.0,
        siou_dist_beta: float = 1.0,
        siou_gamma: float = 0.5,
        eiou_rho: float = 1.0,
        eiou_sigma: float = 1.0,
        wiou_v_threshold: float = 0.7,
        wiou_eps: float = 1e-6,
        alphaiou_alpha: float = 2.0,
        nwd_sigma: float = 0.5,
    ):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.iou_type = self._validate_iou_type(iou_type)
        self.iou_kwargs = self._get_iou_kwargs(self.iou_type)
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

        # Store extended IoU parameters
        self.extended_iou_params = {
            "siou_angle_alpha": siou_angle_alpha,
            "siou_dist_beta": siou_dist_beta,
            "siou_gamma": siou_gamma,
            "eiou_rho": eiou_rho,
            "eiou_sigma": eiou_sigma,
            "wiou_v_threshold": wiou_v_threshold,
            "wiou_eps": wiou_eps,
            "alphaiou_alpha": alphaiou_alpha,
            "nwd_sigma": nwd_sigma,
        }

        # Initialize extended loss modules if needed
        self._init_extended_losses()

    def _init_extended_losses(self):
        """Initialize extended IoU loss modules based on iou_type."""
        if self.iou_type in ["siou", "SIoU"]:
            self.siou_loss = SIoULoss(
                angle_alpha=self.extended_iou_params["siou_angle_alpha"],
                dist_beta=self.extended_iou_params["siou_dist_beta"],
                gamma=self.extended_iou_params["siou_gamma"],
            )
        elif self.iou_type in ["eiou", "EIoU"]:
            self.eiou_loss = EIoULoss(
                rho=self.extended_iou_params["eiou_rho"],
                sigma=self.extended_iou_params["eiou_sigma"],
            )
        elif self.iou_type in ["wiou", "WIoU"]:
            self.wiou_loss = WIoULoss(
                v_threshold=self.extended_iou_params["wiou_v_threshold"],
                eps=self.extended_iou_params["wiou_eps"],
            )
        elif self.iou_type in ["alphaiou", "alpha_iou", "AlphaIoU"]:
            self.alphaiou_loss = AlphaIoULoss(
                alpha=self.extended_iou_params["alphaiou_alpha"],
            )
        elif self.iou_type in ["nwd", "NWD"]:
            self.nwd_loss = NWDLoss(
                sigma=self.extended_iou_params["nwd_sigma"],
            )
        elif self.iou_type in ["mpdiou", "MPDIoU"]:
            self.mpdiou_loss = MPDIoULoss()

    @classmethod
    def _validate_iou_type(cls, iou_type: str) -> str:
        """Validate and normalize the configured IoU type."""
        if not isinstance(iou_type, str):
            raise ValueError(f"Unknown iou_type: {iou_type}. Valid: {list(cls.IOU_TYPE_MAPPING.keys())}")
        normalized = iou_type.lower()
        # Handle underscore variants
        if normalized == "alpha_iou":
            normalized = "alphaiou"
        if normalized not in cls.STANDARD_IOU_TYPES and normalized not in cls.EXTENDED_IOU_TYPES:
            raise ValueError(f"Unknown iou_type: {normalized}. Valid: {list(cls.IOU_TYPE_MAPPING.keys())}")
        return normalized

    @classmethod
    def _get_iou_kwargs(cls, iou_type: str) -> Dict[str, bool]:
        """Translate IoU type into bbox_iou keyword arguments."""
        return dict(cls.STANDARD_IOU_TYPES.get(iou_type, {}))

    @classmethod
    def get_supported_iou_types(cls) -> List[str]:
        """Return list of all supported IoU loss types."""
        return sorted(set(cls.STANDARD_IOU_TYPES.keys()) | set(cls.EXTENDED_IOU_TYPES.keys()))

    @classmethod
    def is_extended_iou(cls, iou_type: str) -> bool:
        """Check if the IoU type is an extended loss."""
        return iou_type.lower() in cls.EXTENDED_IOU_TYPES or (
            iou_type.lower() == "alpha_iou"
        )

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]

        # Compute IoU based on type
        if self.is_extended_iou(self.iou_type):
            iou = self._compute_extended_iou(pred_fg, target_fg)
        else:
            iou = bbox_iou(pred_fg, target_fg, xywh=False, **self.iou_kwargs)

        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    def _compute_extended_iou(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute extended IoU loss."""
        if self.iou_type == "siou":
            return self.siou_loss(pred_boxes, target_boxes, xywh=False)
        elif self.iou_type == "eiou":
            return self.eiou_loss(pred_boxes, target_boxes, xywh=False)
        elif self.iou_type == "wiou":
            return self.wiou_loss(pred_boxes, target_boxes, xywh=False)
        elif self.iou_type == "alphaiou":
            return self.alphaiou_loss(pred_boxes, target_boxes, xywh=False)
        elif self.iou_type == "nwd":
            return self.nwd_loss(pred_boxes, target_boxes, xywh=False)
        elif self.iou_type == "mpdiou":
            return self.mpdiou_loss(pred_boxes, target_boxes, xywh=False)
        else:
            raise ValueError(f"Extended IoU type {self.iou_type} not initialized.")


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    IOU_TYPE_MAPPING = {
        "prob": {},
        "ciou": {"CIoU": True},
    }

    def __init__(self, reg_max: int, iou_type: str = "prob"):
        """
        Initialize the RotatedBboxLoss module with regularization maximum and DFL settings.

        Args:
            reg_max (int): Maximum value for distribution focal loss.
            iou_type (str): IoU loss type for OBB. Options: 'prob' (default), 'ciou', 'none'.
        """
        super().__init__(reg_max, iou_type=iou_type)

    @classmethod
    def _validate_iou_type(cls, iou_type: str) -> str:
        """Validate and normalize the IoU type for OBB."""
        normalized = iou_type.lower() if isinstance(iou_type, str) else "prob"
        if normalized not in cls.IOU_TYPE_MAPPING:
            raise ValueError(
                f"Unknown iou_type for OBB: {iou_type}. "
                f"Valid options: {list(cls.IOU_TYPE_MAPPING.keys())}"
            )
        return normalized

    def _get_iou_kwargs(self, iou_type: str = None) -> dict:
        """Get IoU kwargs based on iou_type for OBB."""
        # If 'none', return empty dict (will be handled separately to disable loss)
        _type = iou_type if iou_type is not None else self.iou_type
        if _type == "none":
            return {}
        return self.IOU_TYPE_MAPPING.get(_type, {})

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        # Handle disabled bbox loss
        if self.iou_type == "none":
            loss_iou = torch.tensor(0.0, device=pred_bboxes.device)
            loss_dfl = torch.tensor(0.0, device=pred_bboxes.device)
            return loss_iou, loss_dfl

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou_kwargs = self._get_iou_kwargs()
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], **iou_kwargs)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Attributes:
        size_average (bool): Option to average the loss by the batch_size.
        use_target_weight (bool): Option to use weighted loss.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options."""
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute RLE loss.

        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


def _as_bool(value):
    """Resolve bool-like config values from CLI/default namespaces."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def resolve_detection_loss_profile(args, model=None):
    """Resolve detection loss profile and YOLO26 knobs from args.

    Returns:
        dict: {
            "profile": "standard" | "yolo26",
            "pred_adapter": "auto" | "standard" | "lspcd",
            "use_prog": bool,
            "tal_topk2": int | None,
            "tal_stride": list[int] | None,
        }
    """
    profile = getattr(args, "loss_profile", "standard") if args is not None else "standard"
    pred_adapter = getattr(args, "loss_pred_adapter", "auto") if args is not None else "auto"

    legacy_loss26 = _as_bool(getattr(args, "loss26", False)) if args is not None else False
    if legacy_loss26:
        profile = "yolo26"

    if profile in (None, ""):
        profile = "standard"
    if pred_adapter in (None, ""):
        pred_adapter = "auto"

    profile = str(profile).lower()
    pred_adapter = str(pred_adapter).lower()

    valid_profiles = {"standard", "yolo26"}
    valid_adapters = {"auto", "standard", "lspcd"}
    if profile not in valid_profiles:
        raise ValueError(f"Unknown loss_profile={profile}. Valid values: {sorted(valid_profiles)}")
    if pred_adapter not in valid_adapters:
        raise ValueError(f"Unknown loss_pred_adapter={pred_adapter}. Valid values: {sorted(valid_adapters)}")

    if profile == "yolo26":
        use_prog = _as_bool(getattr(args, "loss_prog", True))
        use_stal = _as_bool(getattr(args, "loss_stal", True))
        use_stride = _as_bool(getattr(args, "loss_stride_enhance", True))
        tal_topk2 = 1 if use_stal else None
        tal_stride = [8, 16] if use_stride else None
    else:
        use_prog = False
        tal_topk2 = None
        tal_stride = None

    return {
        "profile": profile,
        "pred_adapter": pred_adapter,
        "use_prog": use_prog,
        "tal_topk2": tal_topk2,
        "tal_stride": tal_stride,
    }


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    # Supported classification loss types
    SUPPORTED_CLS_LOSSES = {
        "bce": "Binary Cross Entropy",
        "focal": "Focal Loss",
        "varifocal": "Varifocal Loss",
        "efl": "Equalized Focal Loss",
        "qfl": "Quality Focal Loss",
    }

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        tal_topk2: int = None,
        tal_stride: list = None
    ):  # model must be de-paralleled
        """
        Initialize v8DetectionLoss with model parameters and task-aligned assignment settings.

        Args:
            model: The detection model (must be de-paralleled).
            tal_topk (int): Number of top candidates for TaskAlignedAssigner.
            tal_topk2 (int, optional): Secondary topk for STAL filtering. None to disable.
            tal_stride (list, optional): Stride values for small object enhancement. None to disable.
        """
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.loss_cls_type = self._resolve_loss_cls(getattr(h, "loss_cls", None))
        self.loss_box_type = BboxLoss._validate_iou_type(getattr(h, "loss_box", "ciou"))
        self.loss_dfl_enabled = bool(getattr(h, "loss_dfl", True))

        # Initialize classification loss function
        self.cls_loss_fn = self._init_cls_loss_fn(
            self.loss_cls_type,
            efl_gamma=2.0,
            efl_alpha=0.25,
            qfl_beta=2.0,
            focal_gamma=1.5,
            focal_alpha=0.25,
            label_smoothing=0.0,
        )

        # Build TaskAlignedAssigner with optional YOLO26 enhancements
        assigner_kwargs = {
            "topk": tal_topk,
            "num_classes": self.nc,
            "alpha": 0.5,
            "beta": 6.0,
        }
        if tal_topk2 is not None:
            assigner_kwargs["topk2"] = tal_topk2
        if tal_stride is not None:
            assigner_kwargs["stride"] = tal_stride
        self.assigner = TaskAlignedAssigner(**assigner_kwargs)

        # Initialize BboxLoss with extended parameters
        bbox_loss_kwargs = {"iou_type": self.loss_box_type}
        # Add extended IoU parameters (hardcoded defaults)
        if self.loss_box_type == "siou":
            bbox_loss_kwargs.update({
                "siou_angle_alpha": 3.0,
                "siou_dist_beta": 1.0,
                "siou_gamma": 0.5,
            })
        elif self.loss_box_type == "eiou":
            bbox_loss_kwargs.update({
                "eiou_rho": 1.0,
                "eiou_sigma": 1.0,
            })
        elif self.loss_box_type == "wiou":
            bbox_loss_kwargs.update({
                "wiou_v_threshold": 0.7,
                "wiou_eps": 1e-6,
            })
        elif self.loss_box_type in ["alphaiou", "alpha_iou"]:
            bbox_loss_kwargs.update({
                "alphaiou_alpha": 2.0,
            })
        elif self.loss_box_type == "nwd":
            bbox_loss_kwargs.update({
                "nwd_sigma": 0.5,
            })
        # mpdiou has no extra parameters, no update needed

        self.bbox_loss = BboxLoss(m.reg_max, **bbox_loss_kwargs).to(device)
        if not self.loss_dfl_enabled:
            self.bbox_loss.dfl_loss = None
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    @staticmethod
    def _resolve_loss_cls(loss_cls: Any) -> str:
        """Resolve YOLO detection classification loss type with model-specific defaults."""
        if loss_cls is None:
            return "bce"
        if not isinstance(loss_cls, str):
            raise ValueError(
                f"Unknown loss_cls for YOLOMM: expected string, got {type(loss_cls)}"
            )
        normalized = loss_cls.lower()
        valid = v8DetectionLoss.SUPPORTED_CLS_LOSSES.keys()
        if normalized not in valid:
            raise ValueError(
                f"Unknown loss_cls for YOLOMM: {normalized}. "
                f"Valid: {sorted(valid)}"
            )
        return normalized

    def _init_cls_loss_fn(
        self,
        loss_type: str,
        efl_gamma: float = 2.0,
        efl_alpha: float = 0.25,
        qfl_beta: float = 2.0,
        focal_gamma: float = 1.5,
        focal_alpha: float = 0.25,
        label_smoothing: float = 0.0,
    ) -> nn.Module:
        """Initialize the classification loss function based on type."""
        if loss_type == "bce":
            if label_smoothing > 0:
                return LabelSmoothingBCELoss(smoothing=label_smoothing, reduction="none")
            return self.bce
        elif loss_type == "focal":
            return FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        elif loss_type == "varifocal":
            return VarifocalLoss(gamma=2.0, alpha=0.75)
        elif loss_type == "efl":
            return EFClass(gamma=efl_gamma, alpha=efl_alpha)
        elif loss_type == "qfl":
            return QualityFocalLoss(beta=qfl_beta)
        else:
            raise ValueError(f"Unsupported loss_cls type: {loss_type}")

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _is_lspcd_dict(self, preds: Any) -> bool:
        """Return True when preds uses LSPCD structured output format."""
        return isinstance(preds, dict) and {"boxes", "scores", "feats"}.issubset(preds.keys())

    def _unpack_detection_preds(self, preds: Any):
        """Normalize standard Detect and LSPCD predictions for detection loss.

        Returns:
            tuple: (feats, pred_distri, pred_scores)
                feats: list[Tensor] used for anchors and image size
                pred_distri: Tensor[B, N, 4 * reg_max]
                pred_scores: Tensor[B, N, nc]
        """
        preds = preds[1] if isinstance(preds, tuple) else preds

        adapter = str(getattr(self.hyp, "loss_pred_adapter", "auto") or "auto").lower()
        if adapter not in {"auto", "standard", "lspcd"}:
            raise ValueError(f"Unknown loss_pred_adapter={adapter}. Valid values: auto, standard, lspcd")

        if self._is_lspcd_dict(preds):
            if adapter == "standard":
                raise TypeError("loss_pred_adapter=standard cannot consume LSPCD dict predictions")
            feats = preds["feats"]
            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
            pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
            return feats, pred_distri, pred_scores

        if adapter == "lspcd":
            raise TypeError("loss_pred_adapter=lspcd expected dict with boxes/scores/feats")

        feats = preds
        if not isinstance(feats, (list, tuple)):
            raise TypeError(
                f"Detection loss expected list/tuple feature maps or LSPCD dict, got {type(feats).__name__}"
            )
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        return feats, pred_distri, pred_scores

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size.

        保持前向在 AMP 下，但将损失计算强制在 FP32 中执行，提升数值稳定性。
        """
        feats, pred_distri, pred_scores = self._unpack_detection_preds(preds)

        batch_size = pred_scores.shape[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        from ultralytics.utils.torch_utils import autocast
        with autocast(enabled=False):
            loss = torch.zeros(3, device=self.device, dtype=torch.float32)  # box, cls, dfl

            # 统一 FP32 计算
            pred_scores_f = pred_scores.float()
            pred_distri_f = pred_distri.float()
            imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=torch.float32) * self.stride[0]
            anchor_points_f = anchor_points.float()
            stride_tensor_f = stride_tensor.float()

            # Targets
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

            # Pboxes
            pred_bboxes_f = self.bbox_decode(anchor_points_f, pred_distri_f)  # xyxy, (b, h*w, 4)

            _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores_f.detach().sigmoid(),
                (pred_bboxes_f.detach() * stride_tensor_f).type(gt_bboxes.dtype),
                anchor_points_f * stride_tensor_f,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

            target_scores_sum = max(target_scores.sum(), 1)
            target_scores_f = target_scores.to(torch.float32)

            # Cls loss
            if self.loss_cls_type == "focal":
                loss[1] = self.cls_loss_fn(pred_scores_f, target_scores_f) / target_scores_sum
            else:
                loss[1] = self.cls_loss_fn(pred_scores_f, target_scores_f).sum() / target_scores_sum

            # Bbox + DFL loss
            if fg_mask.sum():
                target_bboxes /= stride_tensor_f
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri_f,
                    pred_bboxes_f,
                    anchor_points_f,
                    target_bboxes,
                    target_scores,
                    target_scores_sum,
                    fg_mask,
                )

            loss[0] *= self.hyp.box  # box gain
            loss[1] *= self.hyp.cls  # cls gain
            loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class YOLOVOLoss(v8DetectionLoss):
    """YOLOVO open-vocabulary detection loss.

    继承 v8DetectionLoss，复用 box/DFL/assigner/preprocess/bbox_decode 流程。
    核心差异：分类 logits 不由模型直接输出，而是 embedding 与 t_enhanced
    通过 einsum 动态计算（开放词汇）。新增 objectness loss。

    loss 输出 4 项: (box_loss, cls_loss, dfl_loss, obj_loss)。
    """

    def __init__(self, model):
        """初始化 YOLOVO loss。

        Args:
            model: YOLOVOModel 实例（必须已 de-parallel）。
        """
        super().__init__(model)
        self.model = model  # 保留引用，用于 debug logger 访问
        # 从 YOLOVODetect head 获取 embedding 相关参数
        head = model.model[-1]
        self.embed_dim = head.embed_dim
        # logit_scale / logit_bias 是可学习参数，直接引用（共享梯度）
        self.logit_scale = head.logit_scale
        self.logit_bias = head.logit_bias
        # NOTE: self.nc=1 / self.no=65 继承自 v8DetectionLoss，在 YOLOVO 中
        # 不适用（使用动态 K）。仅被 _unpack_detection_preds 使用，
        # YOLOVOLoss 完全重写 __call__，不调用该方法。

    def _unpack_detection_preds(self, preds):
        """YOLOVO 不使用标准 unpack 路径，防止意外调用。"""
        raise RuntimeError(
            "YOLOVOLoss must unpack predictions manually in __call__. "
            "Do not call _unpack_detection_preds."
        )

    def _compute_cls_logits(self, embeddings, t_enhanced):
        """计算 embedding-text 分类 logits（训练路径）。

        Args:
            embeddings: [B, D, N] 来自 YOLOVODetect 的 embedding 输出。
            t_enhanced: [B, K, D] 增强后的文本特征。

        Returns:
            cls_logits: [B, N, K] 缩放后的分类 logits。
        """
        emb_norm = F.normalize(embeddings.permute(0, 2, 1), dim=-1)  # [B, N, D]
        t_norm = F.normalize(t_enhanced, dim=-1)                      # [B, K, D]
        cls_logits = torch.einsum("bnd,bkd->bnk", emb_norm, t_norm)
        # 训练时 clamp logit_scale 防止数值爆炸
        scale = self.logit_scale.clamp(max=math.log(100)).exp()
        return cls_logits * scale + self.logit_bias

    def __call__(self, preds, batch):
        """计算 YOLOVO 训练 loss。

        复用 v8DetectionLoss 的 box/dfl/assigner 流程，替换 cls_logits 计算，
        新增 objectness loss。

        Args:
            preds: YOLOVODetect 训练模式输出 dict，包含:
                - boxes: [B, 4*reg_max, N]
                - objectness: [B, 1, N]
                - embeddings: [B, D, N] (已 normalize)
                - feats: list[Tensor] 特征图
                - text_condition: (t_enhanced [B,K,D], text_context [B,D])
            batch: 标准训练 batch dict。

        Returns:
            (total_loss, loss_detach): total_loss=[box,cls,dfl,obj]*batch_size,
                                       loss_detach=未乘 batch_size 的各项。
        """
        # 从 dict 中手动取值（不调用 _unpack_detection_preds）
        boxes = preds["boxes"].permute(0, 2, 1)           # [B, N, 4*reg_max]
        obj = preds["objectness"].permute(0, 2, 1)         # [B, N, 1]
        embeddings = preds["embeddings"]                    # [B, D, N]
        feats = preds["feats"]
        t_enhanced = preds["text_condition"][0]            # [B, K, D]

        batch_size = boxes.shape[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 动态计算 cls_logits
        cls_logits = self._compute_cls_logits(embeddings, t_enhanced)  # [B, N, K]

        # 动态 K + 一致性校验（A1 防护）
        K = int(t_enhanced.shape[1])
        if cls_logits.shape[-1] != K:
            raise ValueError(
                f"cls_logits last dim {cls_logits.shape[-1]} != t_enhanced K={K}"
            )
        self.assigner.num_classes = K

        with autocast(enabled=False):
            # 4 项 loss: box, cls, dfl, obj
            loss = torch.zeros(4, device=self.device, dtype=torch.float32)

            # 统一 FP32 计算
            pred_distri_f = boxes.float()
            pred_scores_f = cls_logits.float()
            obj_f = obj.float()
            imgsz = (
                torch.tensor(feats[0].shape[2:], device=self.device, dtype=torch.float32)
                * self.stride[0]
            )
            anchor_points_f = anchor_points.float()
            stride_tensor_f = stride_tensor.float()

            # Targets
            targets = torch.cat(
                (
                    batch["batch_idx"].view(-1, 1),
                    batch["cls"].view(-1, 1),
                    batch["bboxes"],
                ),
                1,
            )
            targets = self.preprocess(
                targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
            )
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

            # 防御性检查: gt_labels 不应超过 K (skip on empty GT)
            if gt_labels.numel() > 0 and gt_labels.max() >= K:
                raise ValueError(
                    f"gt_labels max {gt_labels.max().item()} >= K={K}. "
                    f"Check remap_cls or text prompt setup."
                )

            # 预测框解码
            pred_bboxes_f = self.bbox_decode(anchor_points_f, pred_distri_f)

            # TaskAlignedAssigner
            _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores_f.detach().sigmoid(),
                (pred_bboxes_f.detach() * stride_tensor_f).type(gt_bboxes.dtype),
                anchor_points_f * stride_tensor_f,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

            target_scores_sum = max(target_scores.sum(), 1)
            target_scores_f = target_scores.to(torch.float32)

            # Cls loss (BCE)
            loss[1] = (
                self.cls_loss_fn(pred_scores_f, target_scores_f).sum()
                / target_scores_sum
            )

            # Bbox + DFL loss
            if fg_mask.sum():
                target_bboxes /= stride_tensor_f
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri_f,
                    pred_bboxes_f,
                    anchor_points_f,
                    target_bboxes,
                    target_scores,
                    target_scores_sum,
                    fg_mask,
                )

            # Objectness loss（YOLOVO 新增）
            # target: target_scores 每个锚点上的最大类分数作为 objectness 监督
            obj_target = target_scores.sum(-1, keepdim=True).clamp(0, 1)
            loss[3] = self.bce(obj_f, obj_target).sum() / target_scores_sum

            loss[0] *= self.hyp.box  # box gain
            loss[1] *= self.hyp.cls  # cls gain
            loss[2] *= self.hyp.dfl  # dfl gain
            # obj loss 使用 cls gain 作为权重
            loss[3] *= self.hyp.cls

        # YOLOVO debug: loss and text-alignment diagnostics.
        dbg = getattr(getattr(self, 'model', None), '_debug_logger', None)
        if dbg:
            tag = getattr(getattr(self, "model", None), "_debug_tag", "")
            trace_batch = bool(getattr(getattr(self, "model", None), "_debug_trace_batch", False))
            meta = dict(
                box=loss[0].item(),
                cls=loss[1].item(),
                dfl=loss[2].item(),
                obj=loss[3].item(),
                total=loss.sum().item(),
                batch_size=batch_size,
                K=K,
                fg_count=int(fg_mask.sum().item()),
                target_scores_sum=float(target_scores_sum.item() if hasattr(target_scores_sum, "item") else target_scores_sum),
                logit_scale=self.logit_scale.detach().clamp(max=math.log(100)).exp().item(),
                logit_bias=self.logit_bias.detach().item(),
            )
            if trace_batch:
                with torch.no_grad():
                    pos_mask = target_scores_f > 0
                    neg_mask = ~pos_mask
                    pos_logits = pred_scores_f[pos_mask]
                    neg_logits = pred_scores_f[neg_mask]
                    emb_norm = embeddings.detach().float().norm(dim=1)
                    pos_logit_mean = float(pos_logits.mean().item()) if pos_logits.numel() else 0.0
                    neg_logit_mean = float(neg_logits.mean().item()) if neg_logits.numel() else 0.0
                meta.update(
                    pos_logit_mean=pos_logit_mean,
                    neg_logit_mean=neg_logit_mean,
                    pos_neg_margin=pos_logit_mean - neg_logit_mean,
                    emb_norm_min=emb_norm.min().item(),
                    emb_norm_max=emb_norm.max().item(),
                )
            dbg.log(tag, "loss", **meta)

        return loss * batch_size, loss.detach()


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

        # Initialize segmentation loss function
        self.loss_mask_type = getattr(model.args, "loss_mask", "bce").lower()
        self._init_mask_loss_fn(model.args)

    def _init_mask_loss_fn(self, h):
        """Initialize the mask loss function based on loss_mask configuration."""
        if self.loss_mask_type == "bce":
            self.mask_loss_fn = None  # Use default BCE in single_mask_loss
        elif self.loss_mask_type == "dice":
            self.mask_loss_fn = DiceLoss(
                smooth=1.0,
                reduction="mean",
            )
        elif self.loss_mask_type == "focal_tversky":
            self.mask_loss_fn = FocalTverskyLoss(
                alpha=0.5,
                beta=0.5,
                gamma=1.0,
                reduction="mean",
            )
        elif self.loss_mask_type == "combo":
            self.mask_loss_fn = ComboLoss(
                alpha=0.5,
                beta=0.5,
                smooth=1.0,
                reduction="mean",
            )
        else:
            raise ValueError(f"Unknown loss_mask type: {self.loss_mask_type}. Valid: bce, dice, focal_tversky, combo")

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        # loss: [box, seg(inst), cls, dfl, semseg(aux)]
        loss = torch.zeros(5, device=self.device)
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]

        # YOLO26 Proto26 may return (proto, semseg) during training
        semseg = None
        if isinstance(proto, (tuple, list)):
            if len(proto) != 2:
                raise TypeError(f"Expected proto as Tensor or (proto, semseg), got len={len(proto)}")
            proto, semseg = proto

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Optional semantic segmentation auxiliary loss (YOLO26 Proto26)
        semseg_gain = float(getattr(self.hyp, "aux_semseg_gain", 0.1))
        if semseg is not None:
            if semseg.shape[0] != batch_size or semseg.shape[1] != self.nc:
                raise ValueError(f"semseg shape mismatch: {semseg.shape} vs (B={batch_size}, nc={self.nc})")

            if semseg_gain == 0.0:
                loss[4] = (semseg * 0).sum()
            else:
                semseg_h, semseg_w = semseg.shape[-2], semseg.shape[-1]
                inst_map = batch["masks"].to(self.device)
                if tuple(inst_map.shape[-2:]) != (semseg_h, semseg_w):
                    inst_map = F.interpolate(inst_map.float()[:, None], (semseg_h, semseg_w), mode="nearest")[:, 0]
                inst_map = inst_map.long()

                cls_all = batch["cls"].view(-1).to(self.device).long()
                bidx_all = batch["batch_idx"].view(-1).to(self.device).long()
                target_semseg = torch.zeros(
                    (batch_size, self.nc, semseg_h, semseg_w), device=self.device, dtype=torch.bool
                )

                for i in range(batch_size):
                    sel = bidx_all == i
                    if not sel.any():
                        continue
                    cls_i = cls_all[sel]
                    imap_i = inst_map[i]
                    for j, c in enumerate(cls_i):
                        c = int(c)
                        if 0 <= c < self.nc:
                            target_semseg[i, c].logical_or_(imap_i.eq(j + 1))

                loss[4] = F.binary_cross_entropy_with_logits(semseg.float(), target_semseg.float(), reduction="mean")

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if semseg is not None:
                loss[4] += (semseg * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain
        loss[4] *= semseg_gain  # semseg gain

        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl, semseg)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor,
        loss_fn=None
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).
            loss_fn (callable, optional): Custom loss function. If None, uses BCE.

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        # Crop mask to bounding box region
        pred_mask = crop_mask(pred_mask, xyxy)
        gt_mask = crop_mask(gt_mask, xyxy)

        if loss_fn is None:
            # Default BCE loss
            loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
            loss = loss.mean(dim=(1, 2)) / area
        else:
            # Use custom loss function
            if hasattr(loss_fn, 'forward'):
                # For Dice, FocalTversky, Combo losses
                loss = loss_fn(pred_mask.sigmoid(), gt_mask)
                # These return scalar, need to scale by area
                loss = loss * torch.ones_like(area)  # Match shape
            else:
                loss = loss_fn(pred_mask, gt_mask)

        return loss.sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i],
                    loss_fn=self.mask_loss_fn
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLO26 pose estimation with RLE loss support."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and RLE loss functions."""
        super().__init__(model)
        self.flow_model = model.model[-1].flow_model
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
        self.target_weights = (
            torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
        )

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for YOLO26 pose estimation."""
        # Unpack 3-tuple: (feats_list, kpt, kpt_sigma) from Pose26.forward
        if isinstance(preds[0], list):
            feats, pred_kpts, pred_sigma = preds
        else:
            feats, pred_kpts, pred_sigma = preds[1]

        loss = torch.zeros(6, device=self.device)  # box, kpt_loc, kpt_vis, cls, dfl, rle

        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        # Attach sigma before decode: (b, h*w, 17, 3) -> (b, h*w, 17, 5)
        pred_sigma_perm = pred_sigma.permute(0, 2, 1).contiguous()
        pred_sigma_perm = pred_sigma_perm.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
        pred_kpts = torch.cat([pred_kpts, pred_sigma_perm], dim=-1)  # (b, h*w, 17, 5)

        # Pose26 kpts_decode: direct +anchor, no 2.0 scale or -0.5 offset
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)  # (b, h*w, 17, 5)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2], loss[5] = self._calculate_keypoints_loss26(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain
        loss[5] *= self.hyp.rle  # rle gain

        return loss * batch_size, loss.detach()  # loss(box, kpt_loc, kpt_vis, cls, dfl, rle)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates. Pose26 uses direct +anchor without 2.0 scaling."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def _calculate_keypoints_loss26(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate keypoints loss with RLE support. Returns (kpts_loss, kpts_obj_loss, rle_loss)."""
        batch_idx_flat = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx_flat, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx_flat == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self._calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                rle_loss = rle_loss.clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss

    def _calculate_rle_loss(
        self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

        Args:
            pred_kpt (torch.Tensor): Predicted kpts with sigma, shape (N, num_keypoints, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, num_keypoints, kpts_dim).
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

        Returns:
            (torch.Tensor): The RLE loss.
        """
        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_kpt.device)

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        # Read loss_box for OBB (prob/ciou/none), default to 'prob' for backward compatibility
        loss_box = getattr(self.hyp, "loss_box", "prob")
        self.bbox_loss = RotatedBboxLoss(self.reg_max, iou_type=loss_box).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    @staticmethod
    def calculate_angle_loss(pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta).
            target_bboxes (torch.Tensor): Target bounding boxes with shape [N, 5] (x, y, w, h, theta).
            fg_mask (torch.Tensor): Foreground mask indicating valid predictions.
            weight (torch.Tensor): Loss weights for each prediction.
            target_scores_sum (torch.Tensor): Sum of target scores for normalization.
            lambda_val (int): Controls the sensitivity to aspect ratio.

        Returns:
            (torch.Tensor): The calculated angle loss.
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model, tal_topk2: int = None, tal_stride: list = None):
        """
        Initialize E2EDetectLoss with one-to-many and one-to-one detection losses.

        Args:
            model: The detection model.
            tal_topk2 (int, optional): Secondary topk for STAL filtering. None to disable.
            tal_stride (list, optional): Stride values for small object enhancement. None to disable.
        """
        self.one2many = v8DetectionLoss(model, tal_topk=10, tal_topk2=tal_topk2, tal_stride=tal_stride)
        self.one2one = v8DetectionLoss(model, tal_topk=1, tal_topk2=tal_topk2, tal_stride=tal_stride)

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ELoss:
    """
    Criterion class for computing training losses with ProgLoss (Progressive Loss Decay).

    YOLO26-style progressive loss weighting that gradually shifts from one-to-many (o2m)
    to one-to-one (o2o) supervision during training.

    The weight decay follows: o2m starts at 0.8 and decays to 0.1 over epochs,
    while o2o increases from 0.2 to 0.9.
    """

    def __init__(self, model, loss_fn=None, tal_stride: list = None):
        """
        Initialize E2ELoss with ProgLoss progressive weight decay.

        Args:
            model: The detection model.
            loss_fn: Loss function class to use (default: v8DetectionLoss).
            tal_stride (list, optional): Stride values for small object enhancement. None to disable.
        """
        if loss_fn is None:
            loss_fn = v8DetectionLoss

        # STAL configuration: topk=7 for o2o, topk2=1 for secondary filtering
        self.one2many = loss_fn(model, tal_topk=10, tal_stride=tal_stride)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1, tal_stride=tal_stride)

        # ProgLoss state
        self.updates = 0
        self.total = 1.0
        # Initial weights: o2m=0.8, o2o=0.2
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # Final weights: o2m=0.1, o2o=0.9
        self.final_o2m = 0.1

        # Store hyp for epoch calculation
        self.hyp = self.one2one.hyp

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the weighted sum of one2many and one2one losses.

        Uses current o2m/o2o weights for progressive loss decay.
        """
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        one2one = preds["one2one"]

        loss_one2many = self.one2many(one2many, batch)
        loss_one2one = self.one2one(one2one, batch)

        # Weighted combination with progressive decay
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """
        Update the weights for one-to-many and one-to-one losses based on the decay schedule.

        Should be called at the end of each epoch.
        """
        self.updates += 1
        self.o2m = self._decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def _decay(self, x: int) -> float:
        """
        Calculate the decayed weight for one-to-many loss based on the current update step.

        Linear decay from o2m_copy (0.8) to final_o2m (0.1) over epochs.
        """
        epochs = getattr(self.hyp, 'epochs', 100)
        return max(1 - x / max(epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m

    def get_weights(self) -> Tuple[float, float]:
        """Return current o2m and o2o weights for logging."""
        return self.o2m, self.o2o


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]


class SemSegLoss:
    """Semantic segmentation loss: CE + Dice + Aux CE.

    Supports multiclass (nc>1) and binary (nc=1) segmentation.
    Auxiliary head loss weighted at 0.4 (deep supervision from P4).
    """

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.hyp = model.args
        self.nc = int(model.yaml["nc"])
        if self.nc < 1:
            raise ValueError(f"SemSeg nc must be >= 1, got {self.nc}")
        self.ignore_index = int(getattr(self.hyp, "ignore_index", 255))
        self.aux_weight = 0.4

        # 主 CE Loss
        if self.nc == 1:
            self.ce = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # loss_names
        self.names = ("semseg_ce_loss", "semseg_dice_loss", "semseg_aux_loss")

    def __call__(self, preds, batch):
        # 解包预测
        if isinstance(preds, (tuple, list)):
            main_logits, aux_logits = preds[0], preds[1] if len(preds) > 1 else None
        else:
            main_logits, aux_logits = preds, None

        target = batch["sem_mask"].to(self.device).long()

        # 尺寸对齐
        if main_logits.shape[-2:] != target.shape[-2:]:
            main_logits = F.interpolate(main_logits, target.shape[-2:], mode="bilinear", align_corners=False)

        # CE Loss
        ce_loss = self._ce_loss(main_logits, target)

        # Dice Loss
        dice_loss = self._dice_loss(main_logits, target)

        # Aux Loss (P4 辅助头)
        aux_loss = torch.tensor(0.0, device=self.device)
        if aux_logits is not None:
            if aux_logits.shape[-2:] != target.shape[-2:]:
                aux_logits = F.interpolate(aux_logits, target.shape[-2:], mode="bilinear", align_corners=False)
            aux_loss = self._ce_loss(aux_logits, target)

        total = ce_loss + dice_loss + self.aux_weight * aux_loss
        batch_size = main_logits.shape[0]
        return total * batch_size, torch.stack([ce_loss.detach(), dice_loss.detach(), aux_loss.detach()])

    def _ce_loss(self, logits, target):
        """Cross-entropy loss. Binary mode uses BCEWithLogitsLoss."""
        if self.nc == 1:
            # 二值分割: target 为 [B,H,W]，logits 为 [B,1,H,W]
            probs = logits.squeeze(1)
            valid = target != self.ignore_index
            if not valid.any():
                return logits.sum() * 0.0
            t = target.float()
            t[~valid] = 0.0
            loss = self.ce(probs, t)
            return loss[valid].mean()
        else:
            loss = self.ce(logits, target)
            return loss if not torch.isnan(loss) else logits.sum() * 0.0

    def _dice_loss(self, logits, target):
        """Dice loss. Binary mode uses sigmoid; multiclass uses softmax."""
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0

        if self.nc == 1:
            # 二值 Dice
            probs = logits.squeeze(1).sigmoid()
            t = target.float()
            t[~valid] = 0.0
            intersection = (probs * t * valid.float()).sum()
            union = (probs * valid.float()).sum() + (t * valid.float()).sum()
            return 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        else:
            # 多类 Dice
            nc = self.nc
            safe_target = target.clamp(0, nc - 1)
            onehot = F.one_hot(safe_target, num_classes=nc).permute(0, 3, 1, 2).float()
            valid_f = valid.unsqueeze(1).float()
            onehot = onehot * valid_f
            probs = logits.softmax(dim=1) * valid_f
            intersection = (probs * onehot).sum(dim=(0, 2, 3))
            union = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
            present = union > 0
            if not present.any():
                return logits.sum() * 0.0
            dice = (2.0 * intersection[present] + 1e-6) / (union[present] + 1e-6)
            return 1.0 - dice.mean()
