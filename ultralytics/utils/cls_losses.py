# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""
Extended Classification Loss Functions for YOLOMM

This module implements advanced classification loss functions:
- EFL (Equalized Focal Loss): https://arxiv.org/abs/2201.00486
- QFL (Quality Focal Loss): https://arxiv.org/abs/2108.00884
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EFClass(nn.Module):
    """
    EFL (Equalized Focal Loss) for Classification.

    EFL addresses class imbalance by equalizing the gradient contributions
    from different classes, preventing easy negatives from dominating.

    Paper: https://arxiv.org/abs/2201.00486

    Args:
        gamma (float): Focusing parameter for modulating loss. Default: 2.0
        alpha (float): Balancing factor for positive samples. Default: 0.25
        eps (float): Small value for numerical stability. Default: 1e-7
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculate EFL loss.

        Args:
            pred (torch.Tensor): Predicted logits, shape (N, C) or (N,)
            label (torch.Tensor): Ground truth labels, shape (N, C) or (N,)
            weight (torch.Tensor, optional): Sample weights, shape (N,)

        Returns:
            torch.Tensor: EFL loss values
        """
        # Standard BCE loss
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")

        # Prediction probability
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)

        # Gradient targeting: equalize contributions
        # EFL uses a dynamic threshold to equalize gradients
        with torch.no_grad():
            # Calculate gradient contribution weights
            # For positive samples: focus on hard positives (low p_t)
            # For negative samples: focus on hard negatives (high p_t)
            pos_mask = label > 0.5
            neg_mask = label < 0.5

            # EFL gradient equalization
            # Use inverse probability as weight to equalize
            pt_inv = 1.0 - p_t + self.eps
            gradient_weight = torch.ones_like(p_t)

            # For positives: lower p_t (harder) -> higher weight
            if pos_mask.any():
                gradient_weight[pos_mask] = (1.0 - p_t[pos_mask] + self.eps) ** self.gamma

            # For negatives: higher p_t (harder) -> higher weight
            if neg_mask.any():
                gradient_weight[neg_mask] = (p_t[neg_mask] + self.eps) ** self.gamma

        # Apply modulating factor
        loss = loss * gradient_weight

        # Apply alpha weighting for class balance
        if self.alpha >= 0:
            alpha_weight = torch.ones_like(label)
            alpha_weight[label > 0.5] = self.alpha
            alpha_weight[label < 0.5] = 1.0 - self.alpha
            loss = loss * alpha_weight

        if weight is not None:
            loss = loss * weight

        return loss


class QFL(nn.Module):
    """
    QFL (Quality Focal Loss) for Classification.

    QFL extends focal loss by incorporating quality annotations (IoU scores)
    to focus on high-quality samples and provide better gradient signals.

    Paper: https://arxiv.org/abs/2108.00884

    Args:
        beta (float): Focusing parameter that controls the down-weighting
                      of easy examples. Default: 2.0
        eps (float): Small value for numerical stability. Default: 1e-7
    """

    def __init__(
        self,
        beta: float = 2.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculate QFL loss.

        Args:
            pred (torch.Tensor): Predicted quality scores (logits), shape (N, C) or (N,)
            label (torch.Tensor): Ground truth quality scores (0-1), shape (N, C) or (N,)
                                For standard classification, use binary labels (0 or 1)
            weight (torch.Tensor, optional): Sample weights, shape (N,)

        Returns:
            torch.Tensor: QFL loss values

        Note:
            Unlike standard focal loss which uses binary labels (0 or 1),
            QFL uses continuous quality scores (typically IoU values in [0, 1]).
            For binary classification, labels can still be 0 or 1.
        """
        # Clamp labels to [0, 1] for stability
        label = label.clamp(0.0, 1.0)

        # Prediction probability
        pred_prob = pred.sigmoid().clamp(self.eps, 1.0 - self.eps)

        # QFL: |y - sigmoid(x)|^beta * BCE(x, y)
        # This provides quality-aware focusing
        quality_diff = (label - pred_prob).abs()
        focal_weight = quality_diff ** self.beta

        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")

        # Apply focal weighting
        loss = bce_loss * focal_weight

        if weight is not None:
            loss = loss * weight

        return loss

    def forward_with_iou(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        iou_scores: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculate QFL loss with explicit IoU scores as quality labels.

        Args:
            pred (torch.Tensor): Predicted quality scores (logits), shape (N,)
            label (torch.Tensor): Ground truth class labels (0 or 1), shape (N,)
            iou_scores (torch.Tensor): Quality scores (IoU values) for positive samples, shape (N,)
            weight (torch.Tensor, optional): Sample weights, shape (N,)

        Returns:
            torch.Tensor: QFL loss values
        """
        # For positive samples, use IoU as quality score
        # For negative samples, quality score is 0
        quality = label * iou_scores

        return self.forward(pred, quality, weight)


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss with integrated IoU scores.

    This is a more convenient wrapper that combines classification and
    quality assessment in a single loss function.

    Args:
        beta (float): Focusing parameter. Default: 2.0
        eps (float): Small value for numerical stability. Default: 1e-7
    """

    def __init__(
        self,
        beta: float = 2.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.qfl = QFL(beta=beta, eps=eps)

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        iou_scores: torch.Tensor = None,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculate Quality Focal Loss.

        Args:
            pred (torch.Tensor): Predicted logits, shape (N,)
            label (torch.Tensor): Ground truth labels (0 or 1), shape (N,)
            iou_scores (torch.Tensor, optional): IoU scores for quality. If None,
                                                 binary labels are used as quality.
            weight (torch.Tensor, optional): Sample weights, shape (N,)

        Returns:
            torch.Tensor: QFL loss values
        """
        if iou_scores is None:
            # Use binary labels as quality scores
            quality = label
        else:
            # Use IoU scores for positive samples
            quality = label * iou_scores

        return self.qfl(pred, quality, weight)


# Convenience function to create extended classification losses
def get_extended_cls_loss(
    loss_type: str,
    **kwargs
) -> nn.Module:
    """
    Create an extended classification loss function.

    Args:
        loss_type (str): Type of loss function. Options: 'efl', 'qfl'
        **kwargs: Additional parameters for the specific loss type

    Returns:
        nn.Module: The requested loss function
    """
    loss_type = loss_type.lower()

    if loss_type == "efl":
        return EFClass(
            gamma=kwargs.get("efl_gamma", 2.0),
            alpha=kwargs.get("efl_alpha", 0.25),
            eps=kwargs.get("eps", 1e-7),
        )
    elif loss_type in ("qfl", "quality_focal"):
        return QualityFocalLoss(
            beta=kwargs.get("qfl_beta", 2.0),
            eps=kwargs.get("eps", 1e-7),
        )
    else:
        raise ValueError(f"Unknown extended classification loss type: {loss_type}")
