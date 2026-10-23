# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""
Extended IoU Loss Functions for YOLOMM

This module implements advanced IoU-based loss functions:
- SIoU (Sketch-IoU): https://arxiv.org/abs/2205.12740
- EIoU (Efficient-IoU): https://arxiv.org/abs/2101.08158
- WIoU (Wise-IoU): https://arxiv.org/abs/2301.10046
- Alpha-IoU: https://arxiv.org/abs/2112.05103
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIoULoss(nn.Module):
    """
    SIoU (Sketch-IoU) Loss.

    SIoU considers angle costs, distance costs, and shape costs to improve
    bounding box regression accuracy. It helps the model converge faster
    by focusing on the regression of the nearest axis.

    Paper: https://arxiv.org/abs/2205.12740

    Args:
        angle_alpha (float): Weight for angle cost. Default: 3.0
        dist_beta (float): Exponent for distance cost. Default: 1.0
        gamma (float): Weight for shape cost. Default: 0.5
        eps (float): Small value for numerical stability. Default: 1e-7
    """

    def __init__(
        self,
        angle_alpha: float = 3.0,
        dist_beta: float = 1.0,
        gamma: float = 0.5,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.angle_alpha = angle_alpha
        self.dist_beta = dist_beta
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        xywh: bool = True,
    ) -> torch.Tensor:
        """
        Calculate SIoU loss.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes, shape (N, 4)
            target_boxes (torch.Tensor): Target boxes, shape (N, 4)
            xywh (bool): If True, boxes are in (x, y, w, h) format, else (x1, y1, x2, y2)

        Returns:
            torch.Tensor: SIoU loss values, shape (N,)
        """
        # Convert to xyxy if needed
        if xywh:
            px, py, pw, ph = pred_boxes.unbind(-1)
            tx, ty, tw, th = target_boxes.unbind(-1)
            px1 = px - pw / 2
            py1 = py - ph / 2
            px2 = px + pw / 2
            py2 = py + ph / 2
            tx1 = tx - tw / 2
            ty1 = ty - th / 2
            tx2 = tx + tw / 2
            ty2 = ty + th / 2
        else:
            px1, py1, px2, py2 = pred_boxes.unbind(-1)
            tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)
            pw = px2 - px1
            ph = py2 - py1
            tw = tx2 - tx1
            th = ty2 - ty1

        # Intersection
        ix1 = torch.maximum(px1, tx1)
        iy1 = torch.maximum(py1, ty1)
        ix2 = torch.minimum(px2, tx2)
        iy2 = torch.minimum(py2, ty2)
        inter_w = (ix2 - ix1).clamp(0)
        inter_h = (iy2 - iy1).clamp(0)
        inter = inter_w * inter_h

        # Union
        pred_area = pw * ph
        target_area = tw * th
        union = pred_area + target_area - inter + self.eps

        # IoU
        iou = inter / union

        # Center points
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2

        # Angle cost (focus on regression of the nearest axis)
        delta_x = tcx - pcx
        delta_y = tcy - pcy
        angle = torch.atan2(delta_y, delta_x)

        # Compute sin and cos of angle
        sin_theta = torch.sin(angle) * self.angle_alpha
        cos_theta = torch.cos(angle) * self.angle_alpha

        # Angle cost: prioritize alignment with x or y axis
        angle_cost = 1 - torch.abs(sin_theta) * torch.abs(cos_theta)

        # Distance cost (penalize center distance)
        ch = torch.maximum(py2, ty2) - torch.minimum(py1, ty1)
        cw = torch.maximum(px2, tx2) - torch.minimum(px1, tx1)

        # Distance to the bounding box center
        distance_x = (delta_x / cw) ** 2
        distance_y = (delta_y / ch) ** 2
        distance_cost = distance_x + distance_y

        # Shape cost (aspect ratio consistency)
        width_diff = (pw - tw) ** 2
        height_diff = (ph - th) ** 2
        shape_cost = width_diff + height_diff

        # Combined shape cost with gamma
        shape_loss = shape_cost / (2 * (cw ** 2 + ch ** 2) + self.eps)

        # SIoU loss
        siou = iou - 0.5 * (angle_cost + distance_cost + self.gamma * shape_loss)

        return siou.clamp(0, 1)


class EIoULoss(nn.Module):
    """
    EIoU (Efficient-IoU) Loss.

    EIoU improves CIoU by separately penalizing width and height differences
    between predicted and target boxes, providing better guidance for box regression.

    Paper: https://arxiv.org/abs/2101.08158

    Args:
        rho (float): Weight for the overlap and center distance terms. Default: 1.0
        sigma (float): Weight for the width/height penalty terms. Default: 1.0
        eps (float): Small value for numerical stability. Default: 1e-7
    """

    def __init__(
        self,
        rho: float = 1.0,
        sigma: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.rho = rho
        self.sigma = sigma
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        xywh: bool = True,
    ) -> torch.Tensor:
        """
        Calculate EIoU loss.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes, shape (N, 4)
            target_boxes (torch.Tensor): Target boxes, shape (N, 4)
            xywh (bool): If True, boxes are in (x, y, w, h) format, else (x1, y1, x2, y2)

        Returns:
            torch.Tensor: EIoU loss values, shape (N,)
        """
        # Convert to xyxy if needed
        if xywh:
            px, py, pw, ph = pred_boxes.unbind(-1)
            tx, ty, tw, th = target_boxes.unbind(-1)
            px1 = px - pw / 2
            py1 = py - ph / 2
            px2 = px + pw / 2
            py2 = py + ph / 2
            tx1 = tx - tw / 2
            ty1 = ty - th / 2
            tx2 = tx + tw / 2
            ty2 = ty + th / 2
        else:
            px1, py1, px2, py2 = pred_boxes.unbind(-1)
            tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)
            pw = px2 - px1
            ph = py2 - py1
            tw = tx2 - tx1
            th = ty2 - ty1

        # Intersection
        ix1 = torch.maximum(px1, tx1)
        iy1 = torch.maximum(py1, ty1)
        ix2 = torch.minimum(px2, tx2)
        iy2 = torch.minimum(py2, ty2)
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

        # Union
        union = pw * ph + tw * th - inter + self.eps

        # IoU
        iou = inter / union

        # Smallest enclosing box
        cx1 = torch.minimum(px1, tx1)
        cy1 = torch.minimum(py1, ty1)
        cx2 = torch.maximum(px2, tx2)
        cy2 = torch.maximum(py2, ty2)
        cw = cx2 - cx1
        ch = cy2 - cy1

        # Center distance
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2

        # Distance cost (normalized by diagonal of enclosing box)
        center_dist = ((pcx - tcx) ** 2 + (pcy - tcy) ** 2) / (cw ** 2 + ch ** 2 + self.eps)

        # Width and height penalties
        width_penalty = (pw - tw) ** 2 / (cw ** 2 + self.eps)
        height_penalty = (ph - th) ** 2 / (ch ** 2 + self.eps)

        # EIoU
        eiou = iou - self.rho * center_dist - self.sigma * (width_penalty + height_penalty)

        return eiou.clamp(0, 1)


class WIoULoss(nn.Module):
    """
    WIoU (Wise-IoU) Loss.

    WIoU uses a dynamic gradient gain strategy based on the quality of
    anchor boxes. It applies smaller gradients to high-quality anchor boxes
    to prevent them from dominating the gradient.

    Paper: https://arxiv.org/abs/2301.10046

    Args:
        v_threshold (float): Threshold for outlier detection. Default: 0.7
        eps (float): Small value for numerical stability. Default: 1e-6
    """

    def __init__(
        self,
        v_threshold: float = 0.7,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        xywh: bool = True,
    ) -> torch.Tensor:
        """
        Calculate WIoU loss (version 3).

        Args:
            pred_boxes (torch.Tensor): Predicted boxes, shape (N, 4)
            target_boxes (torch.Tensor): Target boxes, shape (N, 4)
            xywh (bool): If True, boxes are in (x, y, w, h) format, else (x1, y1, x2, y2)

        Returns:
            torch.Tensor: WIoU loss values, shape (N,)
        """
        # Convert to xyxy if needed
        if xywh:
            px, py, pw, ph = pred_boxes.unbind(-1)
            tx, ty, tw, th = target_boxes.unbind(-1)
            px1 = px - pw / 2
            py1 = py - ph / 2
            px2 = px + pw / 2
            py2 = py + ph / 2
            tx1 = tx - tw / 2
            ty1 = ty - th / 2
            tx2 = tx + tw / 2
            ty2 = ty + th / 2
        else:
            px1, py1, px2, py2 = pred_boxes.unbind(-1)
            tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)
            pw = px2 - px1
            ph = py2 - py1
            tw = tx2 - tx1
            th = ty2 - ty1

        # Intersection
        ix1 = torch.maximum(px1, tx1)
        iy1 = torch.maximum(py1, ty1)
        ix2 = torch.minimum(px2, tx2)
        iy2 = torch.minimum(py2, ty2)
        inter_w = (ix2 - ix1).clamp(0)
        inter_h = (iy2 - iy1).clamp(0)
        inter = inter_w * inter_h

        # Union
        union = pw * ph + tw * th - inter + self.eps

        # Smallest enclosing box
        cx1 = torch.minimum(px1, tx1)
        cy1 = torch.minimum(py1, ty1)
        cx2 = torch.maximum(px2, tx2)
        cy2 = torch.maximum(py2, ty2)
        cw = cx2 - cx1
        ch = cy2 - cy1

        # Center distance
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2

        # Distance cost
        dist = ((pcx - tcx) ** 2 + (pcy - tcy) ** 2) / (cw ** 2 + ch ** 2 + self.eps)

        # WIoU v3: IoU - distance cost
        iou = inter / union
        wiou = iou - dist

        # Outlier detection for dynamic gradient gain
        # High-quality anchors (small loss) get smaller gradients
        beta = iou.detach().clone()
        beta = torch.where(beta < self.v_threshold, beta, torch.zeros_like(beta))

        # Apply monotonic gradient focusing
        # r is the gradient gain, inversely related to anchor quality
        r = torch.ones_like(beta)
        mask = beta < self.v_threshold
        r[mask] = beta[mask] / self.v_threshold

        # WIoU loss with gradient gain
        loss = (1 - wiou) * r + (1 - r) * (1 - wiou).detach()

        return loss.clamp(0, 1)


class AlphaIoULoss(nn.Module):
    """
    Alpha-IoU Loss.

    Alpha-IoU generalizes CIoU by introducing a power parameter alpha
    that controls the sensitivity to outliers. When alpha > 1, it provides
    more gradients for well-localized boxes.

    Paper: https://arxiv.org/abs/2112.05103

    Args:
        alpha (float): Power parameter for IoU. Higher values (>1) increase
                       sensitivity to outliers. Default: 2.0
        eps (float): Small value for numerical stability. Default: 1e-7
    """

    def __init__(
        self,
        alpha: float = 2.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        xywh: bool = True,
    ) -> torch.Tensor:
        """
        Calculate Alpha-IoU loss.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes, shape (N, 4)
            target_boxes (torch.Tensor): Target boxes, shape (N, 4)
            xywh (bool): If True, boxes are in (x, y, w, h) format, else (x1, y1, x2, y2)

        Returns:
            torch.Tensor: Alpha-IoU loss values, shape (N,)
        """
        # Convert to xyxy if needed
        if xywh:
            px, py, pw, ph = pred_boxes.unbind(-1)
            tx, ty, tw, th = target_boxes.unbind(-1)
            px1 = px - pw / 2
            py1 = py - ph / 2
            px2 = px + pw / 2
            py2 = py + ph / 2
            tx1 = tx - tw / 2
            ty1 = ty - th / 2
            tx2 = tx + tw / 2
            ty2 = ty + th / 2
        else:
            px1, py1, px2, py2 = pred_boxes.unbind(-1)
            tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)
            pw = px2 - px1
            ph = py2 - py1
            tw = tx2 - tx1
            th = ty2 - ty1

        # Intersection
        ix1 = torch.maximum(px1, tx1)
        iy1 = torch.maximum(py1, ty1)
        ix2 = torch.minimum(px2, tx2)
        iy2 = torch.minimum(py2, ty2)
        inter_w = (ix2 - ix1).clamp(0)
        inter_h = (iy2 - iy1).clamp(0)
        inter = inter_w * inter_h + self.eps

        # Union
        union = pw * ph + tw * th - inter + self.eps

        # Alpha-IoU
        iou_alpha = (inter / union) ** self.alpha

        # Smallest enclosing box for distance and aspect ratio
        cw = torch.maximum(px2, tx2) - torch.minimum(px1, tx1)
        ch = torch.maximum(py2, ty2) - torch.minimum(py1, ty1)

        # Center distance
        c2 = cw ** 2 + ch ** 2 + self.eps
        rho2 = ((tx1 + tx2 - px1 - px2) ** 2 + (ty1 + ty2 - py1 - py2) ** 2) / 4

        # Distance-IoU (alpha power)
        diou_alpha = iou_alpha - (rho2 / c2) ** self.alpha

        # Aspect ratio consistency (CIoU component with alpha)
        v = (4 / math.pi ** 2) * ((tw / th).atan() - (pw / ph).atan()) ** 2
        with torch.no_grad():
            alpha_v = v / (v - iou_alpha + (1 + self.eps))

        # Alpha-CIoU
        alphaciou = iou_alpha - (rho2 / c2) ** self.alpha - v * alpha_v

        return alphaciou.clamp(0, 1)


# Convenience function to compute all extended IoU losses
def compute_extended_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    iou_type: str,
    xywh: bool = True,
    **kwargs
) -> torch.Tensor:
    """
    Compute extended IoU loss.

    Args:
        box1 (torch.Tensor): First set of boxes, shape (N, 4)
        box2 (torch.Tensor): Second set of boxes, shape (N, 4)
        iou_type (str): Type of IoU loss. Options: 'siou', 'eiou', 'wiou', 'alphaiou', 'nwd', 'mpdiou'
        xywh (bool): If True, boxes are in (x, y, w, h) format, else (x1, y1, x2, y2)
        **kwargs: Additional parameters for specific IoU types

    Returns:
        torch.Tensor: IoU values, shape (N,)
    """
    iou_type = iou_type.lower()

    if iou_type == "siou":
        loss_fn = SIoULoss(
            angle_alpha=kwargs.get("siou_angle_alpha", 3.0),
            dist_beta=kwargs.get("siou_dist_beta", 1.0),
            gamma=kwargs.get("siou_gamma", 0.5),
            eps=kwargs.get("eps", 1e-7),
        )
    elif iou_type == "eiou":
        loss_fn = EIoULoss(
            rho=kwargs.get("eiou_rho", 1.0),
            sigma=kwargs.get("eiou_sigma", 1.0),
            eps=kwargs.get("eps", 1e-7),
        )
    elif iou_type == "wiou":
        loss_fn = WIoULoss(
            v_threshold=kwargs.get("wiou_v_threshold", 0.7),
            eps=kwargs.get("wiou_eps", 1e-6),
        )
    elif iou_type in ("alphaiou", "alpha_iou"):
        loss_fn = AlphaIoULoss(
            alpha=kwargs.get("alphaiou_alpha", 2.0),
            eps=kwargs.get("eps", 1e-7),
        )
    elif iou_type == "nwd":
        loss_fn = NWDLoss(
            sigma=kwargs.get("nwd_sigma", 0.5),
            eps=kwargs.get("eps", 1e-7),
        )
    elif iou_type in ("mpdiou", "mpdio"):
        loss_fn = MPDIoULoss(
            eps=kwargs.get("eps", 1e-7),
        )
    else:
        raise ValueError(f"Unknown extended IoU type: {iou_type}")

    return loss_fn(box1, box2, xywh=xywh)


class NWDLoss(nn.Module):
    """
    Normalized Wasserstein Distance Loss for small object detection.

    NWD provides a more stable measure for small objects than IoU, as small objects
    are more sensitive to slight position deviations. It computes the Wasserstein distance
    between predicted and target boxes normalized by the diagonal of the enclosing box.

    Paper: https://arxiv.org/abs/2110.13389

    Args:
        sigma (float): Standard deviation for the Gaussian kernel. Default: 0.5
        eps (float): Small value for numerical stability. Default: 1e-7
    """

    def __init__(self, sigma: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.sigma = sigma
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        xywh: bool = True,
    ) -> torch.Tensor:
        """
        Calculate NWD loss.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes, shape (N, 4)
            target_boxes (torch.Tensor): Target boxes, shape (N, 4)
            xywh (bool): If True, boxes are in (x, y, w, h) format, else (x1, y1, x2, y2)

        Returns:
            torch.Tensor: NWD loss values, shape (N,)
        """
        # Convert to xyxy if needed
        if xywh:
            px, py, pw, ph = pred_boxes.unbind(-1)
            tx, ty, tw, th = target_boxes.unbind(-1)
            px1 = px - pw / 2
            py1 = py - ph / 2
            px2 = px + pw / 2
            py2 = py + ph / 2
            tx1 = tx - tw / 2
            ty1 = ty - th / 2
            tx2 = tx + tw / 2
            ty2 = ty + th / 2
        else:
            px1, py1, px2, py2 = pred_boxes.unbind(-1)
            tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)
            pw = px2 - px1
            ph = py2 - py1
            tw = tx2 - tx1
            th = ty2 - ty1

        # Center points
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2

        # Diagonal of the enclosing box (normalize factor)
        cw = torch.maximum(px2, tx2) - torch.minimum(px1, tx1)
        ch = torch.maximum(py2, ty2) - torch.minimum(py1, ty1)
        diagonal = torch.sqrt(cw ** 2 + ch ** 2 + self.eps)

        # Center distance
        center_dist_sq = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

        # Width and height
        w_dist = (pw - tw) ** 2
        h_dist = (ph - th) ** 2

        # Normalized Wasserstein distance
        wasserstein_dist = torch.sqrt(center_dist_sq + w_dist + h_dist + self.eps)
        nwd = torch.exp(-wasserstein_dist / (diagonal * self.sigma + self.eps))

        return 1 - nwd


class MPDIoULoss(nn.Module):
    """
    MPDIoU (Minimum Point Distance IoU) Loss.

    MPDIoU computes loss based on the minimum point distance between corners of
    predicted and target boxes, providing better gradient flow for bounding box
    regression compared to traditional IoU losses.

    Paper: https://arxiv.org/abs/2307.07693

    Args:
        eps (float): Small value for numerical stability. Default: 1e-7
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        xywh: bool = True,
    ) -> torch.Tensor:
        """
        Calculate MPDIoU loss.

        Args:
            pred_boxes (torch.Tensor): Predicted boxes, shape (N, 4)
            target_boxes (torch.Tensor): Target boxes, shape (N, 4)
            xywh (bool): If True, boxes are in (x, y, w, h) format, else (x1, y1, x2, y2)

        Returns:
            torch.Tensor: MPDIoU loss values, shape (N,)
        """
        # Convert to xyxy if needed
        if xywh:
            px, py, pw, ph = pred_boxes.unbind(-1)
            tx, ty, tw, th = target_boxes.unbind(-1)
            px1 = px - pw / 2
            py1 = py - ph / 2
            px2 = px + pw / 2
            py2 = py + ph / 2
            tx1 = tx - tw / 2
            ty1 = ty - th / 2
            tx2 = tx + tw / 2
            ty2 = ty + th / 2
        else:
            px1, py1, px2, py2 = pred_boxes.unbind(-1)
            tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)

        # Calculate the minimum point distance
        # Corners of predicted box
        pred_corners = torch.stack([px1, py1, px2, py2], dim=-1)  # (N, 4) -> (N, 2, 2) conceptually
        # Actually we need corners as (x1,y1), (x2,y2), (x1,y2), (x2,y1)
        pred_points = torch.stack([
            torch.stack([px1, py1], dim=-1),  # top-left
            torch.stack([px2, py2], dim=-1),  # bottom-right
            torch.stack([px1, py2], dim=-1),  # bottom-left
            torch.stack([px2, py1], dim=-1),  # top-right
        ], dim=1)  # (N, 4, 2)

        target_points = torch.stack([
            torch.stack([tx1, ty1], dim=-1),  # top-left
            torch.stack([tx2, ty2], dim=-1),  # bottom-right
            torch.stack([tx1, ty2], dim=-1),  # bottom-left
            torch.stack([tx2, ty1], dim=-1),  # top-right
        ], dim=1)  # (N, 4, 2)

        # Compute minimum distance between corner sets
        # For each pred point, find min distance to all target points
        pred_points_expand = pred_points.unsqueeze(2)  # (N, 4, 1, 2)
        target_points_expand = target_points.unsqueeze(1)  # (N, 1, 4, 2)
        dist_matrix = torch.sqrt(((pred_points_expand - target_points_expand) ** 2).sum(dim=-1) + self.eps)  # (N, 4, 4)

        # Minimum point distance for each corner
        min_dist_pred, _ = dist_matrix.min(dim=2)  # (N, 4), min over target corners
        min_dist_target, _ = dist_matrix.min(dim=1)  # (N, 4), min over pred corners

        # Maximum of the two minimum distances
        mpdiou_term = torch.max(min_dist_pred, min_dist_target).sum(dim=1)  # (N,)

        # Intersection area
        ix1 = torch.maximum(px1, tx1)
        iy1 = torch.maximum(py1, ty1)
        ix2 = torch.minimum(px2, tx2)
        iy2 = torch.minimum(py2, ty2)
        inter_w = (ix2 - ix1).clamp(0)
        inter_h = (iy2 - iy1).clamp(0)
        inter = inter_w * inter_h

        # Union area
        pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
        target_area = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
        union = pred_area + target_area - inter + self.eps

        # IoU term
        iou = inter / union

        # MPDIoU = 1 - IoU + alpha * point_distance
        # where alpha balances the two terms
        alpha = 1.0
        mpdiou = 1 - iou + alpha * mpdiou_term / 4  # divide by 4 to normalize

        return mpdiou.clamp(0, 1)


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Dice loss is particularly effective for handling class imbalance in segmentation
    tasks by directly maximizing the overlap between predictions and ground truth.

    Args:
        smooth (float): Smoothing factor to avoid division by zero. Default: 1.0
        reduction (str): Reduction method. Options: 'none', 'mean', 'sum'. Default: 'mean'
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.

        Args:
            pred_masks (torch.Tensor): Predicted masks (after sigmoid), shape (B, H, W) or (B, N, H, W)
            target_masks (torch.Tensor): Target masks (binary), shape (B, H, W) or (B, N, H, W)

        Returns:
            torch.Tensor: Dice loss value
        """
        # Ensure binary targets
        target_masks = target_masks.float()

        # Flatten if needed
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)  # (B, 1, H, W)
        if target_masks.dim() == 3:
            target_masks = target_masks.unsqueeze(1)  # (B, 1, H, W)

        # Flatten spatial dimensions
        pred_flat = pred_masks.view(pred_masks.size(0), -1)
        target_flat = target_masks.view(target_masks.size(0), -1)

        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == "mean":
            return 1 - dice.mean()
        elif self.reduction == "sum":
            return (1 - dice).sum()
        else:
            return 1 - dice


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for segmentation.

    Focal Tversky loss is designed for small object segmentation by controlling
    the trade-off between precision and recall through alpha, beta, and gamma parameters.

    Args:
        alpha (float): Weight of false positives. Default: 0.5
        beta (float): Weight of false negatives. Default: 0.5
        gamma (float): Focal parameter. Default: 1.0
        smooth (float): Smoothing factor. Default: 1e-7
        reduction (str): Reduction method. Options: 'none', 'mean', 'sum'. Default: 'mean'
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        smooth: float = 1e-7,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Tversky loss.

        Args:
            pred_masks (torch.Tensor): Predicted masks (after sigmoid), shape (B, H, W) or (B, N, H, W)
            target_masks (torch.Tensor): Target masks (binary), shape (B, H, W) or (B, N, H, W)

        Returns:
            torch.Tensor: Focal Tversky loss value
        """
        # Ensure binary targets
        target_masks = target_masks.float()

        # Flatten if needed
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)  # (B, 1, H, W)
        if target_masks.dim() == 3:
            target_masks = target_masks.unsqueeze(1)  # (B, 1, H, W)

        # Flatten spatial dimensions
        pred_flat = pred_masks.view(pred_masks.size(0), -1)
        target_flat = target_masks.view(target_masks.size(0), -1)

        # Calculate Tversky index components
        tp = (pred_flat * target_flat).sum(dim=1)  # True positives
        fp = (pred_flat * (1 - target_flat)).sum(dim=1)  # False positives
        fn = ((1 - pred_flat) * target_flat).sum(dim=1)  # False negatives

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Focal Tversky loss
        focal_tversky = torch.pow(1 - tversky, self.gamma)

        if self.reduction == "mean":
            return focal_tversky.mean()
        elif self.reduction == "sum":
            return focal_tversky.sum()
        else:
            return focal_tversky


class ComboLoss(nn.Module):
    """
    Combo Loss for segmentation.

    Combo loss combines BCE and Dice loss to leverage the benefits of both:
    - BCE provides pixel-wise classification guidance
    - Dice loss handles class imbalance and boundary precision

    Args:
        alpha (float): Weight for BCE term. Default: 0.5
        beta (float): Weight for Dice term. Default: 0.5
        smooth (float): Smoothing factor for Dice. Default: 1.0
        reduction (str): Reduction method. Options: 'none', 'mean', 'sum'. Default: 'mean'
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """
        Calculate Combo loss.

        Args:
            pred_masks (torch.Tensor): Predicted masks (logits before sigmoid), shape (B, H, W) or (B, N, H, W)
            target_masks (torch.Tensor): Target masks (binary), shape (B, H, W) or (B, N, H, W)

        Returns:
            torch.Tensor: Combo loss value
        """
        # Ensure targets are float
        target_masks = target_masks.float()

        # Flatten if needed
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)
        if target_masks.dim() == 3:
            target_masks = target_masks.unsqueeze(1)

        # BCE term (use binary_cross_entropy_with_logits for numerical stability)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_masks, target_masks, reduction="none"
        )
        if self.reduction == "mean":
            bce_loss = bce_loss.mean()
        elif self.reduction == "sum":
            bce_loss = bce_loss.sum()

        # Dice term
        pred_sigmoid = torch.sigmoid(pred_masks)
        pred_flat = pred_sigmoid.view(pred_sigmoid.size(0), -1)
        target_flat = target_masks.view(target_masks.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean() if self.reduction == "mean" else (1 - dice).sum()

        # Combined loss
        return self.alpha * bce_loss + self.beta * dice_loss


# Extended function to compute all losses
def compute_mask_loss(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    loss_type: str,
    **kwargs
) -> torch.Tensor:
    """
    Compute mask/segmentation loss.

    Args:
        pred_masks (torch.Tensor): Predicted masks
        target_masks (torch.Tensor): Target masks
        loss_type (str): Type of loss. Options: 'bce', 'dice', 'focal_tversky', 'combo'
        **kwargs: Additional parameters for specific loss types

    Returns:
        torch.Tensor: Loss value
    """
    loss_type = loss_type.lower()

    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(pred_masks, target_masks)
    elif loss_type == "dice":
        loss_fn = DiceLoss(
            smooth=kwargs.get("dice_smooth", 1.0),
            reduction=kwargs.get("reduction", "mean"),
        )
        return loss_fn(pred_masks, target_masks)
    elif loss_type == "focal_tversky":
        loss_fn = FocalTverskyLoss(
            alpha=kwargs.get("focal_tversky_alpha", 0.5),
            beta=kwargs.get("focal_tversky_beta", 0.5),
            gamma=kwargs.get("focal_tversky_gamma", 1.0),
            smooth=kwargs.get("focal_tversky_smooth", 1e-7),
            reduction=kwargs.get("reduction", "mean"),
        )
        return loss_fn(pred_masks, target_masks)
    elif loss_type == "combo":
        loss_fn = ComboLoss(
            alpha=kwargs.get("combo_alpha", 0.5),
            beta=kwargs.get("combo_beta", 0.5),
            smooth=kwargs.get("combo_smooth", 1.0),
            reduction=kwargs.get("reduction", "mean"),
        )
        return loss_fn(pred_masks, target_masks)
    else:
        raise ValueError(f"Unknown mask loss type: {loss_type}")
