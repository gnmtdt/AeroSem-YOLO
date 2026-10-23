# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""
Multimodal Loss Functions for YOLOMM

This module implements multimodal-specific loss functions for RGB-X feature alignment
and other multi-modality tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlignmentLoss(nn.Module):
    """
    Feature Alignment Loss for multimodal RGB-X feature alignment.

    This loss encourages the features from RGB and X (e.g., thermal, depth) modalities
    to be aligned in a shared representation space. It uses multiple strategies:
    1. L2 distance between normalized features
    2. Correlation alignment
    3. Multi-level alignment from different network stages

    Args:
        alignment_type (str): Type of alignment loss. Options: 'l2', 'correlation', 'cosine'. Default: 'cosine'
        weight (float): Weight for the alignment loss. Default: 1.0
        reduction (str): Reduction method. Options: 'none', 'mean', 'sum'. Default: 'mean'
        temperature (float): Temperature for cosine similarity. Default: 0.1
        detach_x (bool): If True, detach X features to prevent gradient from flowing to X branch. Default: True
    """

    def __init__(
        self,
        alignment_type: str = "cosine",
        weight: float = 1.0,
        reduction: str = "mean",
        temperature: float = 0.1,
        detach_x: bool = True,
    ):
        super().__init__()
        self.alignment_type = alignment_type.lower()
        self.weight = weight
        self.reduction = reduction
        self.temperature = temperature
        self.detach_x = detach_x

        valid_types = ["l2", "correlation", "cosine", "mmd"]
        if self.alignment_type not in valid_types:
            raise ValueError(f"Unknown alignment_type: {alignment_type}. Valid: {valid_types}")

    def forward(
        self,
        rgb_features: torch.Tensor,
        x_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate feature alignment loss.

        Args:
            rgb_features (torch.Tensor): RGB branch features, shape (B, C, H, W) or list of tensors
            x_features (torch.Tensor): X modality features, shape (B, C, H, W) or list of tensors

        Returns:
            torch.Tensor: Alignment loss value
        """
        if isinstance(rgb_features, (list, tuple)):
            # Multi-level features
            loss = 0
            for rgb_feat, x_feat in zip(rgb_features, x_features):
                loss += self._compute_alignment(rgb_feat, x_feat)
            return loss / len(rgb_features) * self.weight

        return self._compute_alignment(rgb_features, x_features) * self.weight

    def _compute_alignment(self, rgb_feat: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss for a single pair of features."""
        # Detach X features if configured
        if self.detach_x:
            x_feat = x_feat.detach()

        # Ensure same shape
        if rgb_feat.shape != x_feat.shape:
            # Try to interpolate
            x_feat = F.interpolate(
                x_feat, size=rgb_feat.shape[-2:], mode="bilinear", align_corners=False
            )

        if self.alignment_type == "l2":
            return self._l2_alignment(rgb_feat, x_feat)
        elif self.alignment_type == "correlation":
            return self._correlation_alignment(rgb_feat, x_feat)
        elif self.alignment_type == "cosine":
            return self._cosine_alignment(rgb_feat, x_feat)
        elif self.alignment_type == "mmd":
            return self._mmd_alignment(rgb_feat, x_feat)
        return torch.tensor(0.0, device=rgb_feat.device)

    def _l2_alignment(self, rgb_feat: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """L2 distance alignment."""
        # Normalize features
        rgb_norm = F.normalize(rgb_feat.flatten(1), dim=1)
        x_norm = F.normalize(x_feat.flatten(1), dim=1)

        # L2 distance between normalized features
        diff = rgb_norm - x_norm
        loss = (diff ** 2).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _correlation_alignment(self, rgb_feat: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Correlation alignment (CORAL) loss."""
        # Flatten spatial dimensions
        rgb_flat = rgb_feat.flatten(1)
        x_flat = x_feat.flatten(1)

        # Center the features
        rgb_centered = rgb_flat - rgb_flat.mean(dim=1, keepdim=True)
        x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)

        # Compute covariance
        rgb_cov = torch.mm(rgb_centered, rgb_centered.t()) / (rgb_flat.shape[1] - 1)
        x_cov = torch.mm(x_centered, x_centered.t()) / (x_flat.shape[1] - 1)

        # Frobenius norm of difference
        loss = torch.norm(rgb_cov - x_cov, p="fro")

        if self.reduction == "mean":
            return loss / rgb_feat.shape[0]
        return loss

    def _cosine_alignment(self, rgb_feat: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Cosine similarity alignment (maximize similarity = minimize 1 - similarity)."""
        # Flatten spatial dimensions
        rgb_flat = F.normalize(rgb_feat.flatten(1), dim=1)
        x_flat = F.normalize(x_feat.flatten(1), dim=1)

        # Cosine similarity
        similarity = (rgb_flat * x_flat).sum(dim=1)

        # Loss = 1 - similarity (we want to maximize similarity)
        loss = 1 - similarity

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _mmd_alignment(self, rgb_feat: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Maximum Mean Discrepancy (MMD) alignment."""
        # Flatten spatial dimensions
        rgb_flat = rgb_feat.flatten(1)
        x_flat = x_feat.flatten(1)

        # Use Gaussian kernel
        diff_sq = ((rgb_flat.unsqueeze(1) - x_flat.unsqueeze(0)) ** 2).sum(dim=2)

        # Compute MMD with Gaussian kernel
        kernels = torch.exp(-diff_sq / (2 * self.temperature ** 2))

        # Diagonal (same source) and off-diagonal (different source)
        n = rgb_flat.shape[0]
        if n > 1:
            # MMD^2 = E[k(x, x')] + E[k(y, y')] - 2*E[k(x, y)]
            k_xx = kernels[:n, :n]
            k_yy = kernels[n:, n:]
            k_xy = kernels[:n, n:]

            # Average of kernel matrices
            mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        else:
            mmd = torch.tensor(0.0, device=rgb_feat.device)

        return mmd


class CrossModalContrastiveLoss(nn.Module):
    """
    Cross-Modal Contrastive Loss for multimodal feature learning.

    This loss encourages features from the same spatial location across different
    modalities to be closer in feature space, while pushing apart features from
    different locations.

    Args:
        temperature (float): Temperature parameter for softmax. Default: 0.1
        weight (float): Weight for the loss. Default: 1.0
        detach_x (bool): If True, detach X features. Default: True
    """

    def __init__(
        self,
        temperature: float = 0.1,
        weight: float = 1.0,
        detach_x: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.detach_x = detach_x

    def forward(
        self,
        rgb_features: torch.Tensor,
        x_features: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculate cross-modal contrastive loss.

        Args:
            rgb_features (torch.Tensor): RGB branch features, shape (B, C, H, W)
            x_features (torch.Tensor): X modality features, shape (B, C, H, W)
            labels (torch.Tensor, optional): Class labels for same-class sampling. If None, uses all positives.

        Returns:
            torch.Tensor: Contrastive loss value
        """
        if self.detach_x:
            x_features = x_features.detach()

        # Flatten spatial dimensions
        B, C, H, W = rgb_features.shape
        rgb_flat = rgb_features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        x_flat = x_features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # Normalize features
        rgb_norm = F.normalize(rgb_flat, dim=2)
        x_norm = F.normalize(x_flat, dim=2)

        # Compute similarity matrix
        # similarity[i, j, k, l] = sim(rgb_i_k, x_j_l)
        # We want RGB_i to be close to X_i (same sample)
        rgb_flat_2d = rgb_norm.reshape(B * H * W, C)
        x_flat_2d = x_norm.reshape(B * H * W, C)

        # Full similarity matrix
        sim_matrix = torch.mm(rgb_flat_2d, x_flat_2d.t()) / self.temperature  # (B*H*W, B*H*W)

        # Create positive mask (same sample index)
        # For RGB_i_k and X_j_l, they are positive if i == j
        rgb_idx = torch.arange(B, device=rgb_features.device).repeat_interleave(H * W)
        x_idx = torch.arange(B, device=x_features.device).repeat(H * W)
        positive_mask = (rgb_idx.unsqueeze(1) == x_idx.unsqueeze(0)).float()

        # For each RGB feature, we want it to be close to all X features from same sample
        # But for simplicity, we just use positive mask

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean of log-likelihood over positives
        loss = -(positive_mask * log_prob).sum() / (positive_mask.sum() + 1e-8)

        return loss * self.weight


def compute_multimodal_loss(
    rgb_features: torch.Tensor,
    x_features: torch.Tensor,
    loss_type: str = "cosine",
    **kwargs
) -> torch.Tensor:
    """
    Compute multimodal alignment loss.

    Args:
        rgb_features (torch.Tensor): RGB features
        x_features (torch.Tensor): X modality features
        loss_type (str): Type of loss. Options: 'cosine', 'l2', 'correlation', 'mmd', 'contrastive'
        **kwargs: Additional parameters for the loss

    Returns:
        torch.Tensor: Loss value
    """
    loss_type = loss_type.lower()

    if loss_type in ["cosine", "l2", "correlation", "mmd"]:
        loss_fn = FeatureAlignmentLoss(
            alignment_type=loss_type if loss_type != "mmd" else "mmd",
            weight=kwargs.get("weight", 1.0),
            temperature=kwargs.get("temperature", 0.1),
            detach_x=kwargs.get("detach_x", True),
        )
        return loss_fn(rgb_features, x_features)
    elif loss_type == "contrastive":
        loss_fn = CrossModalContrastiveLoss(
            temperature=kwargs.get("temperature", 0.1),
            weight=kwargs.get("weight", 1.0),
            detach_x=kwargs.get("detach_x", True),
        )
        return loss_fn(rgb_features, x_features, kwargs.get("labels", None))
    else:
        raise ValueError(f"Unknown multimodal loss type: {loss_type}")