# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
VCTE (Visual-Conditioned Text Enhancement) modules for YOLOVO.

This module provides the core components for text-visual feature interaction
in the YOLOVO open-vocabulary multimodal detection framework:

- VCTETextInput: Text data input socket
- VCTETextAdapter: Text feature fine-tuning with residual bottleneck
- VCTEVisualAdapter: Visual global context extraction from backbone features
- VCTETextEnhancer: Text-visual gated fusion for text enhancement
- VCTENeckModulator: Text-conditioned FiLM modulation for neck features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VCTETextInput(nn.Module):
    """Text data input socket for the VCTE pipeline.

    Passes through CLIP text embeddings unchanged. Serves as a named entry point
    in the model YAML for text modality routing.

    Args:
        dim (int): Text embedding dimension (e.g. 512 for MobileCLIP).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # P0-1: 标记为 TEXT 模态源，供 _predict_once 路由使用
        self._mm_input_source = "TEXT"

    def forward(self, t_clip: torch.Tensor) -> torch.Tensor:
        """Pass through CLIP text embeddings.

        Args:
            t_clip: Text embeddings of shape [K, D] or [B, K, D].

        Returns:
            Same tensor, unchanged.
        """
        return t_clip


class VCTETextAdapter(nn.Module):
    """Text feature fine-tuning adapter with residual bottleneck.

    Applies LayerNorm -> compress to bottleneck -> GELU -> restore to dim,
    then adds residual and L2-normalizes. The up-projection is initialized
    with small weights (std=1e-4) to start near-identity.

    Args:
        dim (int): Text embedding dimension.
        bottleneck (int): Bottleneck dimension. Default 64.
    """

    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)
        nn.init.normal_(self.up.weight, std=1e-4)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, t_clip: torch.Tensor) -> torch.Tensor:
        """Adapt text features with residual bottleneck.

        Args:
            t_clip: Text embeddings [K, D] or [B, K, D].

        Returns:
            L2-normalized adapted text features [K, D] or [B, K, D].
        """
        delta = self.up(self.act(self.down(self.norm(t_clip))))
        return F.normalize(t_clip + delta, dim=-1)


class VCTEVisualAdapter(nn.Module):
    """Visual global context extraction from backbone feature maps.

    Pipeline: BN -> 1x1 reduce -> multi-scale depth-wise conv branches
    (3x3 + 5x5 + dilated 3x3 + identity) -> ReLU -> avg+max pool ->
    ECA 1D channel mixing -> linear projection -> L2 normalize.

    Args:
        in_channels (int): Number of input channels from backbone.
        out_dim (int): Output embedding dimension. Default 512.
        reduce_dim (int): Intermediate reduction dimension. Default 128.
    """

    def __init__(self, in_channels: int, out_dim: int = 512, reduce_dim: int = 128):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.reduce = nn.Conv2d(in_channels, reduce_dim, 1, bias=False)
        self.dw3 = nn.Conv2d(reduce_dim, reduce_dim, 3, padding=1, groups=reduce_dim, bias=False)
        self.dw5 = nn.Conv2d(reduce_dim, reduce_dim, 5, padding=2, groups=reduce_dim, bias=False)
        self.dw_dilated = nn.Conv2d(reduce_dim, reduce_dim, 3, padding=2, dilation=2, groups=reduce_dim, bias=False)
        self.eca = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.proj = nn.Linear(reduce_dim, out_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global visual context from feature map.

        Args:
            x: Backbone feature map [B, C, H, W].

        Returns:
            L2-normalized visual context vector [B, out_dim].
        """
        x = self.reduce(self.norm(x))
        branch_sum = x + self.dw3(x) + self.dw5(x) + self.dw_dilated(x)
        branch_sum = F.relu(branch_sum)
        pool = branch_sum.mean(dim=[2, 3]) + branch_sum.amax(dim=[2, 3])
        pool = self.eca(pool.unsqueeze(1)).squeeze(1)
        return F.normalize(self.proj(pool), dim=-1)


class VCTETextEnhancer(nn.Module):
    """Text-visual gated fusion for text enhancement.

    Projects text and visual features to a shared low-rank space, computes
    additive + multiplicative interactions, then applies a learned gate
    with learnable alpha scaling. Outputs both enhanced text features and
    a mean-pooled text context vector.

    Args:
        text_dim (int): Input text feature dimension. Default 512.
        visual_dim (int): Input visual context dimension. Default 512.
        out_dim (int): Output text feature dimension. Default 512.
        rank (int): Low-rank projection dimension. Default 64.
    """

    def __init__(self, text_dim: int = 512, visual_dim: int = 512, out_dim: int = 512, rank: int = 64):
        super().__init__()
        self.Wt = nn.Linear(text_dim, rank, bias=False)
        self.Wv = nn.Linear(visual_dim, rank, bias=False)
        self.Wup = nn.Linear(rank, out_dim, bias=False)
        self.Wg = nn.Linear(rank, 1, bias=False)
        nn.init.normal_(self.Wup.weight, std=1e-4)
        nn.init.normal_(self.Wg.weight, std=1e-4)
        nn.init.normal_(self.Wt.weight, std=0.02)
        nn.init.normal_(self.Wv.weight, std=0.02)
        self.register_buffer("alpha", torch.tensor(0.0))

    def forward(self, inputs: list) -> tuple:
        """Enhance text features with visual context.

        Args:
            inputs: List of [t_base, v_ctx] where
                t_base: Adapted text features [K, D] or [B, K, D].
                v_ctx: Visual context vector [B, D].

        Returns:
            Tuple of (t_enhanced, text_context) where
                t_enhanced: Enhanced text features [B, K, D].
                text_context: Mean-pooled context [B, D].
        """
        t_base, v_ctx = inputs
        if t_base.dim() == 2:
            t_base = t_base.unsqueeze(0)  # [1, K, D]
        t_r = self.Wt(t_base)              # [B, K, r]
        v_r = self.Wv(v_ctx).unsqueeze(1)  # [B, 1, r]
        h = F.gelu(t_r + v_r + t_r * v_r)  # [B, K, r]
        delta = self.Wup(h)                # [B, K, D]
        gate = torch.sigmoid(self.Wg(h))   # [B, K, 1]
        t_enhanced = F.normalize(t_base + self.alpha * gate * delta, dim=-1)
        text_context = t_enhanced.mean(dim=1) if t_base.shape[1] > 0 else torch.zeros(
            t_base.shape[0], t_base.shape[2], device=t_base.device, dtype=t_base.dtype
        )
        return (t_enhanced, text_context)


class VCTENeckModulator(nn.Module):
    """Text-conditioned FiLM modulation for neck feature maps.

    Applies Feature-wise Linear Modulation (FiLM) conditioned on text context:
    gamma * feat + beta, where gamma and beta are derived from the text context
    vector through a two-layer MLP with learnable alpha scaling.

    Args:
        feat_channels (int): Number of channels in the input feature map.
        text_dim (int): Text context vector dimension. Default 512.
        hidden_dim (int): Hidden layer dimension. Default 128.
    """

    def __init__(self, feat_channels: int, text_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(text_dim, hidden_dim, bias=False)
        self.fc_gamma = nn.Linear(hidden_dim, feat_channels, bias=False)
        self.fc_beta = nn.Linear(hidden_dim, feat_channels, bias=False)
        nn.init.normal_(self.fc_gamma.weight, std=1e-4)
        nn.init.normal_(self.fc_beta.weight, std=1e-4)
        self.register_buffer("alpha", torch.tensor(0.0))

    def forward(self, inputs: list) -> torch.Tensor:
        """Apply text-conditioned modulation to feature map.

        Args:
            inputs: List of [feat, text_condition] where
                feat: Neck feature map [B, C, H, W].
                text_condition: Tuple of (t_enhanced, text_context).

        Returns:
            Modulated feature map [B, C, H, W].
        """
        feat, text_condition = inputs
        _, text_context = text_condition  # [B, D]
        h = F.gelu(self.fc1(text_context))
        gamma = self.fc_gamma(h).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.fc_beta(h).unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + self.alpha * gamma) + self.alpha * beta
