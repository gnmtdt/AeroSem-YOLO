"""
MaskUnitAttention - Mask Unit Attention (Hiera/ViTDet style)

论文: Hiera: A Hierarchical Vision Transformer without the Bells and Whistles
      ViTDet: Exploring Plain Vision Transformer Backbones for Object Detection
风格: Hiera / ViTDet 掩码单元注意力

支持掩码单元(Mask Unit)局部注意力和全局注意力的统一实现。
可配置Q池化步长(q_stride)降低计算量，支持窗口分区注意力。
优先使用 PyTorch scaled_dot_product_attention 加速。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        """
        Args:
        - dim: The input feature dimension.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.q_stride = q_stride

        self.head_dim = dim // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [batch, tokens, channels]. """
        restore_bchw = x.ndim == 4
        if x.ndim == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
        elif x.ndim != 3:
            raise ValueError(
                f"MaskUnitAttention expects input as [B, N, C] or [B, C, H, W], got {tuple(x.shape)}"
            )

        B, N, _ = x.shape
        num_windows = (
            (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        )

        qkv = (
            self.qkv(x)
            .reshape(B, -1, num_windows, 3, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = (
                q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            attn = attn.softmax(dim=-1)
            x = (attn @ v)

        x = x.transpose(1, 3).reshape(B, -1, self.dim)
        x = self.proj(x)
        if restore_bchw:
            if x.shape[1] != H * W:
                raise ValueError(
                    f"Cannot restore BCHW output because token count changed from {H * W} to {x.shape[1]}."
                )
            return x.transpose(1, 2).reshape(B, self.dim, H, W).contiguous()
        return x


__all__ = ['MaskUnitAttention']
