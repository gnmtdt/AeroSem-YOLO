"""
MALA - 多轴线性注意力 + RoPE 位置编码 (Multi-Axis Linear Attention with RoPE)
RoPE - 2D 旋转位置编码

论文: Multi-Axis Linear Attention for Vision Transformers
期刊/会议: arXiv 2507.00698
论文链接: https://arxiv.org/pdf/2507.00698
"""

try:
    from einops import rearrange
except ImportError:
    rearrange = None

import torch
import torch.nn as nn
from typing import Tuple

__all__ = ['RoPE', 'MALA']


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class RoPE(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, slen: Tuple[int]):
        """
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        """
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1)
        sin = torch.cat([sin_h, sin_w], -1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :])
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1)
        cos = torch.cat([cos_h, cos_w], -1)

        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))

        return retention_rel_pos


class MALA(nn.Module):

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = self.head_dim ** -0.5
        self.elu = nn.ELU()

        self.repo = RoPE(dim, num_heads)

    def forward(self, x: torch.Tensor):
        """
        x: (b c h w)
        sin: ((h w) d1)
        cos: ((h w) d1)
        """
        B, C, H, W = x.shape
        sin, cos = self.repo((H, W))
        qkvo = self.qkvo(x)  # (b 4*c h w)
        qkv = qkvo[:, :3 * self.dim, :, :]
        o = qkvo[:, 3 * self.dim:, :, :]
        lepe = self.lepe(qkv[:, 2 * self.dim:, :, :])  # (b c h w)

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads)

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        kv = (k.transpose(-2, -1) * (self.scale / (H * W)) ** 0.5) @ (v * (self.scale / (H * W)) ** 0.5)

        res = q @ kv * (1 + 1 / (z + 1e-6)) - z * v.mean(dim=2, keepdim=True)

        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        res = res + lepe
        return self.proj(res * o)
