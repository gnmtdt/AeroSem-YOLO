"""
CBSA - Context Bridge Sparse Attention

论文: Context Bridge Sparse Attention
会议: NeurIPS 2025
论文链接: https://arxiv.org/abs/2509.16875

上下文桥接稀疏注意力机制，通过可学习的稀疏表示(rep)桥接全局上下文。
利用自适应平均池化压缩token数量，通过两阶段注意力(rep->tokens, rep->rep)
实现高效的稀疏注意力计算。

依赖: einops
"""

try:
    from einops import rearrange
except ImportError:
    rearrange = None

import torch
import torch.nn as nn


class CBSA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.step_x = nn.Parameter(torch.randn(num_heads, 1, 1))
        self.step_rep = nn.Parameter(torch.randn(num_heads, 1, 1))

        self.to_out = nn.Linear(dim, dim)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(8, 8))

    def attention(self, query, key, value):
        dots = (query @ key.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = attn @ value
        return out, attn

    def forward(self, x, return_attn=False):
        if rearrange is None:
            raise ImportError(
                "CBSA 模块需要 einops 库。请安装: pip install einops"
            )

        restore_bchw = x.ndim == 4
        if x.ndim == 4:
            b, c, h, width = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
        elif x.ndim == 3:
            b, n, c = x.shape
            h = width = int(n ** 0.5)
            if h * width != n:
                raise ValueError(f"CBSA token input requires square token count, got N={n}.")
        else:
            raise ValueError(f"CBSA expects input as [B, N, C] or [B, C, H, W], got {tuple(x.shape)}")

        b, n, c = x.shape

        w = self.proj(x)
        rep = self.pool(w[:, :, :].reshape(b, h, width, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)

        w = w.reshape(b, n, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        rep = rep.reshape(b, 64, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        rep_delta, attn = self.attention(rep, w, w)

        if return_attn:
            return attn.transpose(-1, -2) @ attn

        rep = rep + self.step_rep * rep_delta

        x_delta, _ = self.attention(rep, rep, rep)
        x_delta = attn.transpose(-1, -2) @ x_delta
        x_delta = self.step_x * x_delta

        x_delta = rearrange(x_delta, 'b h n k -> b n (h k)')
        out = self.to_out(x_delta)
        if restore_bchw:
            return out.transpose(1, 2).reshape(b, c, h, width).contiguous()
        return out


__all__ = ['CBSA']
