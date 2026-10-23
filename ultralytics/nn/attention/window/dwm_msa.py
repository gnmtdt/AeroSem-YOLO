"""
DWM_MSA - 双窗口多头自注意力机制 (Dual Window Multi-Head Self-Attention)

论文: Dual Window Multi-Head Self-Attention
期刊/会议: IEEE TIP (2025)
论文链接: https://ieeexplore.ieee.org/document/10955125
"""

import torch
import torch.nn as nn

try:
    from einops import rearrange
except ImportError:
    rearrange = None

from torch import einsum

__all__ = ['DWM_MSA']


class DWM_MSA(nn.Module):
    """Dual Window Multi-Head Self-Attention

    Args:
        dim: input feature dimension
        window_size1: first window size (tuple)
        window_size2: second window size (tuple)
        dim_head: dimension per head
        heads: number of attention heads
    """

    def __init__(
            self,
            dim,
            window_size1=(4, 4),
            window_size2=(10, 10),
            dim_head=32,
            heads=2,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size1 = window_size1
        self.window_size2 = window_size2

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        x: [B, C, H, W] (NCHW format)
        return out: [B, C, H, W]
        """
        x = x.permute(0, 2, 3, 1)
        b, h, w, _ = x.shape
        w_size1 = self.window_size1
        w_size2 = self.window_size2
        assert h % w_size1[0] == 0 and w % w_size1[1] == 0, \
            'fmap dimensions must be divisible by the window size 1'
        assert h % w_size2[0] == 0 and w % w_size2[1] == 0, \
            'fmap dimensions must be divisible by the window size 2'

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        _, _, _, c = q.shape
        q1, q2, q3, q4 = q[:, :, :, :c // 4], q[:, :, :, c // 4:c // 2], \
                         q[:, :, :, c // 2:c // 4 * 3], q[:, :, :, c // 4 * 3:]
        k1, k2, k3, k4 = k[:, :, :, :c // 4], k[:, :, :, c // 4:c // 2], \
                         k[:, :, :, c // 2:c // 4 * 3], k[:, :, :, c // 4 * 3:]
        v1, v2, v3, v4 = v[:, :, :, :c // 4], v[:, :, :, c // 4:c // 2], \
                         v[:, :, :, c // 2:c // 4 * 3], v[:, :, :, c // 4 * 3:]

        # local branch of window size 1
        q1, k1, v1 = map(
            lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                b0=w_size1[0], b1=w_size1[1]),
            (q1, k1, v1))
        q1, k1, v1 = map(
            lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q1, k1, v1))
        q1 *= self.scale
        sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
        attn1 = sim1.softmax(dim=-1)
        out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
        out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

        # non-local branch of window size 1
        q2, k2, v2 = map(
            lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                b0=w_size1[0], b1=w_size1[1]),
            (q2, k2, v2))
        q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
        q2, k2, v2 = map(
            lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q2, k2, v2))
        q2 *= self.scale
        sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
        attn2 = sim2.softmax(dim=-1)
        out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
        out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
        out2 = out2.permute(0, 2, 1, 3)

        out_1 = torch.cat([out1, out2], dim=-1).contiguous()
        out_1 = rearrange(out_1, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c',
                          h=h // w_size1[0], w=w // w_size1[1], b0=w_size1[0])

        # local branch of window size 2
        q3, k3, v3 = map(
            lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                b0=w_size2[0], b1=w_size2[1]),
            (q3, k3, v3))
        q3, k3, v3 = map(
            lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q3, k3, v3))
        q3 *= self.scale
        sim3 = einsum('b n h i d, b n h j d -> b n h i j', q3, k3)
        attn3 = sim3.softmax(dim=-1)
        out3 = einsum('b n h i j, b n h j d -> b n h i d', attn3, v3)
        out3 = rearrange(out3, 'b n h mm d -> b n mm (h d)')

        # non-local of window size 2
        q4, k4, v4 = map(
            lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                b0=w_size2[0], b1=w_size2[1]),
            (q4, k4, v4))
        q4, k4, v4 = map(lambda t: t.permute(0, 2, 1, 3), (q4.clone(), k4.clone(), v4.clone()))
        q4, k4, v4 = map(
            lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q4, k4, v4))
        q4 *= self.scale
        sim4 = einsum('b n h i d, b n h j d -> b n h i j', q4, k4)
        attn4 = sim4.softmax(dim=-1)
        out4 = einsum('b n h i j, b n h j d -> b n h i d', attn4, v4)
        out4 = rearrange(out4, 'b n h mm d -> b n mm (h d)')
        out4 = out4.permute(0, 2, 1, 3)

        out_2 = torch.cat([out3, out4], dim=-1).contiguous()
        out_2 = rearrange(out_2, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c',
                          h=h // w_size2[0], w=w // w_size2[1], b0=w_size2[0])

        out = torch.cat([out_1, out_2], dim=-1).contiguous()
        out = self.to_out(out)

        return out.permute(0, 3, 1, 2)
