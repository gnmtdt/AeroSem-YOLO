"""
EfficientGlobalSA - 高效全局自注意力

论文: Efficient Global Self-Attention for Mobile Visual Transformers
期刊/会议: ACM MM 2025
论文链接: https://dl.acm.org/doi/epdf/10.1145/3746027.3755657
"""

import torch
import torch.nn as nn

__all__ = ['EfficientGlobalSA']


class EfficientGlobalSA(nn.Module):

    def __init__(self, dim, heads=8, reduction_ratio=4):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.rd = reduction_ratio

        self.to_q = nn.Conv2d(dim, dim, 1, bias=True, groups=dim)
        self.to_k = nn.Conv2d(dim, dim, reduction_ratio, stride=reduction_ratio, bias=True, groups=dim)
        self.to_v = nn.Conv2d(dim, dim, reduction_ratio, stride=reduction_ratio, bias=True, groups=dim)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W and (W % self.rd == 0)
        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        _q = q.reshape(B, self.heads, -1, H * W).transpose(-2, -1)
        _k = k.reshape(B, self.heads, -1, (H // self.rd) ** 2)
        _v = v.reshape(B, self.heads, -1, (H // self.rd) ** 2).transpose(-2, -1)
        attn = (_q @ _k) * self.scale

        attn = self.softmax(attn)
        res = (attn @ _v)
        res = res.transpose(-2, -1).reshape(B, -1, H, W)
        return self.to_out(res)
