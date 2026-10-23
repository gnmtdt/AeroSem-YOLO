"""
DilatedMWSA - 膨胀多窗口自注意力机制

论文: Dilated Multi-Window Self-Attention
期刊/会议: arXiv (2024)
论文链接: https://arxiv.org/abs/2404.07846
"""

import torch
import torch.nn as nn

try:
    from einops import rearrange
except ImportError:
    rearrange = None

__all__ = ['DilatedMWSA']


class DilatedMWSA(nn.Module):
    """Dilated Multi-Window Self-Attention

    Args:
        dim: input feature dimension
        num_heads: number of attention heads
        bias: whether to use bias in convolutions
    """

    def __init__(self, dim, num_heads=8, bias=False):
        super(DilatedMWSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1,
                                    dilation=2, padding=2, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
