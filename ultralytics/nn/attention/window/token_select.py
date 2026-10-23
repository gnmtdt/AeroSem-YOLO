"""
Token_Selective_Attention - top-k token选择注意力机制

论文: Token Selective Attention
期刊/会议: Neural Networks (2025)
论文链接: https://arxiv.org/pdf/2410.03171v3
"""

import torch
import torch.nn as nn

try:
    from einops import rearrange
except ImportError:
    rearrange = None

__all__ = ['Token_Selective_Attention']


class Token_Selective_Attention(nn.Module):
    """Token Selective Attention with top-k token selection

    Args:
        dim: input feature dimension
        num_heads: number of attention heads
        bias: whether to use bias in convolutions
        k: ratio of tokens to select (0 < k <= 1.0)
        group_num: number of groups for channel splitting
    """

    def __init__(self, dim, num_heads=8, bias=False, k=0.8, group_num=4):
        super(Token_Selective_Attention, self).__init__()
        self.num_heads = num_heads
        self.k = k
        self.group_num = group_num
        self.dim_group = dim // group_num
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv3d(self.group_num, self.group_num * 3, kernel_size=(1, 1, 1), bias=False)
        self.qkv_conv = nn.Conv3d(self.group_num * 3, self.group_num * 3,
                                  kernel_size=(1, 3, 3), padding=(0, 1, 1),
                                  groups=self.group_num * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, self.group_num, c // self.group_num, h, w)
        b, t, c, h, w = x.shape

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = rearrange(q, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        k = rearrange(k, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        v = rearrange(v, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, _, N = q.shape

        mask = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)

        attn = (q.transpose(-2, -1) @ k) * self.temperature

        index = torch.topk(attn, k=int(N * self.k), dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
        attn = attn.softmax(dim=-1)

        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)

        out = rearrange(out, 'b head c (h w t) -> b t (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)

        return out
