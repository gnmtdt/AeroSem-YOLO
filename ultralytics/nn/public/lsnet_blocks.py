"""LSNet 系列公共模块（迁移自 RTDETR-main `nn/backbone/lsnet.py`）。

注意：
- 上游版本依赖 triton 实现 SKA 加速，并包含 try/except 兼容分支；
- 本仓库遵循“不得新增优雅降级机制”的约束，统一采用纯 PyTorch 实现的 SKA（不做自动切换）。
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import SqueezeExcite

__all__ = ["LSConv", "LSBlock"]


class SKA(nn.Module):
    """Spatial Kernel Aggregation（纯 PyTorch 版本）。"""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        # w: [B, wc, k*k, H, W]，其中对 channel 采用 c % wc 的共享映射（与上游 triton 实现一致）
        if x.ndim != 4:
            raise ValueError(f"SKA expects NCHW input, got {tuple(x.shape)}")
        if w.ndim != 5:
            raise ValueError(f"SKA expects weight shaped [B, wc, k2, H, W], got {tuple(w.shape)}")

        b, c, h, w_ = x.shape
        _, wc, k2, wh, ww = w.shape
        if wh != h or ww != w_:
            raise ValueError(f"SKA weight spatial {(wh, ww)} must match input {(h, w_)}")
        ks = int(math.isqrt(k2))
        if ks * ks != k2:
            raise ValueError(f"SKA expects k2 to be perfect square, got k2={k2}")
        pad = (ks - 1) // 2

        patches = F.unfold(x, kernel_size=ks, padding=pad)  # [B, C*k2, H*W]
        patches = patches.view(b, c, k2, h * w_)
        w_flat = w.view(b, wc, k2, h * w_)

        idx = torch.arange(c, device=x.device) % wc
        w_expanded = w_flat[:, idx]  # [B, C, k2, H*W]
        out = (patches * w_expanded).sum(dim=2).view(b, c, h, w_)
        return out


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module("c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module("bn", nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1) * c.groups,
            w.size(0),
            w.shape[2:],
            stride=c.stride,
            padding=c.padding,
            dilation=c.dilation,
            groups=c.groups,
            device=c.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    def __init__(self, m: nn.Module, drop: float = 0.0):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            gate = torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
            return x + self.m(x) * gate
        return x + self.m(x)


class FFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


class Attention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.dh = int(attn_ratio * key_dim) * num_heads
        h = self.dh + nh_kd * 2
        self.qkv = Conv2d_BN(dim, h, ks=1)
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(self.dh, dim, bn_weight_init=0))
        self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)

        points = list(itertools.product(range(resolution), range(resolution)))
        n = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(n, n))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        b, _, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, -1, h, w).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
        q = self.dw(q)
        q = q.view(b, self.num_heads, -1, n)
        k = k.view(b, self.num_heads, -1, n)
        v = v.view(b, self.num_heads, -1, n)
        bias = self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        attn = (q.transpose(-2, -1) @ k) * self.scale + bias
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(b, -1, h, w)
        return self.proj(x)


class RepVGGDW(nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x


class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks**2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks**2 * dim // groups)
        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        return w.view(b, self.dim // self.groups, self.sks**2, h, width)


class LSConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


class LSBlock(nn.Module):
    """上游 `lsnet.Block` 的兼容实现（保留相同签名）。"""

    def __init__(self, ed, kd=16, nh=8, ar=4, resolution=14, stage=-1, depth=-1):
        super().__init__()
        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = nn.Identity()
            if stage == 3:
                self.mixer = Residual(Attention(ed, kd, nh, ar, resolution=resolution))
            else:
                self.mixer = LSConv(ed)
        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))

