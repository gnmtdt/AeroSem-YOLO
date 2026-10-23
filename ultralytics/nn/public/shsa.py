import torch
import torch.nn as nn
from timm.models.layers import DropPath
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.public.common_glu import ConvolutionalGLU

__all__ = [
    "SHSA_GroupNorm",
    "SHSABlock_FFN",
    "SHSA",
    "SHSABlock",
    "SHSABlock_CGLU",
]


class SHSA_GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class SHSABlock_FFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = nn.SiLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


def Conv2d_BN(in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bn_weight_init=1.0):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=False)
    bn = nn.BatchNorm2d(out_ch)
    nn.init.constant_(bn.weight, bn_weight_init)
    nn.init.constant_(bn.bias, 0)
    return nn.Sequential(conv, bn)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class SHSA(nn.Module):
    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.pre_norm = SHSA_GroupNorm(pdim)
        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = nn.Sequential(nn.SiLU(), Conv2d_BN(dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim=1))
        return x


class SHSABlock(nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
        self.mixer = Residual(SHSA(dim, qk_dim, pdim))
        self.ffn = Residual(SHSABlock_FFN(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))


class SHSABlock_CGLU(nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
        self.mixer = Residual(SHSA(dim, qk_dim, pdim))
        self.ffn = ConvolutionalGLU(dim, int(dim * 2))

    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))
