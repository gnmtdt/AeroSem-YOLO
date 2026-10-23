import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.public.common_glu import ConvolutionalGLU

__all__ = ["MutilScal", "Mutilscal_MHSA", "MSMHSA_CGLU"]


class MutilScal(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super().__init__()
        self.conv0_1 = Conv(dim, dim // fc_ratio)
        self.conv0_2 = Conv(dim // fc_ratio, dim // fc_ratio, 3, d=dilation[-3], g=dim // fc_ratio)
        self.conv0_3 = Conv(dim // fc_ratio, dim, 1)

        self.conv1_2 = Conv(dim // fc_ratio, dim // fc_ratio, 3, d=dilation[-2], g=dim // fc_ratio)
        self.conv1_3 = Conv(dim // fc_ratio, dim, 1)

        self.conv2_2 = Conv(dim // fc_ratio, dim // fc_ratio, 3, d=dilation[-1], g=dim // fc_ratio)
        self.conv2_3 = Conv(dim // fc_ratio, dim, 1)

        self.conv3 = Conv(dim, dim, 1)
        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x):
        u = x.clone()
        attn0_1 = self.conv0_1(x)
        attn0_2 = self.conv0_2(attn0_1)
        attn0_3 = self.conv0_3(attn0_2)

        attn1_2 = self.conv1_2(attn0_1)
        attn1_3 = self.conv1_3(attn1_2)

        attn2_2 = self.conv2_2(attn0_1)
        attn2_3 = self.conv2_3(attn2_2)

        attn = attn0_3 + attn1_3 + attn2_3
        attn = self.conv3(attn)
        attn = attn * u
        pool = self.Avg(attn)
        return pool


class Mutilscal_MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, atten_drop=0.0, proj_drop=0.0, dilation=[3, 5, 7], fc_ratio=4, pool_ratio=16):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.MSC = MutilScal(dim=dim, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim // fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.kv = Conv(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        B, C, H, W = x.shape
        kv = self.MSC(x)
        kv = self.kv(kv)
        B1, C1, H1, W1 = kv.shape

        q = rearrange(x, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads, d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) hh ww -> kv b h (hh ww) d', h=self.num_heads, d=C // self.num_heads, hh=H1, ww=W1, kv=2)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, 'b h (hh ww) d -> b (h d) hh ww', h=self.num_heads, d=C // self.num_heads, hh=H, ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
        return attn + c_attn


class MSMHSA_CGLU(nn.Module):
    def __init__(self, inc, drop_path=0.1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = ConvolutionalGLU(inc)
        self.msmhsa = nn.Sequential(Mutilscal_MHSA(inc), nn.BatchNorm2d(inc))

    def forward(self, x):
        x = x + self.drop_path(self.msmhsa(x))
        x = x + self.drop_path(self.mlp(x))
        return x
