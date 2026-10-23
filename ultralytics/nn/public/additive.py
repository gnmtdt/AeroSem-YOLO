"""CAS-ViT Additive blocks (shared base)."""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from ultralytics.nn.public.common_glu import ConvolutionalGLU


class Mlp_CASVIT(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatialOperation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class ChannelOperation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class LocalIntegration(nn.Module):
    def __init__(self, dim: int, ratio: float = 1.0, act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        mid_dim = round(ratio * dim)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            norm_layer(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AdditiveTokenMixer(nn.Module):
    def __init__(self, dim: int = 512, attn_bias: bool = False, proj_drop: float = 0.0):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.oper_k = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out


class AdditiveBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        attn_bias: bool = False,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.local_perception = LocalIntegration(dim, ratio=1, act_layer=act_layer, norm_layer=norm_layer)
        self.norm1 = norm_layer(dim)
        self.attn = AdditiveTokenMixer(dim, attn_bias=attn_bias, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_CASVIT(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.local_perception(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AdditiveBlock_CGLU(AdditiveBlock):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        attn_bias: bool = False,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__(dim, mlp_ratio, attn_bias, drop, drop_path, act_layer, norm_layer)
        self.mlp = ConvolutionalGLU(dim)

