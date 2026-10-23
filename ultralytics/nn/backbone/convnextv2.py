# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# NOTE(ultralyticsmm):
# - 本文件从上游 RTDETR-main 迁移，尽可能保持原实现。
# - 为满足本项目“多模态入口先 Conv 接收 -> 单模态多输出主干”的规范，
#   在工厂函数 convnextv2_* 增加了输入投影包装（in_chans -> 3ch）。

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from ..modules.conv import Conv

__all__ = [
    "convnextv2_atto",
    "convnextv2_femto",
    "convnextv2_pico",
    "convnextv2_nano",
    "convnextv2_tiny",
    "convnextv2_base",
    "convnextv2_large",
    "convnextv2_huge",
]


class LayerNorm(nn.Module):
    """LayerNorm that supports channels_last or channels_first."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        raise NotImplementedError


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXtV2 Block."""

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = inp + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ConvNeXtV2 backbone that returns 4 stage features (P2/4..P5/32)."""

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        head_init_scale=1.0,
    ):
        super().__init__()
        self.depths = list(depths)
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]))
            cur += self.depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        # 迁移保持一致：channel 通过一次 dummy forward 计算（固定 3ch，因为上层会做输入投影）
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            res.append(x)
        return res


def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f"loading weights... {idx}/{len(model_dict)} items")
    return model_dict


class _InputProjBackbone(nn.Module):
    """多模态输入投影包装：in_chans -> 3ch，然后进入单模态主干。"""

    def __init__(self, in_chans: int, backbone: nn.Module, proj_out_chans: int = 3):
        super().__init__()
        self.in_chans = int(in_chans)
        self.proj_out_chans = int(proj_out_chans)
        self.input_proj = Conv(self.in_chans, self.proj_out_chans, k=1, s=1, act=False)
        self.backbone_impl = backbone
        self.channel = list(getattr(backbone, "channel"))
        self.backbone = True

    def forward(self, x):
        x = self.input_proj(x)
        return self.backbone_impl(x)


def _wrap_convnextv2(in_chans: int, model: nn.Module) -> nn.Module:
    return _InputProjBackbone(in_chans=in_chans, backbone=model, proj_out_chans=3)


def convnextv2_atto(in_chans: int = 3, weights: str = "", **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_convnextv2(in_chans, model)


def convnextv2_femto(in_chans: int = 3, weights: str = "", **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_convnextv2(in_chans, model)


def convnextv2_pico(in_chans: int = 3, weights: str = "", **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_convnextv2(in_chans, model)


def convnextv2_nano(in_chans: int = 3, weights: str = "", **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_convnextv2(in_chans, model)


def convnextv2_tiny(in_chans: int = 3, weights: str = "", **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_convnextv2(in_chans, model)


def convnextv2_base(in_chans: int = 3, weights: str = "", **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_convnextv2(in_chans, model)


def convnextv2_large(in_chans: int = 3, weights: str = "", **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_convnextv2(in_chans, model)


def convnextv2_huge(in_chans: int = 3, weights: str = "", **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_convnextv2(in_chans, model)

