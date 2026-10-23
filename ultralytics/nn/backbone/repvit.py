# NOTE(ultralyticsmm):
# - 本文件从上游 RTDETR-main 迁移，尽可能保持原实现。
# - 为满足本项目“多模态入口先 Conv 接收 -> 单模态多输出主干”的规范，
#   在工厂函数 repvit_* 增加了输入投影包装（in_chans -> 3ch）。

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import SqueezeExcite

from ..modules.conv import Conv

__all__ = ["repvit_m0_9", "repvit_m1_0", "repvit_m1_1", "repvit_m1_5", "repvit_m2_3"]


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, "fuse_self"):
            fused = child.fuse_self()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2d_BN(torch.nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        resolution=-10000,
    ):
        super().__init__()
        self.add_module("c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module("bn", torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse_self(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.0):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div_(1 - self.drop)
        return x + self.m(x)


class RepViTBlock(nn.Module):
    def __init__(self, in_channels, exp_size, out_channels, kernel_size, stride, use_se, use_hs):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        act = nn.Hardswish if use_hs else nn.ReLU
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if in_channels != exp_size:
            layers.append(Conv2d_BN(in_channels, exp_size, ks=1))
            layers.append(act())
        layers.extend(
            [
                Conv2d_BN(exp_size, exp_size, kernel_size, stride, (kernel_size - 1) // 2, groups=exp_size),
                act(),
            ]
        )
        if use_se:
            layers.append(SqueezeExcite(exp_size, 0.25))
        layers.append(Conv2d_BN(exp_size, out_channels, ks=1, bn_weight_init=0))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class RepViT(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

        input_channel = self.cfgs[0][2]
        patch_embed = nn.Sequential(
            Conv2d_BN(3, input_channel // 2, 3, 2, 1),
            nn.GELU(),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1),
        )
        layers = [patch_embed]
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(RepViTBlock(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)

        # 迁移保持一致：channel 通过一次 dummy forward 计算（固定 3ch，因为上层会做输入投影）
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        input_size = x.size(2)
        scale = [4, 8, 16, 32]
        features = [None, None, None, None]
        for f in self.features:
            x = f(x)
            if input_size // x.size(2) in scale:
                features[scale.index(input_size // x.size(2))] = x
        return features

    def switch_to_deploy(self):
        replace_batchnorm(self)


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


def _wrap_repvit(in_chans: int, model: nn.Module) -> nn.Module:
    return _InputProjBackbone(in_chans=in_chans, backbone=model, proj_out_chans=3)


def repvit_m0_9(in_chans: int = 3, weights: str = ""):
    cfgs = [
        [3, 2, 48, 1, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 96, 0, 0, 2],
        [3, 2, 96, 1, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 192, 0, 1, 2],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 384, 0, 1, 2],
        [3, 2, 384, 1, 1, 1],
        [3, 2, 384, 1, 1, 1],
        [3, 2, 384, 1, 1, 1],
    ]
    model = RepViT(cfgs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_repvit(in_chans, model)


def repvit_m1_0(in_chans: int = 3, weights: str = ""):
    cfgs = [
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 1, 1, 1],
    ]
    model = RepViT(cfgs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_repvit(in_chans, model)


def repvit_m1_1(in_chans: int = 3, weights: str = ""):
    cfgs = [
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 1, 1, 1],
    ]
    model = RepViT(cfgs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_repvit(in_chans, model)


def repvit_m1_5(in_chans: int = 3, weights: str = ""):
    cfgs = [
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 1, 1, 1],
    ]
    model = RepViT(cfgs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_repvit(in_chans, model)


def repvit_m2_3(in_chans: int = 3, weights: str = ""):
    cfgs = [
        [3, 2, 80, 1, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 160, 0, 0, 2],
        [3, 2, 160, 1, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 320, 0, 1, 2],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 640, 0, 1, 2],
        [3, 2, 640, 1, 1, 1],
        [3, 2, 640, 1, 1, 1],
        [3, 2, 640, 1, 1, 1],
        [3, 2, 640, 1, 1, 1],
        [3, 2, 640, 1, 1, 1],
    ]
    model = RepViT(cfgs)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)["model"]))
    return _wrap_repvit(in_chans, model)

