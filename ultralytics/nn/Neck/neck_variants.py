"""
Neck Modules (AFPN / HS-FPN / CFPT / Fusion / RepPAN / GFPN / Multi-Branch FPN)
迁移自 upstream `ultralytics/nn/extra_modules`，仅包含纯 PyTorch 实现。
"""

from __future__ import annotations

import math
from typing import List, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_dct as DCT
except Exception:  # 可选依赖缺失时自动降级
    DCT = None
from timm.layers import trunc_normal_, to_2tuple

from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import C2f, C3k2, RepConv

__all__ = [
    # AFPN / ASFF
    "AFPN_P345",
    "AFPN_P345_Custom",
    "AFPN_P2345",
    "AFPN_P2345_Custom",
    # HS-FPN
    "HFP",
    "SDP",
    "SDP_Improved",
    "ChannelAttention_HSFPN",
    "ELA_HSFPN",
    "CA_HSFPN",
    "CAA_HSFPN",
    # CFPT
    "CrossLayerSpatialAttention",
    "CrossLayerChannelAttention",
    # 频域融合
    "FreqFusion",
    "LocalSimGuidedSampler",
    # BIFPN / 加权融合
    "Fusion",
    "SDI",
    # GFPN / RepPAN
    "CSPStage",
    "BiFusion",
    "OREPANCSPELAN4",
    # Re-Calibration FPN
    "SBA",
    # Efficient Multi-Branch & Scale FPN
    "EUCB",
    "MSDC",
    "MSCB",
    "CSP_MSCB",
]

# ---------------- 基础块 ----------------


class GSConv(nn.Module):
    """GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = x2.size()
        y = x2.reshape(b * n // 2, 2, h * w).permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)


class SDI(nn.Module):
    """Semantics and Detail Infusion"""

    def __init__(self, channels):
        super().__init__()
        self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[2:]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, target_size)
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
            ans = ans * self.convs[i](x)
        return ans


class Fusion(nn.Module):
    """可选 weight/adaptive/concat/bifpn/SDI"""

    def __init__(self, inc_list, fusion="bifpn") -> None:
        super().__init__()
        assert fusion in ["weight", "adaptive", "concat", "bifpn", "SDI"]
        self.fusion = fusion

        if self.fusion == "bifpn":
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == "SDI":
            self.SDI = SDI(inc_list)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])
            if self.fusion == "adaptive":
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)

    def forward(self, x):
        if self.fusion in ["weight", "adaptive"]:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == "weight":
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == "adaptive":
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == "concat":
            return torch.cat(x, dim=1)
        elif self.fusion == "bifpn":
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == "SDI":
            return self.SDI(x)


# ---------------- AFPN / ASFF ----------------


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, filter_in, filter_out):
        super().__init__()
        self.conv1 = Conv(filter_in, filter_out, 3)
        self.conv2 = Conv(filter_out, filter_out, 3, act=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return self.conv1.act(out)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Sequential(Conv(in_channels, out_channels, 1), nn.Upsample(scale_factor=scale_factor, mode="bilinear"))

    def forward(self, x):
        return self.upsample(x)


class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = Conv(in_channels, out_channels, 2, 2, 0)

    def forward(self, x):
        return self.downsample(x)


class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = Conv(in_channels, out_channels, 4, 4, 0)

    def forward(self, x):
        return self.downsample(x)


class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = Conv(in_channels, out_channels, 8, 8, 0)

    def forward(self, x):
        return self.downsample(x)


class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super().__init__()
        compress_c = 8
        self.weight_level_1 = Conv(inter_dim, compress_c, 1)
        self.weight_level_2 = Conv(inter_dim, compress_c, 1)
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(inter_dim, inter_dim, 3)

    def forward(self, input1, input2):
        w1 = self.weight_level_1(input1)
        w2 = self.weight_level_2(input2)
        levels_weight = self.weight_levels(torch.cat((w1, w2), 1))
        levels_weight = F.softmax(levels_weight, dim=1)
        fused = input1 * levels_weight[:, 0:1] + input2 * levels_weight[:, 1:2]
        return self.conv(fused)


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super().__init__()
        compress_c = 8
        self.weight_level_1 = Conv(inter_dim, compress_c, 1)
        self.weight_level_2 = Conv(inter_dim, compress_c, 1)
        self.weight_level_3 = Conv(inter_dim, compress_c, 1)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(inter_dim, inter_dim, 3)

    def forward(self, input1, input2, input3):
        w1 = self.weight_level_1(input1)
        w2 = self.weight_level_2(input2)
        w3 = self.weight_level_3(input3)
        levels_weight = self.weight_levels(torch.cat((w1, w2, w3), 1))
        levels_weight = F.softmax(levels_weight, dim=1)
        fused = input1 * levels_weight[:, 0:1] + input2 * levels_weight[:, 1:2] + input3 * levels_weight[:, 2:]
        return self.conv(fused)


class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512):
        super().__init__()
        compress_c = 8
        self.weight_level_0 = Conv(inter_dim, compress_c, 1)
        self.weight_level_1 = Conv(inter_dim, compress_c, 1)
        self.weight_level_2 = Conv(inter_dim, compress_c, 1)
        self.weight_level_3 = Conv(inter_dim, compress_c, 1)
        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(inter_dim, inter_dim, 3)

    def forward(self, input0, input1, input2, input3):
        w0 = self.weight_level_0(input0)
        w1 = self.weight_level_1(input1)
        w2 = self.weight_level_2(input2)
        w3 = self.weight_level_3(input3)
        levels_weight = self.weight_levels(torch.cat((w0, w1, w2, w3), 1))
        levels_weight = F.softmax(levels_weight, dim=1)
        fused = input0 * levels_weight[:, 0:1] + input1 * levels_weight[:, 1:2] + input2 * levels_weight[:, 2:3] + input3 * levels_weight[:, 3:]
        return self.conv(fused)


class BlockBody_P345(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        channels = list(channels)

        self.blocks_scalezero1 = nn.Sequential(Conv(channels[0], channels[0], 1))
        self.blocks_scaleone1 = nn.Sequential(Conv(channels[1], channels[1], 1))
        self.blocks_scaletwo1 = nn.Sequential(Conv(channels[2], channels[2], 1))

        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        self.blocks_scalezero2 = nn.Sequential(*[BasicBlock(channels[0], channels[0]) for _ in range(4)])
        self.blocks_scaleone2 = nn.Sequential(*[BasicBlock(channels[1], channels[1]) for _ in range(4)])

        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])

        self.blocks_scalezero3 = nn.Sequential(*[BasicBlock(channels[0], channels[0]) for _ in range(4)])
        self.blocks_scaleone3 = nn.Sequential(*[BasicBlock(channels[1], channels[1]) for _ in range(4)])
        self.blocks_scaletwo3 = nn.Sequential(*[BasicBlock(channels[2], channels[2]) for _ in range(4)])

        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)

    def forward(self, x):
        x0, x1, x2 = x

        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)

        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)

        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)

        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)

        return x0, x1, x2


class BlockBody_P345_Custom(BlockBody_P345):
    def __init__(self, channels=(64, 128, 256, 512), block_type: object = "C2f"):
        super().__init__(channels)
        block = block_type
        if isinstance(block_type, str):
            if block_type not in globals():
                raise ValueError(f"AFPN Custom block_type 未注册：{block_type}")
            block = globals()[block_type]

        channels = list(channels)
        self.blocks_scalezero2 = block(channels[0], channels[0])
        self.blocks_scaleone2 = block(channels[1], channels[1])
        self.blocks_scalezero3 = block(channels[0], channels[0])
        self.blocks_scaleone3 = block(channels[1], channels[1])
        self.blocks_scaletwo3 = block(channels[2], channels[2])


class BlockBody_P2345(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        channels = list(channels)

        self.blocks_scalezero1 = nn.Sequential(Conv(channels[0], channels[0], 1))
        self.blocks_scaleone1 = nn.Sequential(Conv(channels[1], channels[1], 1))
        self.blocks_scaletwo1 = nn.Sequential(Conv(channels[2], channels[2], 1))
        self.blocks_scalethree1 = nn.Sequential(Conv(channels[3], channels[3], 1))

        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        self.blocks_scalezero2 = nn.Sequential(*[BasicBlock(channels[0], channels[0]) for _ in range(4)])
        self.blocks_scaleone2 = nn.Sequential(*[BasicBlock(channels[1], channels[1]) for _ in range(4)])

        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])

        self.blocks_scalezero3 = nn.Sequential(*[BasicBlock(channels[0], channels[0]) for _ in range(4)])
        self.blocks_scaleone3 = nn.Sequential(*[BasicBlock(channels[1], channels[1]) for _ in range(4)])
        self.blocks_scaletwo3 = nn.Sequential(*[BasicBlock(channels[2], channels[2]) for _ in range(4)])

        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scalezero3_8 = Downsample_x8(channels[0], channels[3])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaleone3_4 = Downsample_x4(channels[1], channels[3])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo3_2 = Downsample_x2(channels[2], channels[3])
        self.upsample_scalethree3_8 = Upsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree3_4 = Upsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree3_2 = Upsample(channels[3], channels[2], scale_factor=2)

        self.asff_scalezero3 = ASFF_4(inter_dim=channels[0])
        self.asff_scaleone3 = ASFF_4(inter_dim=channels[1])
        self.asff_scaletwo3 = ASFF_4(inter_dim=channels[2])
        self.asff_scalethree3 = ASFF_4(inter_dim=channels[3])

        self.blocks_scalezero4 = nn.Sequential(*[BasicBlock(channels[0], channels[0]) for _ in range(4)])
        self.blocks_scaleone4 = nn.Sequential(*[BasicBlock(channels[1], channels[1]) for _ in range(4)])
        self.blocks_scaletwo4 = nn.Sequential(*[BasicBlock(channels[2], channels[2]) for _ in range(4)])
        self.blocks_scalethree4 = nn.Sequential(*[BasicBlock(channels[3], channels[3]) for _ in range(4)])

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)
        x3 = self.blocks_scalethree1(x3)

        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)

        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)

        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)

        x0 = self.blocks_scalezero3(scalezero)
        x1 = self.blocks_scaleone3(scaleone)
        x2 = self.blocks_scaletwo3(scaletwo)

        scalezero = self.asff_scalezero3(
            x0,
            self.upsample_scaleone3_2(x1),
            self.upsample_scaletwo3_4(x2),
            self.upsample_scalethree3_8(x3),
        )
        scaleone = self.asff_scaleone3(
            self.downsample_scalezero3_2(x0),
            x1,
            self.upsample_scaletwo3_2(x2),
            self.upsample_scalethree3_4(x3),
        )
        scaletwo = self.asff_scaletwo3(
            self.downsample_scalezero3_4(x0),
            self.downsample_scaleone3_2(x1),
            x2,
            self.upsample_scalethree3_2(x3),
        )
        scalethree = self.asff_scalethree3(
            self.downsample_scalezero3_8(x0),
            self.downsample_scaleone3_4(x1),
            self.downsample_scaletwo3_2(x2),
            x3,
        )

        scalezero = self.blocks_scalezero4(scalezero)
        scaleone = self.blocks_scaleone4(scaleone)
        scaletwo = self.blocks_scaletwo4(scaletwo)
        scalethree = self.blocks_scalethree4(scalethree)

        return scalezero, scaleone, scaletwo, scalethree


class BlockBody_P2345_Custom(BlockBody_P2345):
    def __init__(self, channels=(64, 128, 256, 512), block_type: object = "C2f"):
        super().__init__(channels)
        block = block_type
        if isinstance(block_type, str):
            if block_type not in globals():
                raise ValueError(f"AFPN Custom block_type 未注册：{block_type}")
            block = globals()[block_type]

        channels = list(channels)
        self.blocks_scalezero2 = block(channels[0], channels[0])
        self.blocks_scaleone2 = block(channels[1], channels[1])

        self.blocks_scalezero3 = block(channels[0], channels[0])
        self.blocks_scaleone3 = block(channels[1], channels[1])
        self.blocks_scaletwo3 = block(channels[2], channels[2])

        self.blocks_scalezero4 = block(channels[0], channels[0])
        self.blocks_scaleone4 = block(channels[1], channels[1])
        self.blocks_scaletwo4 = block(channels[2], channels[2])
        self.blocks_scalethree4 = block(channels[3], channels[3])


class AFPN_P345(nn.Module):
    def __init__(self, ch=(256, 512, 1024), hidc: int = 256, factor: int = 4):
        super().__init__()
        ch = list(ch)

        self.conv0 = Conv(ch[0], ch[0] // factor, 1)
        self.conv1 = Conv(ch[1], ch[1] // factor, 1)
        self.conv2 = Conv(ch[2], ch[2] // factor, 1)

        self.body = nn.Sequential(BlockBody_P345([ch[0] // factor, ch[1] // factor, ch[2] // factor]))

        self.conv00 = Conv(ch[0] // factor, hidc, 1)
        self.conv11 = Conv(ch[1] // factor, hidc, 1)
        self.conv22 = Conv(ch[2] // factor, hidc, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        out0, out1, out2 = self.body([x0, x1, x2])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        return [out0, out1, out2]


class AFPN_P345_Custom(nn.Module):
    def __init__(self, ch=(256, 512, 1024), hidc: int = 256, block_type: object = "C2f", factor: int = 4):
        super().__init__()
        ch = list(ch)

        self.conv0 = Conv(ch[0], ch[0] // factor, 1)
        self.conv1 = Conv(ch[1], ch[1] // factor, 1)
        self.conv2 = Conv(ch[2], ch[2] // factor, 1)

        self.body = nn.Sequential(
            BlockBody_P345_Custom([ch[0] // factor, ch[1] // factor, ch[2] // factor], block_type=block_type)
        )

        self.conv00 = Conv(ch[0] // factor, hidc, 1)
        self.conv11 = Conv(ch[1] // factor, hidc, 1)
        self.conv22 = Conv(ch[2] // factor, hidc, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        out0, out1, out2 = self.body([x0, x1, x2])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        return [out0, out1, out2]


class AFPN_P2345(nn.Module):
    def __init__(self, ch=(256, 512, 1024, 2048), hidc: int = 256, factor: int = 4):
        super().__init__()
        ch = list(ch)

        self.conv0 = Conv(ch[0], ch[0] // factor, 1)
        self.conv1 = Conv(ch[1], ch[1] // factor, 1)
        self.conv2 = Conv(ch[2], ch[2] // factor, 1)
        self.conv3 = Conv(ch[3], ch[3] // factor, 1)

        self.body = nn.Sequential(
            BlockBody_P2345([ch[0] // factor, ch[1] // factor, ch[2] // factor, ch[3] // factor])
        )

        self.conv00 = Conv(ch[0] // factor, hidc, 1)
        self.conv11 = Conv(ch[1] // factor, hidc, 1)
        self.conv22 = Conv(ch[2] // factor, hidc, 1)
        self.conv33 = Conv(ch[3] // factor, hidc, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)
        return [out0, out1, out2, out3]


class AFPN_P2345_Custom(nn.Module):
    def __init__(self, ch=(256, 512, 1024, 2048), hidc: int = 256, block_type: object = "C2f", factor: int = 4):
        super().__init__()
        ch = list(ch)

        self.conv0 = Conv(ch[0], ch[0] // factor, 1)
        self.conv1 = Conv(ch[1], ch[1] // factor, 1)
        self.conv2 = Conv(ch[2], ch[2] // factor, 1)
        self.conv3 = Conv(ch[3], ch[3] // factor, 1)

        self.body = nn.Sequential(
            BlockBody_P2345_Custom(
                [ch[0] // factor, ch[1] // factor, ch[2] // factor, ch[3] // factor],
                block_type=block_type,
            )
        )

        self.conv00 = Conv(ch[0] // factor, hidc, 1)
        self.conv11 = Conv(ch[1] // factor, hidc, 1)
        self.conv22 = Conv(ch[2] // factor, hidc, 1)
        self.conv33 = Conv(ch[3] // factor, hidc, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2, x3 = x

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)
        return [out0, out1, out2, out3]


# ---------------- HS-FPN & 注意力增强 ----------------


class DctSpatialInteraction(nn.Module):
    def __init__(self, in_channels, ratio, isdct=True):
        super().__init__()
        self.ratio = ratio
        self.isdct = isdct
        if (not self.isdct) or DCT is None:
            self.spatial1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if (not self.isdct) or DCT is None:
            return x * torch.sigmoid(self.spatial1x1(x))
        idct = DCT.dct_2d(x, norm="ortho")
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)
        dct = idct * weight
        dct_ = DCT.idct_2d(dct, norm="ortho")
        return x * dct_

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


class DctChannelInteraction(nn.Module):
    def __init__(self, in_channels, patch, ratio, isdct=True):
        super().__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Conv2d(in_channels, in_channels, 1, groups=32)
        self.channel2x1 = nn.Conv2d(in_channels, in_channels, 1, groups=32)
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if (not self.isdct) or DCT is None:
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm="ortho")
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight
        dct_ = DCT.idct_2d(dct, norm="ortho")

        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


class HFP(nn.Module):
    def __init__(self, in_channels, ratio=(0.25, 0.25), patch=(8, 8), isdct=True):
        super().__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels),
        )

    def forward(self, x):
        spatial = self.spatial(x)
        channel = self.channel(x)
        return self.out(spatial + channel)


class SDP(nn.Module):
    def __init__(self, in_dim, dim=256, patch_size=None, inter_dim=None):
        super().__init__()
        self.conv1x1_0 = Conv(in_dim[0], dim) if in_dim[0] != dim else nn.Identity()
        self.conv1x1_1 = Conv(in_dim[1], dim) if in_dim[1] != dim else nn.Identity()
        self.inter_dim = inter_dim or dim
        self.conv_q = nn.Sequential(nn.Conv2d(dim, self.inter_dim, 1, bias=False), nn.GroupNorm(32, self.inter_dim))
        self.conv_k = nn.Sequential(nn.Conv2d(dim, self.inter_dim, 1, bias=False), nn.GroupNorm(32, self.inter_dim))
        self.softmax = nn.Softmax(dim=-1)
        self.patch_size = patch_size

    def forward(self, x):
        x_low, x_high = x
        x_low = self.conv1x1_0(x_low)
        x_high = self.conv1x1_1(x_high)
        b_, _, h_, w_ = x_low.size()
        q = einops.rearrange(
            self.conv_q(x_low), "b c (h p1) (w p2) -> (b h w) c (p1 p2)", p1=self.patch_size[0], p2=self.patch_size[1]
        ).transpose(1, 2)
        k = einops.rearrange(
            self.conv_k(x_high), "b c (h p1) (w p2) -> (b h w) c (p1 p2)", p1=self.patch_size[0], p2=self.patch_size[1]
        )
        attn = torch.matmul(q, k) / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1, 2)
        output = torch.matmul(attn, v)
        output = einops.rearrange(
            output.transpose(1, 2).contiguous(),
            "(b h w) c (p1 p2) -> b c (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=h_ // self.patch_size[0],
            w=w_ // self.patch_size[1],
        )
        return output + x_low


class SDP_Improved(nn.Module):
    def __init__(self, dim=256, inter_dim=None):
        super().__init__()
        self.inter_dim = inter_dim or dim
        self.conv_q = nn.Sequential(nn.Conv2d(dim, self.inter_dim, 3, padding=1, bias=False), nn.GroupNorm(32, self.inter_dim))
        self.conv_k = nn.Sequential(nn.Conv2d(dim, self.inter_dim, 3, padding=1, bias=False), nn.GroupNorm(32, self.inter_dim))
        self.conv = nn.Sequential(nn.Conv2d(self.inter_dim, dim, 3, padding=1, bias=False), nn.GroupNorm(32, dim))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_low, x_high, patch_size):
        b_, _, h_, w_ = x_low.size()
        q = einops.rearrange(
            self.conv_q(x_low), "b c (h p1) (w p2) -> (b h w) c (p1 p2)", p1=patch_size[0], p2=patch_size[1]
        ).transpose(1, 2)
        k = einops.rearrange(
            self.conv_k(x_high), "b c (h p1) (w p2) -> (b h w) c (p1 p2)", p1=patch_size[0], p2=patch_size[1]
        )
        attn = torch.matmul(q, k) / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1, 2)
        output = torch.matmul(attn, v)
        output = einops.rearrange(
            output.transpose(1, 2).contiguous(),
            "(b h w) c (p1 p2) -> b c (h p1) (w p2)",
            p1=patch_size[0],
            p2=patch_size[1],
            h=h_ // patch_size[0],
            w=w_ // patch_size[1],
        )
        return self.conv(output) + x_low


class ChannelAttention_HSFPN(nn.Module):
    def __init__(self, c1, k=3, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(c1 // reduction, 1)
        self.fc = nn.Sequential(Conv(c1, mid, 1), Conv(mid, c1, 1, act=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class ELA_HSFPN(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        nn.init.constant_(self.conv.weight, 1.0 / 9)

    def forward(self, x):
        edge = x - self.conv(x)
        return x + edge


class CA_HSFPN(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv = nn.Conv2d(c1, c1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv(y)
        x_h, x_w = torch.split(y, [x.shape[2], x.shape[3]], dim=2)
        return x * self.sigmoid(x_h) * self.sigmoid(x_w.permute(0, 1, 3, 2))


class CAA_HSFPN(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.channel = ChannelAttention_HSFPN(c1)
        self.spatial = ELA_HSFPN(c1)

    def forward(self, x):
        return self.spatial(self.channel(x))


# ---------------- Cross-Layer Feature Pyramid Transformer ----------------


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return einops.rearrange(x, "b h w c -> b c h w")


class CrossLayerPosEmbedding3D(nn.Module):
    def __init__(self, num_heads=4, window_size=(5, 3, 1), spatial=True):
        super().__init__()
        self.spatial = spatial
        self.num_heads = num_heads
        self.layer_num = len(window_size)
        if self.spatial:
            self.num_token = sum([i**2 for i in window_size])
            self.num_token_per_level = [i**2 for i in window_size]
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[0] - 1), num_heads))
            coords_h = [torch.arange(ws) - ws // 2 for ws in window_size]
            coords_w = [torch.arange(ws) - ws // 2 for ws in window_size]
            coords_h = [coords_h[i] * window_size[0] / window_size[i] for i in range(len(coords_h) - 1)] + [coords_h[-1]]
            coords_w = [coords_w[i] * window_size[0] / window_size[i] for i in range(len(coords_w) - 1)] + [coords_w[-1]]
            coords = [torch.stack(torch.meshgrid([coord_h, coord_w])) for coord_h, coord_w in zip(coords_h, coords_w)]
            coords_flatten = torch.cat([torch.flatten(coord, 1) for coord in coords], dim=-1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[0] - 1
            relative_coords[:, :, 0] *= 2 * window_size[0] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=0.02)
        else:
            self.num_token = sum([i for i in window_size])
            self.num_token_per_level = [i for i in window_size]
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[0] - 1), num_heads))
            coords_c = [torch.arange(ws) - ws // 2 for ws in window_size]
            coords_c = [coords_c[i] * window_size[0] / window_size[i] for i in range(len(coords_c) - 1)] + [coords_c[-1]]
            coords = torch.cat(coords_c, dim=0)
            coords_flatten = torch.stack([torch.flatten(coord, 0) for coord in coords], dim=-1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.absolute_position_bias = nn.Parameter(torch.zeros(len(window_size), num_heads, 1, 1, 1))
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self):
        pos_indicies = self.relative_position_index.view(-1)
        pos_indicies_floor = torch.floor(pos_indicies).long()
        pos_indicies_ceil = torch.ceil(pos_indicies).long()
        value_floor = self.relative_position_bias_table[pos_indicies_floor]
        value_ceil = self.relative_position_bias_table[pos_indicies_ceil]
        weights_ceil = pos_indicies - pos_indicies_floor.float()
        weights_floor = 1.0 - weights_ceil

        pos_embed = weights_floor.unsqueeze(-1) * value_floor + weights_ceil.unsqueeze(-1) * value_ceil
        pos_embed = pos_embed.reshape(1, 1, self.num_token, -1, self.num_heads).permute(0, 4, 1, 2, 3)
        pos_embed = pos_embed.split(self.num_token_per_level, 3)
        layer_embed = self.absolute_position_bias.split([1 for _ in range(self.layer_num)], 0)
        pos_embed = torch.cat([i + j for (i, j) in zip(pos_embed, layer_embed)], dim=-2)
        return pos_embed


class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3, act=True):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, to_2tuple(k), to_2tuple(1), to_2tuple(k // 2), groups=dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return x + self.activation(self.proj(x))


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def overlaped_window_partition(x, window_size, stride, pad):
    B, C, H, W = x.shape
    out = F.unfold(x, kernel_size=(window_size, window_size), stride=stride, padding=pad)
    return out.reshape(B, C, window_size * window_size, -1).permute(0, 3, 2, 1)


def overlaped_window_reverse(x, H, W, window_size, stride, padding):
    B, Wm, Wsm, C = x.shape
    Ws, S, P = window_size, stride, padding
    x = x.permute(0, 3, 2, 1).reshape(B, C * Wsm, Wm)
    return F.fold(x, output_size=(H, W), kernel_size=(Ws, Ws), padding=P, stride=S)


def overlaped_channel_partition(x, window_size, stride, pad):
    B, HW, C, _ = x.shape
    out = F.unfold(x, kernel_size=(window_size, 1), stride=(stride, 1), padding=(pad, 0))
    return out.reshape(B, HW, window_size, -1)


def overlaped_channel_reverse(x, window_size, stride, pad, outC):
    B, C, Ws, HW = x.shape
    x = x.permute(0, 3, 2, 1).reshape(B, HW * Ws, C)
    return F.fold(x, output_size=(outC, 1), kernel_size=(window_size, 1), padding=(pad, 0), stride=(stride, 1))


class CrossLayerSpatialAttention(nn.Module):
    def __init__(self, in_dim, layer_num=3, beta=1, num_heads=4, mlp_ratio=2, reduction=4):
        super().__init__()
        assert beta % 2 != 0, "beta must be odd"
        self.num_heads = num_heads
        self.reduction = reduction
        self.window_sizes = [(2**i + beta) if i != 0 else (2**i + beta - 1) for i in range(layer_num)][::-1]
        self.token_num_per_layer = [i**2 for i in self.window_sizes]
        self.token_num = sum(self.token_num_per_layer)

        self.stride_list = [2**i for i in range(layer_num)][::-1]
        self.padding_list = [[0, 0] for _ in self.window_sizes]
        self.shape_list = [[0, 0] for _ in range(layer_num)]

        self.hidden_dim = in_dim // reduction
        self.head_dim = self.hidden_dim // num_heads

        self.cpe = nn.ModuleList(nn.ModuleList([ConvPosEnc(dim=in_dim, k=3), ConvPosEnc(dim=in_dim, k=3)]) for _ in range(layer_num))
        self.norm1 = nn.ModuleList(LayerNormProxy(in_dim) for _ in range(layer_num))
        self.norm2 = nn.ModuleList(nn.LayerNorm(in_dim) for _ in range(layer_num))
        self.qkv = nn.ModuleList(nn.Conv2d(in_dim, self.hidden_dim * 3, kernel_size=1, stride=1, padding=0) for _ in range(layer_num))

        mlp_hidden_dim = int(in_dim * mlp_ratio)
        self.mlp = nn.ModuleList(Mlp(in_features=in_dim, hidden_features=mlp_hidden_dim) for _ in range(layer_num))

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.ModuleList(nn.Conv2d(self.hidden_dim, in_dim, kernel_size=1, stride=1, padding=0) for _ in range(layer_num))
        self.pos_embed = CrossLayerPosEmbedding3D(num_heads=num_heads, window_size=self.window_sizes, spatial=True)

    def forward(self, x_list, extra=None):
        WmH, WmW = x_list[-1].shape[-2:]
        shortcut_list = []
        q_list, k_list, v_list = [], [], []

        for i, x in enumerate(x_list):
            B, C, H, W = x.shape
            ws_i, stride_i = self.window_sizes[i], self.stride_list[i]
            pad_i = (
                math.ceil((stride_i * (WmH - 1.0) - H + ws_i) / 2.0),
                math.ceil((stride_i * (WmW - 1.0) - W + ws_i) / 2.0),
            )
            self.padding_list[i] = pad_i
            self.shape_list[i] = [H, W]

            x = self.cpe[i][0](x)
            shortcut_list.append(x)
            qkv = self.qkv[i](x)
            qkv_windows = overlaped_window_partition(qkv, ws_i, stride=stride_i, pad=pad_i)
            qkv_windows = qkv_windows.reshape(B, WmH * WmW, ws_i * ws_i, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
            q_windows, k_windows, v_windows = qkv_windows[0], qkv_windows[1], qkv_windows[2]
            q_list.append(q_windows)
            k_list.append(k_windows)
            v_list.append(v_windows)

        q_stack = torch.cat(q_list, dim=-2)
        k_stack = torch.cat(k_list, dim=-2)
        v_stack = torch.cat(v_list, dim=-2)

        attn = F.normalize(q_stack, dim=-1) @ F.normalize(k_stack, dim=-1).transpose(-1, -2)
        attn = attn + self.pos_embed()
        attn = self.softmax(attn)

        out = attn.to(v_stack.dtype) @ v_stack
        out = out.permute(0, 2, 3, 1, 4).reshape(B, WmH * WmW, self.token_num, self.hidden_dim)

        out_split = out.split(self.token_num_per_layer, dim=-2)
        out_list = []
        for i, out_i in enumerate(out_split):
            ws_i, stride_i, pad_i = self.window_sizes[i], self.stride_list[i], self.padding_list[i]
            H, W = self.shape_list[i]
            out_i = overlaped_window_reverse(out_i, H, W, ws_i, stride_i, pad_i)
            out_i = shortcut_list[i] + self.norm1[i](self.proj[i](out_i))
            out_i = self.cpe[i][1](out_i)
            out_i = out_i.permute(0, 2, 3, 1)
            out_i = out_i + self.mlp[i](self.norm2[i](out_i))
            out_i = out_i.permute(0, 3, 1, 2)
            out_list.append(out_i)
        return out_list


class CrossLayerChannelAttention(nn.Module):
    def __init__(self, in_dim, layer_num=3, alpha=1, num_heads=4, mlp_ratio=2, reduction=4):
        super().__init__()
        assert alpha % 2 != 0, "alpha must be odd"
        self.num_heads = num_heads
        self.reduction = reduction
        self.hidden_dim = in_dim // reduction
        self.in_dim = in_dim
        self.window_sizes = [(4**i + alpha) if i != 0 else (4**i + alpha - 1) for i in range(layer_num)][::-1]
        self.token_num_per_layer = [i for i in self.window_sizes]
        self.token_num = sum(self.token_num_per_layer)

        self.stride_list = [(4**i) for i in range(layer_num)][::-1]
        self.padding_list = [0 for _ in self.window_sizes]
        self.shape_list = [[0, 0] for _ in range(layer_num)]
        self.unshuffle_factor = [(2**i) for i in range(layer_num)][::-1]

        self.cpe = nn.ModuleList(nn.ModuleList([ConvPosEnc(dim=in_dim, k=3), ConvPosEnc(dim=in_dim, k=3)]) for _ in range(layer_num))
        self.norm1 = nn.ModuleList(LayerNormProxy(in_dim) for _ in range(layer_num))
        self.norm2 = nn.ModuleList(nn.LayerNorm(in_dim) for _ in range(layer_num))

        self.qkv = nn.ModuleList(nn.Conv2d(in_dim, self.hidden_dim * 3, kernel_size=1, stride=1, padding=0) for _ in range(layer_num))

        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.ModuleList(nn.Conv2d(self.hidden_dim, in_dim, kernel_size=1, stride=1, padding=0) for _ in range(layer_num))

        mlp_hidden_dim = int(in_dim * mlp_ratio)
        self.mlp = nn.ModuleList(Mlp(in_features=in_dim, hidden_features=mlp_hidden_dim) for _ in range(layer_num))

        self.pos_embed = CrossLayerPosEmbedding3D(num_heads=num_heads, window_size=self.window_sizes, spatial=False)

    def forward(self, x_list, extra=None):
        shortcut_list, reverse_shape = [], []
        q_list, k_list, v_list = [], [], []
        for i, x in enumerate(x_list):
            B, C, H, W = x.shape
            self.shape_list[i] = [H, W]
            ws_i, stride_i = self.window_sizes[i], self.stride_list[i]
            pad_i = math.ceil((stride_i * (self.hidden_dim - 1.0) - (self.unshuffle_factor[i]) ** 2 * self.hidden_dim + ws_i) / 2.0)
            self.padding_list[i] = pad_i
            x = self.cpe[i][0](x)
            shortcut_list.append(x)

            qkv = self.qkv[i](x)
            qkv = F.pixel_unshuffle(qkv, downscale_factor=self.unshuffle_factor[i])
            reverse_shape.append(qkv.size(1) // 3)

            qkv_window = einops.rearrange(qkv, "b c h w -> b (h w) c ()")
            qkv_window = overlaped_channel_partition(qkv_window, ws_i, stride=stride_i, pad=pad_i)
            qkv_window = qkv_window.reshape(B, qkv_window.size(1), ws_i, 3, self.num_heads, self.hidden_dim // self.num_heads)
            q, k, v = qkv_window[:, :, :, 0], qkv_window[:, :, :, 1], qkv_window[:, :, :, 2]
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)

        q_stack = torch.cat(q_list, dim=-2)
        k_stack = torch.cat(k_list, dim=-2)
        v_stack = torch.cat(v_list, dim=-2)

        attn = F.normalize(q_stack, dim=-1) @ F.normalize(k_stack, dim=-1).transpose(-1, -2)
        attn = attn + self.pos_embed()
        attn = self.softmax(attn)

        out = attn.to(v_stack.dtype) @ v_stack
        out = out.permute(0, 2, 3, 1, 4).reshape(B, -1, self.token_num, self.hidden_dim)

        out_split = out.split(self.token_num_per_layer, dim=-2)
        out_list = []
        for i, out_i in enumerate(out_split):
            ws_i, stride_i, pad_i = self.window_sizes[i], self.stride_list[i], self.padding_list[i]
            H, W = self.shape_list[i]
            out_i = overlaped_channel_reverse(out_i.permute(0, 2, 3, 1), ws_i, stride_i, pad_i, reverse_shape[i])
            out_i = shortcut_list[i] + self.norm1[i](self.proj[i](out_i))
            out_i = self.cpe[i][1](out_i)
            out_i = out_i.permute(0, 2, 3, 1)
            out_i = out_i + self.mlp[i](self.norm2[i](out_i))
            out_i = out_i.permute(0, 3, 1, 2)
            out_list.append(out_i)
        return out_list


# ---------------- 频域融合 ----------------


class FreqFusion(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels // reduction, bias=False), nn.ReLU(inplace=True))
        self.channel_att = nn.Sequential(nn.Linear(in_channels // reduction, in_channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y)
        y = self.channel_att(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LocalSimGuidedSampler(nn.Module):
    def __init__(self, in_channels, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, k, padding=k // 2, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn = self.act(self.bn(self.conv(x)))
        return x * attn


# ---------------- OREPA / RepPAN ----------------


def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std


def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [W_pixels_to_pad, W_pixels_to_pad, H_pixels_to_pad, H_pixels_to_pad])


class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class OREPA(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        act=True,
        internal_channels_1x1_3x3=None,
        deploy=False,
        single_init=False,
        weight_only=False,
        init_hyper_para=1.0,
        init_hyper_gamma=1.0,
    ):
        super().__init__()
        self.deploy = deploy
        self.nonlinear = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.weight_only = weight_only
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        padding = autopad(kernel_size, padding, dilation)
        self.padding = padding
        self.dilation = dilation

        if deploy:
            self.orepa_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.branch_counter = 0
            self.weight_orepa_origin = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), kernel_size, kernel_size))
            nn.init.kaiming_uniform_(self.weight_orepa_origin, a=math.sqrt(0.0))
            self.branch_counter += 1

            self.weight_orepa_avg_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            self.weight_orepa_pfir_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_orepa_avg_conv, a=0.0)
            nn.init.kaiming_uniform_(self.weight_orepa_pfir_conv, a=0.0)
            self.register_buffer("weight_orepa_avg_avg", torch.ones(kernel_size, kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 2

            self.weight_orepa_1x1 = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_orepa_1x1, a=0.0)
            self.branch_counter += 1

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups <= 4 else 2 * in_channels

            if internal_channels_1x1_3x3 == in_channels:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
                id_value = np.zeros((in_channels, int(in_channels / self.groups), 1, 1))
                for i in range(in_channels):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer("id_tensor", id_tensor)
            else:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1)
                )
                id_value = np.zeros((internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                for i in range(internal_channels_1x1_3x3):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer("id_tensor", id_tensor)

            self.weight_orepa_1x1_kxk_conv2 = nn.Parameter(
                torch.Tensor(out_channels, int(internal_channels_1x1_3x3 / self.groups), kernel_size, kernel_size)
            )
            nn.init.kaiming_uniform_(self.weight_orepa_1x1_kxk_conv2, a=math.sqrt(0.0))
            self.branch_counter += 1

            expand_ratio = 8
            self.weight_orepa_gconv_dw = nn.Parameter(torch.Tensor(in_channels * expand_ratio, 1, kernel_size, kernel_size))
            self.weight_orepa_gconv_pw = nn.Parameter(torch.Tensor(out_channels, int(in_channels * expand_ratio / self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_orepa_gconv_dw, a=math.sqrt(0.0))
            nn.init.kaiming_uniform_(self.weight_orepa_gconv_pw, a=math.sqrt(0.0))
            self.branch_counter += 1

            self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            if weight_only is False:
                self.bn = nn.BatchNorm2d(self.out_channels)

            self.fre_init()

            nn.init.constant_(self.vector[0, :], 0.25 * math.sqrt(init_hyper_gamma))
            nn.init.constant_(self.vector[1, :], 0.25 * math.sqrt(init_hyper_gamma))
            nn.init.constant_(self.vector[2, :], 0.0 * math.sqrt(init_hyper_gamma))
            nn.init.constant_(self.vector[3, :], 0.5 * math.sqrt(init_hyper_gamma))
            nn.init.constant_(self.vector[4, :], 1.0 * math.sqrt(init_hyper_gamma))
            nn.init.constant_(self.vector[5, :], 0.5 * math.sqrt(init_hyper_gamma))

            self.weight_orepa_1x1.data = self.weight_orepa_1x1.mul(init_hyper_para)
            self.weight_orepa_origin.data = self.weight_orepa_origin.mul(init_hyper_para)
            self.weight_orepa_1x1_kxk_conv2.data = self.weight_orepa_1x1_kxk_conv2.mul(init_hyper_para)
            self.weight_orepa_avg_conv.data = self.weight_orepa_avg_conv.mul(init_hyper_para)
            self.weight_orepa_pfir_conv.data = self.weight_orepa_pfir_conv.mul(init_hyper_para)
            self.weight_orepa_gconv_dw.data = self.weight_orepa_gconv_dw.mul(math.sqrt(init_hyper_para))
            self.weight_orepa_gconv_pw.data = self.weight_orepa_gconv_pw.mul(math.sqrt(init_hyper_para))

            if single_init:
                self.single_init()

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) * (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) * (i + 1 - half_fg) / 3)
        self.register_buffer("weight_orepa_prior", prior_tensor)

    def weight_gen(self):
        weight_orepa_origin = torch.einsum("oihw,o->oihw", self.weight_orepa_origin, self.vector[0, :])
        weight_orepa_avg = torch.einsum(
            "oi,hw->oihw", self.weight_orepa_avg_conv.squeeze(3).squeeze(2), self.weight_orepa_avg_avg
        )
        weight_orepa_avg = torch.einsum("oihw,o->oihw", weight_orepa_avg, self.vector[1, :])
        weight_orepa_pfir = torch.einsum(
            "oi,ohw->oihw", self.weight_orepa_pfir_conv.squeeze(3).squeeze(2), self.weight_orepa_prior
        )
        weight_orepa_pfir = torch.einsum("oihw,o->oihw", weight_orepa_pfir, self.vector[2, :])

        if hasattr(self, "weight_orepa_1x1_kxk_idconv1"):
            weight_orepa_1x1_kxk_conv1 = (self.weight_orepa_1x1_kxk_idconv1 + self.id_tensor).squeeze(3).squeeze(2)
        else:
            weight_orepa_1x1_kxk_conv1 = self.weight_orepa_1x1_kxk_conv1.squeeze(3).squeeze(2)
        weight_orepa_1x1_kxk_conv2 = self.weight_orepa_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_orepa_1x1_kxk_conv1.size()
            o, tg, h, w = weight_orepa_1x1_kxk_conv2.size()
            weight_orepa_1x1_kxk_conv1 = weight_orepa_1x1_kxk_conv1.view(g, int(t / g), ig)
            weight_orepa_1x1_kxk_conv2 = weight_orepa_1x1_kxk_conv2.view(g, int(o / g), tg, h, w)
            weight_orepa_1x1_kxk = torch.einsum("gti,gothw->goihw", weight_orepa_1x1_kxk_conv1, weight_orepa_1x1_kxk_conv2).reshape(o, ig, h, w)
        else:
            weight_orepa_1x1_kxk = torch.einsum("ti,othw->oihw", weight_orepa_1x1_kxk_conv1, weight_orepa_1x1_kxk_conv2)
        weight_orepa_1x1_kxk = torch.einsum("oihw,o->oihw", weight_orepa_1x1_kxk, self.vector[3, :])

        weight_orepa_1x1 = transVI_multiscale(self.weight_orepa_1x1, self.kernel_size)
        weight_orepa_1x1 = torch.einsum("oihw,o->oihw", weight_orepa_1x1, self.vector[4, :])

        weight_orepa_gconv = torch.einsum("oihw,o->oihw", self.weight_orepa_gconv_pw, self.vector[5, :])
        weight_orepa_gconv_dw = transVI_multiscale(self.weight_orepa_gconv_dw, self.kernel_size)
        weight_orepa_gconv = torch.einsum("oihw,o->oihw", weight_orepa_gconv, self.vector[5, :])
        weight_orepa_gconv = torch.einsum("oihw,ohw->oihw", weight_orepa_gconv, weight_orepa_gconv_dw.squeeze(1))

        weight = weight_orepa_origin + weight_orepa_avg + weight_orepa_pfir + weight_orepa_1x1_kxk + weight_orepa_1x1 + weight_orepa_gconv
        return weight

    def forward(self, x):
        if hasattr(self, "orepa_reparam"):
            return self.nonlinear(self.orepa_reparam(x))
        weight = self.weight_gen()
        out = F.conv2d(x, weight, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
        if hasattr(self, "bn"):
            out = self.bn(out)
        return self.nonlinear(out)


class RepConvN(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)


class RepNBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class OREPANBottleneck(RepNBottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = OREPA(c1, c_, k[0], 1)


class RepNCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class OREPANCSP(RepNCSP):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(OREPANBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class OREPANCSPELAN4(RepNCSPELAN4):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__(c1, c2, c3, c4, c5)
        self.cv2 = nn.Sequential(OREPANCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(OREPANCSP(c4, c4, c5), Conv(c4, c4, 3, 1))


# ---------------- DAMO-YOLO GFPN ----------------


class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self, ch_in, ch_hidden_ratio, ch_out, shortcut=True):
        super().__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = Conv(ch_hidden, ch_out, 3, s=1)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, s=1)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        return x + y if self.shortcut else y


class SPP_DAMO(nn.Module):
    def __init__(self, ch_in, ch_out, k, pool_size):
        super().__init__()
        self.pool = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
                for size in pool_size
            ]
        )
        self.conv = Conv(ch_in, ch_out, k)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)
        return self.conv(y)


class CSPStage(nn.Module):
    def __init__(self, ch_in, ch_out, n, block_fn="BasicBlock_3x3_Reverse", ch_hidden_ratio=1.0, spp=False):
        super().__init__()
        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = Conv(ch_in, ch_first, 1)
        self.conv2 = Conv(ch_in, ch_mid, 1)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == "BasicBlock_3x3_Reverse":
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in, ch_hidden_ratio, ch_mid, shortcut=True),
                )
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module("spp", SPP_DAMO(ch_mid * 4, ch_mid, 1, [5, 9, 13]))
            next_ch_in = ch_mid
        self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        return self.conv3(y)


# ---------------- EfficientRepBiPAN / RepPAN ----------------


class Transpose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=True)

    def forward(self, x):
        return self.upsample_transpose(x)


class BiFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = Conv(in_channels[1], out_channels, 1, 1)
        self.cv2 = Conv(in_channels[2], out_channels, 1, 1)
        self.cv3 = Conv(out_channels * 3, out_channels, 1, 1)
        self.upsample = Transpose(in_channels=out_channels, out_channels=out_channels)
        self.downsample = Conv(out_channels, out_channels, 3, 2)

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))


# ---------------- Re-Calibration FPN ----------------


def Upsample_bilinear(x, size, align_corners=False):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


class SBA(nn.Module):
    def __init__(self, inc, input_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.d_in1 = Conv(input_dim // 2, input_dim // 2, 1)
        self.d_in2 = Conv(input_dim // 2, input_dim // 2, 1)
        self.conv = Conv(input_dim, input_dim, 3)
        self.fc1 = nn.Conv2d(inc[1], input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], input_dim // 2, kernel_size=1, bias=False)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        H_feature, L_feature = x
        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)
        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)
        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)
        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample_bilinear(
            g_H_feature * H_feature, size=L_feature.size()[2:], align_corners=False
        )
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample_bilinear(
            g_L_feature * L_feature, size=H_feature.size()[2:], align_corners=False
        )
        H_feature = Upsample_bilinear(H_feature, size=L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out


# ---------------- Efficient Multi-Branch & Scale FPN ----------------


class EUCB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.up_dwc = nn.Sequential(nn.Upsample(scale_factor=2), Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels, s=stride))
        self.pwc = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, dw_parallel=True):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([nn.Sequential(Conv(self.in_channels, self.in_channels, kernel_size, s=stride, g=self.in_channels)) for kernel_size in self.kernel_sizes])

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MSCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5], stride=1, expansion_factor=2, dw_parallel=True, add=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.n_scales = len(self.kernel_sizes)
        assert self.stride in [1, 2]
        self.use_skip_connection = True if self.stride == 1 else False
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(Conv(self.in_channels, self.ex_channels, 1))
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, dw_parallel=self.dw_parallel)
        self.combined_channels = self.ex_channels if self.add else self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(Conv(self.combined_channels, self.out_channels, 1, act=False))
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = self.channel_shuffle(dout, math.gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x


class CSP_MSCB(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=[1, 3, 5], shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MSCB(self.c, self.c, kernel_sizes=kernel_sizes) for _ in range(n))
