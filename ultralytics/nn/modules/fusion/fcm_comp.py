# 论文: FBRT-YOLO: Faster and Better for Real-Time Aerial Image Detection (AAAI 2025)
# 链接: https://arxiv.org/pdf/2504.20670
# 模块作用: 对单路融合特征进行主/辅支路互补映射，结合通道与空间注意力强化互补信息并保持通道一致。

import torch
import torch.nn as nn


class FCMCompConv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, groups: int = 1,
                 act: bool | nn.Module = True) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FCMCompChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.gap(x)
        return self.sigmoid(x)


class FCMCompSpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.sigmoid(x)


class FeatureComplementaryMapping(nn.Module):
    def __init__(self, channels: int | None = None, ratio: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.ratio = int(ratio)
        self._built = False
        self._c = None
        self.main_branch_conv1: nn.Module | None = None
        self.main_branch_conv2: nn.Module | None = None
        self.main_branch_conv3: nn.Module | None = None
        self.sub_branch_conv: nn.Module | None = None
        self.spatial_attn: nn.Module | None = None
        self.channel_attn: nn.Module | None = None

    def _build_if_needed(self, c: int) -> None:
        if self._built and self._c == c:
            return
        sub_c = max(c // self.ratio, 1)
        main_c = c - sub_c
        self.main_branch_conv1 = FCMCompConv(main_c, main_c, k=3, s=1, p=1)
        self.main_branch_conv2 = FCMCompConv(main_c, main_c, k=3, s=1, p=1)
        self.main_branch_conv3 = FCMCompConv(main_c, c, k=1, s=1, p=0)
        self.sub_branch_conv = FCMCompConv(sub_c, c, k=1, s=1, p=0)
        self.spatial_attn = FCMCompSpatialAttention(c)
        self.channel_attn = FCMCompChannelAttention(c)
        self._built = True
        self._c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise TypeError("FeatureComplementaryMapping 期望输入 [B, C, H, W]")
        _, c, _, _ = x.shape
        self._build_if_needed(c)
        sub_c = max(c // self.ratio, 1)
        main_c = c - sub_c
        main_x, sub_x = torch.split(x, [main_c, sub_c], dim=1)
        main = self.main_branch_conv1(main_x)
        main = self.main_branch_conv2(main)
        main = self.main_branch_conv3(main)
        sub = self.sub_branch_conv(sub_x)
        spa = self.spatial_attn(sub) * main
        cha = self.channel_attn(main) * sub
        return spa + cha
