"""LFEM blocks (Local Feature Enhancement Module) used by C2f_LFEM."""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
from timm.layers import DropPath

from ultralytics.nn.modules.conv import Conv


class Conv_Extra(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.block = nn.Sequential(Conv(channel, 64, 1), Conv(64, 64, 3), Conv(64, channel, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Scharr(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        scharr_x = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)

        self.norm = nn.BatchNorm2d(channel)
        self.act = nn.SiLU()
        self.conv_extra = Conv_Extra(channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        edge = torch.sqrt(edges_x**2 + edges_y**2)
        edge = self.act(self.norm(edge))
        out = self.conv_extra(x + edge)
        return out


class Gaussian(nn.Module):
    def __init__(self, dim: int, size: int, sigma: float, feature_extra: bool = True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        if feature_extra:
            self.conv_extra = Conv_Extra(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra:
            return self.conv_extra(x + gaussian)
        return gaussian

    @staticmethod
    def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        kernel = torch.FloatTensor(
            [
                [(1 / (2 * math.pi * sigma**2)) * math.exp(-(x**2 + y**2) / (2 * sigma**2)) for x in range(-size // 2 + 1, size // 2 + 1)]
                for y in range(-size // 2 + 1, size // 2 + 1)
            ]
        ).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


class LFEA(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = Conv(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(channel)

    def forward(self, c: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        att = c * att + c
        att = self.conv2d(att)
        wei = self.avg_pool(att)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = self.norm(c + att * wei)
        return x


class LFE_Module(nn.Module):
    def __init__(self, dim: int, stage: int = 1, mlp_ratio: float = 2.0, drop_path: float = 0.1):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer: List[nn.Module] = [Conv(dim, mlp_hidden_dim, 1), nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)]
        self.mlp = nn.Sequential(*mlp_layer)
        self.lfea = LFEA(dim)

        if stage == 0:
            self.scharr_edge = Scharr(dim)
        else:
            self.gaussian = Gaussian(dim, 5, 1.0)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.scharr_edge(x) if self.stage == 0 else self.gaussian(x)
        x_att = self.lfea(x, att)
        x = x + self.norm(self.drop_path(self.mlp(x_att)))
        return x

