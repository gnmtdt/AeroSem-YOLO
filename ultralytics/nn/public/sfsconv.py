"""Space-Frequency Selection Convolution (SFS_Conv).

迁移自 RTDETR-main 的 `nn/extra_modules/SFSConv.py`，用于 C2f_FasterSFSConv。
说明：
- 移除原文件中的 profiling 依赖（thop）与 __main__ 示例；
- 保留 numpy 作为滤波器生成依赖（本仓库已广泛依赖 numpy）。
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class FractionalGaborFilter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], order: float, angles: Iterable[float], scales: Iterable[float]):
        super().__init__()
        self.real_weights = nn.ParameterList()
        for angle in angles:
            for scale in scales:
                real_weight = self.generate_fractional_gabor(in_channels, out_channels, kernel_size, order, angle, scale)
                self.real_weights.append(nn.Parameter(real_weight))

    @staticmethod
    def generate_fractional_gabor(in_channels: int, out_channels: int, size: tuple[int, int], order: float, angle: float, scale: float) -> torch.Tensor:
        x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
        x_theta = x * np.cos(angle) + y * np.sin(angle)
        y_theta = -x * np.sin(angle) + y * np.cos(angle)

        real_part = np.exp(-((x_theta**2 + (y_theta / scale) ** 2) ** order)) * np.cos(2 * np.pi * x_theta / scale)
        real_weight = torch.tensor(real_part, dtype=torch.float32).view(1, 1, size[0], size[1])
        return real_weight.repeat(out_channels, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real_result = sum(weight * x for weight in self.real_weights)
        return real_result


class GaborSingle(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], order: float, angles: Iterable[float], scales: Iterable[float]):
        super().__init__()
        self.gabor = FractionalGaborFilter(in_channels, out_channels, kernel_size, order, angles, scales)
        self.t = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]), requires_grad=True)
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gabor(self.t)
        out = F.conv2d(x, out, stride=1, padding=(out.shape[-2] - 1) // 2)
        out = self.relu(out)
        out = F.dropout(out, 0.3)
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out


class GaborFPU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, order: float = 0.25, angles: Iterable[float] = (0, 45, 90, 135), scales: Iterable[float] = (1, 2, 3, 4)):
        super().__init__()
        self.gabor = GaborSingle(in_channels // 4, out_channels // 4, (3, 3), order, angles, scales)
        self.fc = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, 1)
        x_out = torch.cat([self.gabor(x1), self.gabor(x2), self.gabor(x3), self.gabor(x4)], dim=1)
        x_out = self.fc(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


class FrFTFilter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], f: float, order: float):
        super().__init__()
        self.register_buffer("weight", self.generate_FrFT_filter(in_channels, out_channels, kernel_size, f, order))

    @staticmethod
    def generate_FrFT_filter(in_channels: int, out_channels: int, kernel: tuple[int, int], f: float, p: float) -> torch.Tensor:
        N = out_channels
        d_x, d_y = kernel[0], kernel[1]
        x = np.linspace(1, d_x, d_x)
        y = np.linspace(1, d_y, d_y)
        X, Y = np.meshgrid(x, y)

        real_FrFT_filterX = np.zeros([d_x, d_y, out_channels])
        real_FrFT_filterY = np.zeros([d_x, d_y, out_channels])
        real_FrFT_filter = np.zeros([d_x, d_y, out_channels])
        for i in range(N):
            real_FrFT_filterX[:, :, i] = np.cos(-f * (X) / math.sin(p) + (f * f + X * X) / (2 * math.tan(p)))
            real_FrFT_filterY[:, :, i] = np.cos(-f * (Y) / math.sin(p) + (f * f + Y * Y) / (2 * math.tan(p)))
            real_FrFT_filter[:, :, i] = real_FrFT_filterY[:, :, i] * real_FrFT_filterX[:, :, i]

        g_f = np.zeros((kernel[0], kernel[1], in_channels, out_channels))
        for i in range(N):
            g_f[:, :, :, i] = np.repeat(real_FrFT_filter[:, :, i : i + 1], in_channels, axis=2)
        g_f_real = np.array(g_f).reshape((out_channels, in_channels, kernel[0], kernel[1]))
        return torch.tensor(g_f_real).type(torch.FloatTensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class FrFTSingle(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], f: float, order: float):
        super().__init__()
        self.fft = FrFTFilter(in_channels, out_channels, kernel_size, f, order)
        self.t = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]), requires_grad=True)
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fft(self.t)
        out = F.conv2d(x, out, stride=1, padding=(out.shape[-2] - 1) // 2)
        out = self.relu(out)
        out = F.dropout(out, 0.3)
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out


class FourierFPU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, order: float = 0.25):
        super().__init__()
        self.fft1 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.25, order)
        self.fft2 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.50, order)
        self.fft3 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.75, order)
        self.fft4 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 1.00, order)
        self.fc = Conv(out_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, 1)
        x_out = torch.cat([self.fft1(x1), self.fft2(x2), self.fft3(x3), self.fft4(x4)], dim=1)
        x_out = self.fc(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


class SPU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.c1 = Conv(in_channels // 2, in_channels // 2, 3, g=in_channels // 2)
        self.c2 = Conv(in_channels // 2, in_channels // 2, 5, g=in_channels // 2)
        self.c3 = Conv(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x1 = self.c1(x1)
        x2 = self.c2(x2 + x1)
        x_out = torch.cat([x1, x2], dim=1)
        x_out = self.c3(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


class SFS_Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, order: float = 0.25, filter: str = "FrGT"):
        super().__init__()
        self.PWC0 = Conv(in_channels, in_channels // 2, 1)
        self.PWC1 = Conv(in_channels, in_channels // 2, 1)
        self.SPU = SPU(in_channels // 2, out_channels)

        if filter not in ("FrFT", "FrGT"):
            raise ValueError("filter must be 'FrFT' or 'FrGT'.")
        self.FPU = FourierFPU(in_channels // 2, out_channels, order) if filter == "FrFT" else GaborFPU(in_channels // 2, out_channels, order)

        self.PWC_o = Conv(out_channels, out_channels, 1)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_spa = self.SPU(self.PWC0(x))
        x_fre = self.FPU(self.PWC1(x))
        out = torch.cat([x_spa, x_fre], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return self.PWC_o(out1 + out2)

