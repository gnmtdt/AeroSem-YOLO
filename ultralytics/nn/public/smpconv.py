"""SMPConv (Sparse Meta-Point Convolution) - pure PyTorch implementation.

源库实现依赖自定义 CUDA 扩展（depthwise_conv2d_implicit_gemm）。
本工程为标准发行版本迁移：采用纯 PyTorch `F.conv2d` 实现，不引入自动降级分支。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


def _rel_pos(kernel_size: int) -> torch.Tensor:
    tensors = [torch.linspace(-1, 1, steps=kernel_size) for _ in range(2)]
    kernel_coord = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-0)
    return kernel_coord.unsqueeze(0)


class SMPConv(nn.Module):
    def __init__(self, planes: int, kernel_size: int, n_points: int, stride: int = 1, padding: int | None = None, groups: int = 1):
        super().__init__()
        if planes <= 0:
            raise ValueError(f"planes must be > 0, got {planes}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if n_points <= 0:
            raise ValueError(f"n_points must be > 0, got {n_points}")

        self.planes = planes
        self.kernel_size = kernel_size
        self.n_points = n_points
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding
        self.groups = groups  # kept for API compatibility; SMPConv is depthwise by design.
        self.init_radius = 2 * (2 / kernel_size)

        kernel_coord = _rel_pos(kernel_size)
        self.register_buffer("kernel_coord", kernel_coord)

        weight_coord = torch.empty(1, n_points, 2)
        nn.init.trunc_normal_(weight_coord, std=0.2, a=-1.0, b=1.0)
        self.weight_coord = nn.Parameter(weight_coord)

        self.radius = nn.Parameter(torch.empty(1, n_points).unsqueeze(-1).unsqueeze(-1))
        self.radius.data.fill_(value=self.init_radius)

        weights = torch.empty(1, planes, n_points)
        trunc_normal_(weights, std=0.02)
        self.weights = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.planes:
            raise ValueError(f"SMPConv expects input channels == {self.planes}, got {x.shape[1]}")
        kernels = self.make_kernels().unsqueeze(1)  # [C, 1, k, k]
        return F.conv2d(x, kernels, stride=self.stride, padding=self.padding, groups=self.planes)

    def make_kernels(self) -> torch.Tensor:
        # diff: [1, n_points, kernel_size^2, 2]
        diff = self.weight_coord.unsqueeze(-2) - self.kernel_coord.reshape(1, 2, -1).transpose(1, 2)
        diff = diff.transpose(2, 3).reshape(1, self.n_points, 2, self.kernel_size, self.kernel_size)
        diff = F.relu(1 - torch.sum(torch.abs(diff), dim=2) / self.radius)  # [1, n_points, k, k]

        kernels = torch.matmul(self.weights, diff.reshape(1, self.n_points, -1))  # [1, planes, k*k]
        kernels = kernels.reshape(1, self.planes, *self.kernel_coord.shape[2:])  # [1, planes, k, k]
        kernels = kernels.squeeze(0)  # [planes, k, k]
        kernels = torch.flip(kernels.permute(0, 2, 1), dims=(1,))
        return kernels

    def radius_clip(self, min_radius: float = 1e-3, max_radius: float = 1.0) -> None:
        self.radius.data = self.radius.data.clamp(min_radius, max_radius)

