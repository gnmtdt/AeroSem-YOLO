"""DySample: 动态上采样模块.

论文: Learning to Upsample by Learning to Sample (ICCV 2023)
论文链接: https://arxiv.org/abs/2308.15085
迁移自参考库 Ultralytics_674595707/nn/extra_modules/upsample/DySample.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DySample"]


class DySample(nn.Module):
    """动态采样上采样模块 (Dynamic Sample Upsampling).

    通过学习采样偏移量实现动态上采样，支持 'lp' (先线性后像素重组) 和
    'pl' (先像素重组后线性) 两种风格。

    Args:
        in_channels: 输入通道数（同时也是输出通道数）。
        scale: 上采样倍率，默认 2。
        style: 上采样风格，'lp' 或 'pl'，默认 'lp'。
        groups: 分组数，默认 4。
        dyscope: 是否使用动态范围控制，默认 True。
    """

    def __init__(
        self,
        in_channels: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        dyscope: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ["lp", "pl"]
        if style == "pl":
            assert in_channels >= scale**2 and in_channels % scale**2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == "pl":
            in_channels = in_channels // scale**2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale**2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.0)

        self.register_buffer("init_pos", self._init_pos())

    @staticmethod
    def normal_init(module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0) -> None:
        """正态分布初始化权重."""
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
        """常数初始化权重."""
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self) -> torch.Tensor:
        """初始化采样位置."""
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h]))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """执行动态采样操作."""
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h]))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device)
        )
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).reshape((B, -1, self.scale * H, self.scale * W))

    def forward_lp(self, x: torch.Tensor) -> torch.Tensor:
        """lp 风格前向传播 (先线性后像素重组)."""
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x: torch.Tensor) -> torch.Tensor:
        """pl 风格前向传播 (先像素重组后线性)."""
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = (
                F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5
                + self.init_pos
            )
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，根据 style 选择分支."""
        if self.style == "pl":
            return self.forward_pl(x)
        return self.forward_lp(x)
