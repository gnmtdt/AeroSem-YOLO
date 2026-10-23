"""Converse2D_Up: 可学习变换上采样模块.

论文: Converse2D (ICCV 2025)
论文链接: https://www.arxiv.org/abs/2508.09824
迁移自参考库 Ultralytics_674595707/nn/extra_modules/upsample/Converse2D_Up.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Converse2D_Up"]


class Converse2D_Up(nn.Module):
    """基于可学习变换的上采样算子 (Converse2D Upsampling).

    利用频域运算实现可学习的上采样，适用于图像恢复任务。

    Args:
        in_channels: 输入通道数。
        kernel_size: 卷积核大小，默认 3。
        scale: 上采样倍率，默认 2。
        padding_mode: 填充模式，默认 'circular'。
        eps: 数值稳定性常数，默认 1e-5。
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        scale: int = 2,
        padding_mode: str = "circular",
        eps: float = 1e-5,
    ):
        super(Converse2D_Up, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = kernel_size - 1
        self.padding_mode = padding_mode
        self.eps = eps

        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        self.weight.data = nn.functional.softmax(
            self.weight.data.view(1, self.in_channels, -1), dim=-1
        ).view(1, self.in_channels, self.kernel_size, self.kernel_size)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            x: 输入张量 (B, C, H, W)。

        Returns:
            上采样后的张量 (B, C, H*scale, W*scale)。
        """
        if self.padding > 0:
            x = nn.functional.pad(
                x,
                pad=[self.padding, self.padding, self.padding, self.padding],
                mode=self.padding_mode,
                value=0,
            )

        biaseps = torch.sigmoid(self.bias - 9.0) + self.eps
        _, _, h, w = x.shape
        STy = self._upsample(x, scale=self.scale)
        if self.scale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")

        FB = self._p2o(self.weight, (h * self.scale, w * self.scale))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))

        FR = FBFy + torch.fft.fftn(biaseps * x, dim=(-2, -1))
        x1 = FB.mul(FR)
        FBR = torch.mean(self._splits(x1, self.scale), dim=-1, keepdim=False)
        invW = torch.mean(self._splits(F2B, self.scale), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + biaseps)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale, self.scale)
        FX = (FR - FCBinvWBR) / biaseps
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if self.padding > 0:
            out = out[
                ...,
                self.padding * self.scale : -self.padding * self.scale,
                self.padding * self.scale : -self.padding * self.scale,
            ]

        return self.act(out)

    @staticmethod
    def _splits(a: torch.Tensor, scale: int) -> torch.Tensor:
        """将张量按 scale 分割为 scale*scale 个块.

        Args:
            a: 输入张量 (..., W, H)。
            scale: 分割因子。

        Returns:
            分割后的张量 (..., W/scale, H/scale, scale^2)。
        """
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale

        b = a.view(*leading_dims, scale, W_s, scale, H_s)

        permute_order = list(range(len(leading_dims))) + [
            len(leading_dims) + 1,
            len(leading_dims) + 3,
            len(leading_dims),
            len(leading_dims) + 2,
        ]
        b = b.permute(*permute_order).contiguous()

        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b

    @staticmethod
    def _p2o(psf: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        """将点扩散函数转换为光学传递函数.

        Args:
            psf: 点扩散函数张量 (N, C, h, w)。
            shape: 目标形状 (H, W)。

        Returns:
            光学传递函数张量 (N, C, H, W)。
        """
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[..., : psf.shape[-2], : psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2] / 2), -int(psf.shape[-1] / 2)), dims=(-2, -1))
        otf = torch.fft.fftn(otf, dim=(-2, -1))
        return otf

    @staticmethod
    def _upsample(x: torch.Tensor, scale: int = 3) -> torch.Tensor:
        """零填充上采样.

        Args:
            x: 输入张量 (N, C, W, H)。
            scale: 上采样因子。

        Returns:
            零填充上采样后的张量。
        """
        st = 0
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale)).type_as(x)
        z[..., st::scale, st::scale].copy_(x)
        return z
