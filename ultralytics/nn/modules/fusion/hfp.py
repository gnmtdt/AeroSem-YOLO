# 论文: HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection (AAAI 2025)
# 链接: https://arxiv.org/abs/2412.10116
# 模块作用: 对融合后特征进行频域高频感知与空间/通道门控，突出细粒度目标与边缘信息，抑制跨模态低频冗余。

import torch
import torch.nn as nn


class HighFrequencyPerception(nn.Module):
    def __init__(self, ratio: tuple[float, float] = (0.25, 0.25), patch: tuple[int, int] = (8, 8), groups: int = 32) -> None:
        super().__init__()
        self.ratio = ratio
        self.ph, self.pw = int(patch[0]), int(patch[1])
        self.groups = int(groups)
        self._built = False
        self._c = None
        self.spatial_conv: nn.Module | None = None
        self.channel_conv1: nn.Module | None = None
        self.channel_conv2: nn.Module | None = None
        self.out_conv: nn.Module | None = None

    def _build_if_needed(self, c: int) -> None:
        if self._built and self._c == c:
            return
        g = max(1, min(self.groups, c))
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.channel_conv1 = nn.Conv2d(c, c, kernel_size=1, groups=g)
        self.channel_conv2 = nn.Conv2d(c, c, kernel_size=1, groups=g)
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, c), num_channels=c),
        )
        self._built = True
        self._c = c

    def _mask_fft(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x, dim=(-2, -1))
        h0 = int(H * self.ratio[0])
        w0 = int((W // 2 + 1) * self.ratio[1])
        mask = torch.ones_like(xf, dtype=xf.dtype)
        mask[:, :, :h0, :w0] = 0
        xf = xf * mask
        xh = torch.fft.irfft2(xf, s=(H, W))
        return xh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise TypeError("HighFrequencyPerception 期望输入 [B, C, H, W]")
        B, C, H, W = x.shape
        self._build_if_needed(C)
        hf = self._mask_fft(x)
        spa = self.spatial_conv(hf) * x
        amax = torch.nn.functional.adaptive_max_pool2d(hf, output_size=(self.ph, self.pw))
        aavg = torch.nn.functional.adaptive_avg_pool2d(hf, output_size=(self.ph, self.pw))
        amax = torch.sum(torch.relu(amax), dim=(2, 3), keepdim=True)
        aavg = torch.sum(torch.relu(aavg), dim=(2, 3), keepdim=True)
        ch = self.channel_conv1(amax) + self.channel_conv1(aavg)
        ch = torch.sigmoid(self.channel_conv2(ch))
        cha = ch * x
        out = self.out_conv(spa + cha)
        return out
