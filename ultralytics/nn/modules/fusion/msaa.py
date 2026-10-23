# 论文: Efficient Visual State Space Model for Image Deblurring (CVPR 2025)
# 链接: https://arxiv.org/pdf/2405.10530
# 模块作用: 双路融合后以多尺度卷积与通道/空间注意力聚合，压缩跨模态冗余并输出单路判别表征。

import torch
import torch.nn as nn


class _MSAAChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        red = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, red, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(red, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)


class _MSAASpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)


class FusionConvMSAA(nn.Module):
    def __init__(
        self,
        dim: int | None = None,
        factor: float = 4.0,
        spatial_kernel: int = 7,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.factor = float(factor)
        self.spatial_kernel = int(spatial_kernel)
        self.reduction = int(reduction)
        self._built = False
        self._c = None
        self._mid = None
        self.down: nn.Module | None = None
        self.conv3: nn.Module | None = None
        self.conv5: nn.Module | None = None
        self.conv7: nn.Module | None = None
        self.spatial_attn: nn.Module | None = None
        self.channel_attn: nn.Module | None = None
        self.up: nn.Module | None = None

    def _build_if_needed(self, c: int) -> None:
        if self._built and self._c == c:
            return
        mid = max(int(c // self.factor), 1)
        self.down = nn.Conv2d(c * 2, mid, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(mid, mid, kernel_size=5, stride=1, padding=2)
        self.conv7 = nn.Conv2d(mid, mid, kernel_size=7, stride=1, padding=3)
        self.spatial_attn = _MSAASpatialAttention(kernel_size=self.spatial_kernel)
        self.channel_attn = _MSAAChannelAttention(mid, reduction=self.reduction)
        self.up = nn.Conv2d(mid, c, kernel_size=1, stride=1)
        self._built = True
        self._c = c
        self._mid = mid

    def forward(self, x1, x2=None):
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1
        if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
            raise TypeError("FusionConvMSAA 需要两路输入张量")
        if x1.shape != x2.shape:
            raise ValueError(f"FusionConvMSAA 要求两路输入形状一致，got {x1.shape} vs {x2.shape}")
        _, c, _, _ = x1.shape
        self._build_if_needed(c)
        x = torch.cat([x1, x2], dim=1)
        x = self.down(x)
        res = x
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        xs = x3 + x5 + x7
        xs = xs * self.spatial_attn(xs)
        xc = self.channel_attn(x)
        out = self.up(res + xs * xc)
        return out
