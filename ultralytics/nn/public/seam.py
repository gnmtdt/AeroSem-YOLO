# Ultralytics YOLOMM - SEAM Modules (ported from YOLO11 extra_modules/block.py)
#
# 说明：
# - 本文件仅承载 Detect_SEAM / Detect_MultiSEAM 所需的最小实现集合。
# - 不包含与 SEAM 无关的其它 block.py 内容，避免把上游大文件整块搬入本工程。

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SEAM", "MultiSEAM"]


class _Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class SEAM(nn.Module):
    """
    SEAM: Squeeze-Excitation-like attention with depthwise residual conv blocks.

    源实现：YOLO11 修改库 `ultralytics/nn/extra_modules/block.py`（SEAM start 区域）。
    """

    def __init__(self, c1: int, c2: int, n: int, reduction: int = 16):
        super().__init__()
        if c1 != c2:
            c2 = c1

        self.DCovN = nn.Sequential(
            *[
                nn.Sequential(
                    _Residual(
                        nn.Sequential(
                            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2),
                            nn.GELU(),
                            nn.BatchNorm2d(c2),
                        )
                    ),
                    nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, groups=1),
                    nn.GELU(),
                    nn.BatchNorm2d(c2),
                )
                for _ in range(n)
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid(),
        )

        self._initialize_weights()
        self._initialize_layer(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _initialize_layer(layer: nn.Module) -> None:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(layer.weight, mean=0.0, std=0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def _dcovn(c1: int, c2: int, depth: int, kernel_size: int = 3, patch_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
        nn.SiLU(),
        nn.BatchNorm2d(c2),
        *[
            nn.Sequential(
                _Residual(
                    nn.Sequential(
                        nn.Conv2d(c2, c2, kernel_size=kernel_size, stride=1, padding=1, groups=c2),
                        nn.SiLU(),
                        nn.BatchNorm2d(c2),
                    )
                ),
                nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, groups=1),
                nn.SiLU(),
                nn.BatchNorm2d(c2),
            )
            for _ in range(depth)
        ],
    )


class MultiSEAM(nn.Module):
    """
    MultiSEAM: Multi-patch SEAM aggregation.

    源实现：YOLO11 修改库 `ultralytics/nn/extra_modules/block.py`（MultiSEAM start 区域）。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        depth: int,
        kernel_size: int = 3,
        patch_size: list[int] | tuple[int, int, int] = (3, 5, 7),
        reduction: int = 16,
    ):
        super().__init__()
        if c1 != c2:
            c2 = c1

        self.DCovN0 = _dcovn(c1, c2, depth, kernel_size=kernel_size, patch_size=int(patch_size[0]))
        self.DCovN1 = _dcovn(c1, c2, depth, kernel_size=kernel_size, patch_size=int(patch_size[1]))
        self.DCovN2 = _dcovn(c1, c2, depth, kernel_size=kernel_size, patch_size=int(patch_size[2]))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y0 = self.DCovN0(x)
        y1 = self.DCovN1(x)
        y2 = self.DCovN2(x)
        y0 = self.avg_pool(y0).view(b, c)
        y1 = self.avg_pool(y1).view(b, c)
        y2 = self.avg_pool(y2).view(b, c)
        y4 = self.avg_pool(x).view(b, c)
        y = (y0 + y1 + y2 + y4) / 4
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

