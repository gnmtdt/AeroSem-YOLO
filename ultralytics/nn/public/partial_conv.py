"""Partial Convolution blocks (shared small utilities)."""

from __future__ import annotations

import torch
import torch.nn as nn


class Partial_conv3(nn.Module):
    """部分卷积层 (Partial Convolution).

    说明：
    - 该实现与 FasterNet/ShuffleNetV2 的“partial channels”思想一致；
    - 仅对前 `dim//n_div` 个通道做 3x3 卷积，其余通道直通。
    """

    def __init__(self, dim: int, n_div: int = 4, forward: str = "split_cat"):
        super().__init__()
        if n_div <= 0:
            raise ValueError(f"n_div must be > 0, got {n_div}")
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError(f"Unsupported forward mode: {forward}")

    def forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        # only for inference
        x = x.clone()  # keep the original input intact for residual connection later
        x[:, : self.dim_conv3, :, :] = self.partial_conv3(x[:, : self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)

