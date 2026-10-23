"""Faster-style blocks built on SFS_Conv (used by C2f_FasterSFSConv)."""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.public.sfsconv import SFS_Conv


class Partial_SFSConv(nn.Module):
    def __init__(self, dim: int, n_div: int = 4, forward: str = "split_cat"):
        super().__init__()
        if n_div <= 0:
            raise ValueError(f"n_div must be > 0, got {n_div}")
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = SFS_Conv(self.dim_conv3, self.dim_conv3)

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError(f"Unsupported forward mode: {forward}")

    def forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, : self.dim_conv3, :, :] = self.partial_conv3(x[:, : self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)


class FasterSFSConv(nn.Module):
    def __init__(
        self,
        inc: int,
        dim: int,
        n_div: int = 4,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.1,
        layer_scale_init_value: float = 0.0,
        pconv_fw_type: str = "split_cat",
    ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
        )

        self.spatial_mixing = Partial_SFSConv(dim, n_div, pconv_fw_type)

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

