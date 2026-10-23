"""Faster-style block with KAN MLP (used by C2f_Faster_KAN).

注意：
- 该实现依赖 `kat_rational` 提供的 `KAT_Group`（源库中为可选导入）。
- 按本工程约束：不引入自动降级；缺少依赖时在实例化阶段明确抛出 ImportError。
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.models.layers import DropPath

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.public.partial_conv import Partial_conv3


def _require_kat_group():
    try:
        from kat_rational import KAT_Group  # type: ignore
    except ImportError as e:
        raise ImportError(
            "C2f_Faster_KAN 依赖 `kat_rational`（提供 KAT_Group）。当前环境未安装该依赖。"
        ) from e
    return KAT_Group


class KAN(nn.Module):
    """MLP using KAT_Group activation (from kat_rational)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=None,
        norm_layer=None,
        bias=True,
        drop: float = 0.0,
        use_conv: bool = False,
        act_init: str = "gelu",
    ):
        super().__init__()
        KAT_Group = _require_kat_group()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act1 = KAT_Group(mode="identity")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.act2 = KAT_Group(mode=act_init)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x


class Faster_Block_KAN(nn.Module):
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

        self.mlp = KAN(dim, hidden_features=int(dim * mlp_ratio))
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.size()
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, c, h, w]).contiguous())
        return x

    def forward_layer_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

