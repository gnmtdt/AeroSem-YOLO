"""CSI 公共模块（迁移自 RTDETR-main `nn/extra_modules/CSI.py`）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["CSI"]


def _import_mamba_cls():
    # 不做优雅降级：缺失依赖时明确失败（仅在实例化 CSI 时触发）。
    from mamba_ssm import Mamba  # type: ignore

    return Mamba


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CSI(nn.Module):
    def __init__(self, input_dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.norm1 = nn.LayerNorm(input_dim // 4)
        self.norm = nn.LayerNorm(input_dim)

        try:
            Mamba = _import_mamba_cls()
        except Exception as e:  # noqa: BLE001 - 明确抛出依赖错误
            raise ImportError(
                "CSI 依赖 `mamba_ssm`（Mamba）。请先安装对应依赖后再使用 C2f_CSI。"
            ) from e

        self.mamba = Mamba(
            d_model=input_dim // 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(input_dim // 4, input_dim // 4)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.cpe2 = nn.Conv2d(input_dim // 4, input_dim // 4, 3, padding=1, groups=input_dim // 4)
        self.out = Conv(input_dim, input_dim, 1)
        self.mlp = Mlp(in_features=input_dim // 4, hidden_features=int(input_dim // 4 * 4))

    def forward(self, x):
        b, c = x.shape[:2]
        if c != self.input_dim:
            raise ValueError(f"CSI expects C=={self.input_dim}, got C={c}")

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(b, c, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mlp(self.norm1(self.mamba(x1))) + self.skip_scale * x1
        x_mamba2 = self.mlp(self.norm1(self.mamba(x2))) + self.skip_scale * x2
        x_mamba3 = self.mlp(self.norm1(self.mamba(x3))) + self.skip_scale * x3
        x_mamba4 = self.mlp(self.norm1(self.mamba(x4))) + self.skip_scale * x4

        x_mamba1 = x_mamba1.transpose(-1, -2).reshape(b, self.output_dim // 4, *img_dims)
        x_mamba2 = x_mamba2.transpose(-1, -2).reshape(b, self.output_dim // 4, *img_dims)
        x_mamba3 = x_mamba3.transpose(-1, -2).reshape(b, self.output_dim // 4, *img_dims)
        x_mamba4 = x_mamba4.transpose(-1, -2).reshape(b, self.output_dim // 4, *img_dims)

        split_tensors = []
        for channel in range(x_mamba1.size(1)):
            channel_tensors = [t[:, channel : channel + 1, :, :] for t in [x_mamba1, x_mamba2, x_mamba3, x_mamba4]]
            split_tensors.append(torch.cat(channel_tensors, dim=1))
        x_out = torch.cat(split_tensors, dim=1)
        return self.out(x_out)

