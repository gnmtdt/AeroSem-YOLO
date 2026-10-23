"""LEGM (Local-Enhanced Global Mixer) used by C2f_LEGM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_relative_positions(window_size: int) -> torch.Tensor:
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_positions = relative_positions.permute(1, 2, 0).contiguous()
    relative_positions_log = torch.sign(relative_positions) * torch.log(1.0 + relative_positions.abs())
    return relative_positions_log


class WATT(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(nn.Linear(2, 256, bias=True), nn.ReLU(True), nn.Linear(256, num_heads, bias=True))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        b_, n, _ = qkv.shape
        qkv = qkv.reshape(b_, n, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, self.dim)
        return x


class Att(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int, use_attn: bool = False, conv_type: str | None = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == "Conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode="reflect"),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode="reflect"),
            )

        if self.conv_type == "DWConv":
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode="reflect")

        if self.conv_type == "DWConv" or self.use_attn:
            self.v = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.qk = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WATT(dim, window_size, num_heads)

    def check_size(self, x: torch.Tensor, shift: bool = False) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(
                x,
                (
                    self.shift_size,
                    (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                    self.shift_size,
                    (self.window_size - self.shift_size + mod_pad_h) % self.window_size,
                ),
                mode="reflect",
            )
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        if self.conv_type == "DWConv" or self.use_attn:
            v = self.v(x)

        if self.use_attn:
            qk = self.qk(x)
            qkv = torch.cat([qk, v], dim=1)

            shifted_qkv = self.check_size(qkv, self.shift_size > 0)
            ht, wt = shifted_qkv.shape[2:]
            shifted_qkv = shifted_qkv.permute(0, 2, 3, 1)
            qkv_win = window_partition(shifted_qkv, self.window_size)
            attn_windows = self.attn(qkv_win)

            shifted_out = window_reverse(attn_windows, self.window_size, ht, wt)
            out = shifted_out[:, self.shift_size : (self.shift_size + h), self.shift_size : (self.shift_size + w), :]
            attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ["Conv", "DWConv"]:
                conv_out = self.conv(v)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)
        else:
            if self.conv_type == "Conv":
                out = self.conv(x)
            elif self.conv_type == "DWConv":
                out = self.proj(self.conv(v))
            else:
                raise ValueError("Att requires conv_type in {'Conv','DWConv'} when use_attn=False.")

        return out


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp = nn.Sequential(nn.Conv2d(in_features, hidden_features, 1), nn.ReLU(True), nn.Conv2d(hidden_features, out_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LayNormal(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, detach_grad: bool = False):
        super().__init__()
        self.eps = eps
        self.detach_grad = detach_grad
        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)
        trunc_normal_(self.meta1.weight, std=0.02)
        nn.init.constant_(self.meta1.bias, 1)
        trunc_normal_(self.meta2.weight, std=0.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, x: torch.Tensor):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((x - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        normalized = (x - mean) / std
        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)
        out = normalized * self.weight + self.bias
        return out, rescale, rebias


class LEGM(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        norm_layer=LayNormal,
        mlp_norm: bool = False,
        window_size: int = 8,
        shift_size: int = 0,
        use_attn: bool = True,
        conv_type: str | None = None,
    ):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Att(dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.use_attn:
            x, rescale, rebias = self.norm1(x)
        x = self.attn(x)
        if self.use_attn:
            x = x * rescale + rebias
        x = identity + x

        identity = x
        if self.use_attn and self.mlp_norm:
            x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm:
            x = x * rescale + rebias
        x = identity + x
        return x

