"""FAT_Block (Frequency-Aware Transformer block).

迁移自 RTDETR-main 的 `nn/extra_modules/filc.py`，用于 C2f_FAT。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import DropPath


def img2windows(img: torch.Tensor, h_sp: int, w_sp: int) -> torch.Tensor:
    b, c, h, w = img.shape
    img_reshape = img.view(b, c, h // h_sp, h_sp, w // w_sp, w_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, h_sp * w_sp, c)
    return img_perm


def windows2img(img_splits_hw: torch.Tensor, h_sp: int, w_sp: int, h: int, w: int) -> torch.Tensor:
    b = int(img_splits_hw.shape[0] / (h * w / h_sp / w_sp))
    img = img_splits_hw.view(b, h // h_sp, w // w_sp, h_sp, w_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return img


class WindowAttention(nn.Module):
    def __init__(self, dim: int, idx: int, split_size: int = 8, dim_out: int | None = None, num_heads: int = 6, attn_drop: float = 0.0, proj_drop: float = 0.0, qk_scale=None, position_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        if idx == 0:
            h_sp, w_sp = self.split_size * 2, self.split_size * 2
        elif idx == 1:
            h_sp, w_sp = self.split_size // 2, self.split_size // 2
        elif idx == 2:
            h_sp, w_sp = self.split_size // 2, self.split_size * 2
        elif idx == 3:
            h_sp, w_sp = self.split_size * 2, self.split_size // 2
        else:
            raise ValueError(f"Invalid idx for WindowAttention: {idx}")

        self.h_sp = h_sp
        self.w_sp = w_sp
        window_size = [h_sp, w_sp]
        self.attn_drop = nn.Dropout(attn_drop)
        self.window_size = window_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def im2win(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.transpose(-2, -1).contiguous().view(b, c, h, w)
        x = img2windows(x, self.h_sp, self.w_sp)
        x = x.reshape(-1, self.h_sp * self.w_sp, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor], h: int, w: int, mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = qkv
        b, l, c = q.shape
        if l != h * w:
            raise ValueError("flatten img_tokens has wrong size")

        q = self.im2win(q, h, w)
        k = self.im2win(k, h, w)
        v = self.im2win(v, h, w)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        n_tokens = attn.shape[3]
        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b, n_w, self.num_heads, n_tokens, n_tokens) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n_tokens, n_tokens)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(-1, self.h_sp * self.w_sp, c)
        x = windows2img(x, self.h_sp, self.w_sp, h, w)
        return x


class WindowFrequencyModulation(nn.Module):
    def __init__(self, dim: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.ratio = 1
        self.complex_weight = nn.Parameter(
            torch.cat(
                (
                    torch.ones(self.window_size, self.window_size // 2 + 1, self.ratio * dim, 1, dtype=torch.float32),
                    torch.zeros(self.window_size, self.window_size // 2 + 1, self.ratio * dim, 1, dtype=torch.float32),
                ),
                dim=-1,
            )
        )

    def forward(self, x: torch.Tensor, h: int, w: int, spatial_size=None) -> torch.Tensor:
        b, l, c = x.shape
        x = x.view(b, h, w, self.ratio * c)
        x = rearrange(x, "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c", p1=self.window_size, p2=self.window_size)
        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(3, 4), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.window_size, self.window_size), dim=(3, 4), norm="ortho")

        x = rearrange(x, "b w1 w2 p1 p2 c -> b (w1 p1) (w2 p2) c")
        x = x.view(b, -1, c)
        return x


class Swin_FDWA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        window_size_fm: int = 16,
        shift_size: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.branch_num = 4
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList(
            [
                WindowAttention(
                    dim // self.branch_num,
                    idx=i,
                    split_size=window_size,
                    num_heads=num_heads // self.branch_num,
                    dim_out=dim // self.branch_num,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    position_bias=True,
                )
                for i in range(self.branch_num)
            ]
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.fm = WindowFrequencyModulation(dim, window_size_fm)

        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.norm2 = norm_layer(dim)

    def forward(self, x: torch.Tensor, spatial_size: tuple[int, int]) -> torch.Tensor:
        h, w = spatial_size
        b, l, c = x.shape

        qkv = self.qkv(self.norm1(x)).reshape(b, l, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.view(b, h, w, c)
        k = k.view(b, h, w, c)
        v = v.view(b, h, w, c)

        q = torch.cat(torch.chunk(q, self.branch_num, dim=-1), dim=0).view(-1, h * w, c // self.branch_num)
        k = torch.cat(torch.chunk(k, self.branch_num, dim=-1), dim=0).view(-1, h * w, c // self.branch_num)
        v = torch.cat(torch.chunk(v, self.branch_num, dim=-1), dim=0).view(-1, h * w, c // self.branch_num)

        x_out = []
        for i in range(self.branch_num):
            x_out.append(self.attns[i]((q, k, v), h, w))
        x_out = torch.cat(x_out, dim=2)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        x = x + self.drop_path(x_out)
        x = x + self.fm(self.ffn(self.norm2(x)), h, w)
        return x


class FAT_Block(nn.Module):
    def __init__(self, trans_dim: int, head_dim: int = 4, window_size: int = 4, window_size_fm: int = 4, drop_path: float = 0.01, type: str = "W", hyper: bool = False):
        super().__init__()
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.drop_path = drop_path
        self.type = type
        if self.type not in ("W", "SW"):
            raise ValueError("type must be 'W' or 'SW'")

        self.trans_block = Swin_FDWA(
            dim=trans_dim,
            num_heads=head_dim,
            window_size=window_size,
            window_size_fm=window_size_fm,
            shift_size=0 if (type == "W") else window_size // 2,
        )

        self.conv1_1 = nn.Conv2d(self.trans_dim, self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.trans_dim, self.trans_dim, 1, 1, 0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trans_x = self.conv1_1(x)
        b, c, h, w = trans_x.shape
        trans_x = Rearrange("b c h w -> b (h w) c")(trans_x)
        trans_x = self.trans_block(trans_x, (h, w))
        trans_x = Rearrange("b (h w) c -> b c h w", h=h, w=w)(trans_x)
        res = self.conv1_2(trans_x)
        return x + res

