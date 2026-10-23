"""FreqFormer / FCA 公共模块（迁移自 RTDETR-main `nn/extra_modules/block.py`）。

该文件提供 FCA 及其依赖的注意力/投影组件，供 `C2f_FCA` 变体组装使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from timm.layers import DropPath

__all__ = ["FCA"]


def img2windows(img: torch.Tensor, H_sp: int, W_sp: int) -> torch.Tensor:
    """Input: (B, C, H, W) -> Output: (B', N, C)."""
    b, c, h, w = img.shape
    if h % H_sp != 0 or w % W_sp != 0:
        raise ValueError(f"img2windows expects H%H_sp==0 and W%W_sp==0, got {(h, w)} vs {(H_sp, W_sp)}")
    img_reshape = img.view(b, c, h // H_sp, H_sp, w // W_sp, W_sp)
    return img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, c)


def windows2img(img_splits_hw: torch.Tensor, H_sp: int, W_sp: int, H: int, W: int) -> torch.Tensor:
    """Input: (B', N, C) -> Output: (B, H, W, C)."""
    denom = (H * W) / H_sp / W_sp
    if denom <= 0:
        raise ValueError(f"windows2img invalid denom={denom}")
    B = int(img_splits_hw.shape[0] / denom)
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    return img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class FrequencyProjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(dim, dim // 2, 1, 1, 0)
        self.act = nn.GELU()
        self.res_2 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.GELU(),
        )
        self.conv_out = nn.Conv2d(dim // 2, dim, 1, 1, 0)

    def forward(self, x):
        res = x
        x = self.conv_1(x)
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((self.act(x1), self.res_2(x2)), dim=1)
        out = self.conv_out(out)
        return out + res


class ChannelProjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.pro_in = nn.Conv2d(dim, dim // 6, 1, 1, 0)
        self.CI1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim // 6, dim // 6, kernel_size=1))
        self.CI2 = nn.Sequential(
            nn.Conv2d(dim // 6, dim // 6, kernel_size=3, stride=1, padding=1, groups=dim // 6),
            nn.Conv2d(dim // 6, dim // 6, 7, stride=1, padding=9, groups=dim // 6, dilation=3),
            nn.Conv2d(dim // 6, dim // 6, kernel_size=1),
        )
        self.pro_out = nn.Conv2d(dim // 6, dim, kernel_size=1)

    def forward(self, x):
        x = self.pro_in(x)
        res = x
        ci1 = self.CI1(x)
        ci2 = self.CI2(x)
        return self.pro_out(res * ci1 * ci2)


class SpatialProjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.pro_in = nn.Conv2d(dim, dim // 6, 1, 1, 0)
        self.dwconv = nn.Conv2d(dim // 6, dim // 6, kernel_size=3, stride=1, padding=1, groups=dim // 6)
        self.pro_out = nn.Conv2d(dim // 12, dim, kernel_size=1)

    def forward(self, x):
        x = self.pro_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.pro_out(x)


class DynamicPosBias(nn.Module):
    def __init__(self, dim: int, num_heads: int, residual: bool):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos2 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos3 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.num_heads))

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            return self.pos3(pos)
        return self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))


class Spatial_Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        idx: int,
        split_size=(8, 8),
        dim_out: int | None = None,
        num_heads: int = 6,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_scale=None,
        position_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = list(split_size)
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            raise ValueError(f"Spatial_Attention idx must be 0/1, got {idx}")
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer("rpe_biases", biases)

            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        return x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

    def forward(self, qkv, H, W, mask=None):
        q, k, v = qkv[0], qkv[1], qkv[2]
        B, L, C = q.shape
        if L != H * W:
            raise ValueError("flatten img_tokens has wrong size")

        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        Nw = attn.shape[3]
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, Nw, Nw) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, Nw, Nw)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
        return windows2img(x, self.H_sp, self.W_sp, H, W)


class Spatial_Frequency_Attention(nn.Module):
    def __init__(self, dim, num_heads, reso=64, split_size=(8, 8), shift_size=(1, 2), qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, b_idx=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = list(split_size)
        self.shift_size = list(shift_size)
        self.b_idx = b_idx
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.hf = nn.Linear(dim, dim, bias=qkv_bias)

        if not (0 <= self.shift_size[0] < self.split_size[0] and 0 <= self.shift_size[1] < self.split_size[1]):
            raise ValueError(f"shift_size must satisfy 0<=shift<split, got shift={self.shift_size}, split={self.split_size}")

        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.dw_block = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0), nn.Conv2d(dim, dim, 3, 1, 1, groups=dim))

        self.attns = nn.ModuleList(
            [
                Spatial_Attention(
                    dim // 2,
                    idx=i,
                    split_size=self.split_size,
                    num_heads=num_heads // 2,
                    dim_out=dim // 2,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    position_bias=True,
                )
                for i in range(self.branch_num)
            ]
        )

        if self.b_idx > 0 and (self.b_idx - 2) % 4 == 0:
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.channel_projection = ChannelProjection(dim)
        self.spatial_projection = SpatialProjection(dim)
        self.frequency_projection = FrequencyProjection(dim)

    def calculate_mask(self, H, W):
        img_mask_0 = torch.zeros((1, H, W, 1))
        img_mask_1 = torch.zeros((1, H, W, 1))
        h_slices_0 = (slice(0, -self.split_size[0]), slice(-self.split_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]), slice(-self.split_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]), slice(-self.split_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]), slice(-self.split_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1], self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1], 1)
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0], self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0], 1)
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))
        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        B, L, C = x.shape
        if L != H * W:
            raise ValueError("flatten img_tokens has wrong size")

        hf = self.hf(x).transpose(-2, -1).contiguous().view(B, C, H, W)
        hf = self.frequency_projection(hf)

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        v = qkv[2].transpose(-2, -1).contiguous().view(B, C, H, W)

        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3 * B, H, W, C).permute(0, 3, 1, 2)
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1)
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        if self.b_idx > 0 and (self.b_idx - 2) % 4 == 0:
            qkv = qkv.view(3, B, _H, _W, C)
            qkv_0 = torch.roll(qkv[:, :, :, :, : C // 2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3)).view(3, B, _L, C // 2)
            qkv_1 = torch.roll(qkv[:, :, :, :, C // 2 :], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3)).view(3, B, _L, C // 2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            x1 = self.attns[0](qkv[:, :, :, : C // 2], _H, _W)[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = self.attns[1](qkv[:, :, :, C // 2 :], _H, _W)[:, :H, :W, :].reshape(B, L, C // 2)
            attened_x = torch.cat([x1, x2], dim=2)

        conv_x = self.dw_block(v)
        channel_map = self.channel_projection(conv_x)
        conv_x = conv_x + channel_map
        hf = hf + channel_map
        channel_map = reduce(channel_map, "b c h w -> b c 1 1", "mean").permute(0, 2, 3, 1).contiguous().view(B, 1, C)

        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_projection(attention_reshape)
        hf = hf + attention_reshape

        attened_x = attened_x * torch.sigmoid(channel_map) * torch.sigmoid(
            reduce(hf, "b c h w -> b c 1 1", "mean").permute(0, 2, 3, 1).contiguous().view(B, 1, C)
        )
        conv_x = torch.sigmoid(spatial_map) * conv_x * torch.sigmoid(hf)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x_out = attened_x + conv_x + hf.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        x_out = self.proj(x_out)
        return self.proj_drop(x_out)


class Channel_Transposed_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.channel_projection = ChannelProjection(dim)
        self.spatial_projection = SpatialProjection(dim)
        self.dwconv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        conv_x = self.dwconv(v_)

        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        channel_map = self.channel_projection(attention_reshape)
        attened_x = attened_x + channel_map.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        channel_map = reduce(channel_map, "b c h w -> b c 1 1", "mean")

        spatial_map = self.spatial_projection(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, C)
        attened_x = attened_x * torch.sigmoid(spatial_map)
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x_out = attened_x + conv_x
        x_out = self.proj(x_out)
        return self.proj_drop(x_out)


class FrequencyGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0), nn.Conv2d(dim, dim, 3, 1, 1, groups=dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x1, x2 = x.chunk(2, dim=-1)
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1, -2).contiguous()
        return x1 * x2


class DFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fg = FrequencyGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fg(x, H, W)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class FCA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        reso=64,
        split_size=(2, 4),
        shift_size=(1, 2),
        expansion_factor=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        b_idx=0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        if b_idx % 2 == 0:
            self.attn = Spatial_Frequency_Attention(
                dim,
                num_heads=num_heads,
                reso=reso,
                split_size=split_size,
                shift_size=shift_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                b_idx=b_idx,
            )
        else:
            self.attn = Channel_Transposed_Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn = DFFN(in_features=dim, hidden_features=ffn_hidden_dim, out_features=dim, act_layer=act_layer)

    def forward(self, x):
        b, n, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x.transpose(1, 2).reshape((b, n, H, W))

