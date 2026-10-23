"""DTAB (Dilated Transformer Attention Block) used by C2f_DTAB."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

from ultralytics.nn.public.tsdn import LayerNorm


def _to(x: torch.Tensor) -> dict[str, object]:
    return {"device": x.device, "dtype": x.dtype}


def _expand_dim(t: torch.Tensor, dim: int, k: int) -> torch.Tensor:
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def _rel_to_abs(x: torch.Tensor) -> torch.Tensor:
    b, l, m = x.shape
    r = (m + 1) // 2
    col_pad = torch.zeros((b, l, 1), **_to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, "b l c -> b (l c)")
    flat_pad = torch.zeros((b, m - l), **_to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def _relative_logits_1d(q: torch.Tensor, rel_k: torch.Tensor) -> torch.Tensor:
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2
    logits = einsum("b x y d, r d -> b x y r", q, rel_k)
    logits = rearrange(logits, "b x y r -> (b x) y r")
    logits = _rel_to_abs(logits)
    logits = logits.reshape(b, h, w, r)
    logits = _expand_dim(logits, dim=2, k=r)
    return logits


class RelPosEmb(nn.Module):
    def __init__(self, block_size: int, rel_size: int, dim_head: int):
        super().__init__()
        height = width = rel_size
        scale = dim_head**-0.5
        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        block = self.block_size
        q = rearrange(q, "b (x y) c -> b x y c", x=block)
        rel_logits_w = _relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b x i y j-> b (x y) (i j)")

        q = rearrange(q, "b x y d -> b y x d")
        rel_logits_h = _relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, "b x i y j -> b (y x) (j i)")
        return rel_logits_w + rel_logits_h


class FixedPosEmb(nn.Module):
    def __init__(self, window_size: int, overlap_window_size: int):
        super().__init__()
        self.window_size = window_size
        self.overlap_window_size = overlap_window_size

        attention_mask_table = torch.zeros((window_size + overlap_window_size - 1), (window_size + overlap_window_size - 1))
        attention_mask_table[0::2, :] = float("-inf")
        attention_mask_table[:, 0::2] = float("-inf")
        attention_mask_table = attention_mask_table.view((window_size + overlap_window_size - 1) * (window_size + overlap_window_size - 1))

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten_1 = torch.flatten(coords, 1)

        coords_h = torch.arange(self.overlap_window_size)
        coords_w = torch.arange(self.overlap_window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten_2 = torch.flatten(coords, 1)

        relative_coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.overlap_window_size - 1
        relative_coords[:, :, 1] += self.overlap_window_size - 1
        relative_coords[:, :, 0] *= self.window_size + self.overlap_window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.attention_mask = nn.Parameter(
            attention_mask_table[relative_position_index.view(-1)].view(1, self.window_size**2, self.overlap_window_size**2),
            requires_grad=False,
        )

    def forward(self) -> torch.Tensor:
        return self.attention_mask


class DilatedMDTA(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)
        out = attn @ v

        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class DilatedOCA(nn.Module):
    def __init__(self, dim: int, window_size: int, overlap_ratio: float, num_heads: int, dim_head: int, bias: bool):
        super().__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size - window_size) // 2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(block_size=window_size, rel_size=window_size + (self.overlap_win_size - window_size), dim_head=self.dim_head)
        self.fixed_pos_emb = FixedPosEmb(window_size, self.overlap_win_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        qs = rearrange(qs, "b c (h p1) (w p2) -> (b h w) (p1 p2) c", p1=self.window_size, p2=self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, "b (c j) i -> (b i) j c", c=self.inner_dim), (ks, vs))

        qs, ks, vs = map(lambda t: rearrange(t, "b n (head c) -> (b head) n c", head=self.num_spatial_heads), (qs, ks, vs))
        qs = qs * self.scale
        spatial_attn = qs @ ks.transpose(-2, -1)
        spatial_attn = spatial_attn + self.rel_pos_emb(qs) + self.fixed_pos_emb()
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = spatial_attn @ vs
        out = rearrange(
            out,
            "(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)",
            head=self.num_spatial_heads,
            h=h // self.window_size,
            w=w // self.window_size,
            p1=self.window_size,
            p2=self.window_size,
        )
        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=3, stride=1, dilation=2, padding=2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=3, stride=1, dilation=2, padding=2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


class DTAB(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int = 4,
        overlap_ratio: float = 0.5,
        num_channel_heads: int = 4,
        num_spatial_heads: int = 2,
        spatial_dim_head: int = 16,
        ffn_expansion_factor: float = 1,
        bias: bool = False,
        layernorm_type: str = "BiasFree",
    ):
        super().__init__()

        self.spatial_attn = DilatedOCA(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)
        self.channel_attn = DilatedMDTA(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, layernorm_type)
        self.norm2 = LayerNorm(dim, layernorm_type)
        self.norm3 = LayerNorm(dim, layernorm_type)
        self.norm4 = LayerNorm(dim, layernorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x

