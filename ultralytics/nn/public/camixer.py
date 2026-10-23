import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

__all__ = ["CAMixer"]


def flow_warp(x, flow, interp_mode="bilinear", padding_mode="zeros", align_corners=True):
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)
    return output


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N
    x = x.reshape(B * N, C)
    x[idx1.reshape(-1)] = x1.reshape(-1, C)
    x[idx2.reshape(-1)] = x2.reshape(-1, C)
    return x.reshape(B, N, C)


class PredictorLG(nn.Module):
    def __init__(self, dim, window_size=8, k=4, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.window_size = window_size
        cdim = dim + 2
        embed_dim = window_size ** 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim // 4, 1),
            LayerNorm(cdim // 4),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim // 4, cdim // 8, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cdim // 8, 2, 1),
        )
        self.out_mask = nn.Sequential(
            nn.Linear(embed_dim, window_size),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(window_size, 2),
            nn.Softmax(dim=-1),
        )
        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim // 4, dim, 1),
            nn.Sigmoid(),
        )
        self.out_SA = nn.Sequential(nn.Conv2d(cdim // 4, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):
        x = self.in_conv(input_x)
        offsets = self.out_offsets(x).tanh().mul(8.0)
        ca = self.out_CA(x)
        sa = self.out_SA(x)
        x = torch.mean(x, keepdim=True, dim=1)
        x = rearrange(x, "b c (h dh) (w dw) -> b (h w) (dh dw c)", dh=self.window_size, dw=self.window_size)
        B, N, C = x.size()
        pred_score = self.out_mask(x)
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]
        if self.training or train_mode:
            return mask, offsets, ca, sa
        score = pred_score[:, :, 0]
        _, N = score.shape
        r = torch.mean(mask, dim=(0, 1)) * 1.0
        num_keep_node = N if self.ratio == 1 else min(int(N * r * 2 * self.ratio), N)
        idx = torch.argsort(score, dim=1, descending=True)
        idx1 = idx[:, :num_keep_node]
        idx2 = idx[:, num_keep_node:]
        return [idx1, idx2], offsets, ca, sa


class CAMixer(nn.Module):
    def __init__(self, dim, window_size=8, bias=True, is_deformable=True, ratio=0.5):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.is_deformable = is_deformable
        self.ratio = ratio
        k = 3
        d = 2
        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.project_q = nn.Linear(dim, dim, bias=bias)
        self.project_k = nn.Linear(dim, dim, bias=bias)
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim),
            nn.Conv2d(dim, dim, k, stride=1, padding=((k // 2) * d), groups=dim, dilation=d),
        )
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias)
        self.act = nn.GELU()
        self.route = PredictorLG(dim, window_size, ratio=ratio)

    def forward(self, x, condition_global=None, mask=None, train_mode=False):
        N, C, H, W = x.shape
        v = self.project_v(x)
        condition_wind = torch.stack(
            torch.meshgrid(torch.linspace(-1, 1, self.window_size), torch.linspace(-1, 1, self.window_size))
        ).type_as(x).unsqueeze(0).repeat(N, 1, H // self.window_size, W // self.window_size)
        _condition = torch.cat([v, condition_wind], dim=1) if condition_global is None else torch.cat([v, condition_global, condition_wind], dim=1)
        mask, offsets, ca, sa = self.route(_condition, ratio=self.ratio, train_mode=train_mode)
        q = x
        k = x + flow_warp(x, offsets.permute(0, 2, 3, 1), interp_mode="bilinear", padding_mode="border")
        qk = torch.cat([q, k], dim=1)
        vs = v * sa
        v = rearrange(v, "b c (h dh) (w dw) -> b (h w) (dh dw c)", dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs, "b c (h dh) (w dw) -> b (h w) (dh dw c)", dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk, "b c (h dh) (w dw) -> b (h w) (dh dw c)", dh=self.window_size, dw=self.window_size)
        if self.training or train_mode:
            N_ = v.shape[1]
            v1, v2 = v * mask, vs * (1 - mask)
            qk1 = qk * mask
        else:
            idx1, idx2 = mask
            _, N_ = idx1.shape
            v1, v2 = batch_index_select(v, idx1), batch_index_select(vs, idx2)
            qk1 = batch_index_select(qk, idx1)
        v1 = rearrange(v1, "b n (dh dw c) -> (b n) (dh dw) c", n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1, "b n (dh dw c) -> b (n dh dw) c", n=N_, dh=self.window_size, dw=self.window_size)
        q1, k1 = torch.chunk(qk1, 2, dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1, "b (n dh dw) c -> (b n) (dh dw) c", n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1, "b (n dh dw) c -> (b n) (dh dw) c", n=N_, dh=self.window_size, dw=self.window_size)
        attn = (q1 @ k1.transpose(-2, -1)).softmax(dim=-1)
        f_attn = attn @ v1
        f_attn = rearrange(f_attn, "(b n) (dh dw) c -> b n (dh dw c)", b=N, n=N_, dh=self.window_size, dw=self.window_size)
        if not (self.training or train_mode):
            attn_out = batch_index_fill(v.clone(), f_attn, v2.clone(), idx1, idx2)
        else:
            attn_out = f_attn + v2
        attn_out = rearrange(attn_out, "b (h w) (dh dw c) -> b c (h dh) (w dw)", dh=self.window_size, dw=self.window_size, h=H // self.window_size)
        return self.project_out(attn_out * ca)
