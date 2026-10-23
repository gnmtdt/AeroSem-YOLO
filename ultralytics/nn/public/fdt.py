"""FDT 系列公共模块（迁移自 RTDETR-main `nn/extra_modules/block.py`）。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .tsdn import LayerNorm

__all__ = ["FDT"]


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=bias)

        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=True),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),
            nn.Conv2d(dim // 16, dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.weight1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 16, 1, bias=True),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),
            nn.Conv2d(dim // 16, dim * 2, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_f = torch.abs(self.weight(torch.fft.fft2(x.float()).real) * torch.fft.fft2(x.float()))
        x_f_gelu = F.gelu(x_f) * x_f

        x_s = self.dwconv1(x)
        x_s_gelu = F.gelu(x_s) * x_s

        x_f = torch.fft.fft2(torch.cat((x_f_gelu, x_s_gelu), 1))
        x_f = torch.abs(torch.fft.ifft2(self.weight1(x_f.real) * x_f))

        x_s = self.dwconv2(torch.cat((x_f_gelu, x_s_gelu), 1))
        out = self.project_out(torch.cat((x_f, x_s), 1))
        return out


class GSA(nn.Module):
    def __init__(self, channels, num_heads=8, bias=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=bias
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        if prev_atns is None:
            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
            k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
            v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = self.act(attn)
            out = attn @ v
            y = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
            y = rearrange(y, "b (head c) h w -> b (c head) h w", head=self.num_heads, h=h, w=w)
            y = self.project_out(y)
            return y, attn

        attn = prev_atns
        v = rearrange(x, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        out = attn @ v
        y = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        y = rearrange(y, "b (head c) h w -> b (c head) h w", head=self.num_heads, h=h, w=w)
        y = self.project_out(y)
        return y


class RSA(nn.Module):
    def __init__(self, channels, num_heads, shifts=1, window_sizes=4, bias=False):
        super().__init__()
        self.channels = channels
        self.shifts = shifts
        self.window_sizes = window_sizes

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=bias
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        if prev_atns is None:
            wsize = self.window_sizes
            x_ = x
            if self.shifts > 0:
                x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
            qkv = self.qkv_dwconv(self.qkv(x_))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize)
            k = rearrange(k, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize)
            v = rearrange(v, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q.transpose(-2, -1) @ k) * self.temperature
            attn = self.act(attn)
            out = v @ attn
            out = rearrange(out, "b (h w) (dh dw) c-> b c (h dh) (w dw)", h=h // wsize, w=w // wsize, dh=wsize, dw=wsize)
            if self.shifts > 0:
                out = torch.roll(out, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
            y = self.project_out(out)
            return y, attn

        wsize = self.window_sizes
        if self.shifts > 0:
            x = torch.roll(x, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
        atn = prev_atns
        v = rearrange(x, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize)
        y_ = v @ atn
        y_ = rearrange(y_, "b (h w) (dh dw) c-> b c (h dh) (w dw)", h=h // wsize, w=w // wsize, dh=wsize, dw=wsize)
        if self.shifts > 0:
            y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
        return self.project_out(y_)


class FDT(nn.Module):
    def __init__(self, inp_channels, num_heads=4, window_sizes=4, shifts=0, shared_depth=1, ffn_expansion_factor=2.66):
        super().__init__()
        self.shared_depth = shared_depth

        modules_ffd = {}
        modules_att = {}
        modules_norm = {}
        for i in range(shared_depth):
            modules_ffd[f"ffd{i}"] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modules_att[f"att_{i}"] = RSA(channels=inp_channels, num_heads=num_heads, shifts=shifts, window_sizes=window_sizes)
            modules_norm[f"norm_{i}"] = LayerNorm(inp_channels, "WithBias")
            modules_norm[f"norm_{i + 2}"] = LayerNorm(inp_channels, "WithBias")
        self.modules_ffd = nn.ModuleDict(modules_ffd)
        self.modules_att = nn.ModuleDict(modules_att)
        self.modules_norm = nn.ModuleDict(modules_norm)

        modulec_ffd = {}
        modulec_att = {}
        modulec_norm = {}
        for i in range(shared_depth):
            modulec_ffd[f"ffd{i}"] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modulec_att[f"att_{i}"] = GSA(channels=inp_channels, num_heads=num_heads)
            modulec_norm[f"norm_{i}"] = LayerNorm(inp_channels, "WithBias")
            modulec_norm[f"norm_{i + 2}"] = LayerNorm(inp_channels, "WithBias")
        self.modulec_ffd = nn.ModuleDict(modulec_ffd)
        self.modulec_att = nn.ModuleDict(modulec_att)
        self.modulec_norm = nn.ModuleDict(modulec_norm)

    def forward(self, x):
        atn = None
        for i in range(self.shared_depth):
            if i == 0:
                x_, atn = self.modules_att[f"att_{i}"](self.modules_norm[f"norm_{i}"](x), None)
                x = self.modules_ffd[f"ffd{i}"](self.modules_norm[f"norm_{i + 2}"](x_ + x)) + x_
            else:
                x_ = self.modules_att[f"att_{i}"](self.modules_norm[f"norm_{i}"](x), atn)
                x = self.modules_ffd[f"ffd{i}"](self.modules_norm[f"norm_{i + 2}"](x_ + x)) + x_

        for i in range(self.shared_depth):
            if i == 0:
                x_, atn = self.modulec_att[f"att_{i}"](self.modulec_norm[f"norm_{i}"](x), None)
                x = self.modulec_ffd[f"ffd{i}"](self.modulec_norm[f"norm_{i + 2}"](x_ + x)) + x_
            else:
                x_ = self.modulec_att[f"att_{i}"](self.modulec_norm[f"norm_{i}"](x), atn)
                x = self.modulec_ffd[f"ffd{i}"](self.modulec_norm[f"norm_{i + 2}"](x_ + x)) + x_
        return x

