"""ETB (Frequency-Spatial Entanglement block) used by C2f_ETB."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ultralytics.nn.public.tsdn import LayerNorm


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = torch.abs(self.weight(torch.fft.fft2(x.float()).real) * torch.fft.fft2(x.float()))
        x_f_gelu = F.gelu(x_f) * x_f

        x_s = self.dwconv1(x)
        x_s_gelu = F.gelu(x_s) * x_s

        x_f = torch.fft.fft2(torch.cat((x_f_gelu, x_s_gelu), 1))
        x_f = torch.abs(torch.fft.ifft2(self.weight1(x_f.real) * x_f))

        x_s = self.dwconv2(torch.cat((x_f_gelu, x_s_gelu), 1))
        out = self.project_out(torch.cat((x_f, x_s), 1))
        return out


def custom_complex_normalization(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)
    return torch.complex(norm_real, norm_imag)


class Attention_F(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=True),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),
            nn.Conv2d(dim // 16, dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        q_f = torch.fft.fft2(x.float())
        k_f = torch.fft.fft2(x.float())
        v_f = torch.fft.fft2(x.float())

        q_f = rearrange(q_f, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k_f = rearrange(k_f, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v_f = rearrange(v_f, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q_f = torch.nn.functional.normalize(q_f, dim=-1)
        k_f = torch.nn.functional.normalize(k_f, dim=-1)
        attn_f = (q_f @ k_f.transpose(-2, -1)) * self.temperature
        attn_f = custom_complex_normalization(attn_f, dim=-1)

        out_f = torch.abs(torch.fft.ifft2(attn_f @ v_f))
        out_f = rearrange(out_f, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out_f_l = torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(x.float()).real) * torch.fft.fft2(x.float())))
        out = self.project_out(torch.cat((out_f, out_f_l), 1))
        return out


class Attention_S(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv1conv_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.qkv2conv_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.qkv3conv_1 = nn.Conv2d(dim, dim, kernel_size=1)

        self.qkv1conv_3 = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, groups=dim // 2, bias=bias)
        self.qkv2conv_3 = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, groups=dim // 2, bias=bias)
        self.qkv3conv_3 = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, groups=dim // 2, bias=bias)

        self.qkv1conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim // 2, bias=bias)
        self.qkv2conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim // 2, bias=bias)
        self.qkv3conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim // 2, bias=bias)

        self.conv_3 = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, groups=dim // 2, bias=bias)
        self.conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim // 2, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q_s = torch.cat((self.qkv1conv_3(self.qkv1conv_1(x)), self.qkv1conv_5(self.qkv1conv_1(x))), 1)
        k_s = torch.cat((self.qkv2conv_3(self.qkv2conv_1(x)), self.qkv2conv_5(self.qkv2conv_1(x))), 1)
        v_s = torch.cat((self.qkv3conv_3(self.qkv3conv_1(x)), self.qkv3conv_5(self.qkv3conv_1(x))), 1)

        q_s = rearrange(q_s, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k_s = rearrange(k_s, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v_s = rearrange(v_s, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = attn_s.softmax(dim=-1)
        out_s = attn_s @ v_s
        out_s = rearrange(out_s, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out_s_l = torch.cat((self.conv_3(x), self.conv_5(x)), 1)
        out = self.project_out(torch.cat((out_s, out_s_l), 1))
        return out


class ETB(nn.Module):
    def __init__(self, dim: int = 128, num_heads: int = 4, ffn_expansion_factor: float = 4.0, bias: bool = False, layernorm_type: str = "WithBias"):
        super().__init__()
        self.norm1 = LayerNorm(dim, layernorm_type)
        self.attn_s = Attention_S(dim, num_heads, bias)
        self.attn_f = Attention_F(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layernorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.add(self.attn_f(self.norm1(x)), self.attn_s(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x

