"""SFHF blocks (SFHformer ECCV2024) used by C2f_SFHF."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SFHF_FFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.dim_sp = dim // 2

        self.conv_init = nn.Sequential(nn.Conv2d(dim, dim * 2, 1))
        self.conv1_1 = nn.Sequential(nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1, groups=self.dim_sp))
        self.conv1_2 = nn.Sequential(nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2, groups=self.dim_sp))
        self.conv1_3 = nn.Sequential(nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3, groups=self.dim_sp))

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(nn.Conv2d(dim * 2, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_init(x)
        parts = list(torch.split(x, self.dim_sp, dim=1))
        parts[1] = self.conv1_1(parts[1])
        parts[2] = self.conv1_2(parts[2])
        parts[3] = self.conv1_3(parts[3])
        x = torch.cat(parts, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)
        return x


class TokenMixer_For_Local(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        self.c_dilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.c_dilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        cd1 = self.c_dilated_1(x1)
        cd2 = self.c_dilated_2(x2)
        return torch.cat([cd1, cd2], dim=1)


class SFHF_FourierUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 4):
        super().__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels * 2)

        self.fdc = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2 * self.groups,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=True,
        )
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=self.groups, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1),
        )
        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, stride=1, groups=in_channels * 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm="ortho")
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, "b c h w d -> b (c d) h w").contiguous()
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted
        dy_weight = self.weight(ffted)
        ffted = self.fdc(ffted).view(batch, self.groups, 2 * c, h, -1)
        ffted = torch.einsum("ijkml,ijml->ikml", ffted, dy_weight)
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, "b (c d) h w -> b c h w d", d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm="ortho")
        return output


class TokenMixer_For_Gloal(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(nn.Conv2d(dim, dim * 2, 1), nn.GELU())
        self.conv_fina = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU())
        self.ffc = SFHF_FourierUnit(self.dim * 2, self.dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_init(x)
        x0 = x
        x = self.ffc(x)
        x = self.conv_fina(x + x0)
        return x


class SFHF_Mixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mixer_local = TokenMixer_For_Local(dim=self.dim)
        self.mixer_gloal = TokenMixer_For_Gloal(dim=self.dim)

        self.ca_conv = nn.Sequential(nn.Conv2d(2 * dim, dim, 1))
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * dim, 2 * dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim // 2, 2 * dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(nn.Conv2d(dim, 2 * dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_init(x)
        parts = list(torch.split(x, self.dim, dim=1))
        x_local = self.mixer_local(parts[0])
        x_gloal = self.mixer_gloal(parts[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)
        return x


class SFHF_Block(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mixer = SFHF_Mixer(dim=self.dim)
        self.ffn = SFHF_FFN(dim=self.dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * self.gamma + copy
        return x

