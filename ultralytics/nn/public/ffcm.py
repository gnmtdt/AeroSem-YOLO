import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

__all__ = ["FourierUnit", "Freq_Fusion", "Fused_Fourier_Conv_Mixer"]


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.groups = groups
        self.conv = Conv(in_channels * 2, out_channels * 2, 1, g=groups, act=nn.ReLU(inplace=True))

    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm="ortho")
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv(ffted)
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm="ortho")
        return output


class Freq_Fusion(nn.Module):
    def __init__(self, dim, kernel_size=[1, 3, 5, 7], se_ratio=4, local_size=8, scale_ratio=2, spilt_num=4):
        super().__init__()
        self.dim = dim
        self.conv_init_1 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.conv_init_2 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.conv_mid = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU())
        self.FFC = FourierUnit(self.dim * 2, self.dim * 2)
        self.bn = nn.BatchNorm2d(dim * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)
        x0 = torch.cat([x_1, x_2], dim=1)
        x = self.FFC(x0) + x0
        x = self.relu(self.bn(x))
        return x


class Fused_Fourier_Conv_Mixer(nn.Module):
    def __init__(self, dim, token_mixer_for_gloal=Freq_Fusion, mixer_kernel_size=[1, 3, 5, 7], local_size=8):
        super().__init__()
        self.dim = dim
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, kernel_size=mixer_kernel_size, se_ratio=8, local_size=local_size)
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode="reflect"),
            nn.GELU(),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv_init = nn.Sequential(nn.Conv2d(dim, dim * 2, 1), nn.GELU())
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim, padding_mode="reflect"),
            nn.GELU(),
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=2, groups=self.dim, padding_mode="reflect"),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local_1 = self.dw_conv_1(x[0])
        x_local_2 = self.dw_conv_2(x[0])
        x_gloal = self.mixer_gloal(torch.cat([x_local_1, x_local_2], dim=1))
        x = self.ca_conv(x_gloal)
        x = self.ca(x) * x
        return x
