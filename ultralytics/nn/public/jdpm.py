"""JDPM (Joint Domain Perception Module) used by C2f_JDPM."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class JDPM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        in_channels = channels

        self.conv1 = nn.Sequential(Conv(channels, in_channels))
        self.Dconv3 = nn.Sequential(Conv(in_channels, in_channels, act=False), Conv(in_channels, in_channels, k=3, d=3))
        self.Dconv5 = nn.Sequential(Conv(in_channels, in_channels, act=False), Conv(in_channels, in_channels, k=3, d=5))
        self.Dconv7 = nn.Sequential(Conv(in_channels, in_channels, act=False), Conv(in_channels, in_channels, k=3, d=7))
        self.Dconv9 = nn.Sequential(Conv(in_channels, in_channels, act=False), Conv(in_channels, in_channels, k=3, d=9))

        self.reduce = nn.Sequential(Conv(in_channels * 5, in_channels))
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=True),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = self.conv1(x)

        x3_s = self.Dconv3(x_input)
        x3_f = self.relu(
            self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(x3_s.float()).real) * torch.fft.fft2(x3_s.float()))))
        )
        x3 = torch.add(x3_s, x3_f)

        x5_s = self.Dconv5(x_input + x3)
        x5_f = self.relu(
            self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(x5_s.float()).real) * torch.fft.fft2(x5_s.float()))))
        )
        x5 = torch.add(x5_s, x5_f)

        x7_s = self.Dconv7(x_input + x5)
        x7_f = self.relu(
            self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(x7_s.float()).real) * torch.fft.fft2(x7_s.float()))))
        )
        x7 = torch.add(x7_s, x7_f)

        x9_s = self.Dconv9(x_input + x7)
        x9_f = self.relu(
            self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(x9_s.float()).real) * torch.fft.fft2(x9_s.float()))))
        )
        x9 = torch.add(x9_s, x9_f)

        return self.reduce(torch.cat((x3, x5, x7, x9, x_input), 1)) + x_input

