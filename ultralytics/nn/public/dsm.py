import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DSM_SpatialGate", "DSM_LocalAttention", "DualDomainSelectionMechanism"]


class DSM_SpatialGate(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(channel, channel // 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel // 16),
            nn.ReLU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(channel // 16, channel // 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid(),
        )
        self.dw1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.dw2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

    def forward(self, x):
        out = self.compress(x)
        out = self.spatial(out)
        out = self.dw1(x) * out + self.dw2(x)
        return out


class DSM_LocalAttention(nn.Module):
    def __init__(self, channel, p):
        super().__init__()
        self.num_patch = 2 ** p
        self.sig = nn.Sigmoid()
        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        out = x - torch.mean(x, dim=(2, 3), keepdim=True)
        return self.a * out * x + self.b * x


class DualDomainSelectionMechanism(nn.Module):
    """Dual-domain selection (spatial + local) attention."""
    def __init__(self, channel):
        super().__init__()
        self.spatial_gate = DSM_SpatialGate(channel)
        pyramid = 1
        layers = [DSM_LocalAttention(channel, p=i) for i in range(pyramid - 1, -1, -1)]
        self.local_attention = nn.Sequential(*layers)
        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        out = self.spatial_gate(x)
        out = self.local_attention(out)
        return self.a * out + self.b * x
