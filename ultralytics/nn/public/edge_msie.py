import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.public.dsm import DualDomainSelectionMechanism

__all__ = [
    "EdgeEnhancer",
    "MutilScaleEdgeInformationEnhance",
    "MutilScaleEdgeInformationSelect",
]


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class MutilScaleEdgeInformationEnhance(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()
        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    Conv(inc, inc // len(bins), 1),
                    Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins)),
                )
                for bin in bins
            ]
        )
        self.ees = nn.ModuleList([EdgeEnhancer(inc // len(bins)) for _ in bins])
        self.local_conv = Conv(inc, inc, 3)
        self.final_conv = Conv(inc * 2, inc)

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode="bilinear", align_corners=True)))
        return self.final_conv(torch.cat(out, 1))


class MutilScaleEdgeInformationSelect(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()
        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    Conv(inc, inc // len(bins), 1),
                    Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins)),
                )
                for bin in bins
            ]
        )
        self.ees = nn.ModuleList([EdgeEnhancer(inc // len(bins)) for _ in bins])
        self.local_conv = Conv(inc, inc, 3)
        self.dsm = DualDomainSelectionMechanism(inc * 2)
        self.final_conv = Conv(inc * 2, inc)

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode="bilinear", align_corners=True)))
        return self.final_conv(self.dsm(torch.cat(out, 1)))
