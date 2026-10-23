"""
Wavelet Transform Convolution (WTConv2d)
源自 upstream ultralytics-yolo11-main/ultralytics/nn/extra_modules/wtconv2d.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WTConv2d"]


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=[1, 3, 5], ssm_ratio=0.25, forward_type="D", bias=False, wt_levels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernels = kernels
        self.forward_type = forward_type
        self.bias = bias
        self.wt_levels = wt_levels

        self.dwt = nn.Conv2d(in_channels, in_channels * 4, kernel_size=2, stride=2, groups=in_channels, bias=False)
        self.idwt = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, groups=out_channels, bias=False)
        self.conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels * 4,
                    out_channels * 4,
                    kernel_size=k,
                    padding=k // 2,
                    groups=in_channels * 4,
                    bias=bias,
                )
                for k in kernels
            ]
        )
        self.proj = nn.Conv2d(out_channels * 4 * len(kernels), out_channels * 4, kernel_size=1, bias=bias)

    def _dwt(self, x):
        ll = x[:, :, 0::2, 0::2]
        lh = x[:, :, 0::2, 1::2]
        hl = x[:, :, 1::2, 0::2]
        hh = x[:, :, 1::2, 1::2]
        return torch.cat([ll, lh, hl, hh], dim=1)

    def _idwt(self, x):
        c = x.shape[1] // 4
        ll, lh, hl, hh = torch.split(x, c, dim=1)
        x0 = torch.zeros((x.size(0), c, x.size(2) * 2, x.size(3) * 2), device=x.device, dtype=x.dtype)
        x0[:, :, 0::2, 0::2] = ll
        x0[:, :, 0::2, 1::2] = lh
        x0[:, :, 1::2, 0::2] = hl
        x0[:, :, 1::2, 1::2] = hh
        return x0

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        # Pad odd spatial dims so DWT slicing produces equal-size sub-bands
        if H % 2 != 0 or W % 2 != 0:
            pad_h = H % 2
            pad_w = W % 2
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        # 2D DWT
        x = self._dwt(x)
        outs = [conv(x) for conv in self.conv]
        x = torch.cat(outs, dim=1)
        x = self.proj(x)
        # 2D IDWT
        x = self._idwt(x)
        # Crop back to original spatial size
        if H % 2 != 0 or W % 2 != 0:
            x = x[:, :, :H, :W]
        return x
