"""DarkIR blocks (DBlock) used by C2f_DBlock."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        ctx.eps = eps
        _, c, _, _ = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, c, 1, 1) * y + bias.view(1, c, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        eps = ctx.eps
        _, c, _, _ = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, c, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        gw = (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0)
        gb = grad_output.sum(dim=3).sum(dim=2).sum(dim=0)
        return gx, gw, gb, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Branch(nn.Module):
    def __init__(self, c: int, dw_expand: int, dilation: int = 1):
        super().__init__()
        dw_channel = dw_expand * c
        self.branch = nn.Sequential(
            nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=dilation, stride=1, groups=dw_channel, bias=True, dilation=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class FreMLP(nn.Module):
    def __init__(self, nc: int, expand: int = 2):
        super().__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_dtype = x.dtype
        x_freq = torch.fft.rfft2(x.float(), norm="backward")
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag.to(x_dtype))
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out.float(), s=(h, w), norm="backward")
        return x_out.to(x_dtype)


class DBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, dilations: list[int] | tuple[int, ...] = (1, 2, 3), extra_depth_wise: bool = False):
        super().__init__()
        self.dw_channel = dw_expand * c

        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation=1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity()

        self.branches = nn.ModuleList([Branch(self.dw_channel, dw_expand=1, dilation=d) for d in dilations])

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dw_channel // 2, self.dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(self.dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation=1)

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor, adapter=None) -> torch.Tensor:
        y = inp
        x = self.norm1(inp)
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x

        x = self.conv4(self.norm2(y))
        x = self.sg2(x)
        x = self.conv5(x)
        x = y + x * self.gamma
        return x 


class EBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, dilations: list[int] | tuple[int, ...] = (1,), extra_depth_wise: bool = False):
        super().__init__()
        self.dw_channel = dw_expand * c
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity()
        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation=1)

        self.branches = nn.ModuleList([Branch(c, dw_expand, dilation=d) for d in dilations])
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dw_channel // 2, self.dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(self.dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation=1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc=c, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        y = inp
        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x

        x_step2 = self.norm2(y)
        x_freq = self.freq(x_step2)
        x = y * x_freq
        x = y + x * self.gamma
        return x

