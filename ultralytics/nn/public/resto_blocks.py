import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

__all__ = [
    "DeepPoolLayer",
    "CubicAttention",
    "MultiShapeKernel",
    "MSMBlock",
    "CAB",
    "HDRAB",
    "RAB",
    "ShiftConv2d0",
    "ShiftConv2d1",
    "LFE",
]


# ===== MSM =====
class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation
        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Tanh()
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        pad = self.dilation * (kernel_size - 1) // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.inside_all = nn.Parameter(torch.zeros(inchannels, 1, 1), requires_grad=True)

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.bn(self.conv(low_filter))
        n, c, h, w = x.shape
        x_unfold = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).reshape(
            n, self.group, c // self.group, self.kernel_size ** 2, h * w
        )
        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)
        low_filter = self.act(low_filter)
        low_part = torch.sum(x_unfold * low_filter, dim=3).reshape(n, c, h, w)
        out_low = low_part * (self.inside_all + 1.0) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None, :, None, None]
        out_high = identity_input * (self.lamb_h[None, :, None, None] + 1.0)
        return out_low + out_high


class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=3, dilation=1, group=2, H=True):
        super().__init__()
        self.k = kernel
        pad = dilation * (kernel - 1) // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group * kernel, 1, 1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Tanh()
        self.inside_all = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        gap_kernel = (None, 1) if H else (1, None)
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        identity_input = x.clone()
        filter = self.conv(self.ap(x))
        n, c, h, w = x.shape
        x_unfold = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, -1, c // (filter.shape[1] // self.k), self.k, h * w)
        filter = filter.reshape(n, -1, self.k, filter.shape[2] * filter.shape[3]).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x_unfold * filter, dim=3).reshape(n, c, h, w)
        out_low = out * (self.inside_all + 1.0) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None, :, None, None]
        out_high = identity_input * (self.lamb_h[None, :, None, None] + 1.0)
        return out_low + out_high


class cubic_attention(nn.Module):
    def __init__(self, dim, group, dilation, kernel):
        super().__init__()
        self.H_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class MultiShapeKernel(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super().__init__()
        self.square_att = dynamic_filter(inchannels=dim, dilation=dilation, group=group, kernel_size=kernel_size)
        self.strip_att = cubic_attention(dim, group=group, dilation=dilation, kernel=kernel_size)

    def forward(self, x):
        return self.strip_att(x) + self.square_att(x)


class DeepPoolLayer(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.pools_sizes = [8, 4, 2]
        dilation = [3, 7, 9]
        self.pools = nn.ModuleList(nn.AvgPool2d(i, stride=i) for i in self.pools_sizes)
        self.convs = nn.ModuleList(nn.Conv2d(k, k, 3, 1, 1, bias=False) for _ in self.pools_sizes)
        self.dynas = nn.ModuleList(MultiShapeKernel(dim=k, kernel_size=3, dilation=dilation[j]) for j in range(len(self.pools_sizes)))
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        y_up = None
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x) + y_up))
            resl = resl + F.interpolate(y, x_size[2:], mode="bilinear", align_corners=True)
            if i != len(self.pools_sizes) - 1:
                y_up = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)
        return resl


class MSMBlock(nn.Module):
    """Wrapper for C3k/C3k2 MSM use."""
    def __init__(self, c):
        super().__init__()
        self.m = DeepPoolLayer(c)

    def forward(self, x):
        return self.m(x)


# ===== HDRAB / RAB =====
class CAB(nn.Module):
    def __init__(self, nc, reduction=8, bias=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(nc, nc // reduction, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc // reduction, nc, 1, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class HDRAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super().__init__()
        kernel_size = 3
        reduction_2 = 2
        self.cab = CAB(in_channels, 8, bias)
        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels // reduction_2, 1)
        self.conv1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size, padding=2, dilation=2, bias=bias)
        self.conv3 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size, padding=4, dilation=4, bias=bias)
        self.conv3_1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv2_1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size, padding=2, dilation=2, bias=bias)
        self.conv1_1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv_tail = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size, padding=1, dilation=1, bias=bias)
        self.conv1x1_2 = nn.Conv2d(in_channels // reduction_2, in_channels, 1)

    def forward(self, y):
        y_d = self.conv1x1_1(y)
        y1 = self.conv1(y_d)
        y1_1 = self.relu1(y1)
        y2 = self.conv2(y1_1)
        y2_1 = y2 + y_d
        y3 = self.conv3(y2_1)
        y3_1 = self.relu3(y3)
        y4 = self.conv4(y3_1)
        y4_1 = y4 + y2_1
        y5 = self.conv3_1(y4_1)
        y5_1 = self.relu3_1(y5)
        y6 = self.conv2_1(y5_1 + y3)
        y6_1 = y6 + y4_1
        y7 = self.conv1_1(y6_1 + y2_1)
        y7_1 = self.relu1_1(y7)
        y8 = self.conv_tail(y7_1 + y1)
        y8_1 = y8 + y6_1
        y9 = self.cab(self.conv1x1_2(y8_1))
        return y + y9


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAB(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 5)

    def forward(self, x):
        scale = torch.sigmoid(self.spatial(self.compress(x)))
        return x * scale


class RAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super().__init__()
        kernel_size = 3
        reduction_2 = 2
        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels // reduction_2, 1)
        self.conv1x1_2 = nn.Conv2d(in_channels // reduction_2, in_channels, 1)
        self.res = nn.Sequential(
            nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size, padding=1, bias=bias),
        )
        self.sab = SAB()

    def forward(self, x):
        x_d = self.conv1x1_1(x)
        x1 = x_d + self.res(x_d)
        x2 = x1 + self.res(x1)
        x3 = x2 + self.res(x2)
        x3_1 = x1 + x3
        x4 = x3_1 + self.res(x3_1)
        x4_1 = x_d + x4
        x5 = self.sab(self.conv1x1_2(x4_1))
        return x + x5


# ===== LFE =====
class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div
        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = torch.zeros((out_channels, inp_channels, 3, 3))
        mask[:, 0 * g:1 * g, 1, 2] = 1.0
        mask[:, 1 * g:2 * g, 1, 0] = 1.0
        mask[:, 2 * g:3 * g, 2, 1] = 1.0
        mask[:, 3 * g:4 * g, 0, 1] = 1.0
        mask[:, 4 * g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        return F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div
        conv3x3 = nn.Conv2d(inp_channels, out_channels, 1, 1, 0)
        mask = torch.zeros((out_channels, inp_channels, 1, 1))
        mask[:, 0 * g:1 * g] = conv3x3.weight[:, 0 * g:1 * g]
        mask[:, 1 * g:2 * g] = conv3x3.weight[:, 1 * g:2 * g]
        mask[:, 2 * g:3 * g] = conv3x3.weight[:, 2 * g:3 * g]
        mask[:, 3 * g:4 * g] = conv3x3.weight[:, 3 * g:4 * g]
        mask[:, 4 * g:] = conv3x3.weight[:, 4 * g:]
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        return F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=0)


class LFE(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type="relu"):
        super().__init__()
        self.act = nn.ReLU(inplace=True) if act_type == "relu" else nn.GELU() if act_type == "gelu" else None
        self.conv0 = ShiftConv2d0(inp_channels, out_channels * exp_ratio)
        self.conv1 = ShiftConv2d0(out_channels * exp_ratio, out_channels)

    def forward(self, x):
        y = self.conv0(x)
        if self.act:
            y = self.act(y)
        y = self.conv1(y)
        return y
