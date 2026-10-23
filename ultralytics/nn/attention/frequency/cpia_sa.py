"""
CPIA_SA - Complex Phase Inversion Attention with Spatial Attention (复数相位注意力)

论文: CPIA_SA
期刊/会议: ACM MM 2025
论文链接: https://arxiv.org/abs/2504.16455
依赖: einops, torch.fft
"""

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

try:
    from einops import rearrange
except ImportError:
    rearrange = None

__all__ = ['ComplexFFT', 'ComplexIFFT', 'Stage2_fft', 'CPIA_SA']


def _rearrange_check():
    """检查 einops 是否可用。"""
    if rearrange is None:
        raise ImportError("CPIA_SA 模块需要 einops 库。请安装: pip install einops")


class ComplexFFT(nn.Module):
    """复数 FFT 变换模块，提取频域实部和虚部。"""

    def __init__(self):
        super(ComplexFFT, self).__init__()

    def forward(self, x):
        x_fft = fft.fft2(x, dim=(-2, -1))
        real = x_fft.real
        imag = x_fft.imag
        return real, imag


class ComplexIFFT(nn.Module):
    """复数 IFFT 逆变换模块，从实部和虚部恢复空间域信号。"""

    def __init__(self):
        super(ComplexIFFT, self).__init__()

    def forward(self, real, imag):
        x_complex = torch.complex(real, imag)
        x_ifft = fft.ifft2(x_complex, dim=(-2, -1))
        return x_ifft.real


class Conv1x1(nn.Module):
    """1x1 分组卷积，用于频域实部虚部的通道混合。"""

    def __init__(self, in_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1,
                              padding=0, groups=in_channels * 2)

    def forward(self, x):
        return self.conv(x)


class Stage2_fft(nn.Module):
    """第二阶段 FFT 模块：FFT -> 1x1 Conv -> IFFT。"""

    def __init__(self, in_channels):
        super(Stage2_fft, self).__init__()
        self.c_fft = ComplexFFT()
        self.conv1x1 = Conv1x1(in_channels)
        self.c_ifft = ComplexIFFT()

    def forward(self, x):
        real, imag = self.c_fft(x)

        combined = torch.cat([real, imag], dim=1)
        conv_out = self.conv1x1(combined)

        out_channels = conv_out.shape[1] // 2
        real_out = conv_out[:, :out_channels, :, :]
        imag_out = conv_out[:, out_channels:, :, :]

        output = self.c_ifft(real_out, imag_out)

        return output


class spr_sa(nn.Module):
    """空间感知自注意力分支，使用自适应平均池化生成空间权重。"""

    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x1 = F.adaptive_avg_pool2d(x, (1, 1))
        x1 = F.softmax(x1, dim=1)
        x = x1 * x
        x = self.act(x)
        x = self.conv_1(x)
        return x


class CPIA_SA(nn.Module):
    """Complex Phase Inversion Attention with Spatial Attention (复数相位注意力)。

    结合空间感知分支和通道转置注意力，通过自适应 Top-K 选择机制
    和频率自适应交互模块 (FAIM) 实现高效的频域-空域特征融合。

    Args:
        in_channels (int): 输入通道数。
        num_heads (int): 注意力头数。默认 8。
        bias (bool): 是否使用偏置。默认 False。
    """

    def __init__(self, in_channels, num_heads=8, bias=False):
        super(CPIA_SA, self).__init__()
        _rearrange_check()
        self.num_heads = num_heads
        dim = in_channels

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.spr_sa = spr_sa(dim // 2, 2)
        self.linear_0 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.qkv = nn.Conv2d(dim // 2, dim // 2 * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim // 2 * 3, dim // 2 * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim // 2 * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim // 2, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim // 2, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
        self.fft = Stage2_fft(in_channels=dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        y, x_in = self.linear_0(x).chunk(2, dim=1)

        y_d = self.spr_sa(y)

        qkv = self.qkv_dwconv(self.qkv(x_in))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape
        dynamic_k = int(C * self.gate(x_in).view(b, -1).mean())
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = attn.softmax(dim=-1)
        out1 = (attn @ v)
        out2 = (attn @ v)
        out3 = (attn @ v)
        out4 = (attn @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out_att = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Frequency Adaptive Interaction Module (FAIM)
        # stage1
        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(out_att)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(y_d)

        # S-I
        attened_x = out_att * torch.sigmoid(spatial_map)
        # C-I
        conv_x = y_d * torch.sigmoid(channel_map)

        x_out = torch.cat([attened_x, conv_x], dim=1)
        x_out = self.project_out(x_out)
        # stage 2
        x_out = self.fft(x_out)
        return x_out
