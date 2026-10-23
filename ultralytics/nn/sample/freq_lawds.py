"""
FreqLAWDS - 频率引导局部自适应加权下采样 (Frequency-guided Light Adaptive-weight Downsampling)

来源: 自研模块 (BiliBili: 魔傀面具)
用途: 利用 Haar 小波分解引导的自适应下采样，结合频率信息优化特征保留
核心机制: 通过小波分解获取低频/高频分量，结合局部注意力与频率门控实现自适应下采样
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

__all__ = ["FreqLAWDS"]


def _safe_groups(channels, group):
    """计算安全的分组数，确保分组数能整除通道数"""
    base = max(1, channels // max(1, group))
    return max(1, math.gcd(channels, base))


def _pad_to_even(x):
    """将输入 pad 到偶数尺寸"""
    _, _, h, w = x.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x


def _gather_subpixels(x):
    """收集 2x2 子像素并堆叠"""
    return torch.stack(
        (
            x[:, :, 0::2, 0::2],
            x[:, :, 1::2, 0::2],
            x[:, :, 0::2, 1::2],
            x[:, :, 1::2, 1::2],
        ),
        dim=-1,
    )


class FreqLAWDS(nn.Module):
    """Frequency-guided Light Adaptive-weight Downsampling -- 频率引导局部自适应下采样"""

    def __init__(self, in_ch, out_ch, group=16, freq_ratio=0.5) -> None:
        super().__init__()
        groups = _safe_groups(in_ch, group)

        self.in_ch = in_ch
        self.softmax = nn.Softmax(dim=-1)
        self.freq_scale = nn.Parameter(torch.tensor(float(freq_ratio)))
        self.res_scale = nn.Parameter(torch.tensor(float(freq_ratio) * 0.5))

        self.local_attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_ch, in_ch, k=1),
        )
        self.ds_conv = Conv(in_ch, in_ch * 4, k=3, s=2, g=groups)

        self.low_proj = Conv(in_ch, in_ch, 1, g=groups)
        self.high_proj = Conv(in_ch, in_ch, 1, g=groups)
        self.freq_gate = nn.Sequential(
            Conv(in_ch * 3, in_ch, 1),
            nn.Conv2d(in_ch, in_ch * 4, kernel_size=1, groups=groups, bias=True),
        )
        self.freq_residual = nn.Sequential(
            Conv(in_ch * 2, in_ch, 1),
            Conv(in_ch, in_ch, 3, g=groups),
        )
        self.output = Conv(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        haar = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[-0.5, -0.5], [0.5, 0.5]],
                [[-0.5, 0.5], [-0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("haar_weight", haar, persistent=False)

    def _wavelet_decompose(self, x):
        """使用 Haar 小波进行频域分解"""
        filters = self.haar_weight.repeat(self.in_ch, 1, 1, 1)
        coeffs = F.conv2d(x, filters, stride=2, groups=self.in_ch)
        b, _, h, w = coeffs.shape
        coeffs = coeffs.view(b, self.in_ch, 4, h, w)
        ll = coeffs[:, :, 0]
        hf = coeffs[:, :, 1:].abs().sum(dim=2)
        return ll, hf

    def forward(self, x):
        x = _pad_to_even(x)

        local_logits = _gather_subpixels(self.local_attention(x))

        candidates = self.ds_conv(x)
        b, _, h, w = candidates.shape
        candidates = candidates.view(b, 4, self.in_ch, h, w).permute(0, 2, 3, 4, 1).contiguous()

        ll, hf = self._wavelet_decompose(x)
        coarse = F.avg_pool2d(x, kernel_size=2, stride=2)

        freq_context = torch.cat((self.low_proj(ll), self.high_proj(hf), coarse), dim=1)
        freq_logits = self.freq_gate(freq_context).view(b, 4, self.in_ch, h, w).permute(0, 2, 3, 4, 1).contiguous()

        att = self.softmax(local_logits + self.freq_scale * freq_logits)
        fused = torch.sum(candidates * att, dim=-1)

        residual = self.freq_residual(torch.cat((ll, hf), dim=1))
        return self.output(fused + self.res_scale * residual)
