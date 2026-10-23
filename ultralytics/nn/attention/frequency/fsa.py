"""
FSA - Frequency-Spatial Attention (频率空间注意力)

论文: Frequency-Spatial Attention
期刊/会议: BIBM 2024
论文链接: https://arxiv.org/pdf/2406.07952
注意: 此模块不支持多尺度训练，size 参数为当前特征图的 (height, width) 元组。
      Adaptive_global_filter 的滤波器参数在初始化时固定了空间尺寸，
      运行时会自动插值到实际输入尺寸，但初始化尺寸应尽量匹配典型特征图大小。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Adaptive_global_filter', 'SpatialAttention', 'FSA']


class Adaptive_global_filter(nn.Module):
    """自适应全局滤波器，通过 FFT 进行频域低通滤波。

    在频域中对低频分量应用可学习的复数滤波器，保留高频分量不变。
    初始化时需指定空间尺寸，运行时会自动插值适配。

    Args:
        ratio (int): 低频区域半径比例。默认 10。
        dim (int): 通道数。默认 32。
        H (int): 初始化高度。默认 512。
        W (int): 初始化宽度。默认 512。
    """

    def __init__(self, ratio=10, dim=32, H=512, W=512):
        super().__init__()
        self.ratio = ratio
        self.filter = nn.Parameter(torch.randn(dim, H, W, 2, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        # 保持频域分支在 FP32 以避免 cuFFT 半精度限制和类型不匹配
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            b, c, h, w = x.shape
            crow, ccol = int(h / 2), int(w / 2)
            r_h = min(self.ratio, crow)
            r_w = min(self.ratio, ccol)

            # 构建运行时掩码，避免初始化时的固定尺寸假设
            mask_lowpass = x.new_zeros((h, w))
            mask_lowpass[crow - r_h:crow + r_h, ccol - r_w:ccol + r_w] = 1
            mask_highpass = 1 - mask_lowpass

            x_fre = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm='ortho'))
            weight = self.filter
            if weight.shape[1] != h or weight.shape[2] != w:
                # 将复数权重的实部-虚部表示插值到当前特征图尺寸
                weight = weight.permute(0, 3, 1, 2).contiguous()  # [C, 2, H, W]
                weight = F.interpolate(weight, size=(h, w), mode='bilinear', align_corners=False)
                weight = weight.permute(0, 2, 3, 1).contiguous()  # [C, H, W, 2]
            weight = torch.view_as_complex(weight)

            x_fre_low = torch.mul(x_fre, mask_lowpass)
            x_fre_high = torch.mul(x_fre, mask_highpass)

            x_fre_low = torch.mul(x_fre_low, weight)
            x_fre_new = x_fre_low + x_fre_high
            x_out = torch.fft.ifft2(torch.fft.ifftshift(x_fre_new, dim=(-2, -1))).real
            return x_out


class SpatialAttention(nn.Module):
    """空间注意力模块，通过平均池化和最大池化聚合空间信息。"""

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        result = x * out
        return result


class FSA(nn.Module):
    """Frequency-Spatial Attention (频率空间注意力)。

    结合自适应全局滤波器 (Adaptive_global_filter) 和空间注意力 (SpatialAttention)，
    在频域和空域同时进行特征增强。

    注意: 此模块不支持多尺度训练。size 参数需指定当前特征图的 (height, width)。

    Args:
        input_channel (int): 输入通道数。默认 64。
        size (tuple): 特征图空间尺寸 (H, W)。默认 (20, 20)。
        ratio (int): 低通滤波器半径比例。默认 10。
    """

    def __init__(self, input_channel=64, size=(20, 20), ratio=10):
        super(FSA, self).__init__()
        self.agf = Adaptive_global_filter(ratio=ratio, dim=input_channel, H=size[0], W=size[1])
        self.sa = SpatialAttention()

    def _spatial_forward_fp32(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.sa(x.float())

    def forward(self, x):
        f_out = self.agf(x)
        sa_out = self._spatial_forward_fp32(x)
        result = f_out + sa_out
        return result
