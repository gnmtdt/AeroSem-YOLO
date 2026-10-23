"""
HWD - Haar 小波下采样模块 (Haar Wavelet Downsampling)

来源: 论文 "Haar Wavelet Downsample: A Simple and Effective Downsample Module for Deep Neural Networks"
链接: https://www.sciencedirect.com/science/article/pii/S0031320323005174
期刊: Pattern Recognition, 2023
核心机制: 使用 Haar 小波变换进行下采样，将空间信息分解为4个子带(LL/LH/HL/HH)后融合
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

__all__ = ["HWD"]


class HWD(nn.Module):
    """Haar Wavelet Downsampling -- Haar 小波下采样"""

    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        from pytorch_wavelets import DWTForward

        self.wt = DWTForward(J=1, mode="zero", wave="haar")
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)

    def _wavelet_forward_fp32(self, x):
        """FP32 下执行小波变换，避免 CUDA AMP 兼容性问题"""
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.wt(x.float())

    def forward(self, x):
        yL, yH = self._wavelet_forward_fp32(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)
        return x
