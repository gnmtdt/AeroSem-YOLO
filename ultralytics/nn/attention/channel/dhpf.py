"""
DHPF - Dynamic High-Pass Filter

论文: Dynamic High-Pass Filter for Remote Sensing Image Processing
期刊: IEEE Transactions on Geoscience and Remote Sensing (TGRS 2025)
论文链接: https://ieeexplore.ieee.org/document/11017756

基于频域的动态高通滤波模块。通过FFT将特征变换到频域，根据能量分布动态确定截止频率，
去除低频成分后通过IFFT还原空间域特征。FFT计算强制使用FP32以保证精度。
"""

import torch
import torch.nn as nn


class DHPF(nn.Module):
    def __init__(self, chn, energy=0.4):
        super(DHPF, self).__init__()
        self.energy = energy

    def _determine_cutoff_frequency(self, f_transform, target_ratio):
        total_energy = self._calculate_total_energy(f_transform)
        target_low_freq_energy = total_energy * target_ratio

        for cutoff_frequency in range(1, min(f_transform.shape[0], f_transform.shape[1]) // 2):
            low_freq_energy = self._calculate_low_freq_energy(f_transform, cutoff_frequency)
            if low_freq_energy >= target_low_freq_energy:
                return cutoff_frequency
        return 5

    def _calculate_total_energy(self, f_transform):
        magnitude_spectrum = torch.abs(f_transform)
        total_energy = torch.sum(magnitude_spectrum ** 2)
        return total_energy

    def _calculate_low_freq_energy(self, f_transform, cutoff_frequency):
        magnitude_spectrum = torch.abs(f_transform)
        height, width = magnitude_spectrum.shape

        low_freq_energy = torch.sum(magnitude_spectrum[
            height // 2 - cutoff_frequency:height // 2 + cutoff_frequency,
            width // 2 - cutoff_frequency:width // 2 + cutoff_frequency
        ] ** 2)

        return low_freq_energy

    def _fft_forward_fp32(self, x):
        # cuFFT half precision only supports power-of-two sizes, so keep the FFT branch in FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            f = torch.fft.fft2(x.float())
            fshift = torch.fft.fftshift(f)
            return fshift

    def forward(self, x):
        B, C, H, W = x.shape
        fshift = self._fft_forward_fp32(x)
        crow, ccol = H // 2, W // 2
        for i in range(B):
            cutoff_frequency = self._determine_cutoff_frequency(fshift[i, 0], self.energy)
            fshift[i, :, crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0
        ishift = torch.fft.ifftshift(fshift)
        ideal_high_pass = torch.abs(torch.fft.ifft2(ishift))
        return ideal_high_pass


__all__ = ['DHPF']
