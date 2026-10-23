"""DSUB: 深度可分离上采样块.

论文: Rethinking Decoder Design: Improving Biomarker Segmentation Using
      Depth-to-Space Restoration and Cross-Scale Attention (CVPR 2025)
论文链接: https://openaccess.thecvf.com/content/CVPR2025/papers/Wazir_Rethinking_Decoder_Design_Improving_Biomarker_Segmentation_Using_Depth-to-Space_Restoration_and_CVPR_2025_paper.pdf
迁移自参考库 Ultralytics_674595707/nn/extra_modules/upsample/DSUB.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DSConv

__all__ = ["DSUB"]


class DSUB(nn.Module):
    """深度可分离上采样块 (Depth-to-Space Upsampling Block).

    使用 PixelShuffle 进行 2x 上采样，配合深度可分离卷积增强特征表达。

    Args:
        inc: 输入通道数。输出通道数等于 inc，空间分辨率放大 2 倍。
    """

    def __init__(self, inc: int):
        super().__init__()

        self.conv3x3_1 = Conv(inc, inc, 3)
        self.conv3x3_2 = Conv(inc // 4, inc // 4, 3)
        self.convblock = DSConv(inc // 4, inc, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            x: 输入张量 (B, inc, H, W)。

        Returns:
            上采样后的张量 (B, inc, 2*H, 2*W)。
        """
        x = self.conv3x3_1(x)
        x = F.pixel_shuffle(x, upscale_factor=2)
        x = self.conv3x3_2(x)
        x = self.convblock(x)
        return x
