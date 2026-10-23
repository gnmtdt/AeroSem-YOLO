"""CARAFE: 内容感知特征上采样模块.

论文: CARAFE: Content-Aware ReAssembly of FEatures (ICCV 2019)
论文链接: https://arxiv.org/abs/1905.02188
迁移自参考库 Ultralytics_674595707/nn/extra_modules/upsample/CARAFE.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["CARAFE"]


class CARAFE(nn.Module):
    """内容感知特征上采样模块 (Content-Aware ReAssembly of FEatures).

    通过预测内容感知的重组核实现上采样，相比双线性插值能更好地保留局部结构信息。

    Args:
        inc: 输入通道数（同时也是输出通道数）。
        k_enc: 编码器卷积核大小，默认 3。
        k_up: 重组核大小，默认 5。
        c_mid: 压缩后的中间通道数，默认 64。
        scale: 上采样倍率，默认 2。
    """

    def __init__(self, inc: int, k_enc: int = 3, k_up: int = 5, c_mid: int = 64, scale: int = 2):
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(inc, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode="nearest")
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, padding=k_up // 2 * scale)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            X: 输入特征图 (B, C, H, W)。

        Returns:
            上采样后的特征图 (B, C, H*scale, W*scale)。
        """
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)                                   # b * m * h * w
        W = self.enc(W)                                    # b * 100 * h * w
        W = self.pix_shf(W)                                # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)                        # b * 25 * h_ * w_

        X = self.upsmp(X)                                  # b * c * h_ * w_
        X = self.unfold(X)                                 # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)                       # b * 25 * c * h_ * w_

        X = torch.einsum("bkhw,bckhw->bchw", [W, X])      # b * c * h_ * w_
        return X
