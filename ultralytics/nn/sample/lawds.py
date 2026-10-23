"""
LAWDS - 轻量自适应权重下采样模块 (Light Adaptive-Weight Downsampling)

来源: 自研模块 (BiliBili: 魔傀面具)
用途: 适用于图像分类、目标检测、语义分割等需要高效空间分辨率压缩的视觉任务
核心机制: 通过自适应权重生成机制动态调整下采样策略，结合分组卷积与通道重组操作
"""

import torch
import torch.nn as nn
from einops import rearrange
from ultralytics.nn.modules.conv import Conv

__all__ = ["LAWDS"]


class LAWDS(nn.Module):
    """Light Adaptive-weight downsampling -- 轻量自适应权重下采样"""

    def __init__(self, in_ch, out_ch, group=16) -> None:
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_ch, in_ch, k=1),
        )

        self.ds_conv = Conv(in_ch, in_ch * 4, k=3, s=2, g=(in_ch // group))
        self.conv1x1 = Conv(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4
        att = rearrange(self.attention(x), "bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)", s1=2, s2=2)
        att = self.softmax(att)

        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        x = rearrange(self.ds_conv(x), "bs (s ch) h w -> bs ch h w s", s=4)
        x = torch.sum(x * att, dim=-1)
        return self.conv1x1(x)
