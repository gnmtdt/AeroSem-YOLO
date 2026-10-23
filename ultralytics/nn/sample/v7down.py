"""
V7DownSampling - YOLOv7 风格下采样模块

来源: 论文 "YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors"
链接: https://arxiv.org/pdf/2207.02696
会议: CVPR 2023
核心机制: 并行使用最大池化和步长卷积两条路径，将输出拼接实现2倍下采样
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

__all__ = ["V7DownSampling"]


class V7DownSampling(nn.Module):
    """YOLOv7 风格下采样 -- 并行池化+卷积拼接"""

    def __init__(self, inc, ouc) -> None:
        super(V7DownSampling, self).__init__()

        ouc = ouc // 2
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(inc, ouc, k=1),
        )
        self.conv = nn.Sequential(
            Conv(inc, ouc, k=1),
            Conv(ouc, ouc, k=3, s=2),
        )

    def forward(self, x):
        return torch.cat([self.maxpool(x), self.conv(x)], dim=1)
