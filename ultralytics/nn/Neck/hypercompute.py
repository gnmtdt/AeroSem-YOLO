"""
HyperComputeModule - 超图计算模块 (TPAMI 2025)
论文: Hypergraph Computation for Visual Recognition (TPAMI 2025)
论文链接: https://arxiv.org/pdf/2408.04804

通过构建超图结构（基于特征距离阈值），在顶点和超边之间传播信息，
实现高阶特征交互。HyPConv 作为核心卷积操作，通过超图关联矩阵
进行消息聚合。
"""

import torch
import torch.nn as nn


class MessageAgg(nn.Module):
    """超图消息聚合层，支持 mean 和 sum 两种聚合方式。"""

    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, x, path):
        x = torch.matmul(path, x)
        if self.agg_method == "mean":
            norm = 1 / torch.sum(path, dim=2, keepdim=True)
            norm[torch.isinf(norm)] = 0
            return norm * x
        return x


class HyPConv(nn.Module):
    """超图卷积层，在顶点和超边之间进行消息传播。

    Args:
        c1: 输入通道数
        c2: 输出通道数
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, incidence):
        x = self.fc(x)
        edge_feat = self.v2e(x, incidence.transpose(1, 2).contiguous())
        return self.e2v(edge_feat, incidence)


class HyperComputeModule(nn.Module):
    """超图计算模块，基于特征距离阈值构建超图关联矩阵进行消息传播。

    Args:
        c1: 输入通道数
        c2: 输出通道数
        threshold: 距离阈值，用于构建超图关联矩阵
    """

    def __init__(self, c1, c2, threshold):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyPConv(c1, c2)
        self.residual = nn.Identity() if c1 == c2 else nn.Linear(c1, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x_flat.clone()
        distance = torch.cdist(feature, feature)
        incidence = (distance < self.threshold).to(dtype=x_flat.dtype, device=x_flat.device)
        out = self.hgconv(x_flat, incidence).to(dtype=x_flat.dtype, device=x_flat.device) + self.residual(x_flat)
        out = out.transpose(1, 2).contiguous().view(b, -1, h, w)
        return self.act(self.bn(out))


__all__ = ("HyperComputeModule", "HyPConv", "MessageAgg")
