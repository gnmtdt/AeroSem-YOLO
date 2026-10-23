# 论文: Multiscale Sparse Cross-Attention Network for Remote Sensing Scene Classification (2025)
# 链接: https://ieeexplore.ieee.org/abstract/document/10820553/
# 模块作用: 双路稀疏交叉注意力，利用多尺度上下文对齐跨模态关键 token，以 Top-k 稀疏匹配突出高相关区域融合并输出单路表征。

import torch
import torch.nn as nn


class MSC(nn.Module):
    """Multiscale Sparse Cross-Attention（双输入→单输出）。

    Args:
        dim (int | None): 通道数（自动注入）
        num_heads (int): 注意力头数
        kernel (list[int]): 多尺度 avgpool kernel 列表
        s (list[int]): 多尺度 stride 列表
        pad (list[int]): 多尺度 padding 列表
        k1 (int): Top-k 比例分母（N1/k1）
        k2 (int): Top-k 比例分母（N1/k2）
    """

    def __init__(
        self,
        dim: int | None = None,
        num_heads: int = 8,
        kernel: list[int] = [3, 5, 7],
        s: list[int] = [1, 1, 1],
        pad: list[int] = [1, 2, 3],
        k1: int = 2,
        k2: int = 3,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = int(num_heads)
        self.k1 = int(k1)
        self.k2 = int(k2)
        self.q = nn.Linear(dim if dim is not None else 1, dim if dim is not None else 1, bias=True)
        self.kv = nn.Linear(dim if dim is not None else 1, (dim if dim is not None else 1) * 2, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim if dim is not None else 1, dim if dim is not None else 1)
        self.proj_drop = nn.Dropout(0.0)
        self.attn1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel_size=kernel[2], stride=s[2], padding=pad[2])
        self.layer_norm = nn.LayerNorm(dim if dim is not None else 1)

    def forward(self, x, y=None):
        if y is None and isinstance(x, (list, tuple)):
            x, y = x
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("MSC 需要两路输入张量")
        if x.shape != y.shape:
            raise ValueError(f"MSC 要求两路输入形状一致，got {x.shape} vs {y.shape}")
        B, C, H, W = x.shape
        # 多尺度池化融合上下文 y
        y1 = self.avgpool1(y)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)
        y = y1 + y2 + y3
        y = y.flatten(-2, -1).transpose(1, 2)
        y = self.layer_norm(y)
        N1 = y.shape[1]
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(x.flatten(2).transpose(1, 2))
        N = q.shape[1]
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        # Top-k 两种稀疏度
        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device, dtype=torch.bool)
        idx1 = torch.topk(attn, k=max(N1 // self.k1, 1), dim=-1, largest=True)[1]
        mask1.scatter_(-1, idx1, True)
        attn1 = torch.where(mask1, attn, torch.full_like(attn, float('-inf'))).softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = attn1 @ v
        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device, dtype=torch.bool)
        idx2 = torch.topk(attn, k=max(N1 // self.k2, 1), dim=-1, largest=True)[1]
        mask2.scatter_(-1, idx2, True)
        attn2 = torch.where(mask2, attn, torch.full_like(attn, float('-inf'))).softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = attn2 @ v
        out = out1 * self.attn1 + out2 * self.attn2
        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        hw = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, hw, hw)
        return x
