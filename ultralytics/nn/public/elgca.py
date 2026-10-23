"""ELGCA blocks (Efficient Local-Global Context Aggregation) used by C2f_ELGCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.public.common_glu import ConvolutionalGLU
from ultralytics.nn.public.tsdn import LayerNorm


class ELGCA_MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.pos = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x


class ELGCA(nn.Module):
    """Efficient local global context aggregation module."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"ELGCA requires dim%4==0, got dim={dim}")
        if dim % 2 != 0:
            raise ValueError(f"ELGCA requires dim%2==0, got dim={dim}")
        if heads < 4:
            raise ValueError(f"ELGCA requires heads>=4, got heads={heads}")

        self.heads = heads
        self.dwconv = nn.Conv2d(dim // 2, dim // 2, 3, padding=1, groups=dim // 2)
        self.qkvl = nn.Conv2d(dim // 2, (dim // 4) * heads, 1, padding=0)
        self.pool_q = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_k = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c % 2 != 0:
            raise ValueError(f"ELGCA expects even channels, got C={c}")

        x1, x2 = torch.split(x, [c // 2, c // 2], dim=1)

        x1 = self.act(self.dwconv(x1))
        x2 = self.act(self.qkvl(x2))
        x2 = x2.reshape(b, self.heads, c // 4, h, w)

        q = torch.sum(x2[:, :-3, :, :, :], dim=1)
        k = x2[:, -3, :, :, :]

        q = self.pool_q(q)
        k = self.pool_k(k)

        v = x2[:, -2, :, :, :].flatten(2)
        lfeat = x2[:, -1, :, :, :]

        qk = torch.matmul(q.flatten(2), k.flatten(2).transpose(1, 2))
        qk = torch.softmax(qk, dim=1).transpose(1, 2)

        x2 = torch.matmul(qk, v).reshape(b, c // 4, h, w)
        x = torch.cat([x1, lfeat, x2], dim=1)
        return x


class ELGCA_EncoderBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, heads: int = 4):
        super().__init__()
        self.layer_norm1 = LayerNorm(dim, "BiasFree")
        self.layer_norm2 = LayerNorm(dim, "BiasFree")
        self.mlp = ELGCA_MLP(dim=dim, mlp_ratio=mlp_ratio)
        self.attn = ELGCA(dim, heads=heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_copy = x
        x = self.layer_norm1(inp_copy)
        x = self.attn(x)
        out = x + inp_copy

        x = self.layer_norm2(out)
        x = self.mlp(x)
        out = out + x
        return out


class ELGCA_CGLU(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.layer_norm1 = LayerNorm(dim, "BiasFree")
        self.layer_norm2 = LayerNorm(dim, "BiasFree")
        self.mlp = ConvolutionalGLU(dim)
        self.attn = ELGCA(dim, heads=heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_copy = x
        x = self.layer_norm1(inp_copy)
        x = self.attn(x)
        out = x + inp_copy

        x = self.layer_norm2(out)
        x = self.mlp(x)
        out = out + x
        return out

