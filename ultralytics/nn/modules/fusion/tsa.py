# 论文: Dual selective fusion transformer network for hyperspectral image classification (2025)
# 链接: https://www.sciencedirect.com/science/article/pii/S089360802500190X
# 模块作用: 在单路融合特征中以 Top-k 令牌选择约束注意力，仅保留最相关 token 交互以提升效率与鲁棒性。

import torch
import torch.nn as nn


class TokenSelectiveAttention(nn.Module):
    def __init__(self, num_heads: int = 8, k_ratio: float = 0.8, group: int = 4, bias: bool = False) -> None:
        super().__init__()
        assert 0.0 < k_ratio <= 1.0, f"k_ratio 必须在 (0,1]，got {k_ratio}"
        self.num_heads = int(num_heads)
        self.k_ratio = float(k_ratio)
        self.group = int(group)
        self.bias = bool(bias)
        self._built = False
        self._c = None
        self.q_proj: nn.Module | None = None
        self.k_proj: nn.Module | None = None
        self.v_proj: nn.Module | None = None
        self.q_dw: nn.Module | None = None
        self.k_dw: nn.Module | None = None
        self.v_dw: nn.Module | None = None
        self.out_proj: nn.Module | None = None
        self.temperature = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))

    def _build_if_needed(self, c: int) -> None:
        if self._built and self._c == c:
            return
        if c % self.num_heads != 0:
            raise ValueError(f"TokenSelectiveAttention: 通道数 {c} 不能被 num_heads {self.num_heads} 整除")
        if c % self.group != 0:
            raise ValueError(f"TokenSelectiveAttention: 通道数 {c} 不能被 group {self.group} 整除")
        self.q_proj = nn.Conv2d(c, c, kernel_size=1, groups=self.group, bias=False)
        self.k_proj = nn.Conv2d(c, c, kernel_size=1, groups=self.group, bias=False)
        self.v_proj = nn.Conv2d(c, c, kernel_size=1, groups=self.group, bias=False)
        self.q_dw = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=self.bias)
        self.k_dw = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=self.bias)
        self.v_dw = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=self.bias)
        self.out_proj = nn.Conv2d(c, c, kernel_size=1, bias=self.bias)
        self._built = True
        self._c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise TypeError("TokenSelectiveAttention 期望输入 [B, C, H, W]")
        B, C, H, W = x.shape
        self._build_if_needed(C)
        q = self.q_dw(self.q_proj(x))
        k = self.k_dw(self.k_proj(x))
        v = self.v_dw(self.v_proj(x))
        C_h = C // self.num_heads
        N = H * W
        q = q.view(B, self.num_heads, C_h, H, W).reshape(B, self.num_heads, C_h, N).transpose(2, 3)
        k = k.view(B, self.num_heads, C_h, H, W).reshape(B, self.num_heads, C_h, N)
        v = v.view(B, self.num_heads, C_h, H, W).reshape(B, self.num_heads, C_h, N).transpose(2, 3)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = torch.matmul(q, k) * self.temperature
        k_top = max(int(N * self.k_ratio), 1)
        topk_idx = torch.topk(attn, k=k_top, dim=-1, largest=True)[1]
        mask = torch.zeros_like(attn, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        attn = torch.where(mask, attn, torch.full_like(attn, float('-inf'))).softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.out_proj(out)
        return x + out
