"""PST: Pyramid Sparse Transformer.

This module is migrated from RTDETR-main (ultralytics/nn/extra_modules/block.py) and is used by
RTDETRMM YAML configs such as `rtdetr-r18-mm-mid-pst.yaml`.

Input: a tuple/list of two feature maps (x, upper_feat):
- x:         [B, C1, H, W]
- upper_feat:[B, Cup, H/2, W/2]
Output:
- y:         [B, C2, H, W]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class PSAttn(nn.Module):
    """Pyramid Sparse Attention (cross-attention across adjacent pyramid levels)."""

    def __init__(self, dim: int, num_heads: int, topk: int = 4, tau: float = 1.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"PSAttn: dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)
        self.all_head_dim = int(self.head_dim * self.num_heads)
        self.topk = int(topk)
        self.tau = float(tau)

        self.q = Conv(dim, self.all_head_dim, 1, act=False)
        self.kv = Conv(dim, self.all_head_dim * 2, 1, act=False)
        self.proj = Conv(self.all_head_dim, dim, 1, act=False)
        self.pe = Conv(self.all_head_dim, dim, 7, 1, 3, g=dim, act=False)
        self.gate_conv1d = nn.Conv1d(2 * self.head_dim, self.head_dim, kernel_size=1)

    @staticmethod
    def gumbel_softmax(logits: torch.Tensor) -> torch.Tensor:
        gumbels = -torch.empty_like(logits).exponential_().log()
        logits = logits + gumbels
        return F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor, upper_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        _, _, h_up, w_up = upper_feat.shape

        q = self.q(x).view(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)
        kv = self.kv(upper_feat).view(b, self.num_heads, 2 * self.head_dim, h_up * w_up).permute(0, 1, 3, 2)
        k, v = kv.split(self.head_dim, dim=3)

        sim = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = sim.softmax(dim=-1)
        coarse_out = attn @ v

        # Fine attention: only for inference if topk>0
        if 0 < self.topk <= h_up * w_up:
            f_kv = self.kv(x).view(b, self.num_heads, 2 * self.head_dim, n).permute(0, 1, 3, 2)
            f_k, f_v = f_kv.split(self.head_dim, dim=3)

            global_sim = sim.mean(dim=2)
            soft_weights = self.gumbel_softmax(global_sim)
            _, topk_indices = torch.topk(soft_weights, k=self.topk, dim=-1)

            # Map selected indices from upper_feat to x (assuming 2x downsampling)
            scale = 2
            h_idx = (topk_indices // w_up) * scale
            w_idx = (topk_indices % w_up) * scale
            topk_x_indices = []
            for dh in range(scale):
                for dw in range(scale):
                    idx = (h_idx + dh) * w + (w_idx + dw)
                    topk_x_indices.append(idx)
            topk_x_indices = torch.cat(topk_x_indices, dim=-1)

            topk_k = torch.gather(f_k, dim=2, index=topk_x_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            topk_v = torch.gather(f_v, dim=2, index=topk_x_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))

            fine_attn = (q @ topk_k.transpose(-2, -1)) * (self.head_dim**-0.5)
            fine_attn = fine_attn.softmax(dim=-1)
            refined_out = fine_attn @ topk_v

            fusion_input = torch.cat([coarse_out, refined_out], dim=-1)
            fusion_input = fusion_input.view(b * self.num_heads, n, -1).transpose(1, 2)
            gate = torch.sigmoid(self.gate_conv1d(fusion_input)).transpose(1, 2).view(b, self.num_heads, n, self.head_dim)
            x_out = gate * refined_out + (1.0 - gate) * coarse_out
        else:
            x_out = coarse_out

        x_out = x_out.transpose(2, 3).reshape(b, self.all_head_dim, h, w)

        v_reshaped = v.transpose(2, 3).reshape(b, self.all_head_dim, h_up, w_up)
        v_pe = self.pe(v_reshaped)
        v_pe = F.interpolate(v_pe, size=(h, w), mode="bilinear", align_corners=False)
        x_out = x_out + v_pe

        return self.proj(x_out)


class PSAttnBlock(nn.Module):
    """Residual PSAttn + Conv-MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, topk: int = 0):
        super().__init__()
        self.attn = PSAttn(dim, num_heads=num_heads, topk=topk)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden_dim, 1),
            Conv(mlp_hidden_dim, dim, 1, act=False),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, upper_feat: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x, upper_feat)
        return x + self.mlp(x)


class PST(nn.Module):
    """Pyramid Sparse Transformer (stacked PSAttnBlock fusion)."""

    def __init__(
        self,
        c1: int,
        c_up: int,
        c2: int,
        n: int = 1,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        k: int = 0,
    ):
        super().__init__()
        c_ = int(c2 * e)
        if c_ % 32 != 0:
            raise ValueError(f"PST: hidden channels int(c2*e) must be multiple of 32, got {c_} (c2={c2}, e={e})")

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cvup = Conv(c_up, c_, 1, 1)
        self.cv2 = Conv((1 + int(n)) * c_, c2, 1)

        self.num_layers = int(n)
        for i in range(self.num_layers):
            layer = PSAttnBlock(c_, c_ // 32, mlp_ratio, topk=k)
            self.add_module(f"attnlayer_{i}", layer)

    def forward(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise TypeError(f"PST expects input (x, upper_feat) tuple/list, got {type(x).__name__}")

        upper_feat = x[1]
        x0 = self.cv1(x[0])
        upper_feat = self.cvup(upper_feat)

        y = [x0]
        for i in range(self.num_layers):
            layer = getattr(self, f"attnlayer_{i}")
            y.append(layer(y[-1], upper_feat))

        return self.cv2(torch.cat(y, 1))
