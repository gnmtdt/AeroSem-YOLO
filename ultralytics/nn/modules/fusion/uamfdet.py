"""UAMFDet plug-and-play fusion modules (PyTorch-only, no torchvision).

This file implements a production-oriented, fail-fast adaptation of the core ideas from:
UAMFDet: Acoustic-Optical Fusion for Underwater Multi-Modal Object Detection.

Design goals (per project plan):
- Plug-and-play via YAML (registered in tasks.py:parse_model()).
- Works for YOLOMM and RTDETRMM at P3/P4/P5.
- Instance-level fusion (MIFM) implemented with grid_sample (no roi_align/torchvision).
- Fail-fast: invalid params or incompatible inputs raise ValueError with key parameter names.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MDSFF", "MDF", "MIFM", "UAMFDetFusion"]


def _unpack_two_inputs(x1: torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | list, x2: torch.Tensor | None):
    """Accept (x1, x2) or ([x1, x2], None) or ((x1, x2), None)."""
    if x2 is not None:
        return x1, x2
    if isinstance(x1, (list, tuple)) and len(x1) == 2:
        return x1[0], x1[1]
    raise ValueError("expects 2 inputs")


def _check_4d(x: torch.Tensor, name: str):
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError(f"{name} must be a 4D torch.Tensor [B,C,H,W], got {type(x)} with ndim={getattr(x, 'ndim', None)}")


def _make_base_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Create a base sampling grid for align_corners=False.

    Returns:
        grid: [1, H, W, 2] in (x, y) order, float32, values in [-1, 1].
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"H and W must be >0, got H={h}, W={w}")
    # For align_corners=False: x_norm = 2*(x+0.5)/W - 1
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * (2.0 / w) - 1.0
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * (2.0 / h) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)  # [1,H,W,2]


class MDSFF(nn.Module):
    """Multi-modal deformable self-aligned feature fusion (alignment stage).

    This implementation aligns the *aux* feature onto the *main* feature semantics.
    Inputs must share the same spatial size (H, W). Channels can differ only when
    auto_channel_align=True (then aux is projected to main channels via 1x1).
    """

    def __init__(
        self,
        c_main: int,
        c_aux: int,
        K: int = 8,
        n_heads: int = 4,
        offset_scale: float = 1.0,
        auto_channel_align: bool = False,
    ):
        super().__init__()
        if K <= 0:
            raise ValueError("K must be >0")
        if n_heads <= 0:
            raise ValueError("n_heads must be >0")
        if offset_scale <= 0:
            raise ValueError("offset_scale must be >0")
        if c_main <= 0 or c_aux <= 0:
            raise ValueError(f"channels must be >0, got c_main={c_main}, c_aux={c_aux}")
        if c_main % n_heads != 0:
            raise ValueError(f"c_main must be divisible by n_heads, got c_main={c_main}, n_heads={n_heads}")

        self.c_main = int(c_main)
        self.c_aux = int(c_aux)
        self.K = int(K)
        self.n_heads = int(n_heads)
        self.offset_scale = float(offset_scale)
        self.auto_channel_align = bool(auto_channel_align)

        if self.c_main != self.c_aux and not self.auto_channel_align:
            raise ValueError(
                f"MDSFF expects equal channels unless auto_channel_align=True, got c_main={self.c_main}, c_aux={self.c_aux}"
            )

        self.aux_in_proj = nn.Identity() if self.c_main == self.c_aux else nn.Conv2d(self.c_aux, self.c_main, 1, 1, 0, bias=False)

        # Predict per-location offsets for K sampling points: [B, 2*K, H, W]
        self.offset_conv = nn.Conv2d(self.c_main, 2 * self.K, kernel_size=3, stride=1, padding=1)

        # Q from main, K from sampled aux; V uses sampled aux directly.
        self.q_proj = nn.Conv2d(self.c_main, self.c_main, kernel_size=1, stride=1, padding=0, bias=False)
        self.k_proj = nn.Conv2d(self.c_main, self.c_main, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_proj = nn.Conv2d(self.c_main, self.c_main, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_main: torch.Tensor, x_aux: torch.Tensor | None = None) -> torch.Tensor:
        x_main, x_aux = _unpack_two_inputs(x_main, x_aux)
        _check_4d(x_main, "x_main")
        _check_4d(x_aux, "x_aux")
        if x_main.shape[0] != x_aux.shape[0]:
            raise ValueError(f"batch size mismatch: {x_main.shape[0]} vs {x_aux.shape[0]}")
        if x_main.shape[-2:] != x_aux.shape[-2:]:
            raise ValueError(f"spatial size mismatch: {x_main.shape[-2:]} vs {x_aux.shape[-2:]}")

        b, _, h, w = x_main.shape
        x_aux = self.aux_in_proj(x_aux)

        # Offsets in normalized pixel units. We keep them small via tanh.
        offsets = torch.tanh(self.offset_conv(x_main))  # [B,2K,H,W] in [-1,1]
        offsets = offsets.view(b, self.K, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()  # [B,K,H,W,2]

        # Convert pixel offsets to normalized offsets for align_corners=False.
        # 1 pixel corresponds to 2/W (x) and 2/H (y).
        scale = torch.tensor([2.0 / w, 2.0 / h], device=x_main.device, dtype=torch.float32).view(1, 1, 1, 1, 2)
        grid_base = _make_base_grid(h, w, x_main.device).unsqueeze(1)  # [1,1,H,W,2]
        grid = grid_base + (offsets.to(torch.float32) * self.offset_scale) * scale  # [B,K,H,W,2]
        grid = grid.clamp(-1.0, 1.0)

        # Sample aux K times using one batched grid_sample.
        x_aux_rep = x_aux.unsqueeze(1).expand(b, self.K, self.c_main, h, w).reshape(b * self.K, self.c_main, h, w)
        grid_rep = grid.reshape(b * self.K, h, w, 2)
        sampled = F.grid_sample(x_aux_rep, grid_rep, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.view(b, self.K, self.c_main, h, w)  # [B,K,C,H,W]

        # Attention weights over K sampling points.
        d = self.c_main // self.n_heads
        q = self.q_proj(x_main).view(b, self.n_heads, d, h, w)  # [B,heads,d,H,W]
        k = self.k_proj(sampled.reshape(b * self.K, self.c_main, h, w)).view(b, self.K, self.n_heads, d, h, w)
        sim = (q.unsqueeze(1) * k).sum(dim=3) / math.sqrt(d)  # [B,K,heads,H,W]
        attn = torch.softmax(sim, dim=1)

        # Use head-averaged attention to weight value (sampled aux).
        w_k = attn.mean(dim=2)  # [B,K,H,W]
        out = (sampled * w_k.unsqueeze(2)).sum(dim=1)  # [B,C,H,W]
        return self.out_proj(out)


class MDF(nn.Module):
    """Multi-modal differential fusion (difference + gating + residual)."""

    def __init__(
        self,
        c_main: int,
        c_aux: int,
        reduction: int = 16,
        auto_channel_align: bool = False,
    ):
        super().__init__()
        if c_main <= 0 or c_aux <= 0:
            raise ValueError(f"channels must be >0, got c_main={c_main}, c_aux={c_aux}")
        if reduction <= 0:
            raise ValueError("reduction must be >0")

        self.c_main = int(c_main)
        self.c_aux = int(c_aux)
        self.reduction = int(reduction)
        self.auto_channel_align = bool(auto_channel_align)

        if self.c_main != self.c_aux and not self.auto_channel_align:
            raise ValueError(
                f"MDF expects equal channels unless auto_channel_align=True, got c_main={self.c_main}, c_aux={self.c_aux}"
            )
        self.aux_in_proj = nn.Identity() if self.c_main == self.c_aux else nn.Conv2d(self.c_aux, self.c_main, 1, 1, 0, bias=False)

        hidden = max(self.c_main // self.reduction, 1)
        self.fc1 = nn.Linear(2 * self.c_main, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, self.c_main, bias=True)

    def forward(self, x_main: torch.Tensor, x_aux_aligned: torch.Tensor | None = None) -> torch.Tensor:
        x_main, x_aux_aligned = _unpack_two_inputs(x_main, x_aux_aligned)
        _check_4d(x_main, "x_main")
        _check_4d(x_aux_aligned, "x_aux")
        if x_main.shape[0] != x_aux_aligned.shape[0]:
            raise ValueError(f"batch size mismatch: {x_main.shape[0]} vs {x_aux_aligned.shape[0]}")
        if x_main.shape[-2:] != x_aux_aligned.shape[-2:]:
            raise ValueError(f"spatial size mismatch: {x_main.shape[-2:]} vs {x_aux_aligned.shape[-2:]}")

        x_aux_aligned = self.aux_in_proj(x_aux_aligned)
        diff = x_main - x_aux_aligned

        # Channel gating from diff statistics (avg + max pooling).
        avg = F.adaptive_avg_pool2d(diff, 1).flatten(1)  # [B,C]
        mx = F.adaptive_max_pool2d(diff, 1).flatten(1)  # [B,C]
        s = torch.cat([avg, mx], dim=1)  # [B,2C]
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True))).view(-1, self.c_main, 1, 1)
        return x_main + gate * diff


class MIFM(nn.Module):
    """Multi-modal instance-level feature matching with grid_sample patches and sparse reinjection."""

    def __init__(
        self,
        c_main: int,
        c_aux: int,
        patch_size: int = 7,
        n_train: int = 256,
        n_infer: int = 500,
        reduction: int = 16,
        auto_channel_align: bool = False,
    ):
        super().__init__()
        if c_main <= 0 or c_aux <= 0:
            raise ValueError(f"channels must be >0, got c_main={c_main}, c_aux={c_aux}")
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd positive integer")
        if n_train <= 0 or n_infer <= 0:
            raise ValueError("n_train and n_infer must be >0")
        if reduction <= 0:
            raise ValueError("reduction must be >0")

        self.c_main = int(c_main)
        self.c_aux = int(c_aux)
        self.patch_size = int(patch_size)
        self.n_train = int(n_train)
        self.n_infer = int(n_infer)
        self.reduction = int(reduction)
        self.auto_channel_align = bool(auto_channel_align)

        if self.c_main != self.c_aux and not self.auto_channel_align:
            raise ValueError(
                f"MIFM expects equal channels unless auto_channel_align=True, got c_main={self.c_main}, c_aux={self.c_aux}"
            )
        self.aux_in_proj = nn.Identity() if self.c_main == self.c_aux else nn.Conv2d(self.c_aux, self.c_main, 1, 1, 0, bias=False)

        # Shared scorer per scale (per feature map): select top-N centers.
        self.scorer = nn.Conv2d(self.c_main, 1, kernel_size=1, stride=1, padding=0)

        hidden = max(self.c_main // self.reduction, 1)
        self.inst_fc1 = nn.Linear(3 * self.c_main, hidden, bias=True)
        self.inst_fc2 = nn.Linear(hidden, self.c_main, bias=True)
        self.inject_proj = nn.Linear(self.c_main, self.c_main, bias=True)

    def forward(self, x_main: torch.Tensor, x_aux: torch.Tensor | None = None) -> torch.Tensor:
        x_main, x_aux = _unpack_two_inputs(x_main, x_aux)
        _check_4d(x_main, "x_main")
        _check_4d(x_aux, "x_aux")
        if x_main.shape[0] != x_aux.shape[0]:
            raise ValueError(f"batch size mismatch: {x_main.shape[0]} vs {x_aux.shape[0]}")
        if x_main.shape[-2:] != x_aux.shape[-2:]:
            raise ValueError(f"spatial size mismatch: {x_main.shape[-2:]} vs {x_aux.shape[-2:]}")

        b, _, h, w = x_main.shape
        x_aux = self.aux_in_proj(x_aux)

        n = self.n_train if self.training else self.n_infer
        if n <= 0:
            raise ValueError("n_train/n_infer must be >0")
        if n > h * w:
            raise ValueError(f"N_total exceeds feature map size: N_total={n} > H*W={h*w}")

        # Score and top-k indices.
        score = torch.sigmoid(self.scorer(x_main)).flatten(1)  # [B,HW]
        _, idx = torch.topk(score, k=n, dim=1, largest=True, sorted=False)  # [B,N]

        # Convert flat idx to (x, y) in pixel coordinates.
        idx_y = idx // w
        idx_x = idx - idx_y * w

        # Build patch grid around each center for align_corners=False.
        r = self.patch_size // 2
        dx = torch.arange(-r, r + 1, device=x_main.device, dtype=torch.float32)
        dy = torch.arange(-r, r + 1, device=x_main.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(dy, dx, indexing="ij")  # [ps,ps]
        # Offsets in normalized units (1 pixel = 2/W or 2/H).
        off_x = xx * (2.0 / w)
        off_y = yy * (2.0 / h)
        off = torch.stack([off_x, off_y], dim=-1)  # [ps,ps,2]

        # Center coords in normalized units.
        cx = (idx_x.to(torch.float32) + 0.5) * (2.0 / w) - 1.0  # [B,N]
        cy = (idx_y.to(torch.float32) + 0.5) * (2.0 / h) - 1.0  # [B,N]
        center = torch.stack([cx, cy], dim=-1)  # [B,N,2]

        grid = center[:, :, None, None, :].to(torch.float32) + off[None, None, :, :, :]  # [B,N,ps,ps,2]
        grid = grid.clamp(-1.0, 1.0).reshape(b * n, self.patch_size, self.patch_size, 2)

        # Sample patches from both modalities.
        x_main_rep = x_main.unsqueeze(1).expand(b, n, self.c_main, h, w).reshape(b * n, self.c_main, h, w)
        x_aux_rep = x_aux.unsqueeze(1).expand(b, n, self.c_main, h, w).reshape(b * n, self.c_main, h, w)

        p_main = F.grid_sample(x_main_rep, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # [B*N,C,ps,ps]
        p_aux = F.grid_sample(x_aux_rep, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        # Instance vectors via (avg + max) pooling.
        v_main = 0.5 * (p_main.mean(dim=(2, 3)) + p_main.amax(dim=(2, 3)))  # [B*N,C]
        v_aux = 0.5 * (p_aux.mean(dim=(2, 3)) + p_aux.amax(dim=(2, 3)))  # [B*N,C]
        v_main = v_main.view(b, n, self.c_main)
        v_aux = v_aux.view(b, n, self.c_main)

        # Matching matrix (main-to-aux) and aligned aux instance vectors.
        sim = torch.matmul(v_main, v_aux.transpose(1, 2)) / math.sqrt(self.c_main)  # [B,N,N]
        w_mat = torch.softmax(sim, dim=-1)
        v_aux_hat = torch.matmul(w_mat, v_aux)  # [B,N,C]

        # Gated instance differential fusion.
        z = torch.cat([v_main, v_aux_hat, v_main - v_aux_hat], dim=-1)  # [B,N,3C]
        gate = torch.sigmoid(self.inst_fc2(F.relu(self.inst_fc1(z), inplace=True)))  # [B,N,C]
        v_fused = v_main + gate * (v_main - v_aux_hat)  # [B,N,C]

        v_inj = self.inject_proj(v_fused).transpose(1, 2)  # [B,C,N]
        idx_exp = idx.unsqueeze(1).expand(b, self.c_main, n)  # [B,C,N]
        inject = torch.zeros((b, self.c_main, h * w), device=x_main.device, dtype=x_main.dtype)
        inject.scatter_add_(2, idx_exp, v_inj.to(dtype=x_main.dtype))
        inject = inject.view(b, self.c_main, h, w)

        return x_main + inject


class UAMFDetFusion(nn.Module):
    """Combined fusion module for YAML integration: MDSFF -> MDF -> MIFM."""

    def __init__(
        self,
        c_main: int,
        c_aux: int,
        K: int = 8,
        n_heads: int = 4,
        offset_scale: float = 1.0,
        patch_size: int = 7,
        n_train: int = 256,
        n_infer: int = 500,
        enable_mdsff: bool = True,
        enable_mdf: bool = True,
        enable_mifm: bool = True,
        auto_channel_align: bool = False,
    ):
        super().__init__()
        self.c_main = int(c_main)
        self.c_aux = int(c_aux)
        self.enable_mdsff = bool(enable_mdsff)
        self.enable_mdf = bool(enable_mdf)
        self.enable_mifm = bool(enable_mifm)
        self.auto_channel_align = bool(auto_channel_align)

        # Always build submodules (even if disabled) to keep state_dict stable across toggles.
        self.mdsff = MDSFF(self.c_main, self.c_aux, K=K, n_heads=n_heads, offset_scale=offset_scale, auto_channel_align=self.auto_channel_align)
        self.mdf = MDF(self.c_main, self.c_aux, reduction=16, auto_channel_align=self.auto_channel_align)
        self.mifm = MIFM(
            self.c_main,
            self.c_aux,
            patch_size=patch_size,
            n_train=n_train,
            n_infer=n_infer,
            reduction=16,
            auto_channel_align=self.auto_channel_align,
        )

    def forward(self, x_main: torch.Tensor, x_aux: torch.Tensor | None = None) -> torch.Tensor:
        x_main, x_aux = _unpack_two_inputs(x_main, x_aux)
        _check_4d(x_main, "x_main")
        _check_4d(x_aux, "x_aux")
        if x_main.shape[-2:] != x_aux.shape[-2:]:
            raise ValueError(f"spatial size mismatch: {x_main.shape[-2:]} vs {x_aux.shape[-2:]}")

        x_aux_aligned = self.mdsff(x_main, x_aux) if self.enable_mdsff else self.mdsff.aux_in_proj(x_aux)
        x_fine = self.mdf(x_main, x_aux_aligned) if self.enable_mdf else x_main
        x_out = self.mifm(x_fine, x_aux_aligned) if self.enable_mifm else x_fine
        return x_out

