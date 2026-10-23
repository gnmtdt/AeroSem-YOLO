"""
Unified modality filling utilities for multimodal routing.

This module centralizes missing-modality synthesis so that YOLOMM and RTDETRMM
can share identical behavior through the MultiModalRouter.
"""

from typing import Optional, Dict

import torch
import torch.nn.functional as F


class ModalityFiller:
    """
    Modality filler with simple, fast, and deterministic-optional strategies.

    Strategies are lightweight to avoid adding heavy dependencies in the routing path.
    """

    DEFAULT_STRATEGY_WEIGHTS = {
        "copy": 0.3,
        "noise": 0.25,
        "channel_repeat": 0.2,
        "edge_blur": 0.15,
        "mixed": 0.1,
    }

    def __init__(self, strategy_weights: Optional[Dict[str, float]] = None, noise_std: float = 0.1, blur_kernel_size: int = 5):
        self.strategy_weights = strategy_weights or self.DEFAULT_STRATEGY_WEIGHTS
        self.noise_std = noise_std
        self.blur_kernel_size = blur_kernel_size

    # --- public API ---
    def generate(self, source_tensor: torch.Tensor, source_modality: str, target_modality: str, strategy: Optional[str] = None) -> torch.Tensor:
        if strategy is None:
            strategy = self._select_random_strategy()
        if strategy == "copy":
            return self._create_copy_fill(source_tensor)
        if strategy == "noise":
            return self._create_noise_fill(source_tensor)
        if strategy == "channel_repeat":
            return self._create_channel_repeat_fill(source_tensor)
        if strategy == "edge_blur":
            return self._create_edge_blur_fill(source_tensor)
        if strategy == "mixed":
            return self._create_mixed_fill(source_tensor)
        # default strategy
        return self._create_copy_fill(source_tensor)

    def get_statistics(self, tensor: torch.Tensor) -> Dict[str, float]:
        return {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "max": tensor.max().item(),
            "min": tensor.min().item(),
            "shape": list(tensor.shape),
        }

    # --- strategies ---
    def _select_random_strategy(self) -> str:
        # Lazy import to avoid extra deps at import-time
        import random

        strategies = list(self.strategy_weights.keys())
        weights = list(self.strategy_weights.values())
        return random.choices(strategies, weights=weights)[0]

    def _create_copy_fill(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()

    def _create_noise_fill(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.noise_std
        out = tensor + noise
        return out.clamp_(0.0, 1.0)

    def _create_channel_repeat_fill(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] == 3:
            gray = tensor.mean(dim=1, keepdim=True)
            return gray.repeat(1, 3, 1, 1)
        # if single channel, repeat to 3
        return tensor.repeat(1, 3, 1, 1)

    def _create_edge_blur_fill(self, tensor: torch.Tensor) -> torch.Tensor:
        # Sobel kernels
        device = tensor.device
        dtype = tensor.dtype
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        edges = []
        for c in range(tensor.shape[1]):
            ch = tensor[:, c:c+1]
            ex = F.conv2d(ch, kx, padding=1)
            ey = F.conv2d(ch, ky, padding=1)
            edges.append(torch.sqrt(ex * ex + ey * ey))
        edge = torch.cat(edges, dim=1)
        return _gaussian_blur(edge, self.blur_kernel_size)

    def _create_mixed_fill(self, tensor: torch.Tensor) -> torch.Tensor:
        import random

        candidates = ["copy", "noise", "channel_repeat", "edge_blur"]
        sel = random.sample(candidates, random.randint(2, min(3, len(candidates))))
        outs = []
        for s in sel:
            if s == "copy":
                outs.append(self._create_copy_fill(tensor))
            elif s == "noise":
                outs.append(self._create_noise_fill(tensor))
            elif s == "channel_repeat":
                outs.append(self._create_channel_repeat_fill(tensor))
            elif s == "edge_blur":
                outs.append(self._create_edge_blur_fill(tensor))
        weights = torch.softmax(torch.rand(len(outs), device=tensor.device, dtype=tensor.dtype), dim=0)
        out = torch.zeros_like(tensor)
        for w, o in zip(weights, outs):
            out = out + w * o
        return out


def _gaussian_blur(tensor: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return tensor
    device = tensor.device
    dtype = tensor.dtype
    sigma = k / 3.0
    x = torch.arange(k, dtype=dtype, device=device) - k // 2
    g1 = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    g1 = g1 / g1.sum()
    g2 = (g1.view(1, 1, 1, -1) * g1.view(1, 1, -1, 1))
    outs = []
    for c in range(tensor.shape[1]):
        ch = tensor[:, c:c+1]
        outs.append(F.conv2d(ch, g2, padding=k // 2))
    return torch.cat(outs, dim=1)


_default_filler = ModalityFiller()


def generate_modality_filling(source_tensor: torch.Tensor,
                              source_modality: str,
                              target_modality: str,
                              *,
                              target_ch: Optional[int] = None,
                              strategy: Optional[str] = None,
                              filler: Optional[ModalityFiller] = None) -> torch.Tensor:
    """生成目标模态占位张量。

    Args:
        source_tensor: 源模态张量 (B, C, H, W)
        source_modality: 源模态名称
        target_modality: 目标模态名称
        target_ch: 目标通道数，不为 None 时自动调用 adapt_xch 适配
        strategy: 填充策略
        filler: 自定义填充器
    """
    f = filler or _default_filler
    filled = f.generate(source_tensor, source_modality, target_modality, strategy)
    if target_ch is not None:
        filled = adapt_xch(filled, target_ch)
    return filled


def adapt_xch(tensor: torch.Tensor, xch: int) -> torch.Tensor:
    """
    Adapt channel count to desired Xch or 3 for RGB.
    If current C == xch, return as-is. If 3->1 use mean; If 1->3 repeat.
    """
    c = tensor.shape[1]
    if c == xch:
        return tensor
    if c == 3 and xch == 1:
        return tensor.mean(dim=1, keepdim=True)
    if c == 1 and xch == 3:
        return tensor.repeat(1, 3, 1, 1)
    # generic projection: interpolate across channels by linear projection
    # create simple fixed weights to map C->xch
    device, dtype = tensor.device, tensor.dtype
    W = torch.zeros((xch, c), device=device, dtype=dtype)
    for i in range(min(xch, c)):
        W[i, i] = 1.0
    # (B,C,H,W) -> (B,H,W,C) -> matmul -> (B,H,W,xch) -> (B,xch,H,W)
    B, C, H, W = tensor.shape
    t = tensor.permute(0, 2, 3, 1).reshape(-1, C)
    out = t @ W.T
    out = out.reshape(B, H, W, xch).permute(0, 3, 1, 2).contiguous()
    return out
