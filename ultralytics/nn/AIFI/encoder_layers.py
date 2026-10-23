"""RT-DETR AIFI-related encoder layer variants (ported/adapted from RTDETR-main)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv

from .deps.edffn import EDFFN
from .deps.mona import Mona
from .deps.sefn import SEFN
from .deps.spectral_enhanced_ffn import SpectralEnhancedFFN
from .variants import DynamicTanh


def _lazy_import_c2psa_base():
    """延迟导入，避免与 `ultralytics.nn.modules.block -> transformer -> AIFI` 形成循环依赖。"""
    from ..extraction import c2psa_base

    return c2psa_base


def _lazy_import_common_base():
    """延迟导入，避免循环依赖。"""
    from ..extraction import common_base

    return common_base


def _require_fixed_hw(src: torch.Tensor, *, expected_hw: tuple[int, int] = (20, 20), module_name: str) -> None:
    """Fail-Fast：固定 token 分辨率算子入口校验（避免隐式行为变化/自动适配）。"""
    from ultralytics.utils import LOGGER as _LOGGER

    _, _, h, w = src.shape
    eh, ew = expected_hw
    if h == eh and w == ew:
        return

    n = h * w
    en = eh * ew
    msg = (
        f"{module_name} 仅支持输入特征分辨率 {eh}×{ew}（N={en} tokens）；"
        f"当前为 {h}×{w}（N={n}）。"
        "该变体内部固定假设 P5=20×20（通常 imgsz=640 时成立）。"
        "请将 imgsz 调整为满足该约束，或改用非 DPB/Pola 的 AIFI 变体。"
    )
    _LOGGER.error(msg)
    raise RuntimeError(msg)


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_first (B,C,H,W) with per-pixel normalization over C."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module (SwiftFormer-style), implemented without external deps.

    Input/Output: [B, C, H, W]
    """

    def __init__(self, in_dims: int = 512, num_heads: int = 1):
        super().__init__()
        token_dim = in_dims
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim**-0.5
        self.proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x_4d: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_4d.size()
        n = h * w
        x = x_4d.flatten(2).transpose(2, 1)  # [B, N, C]
        query = F.normalize(self.to_query(x), dim=-1)
        key = F.normalize(self.to_key(x), dim=-1)

        query_weight = query @ self.w_g  # [B, N, 1]
        a = F.normalize(query_weight * self.scale_factor, dim=1)  # [B, N, 1]
        g = torch.sum(a * query, dim=1)  # [B, D]
        g = g.unsqueeze(1).expand(-1, n, -1)  # [B, N, D]

        out = self.proj(g * key) + query
        out = self.final(out)  # [B, N, C]
        return out.transpose(2, 1).reshape((b, c, h, w)).contiguous()


class HiLo(nn.Module):
    """
    HiLo Attention (ported from RTDETR-main/extra_modules/attention.py), no external deps.

    Input/Output: [B, C, H, W]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size: int = 2,
        alpha: float = 0.5,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)
        self.dim = dim

        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim
        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim
        self.ws = int(window_size)

        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim**-0.5

        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=self.ws, stride=self.ws)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _hifi(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        hg, wg = h // self.ws, w // self.ws
        total_groups = hg * wg
        x = x.reshape(b, hg, self.ws, wg, self.ws, c).transpose(2, 3)  # [B, hg, wg, ws, ws, C]

        qkv = self.h_qkv(x).reshape(b, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = (attn @ v).transpose(2, 3).reshape(b, hg, wg, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(b, hg * self.ws, wg * self.ws, self.h_dim)
        return self.h_proj(x)

    def _lofi(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        q = self.l_q(x).reshape(b, h * w, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(b, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(b, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, h, w, self.l_dim)
        return self.l_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        if self.h_heads == 0:
            return self._lofi(x).permute(0, 3, 1, 2).contiguous()
        if self.l_heads == 0:
            return self._hifi(x).permute(0, 3, 1, 2).contiguous()
        hifi_out = self._hifi(x)
        lofi_out = self._lofi(x)
        out = torch.cat((hifi_out, lofi_out), dim=-1)
        return out.permute(0, 3, 1, 2).contiguous()


class AttentionTSSA(nn.Module):
    """Token Statistics Self-Attention (ToST), implemented without einops dependency."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.heads = int(num_heads)
        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.temp = nn.Parameter(torch.ones(self.heads, 1))
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        if c % self.heads != 0:
            raise ValueError(f"AttentionTSSA expects C divisible by heads, got C={c}, heads={self.heads}")
        d = c // self.heads
        w = self.qkv(x).view(b, n, self.heads, d).permute(0, 2, 1, 3).contiguous()  # [B, H, N, D]

        w_normed = F.normalize(w, dim=-2)
        w_sq = w_normed**2
        pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # [B, H, N]
        pi_norm = pi / (pi.sum(dim=-1, keepdim=True) + 1e-8)
        dots = torch.matmul(pi_norm.unsqueeze(-2), w**2)  # [B, H, 1, D]
        attn = 1.0 / (1.0 + dots)
        attn = self.attn_drop(attn)
        out = -w * pi.unsqueeze(-1) * attn  # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, c)
        return self.to_out(out)


class AttentionHistogram(nn.Module):
    """Dynamic-range Histogram Self-Attention (DHSA) ported from RTDETR-main without einops."""

    def __init__(self, dim: int, num_heads: int = 8, bias: bool = False, if_box: bool = True):
        super().__init__()
        self.factor = int(num_heads)
        self.if_box = bool(if_box)
        self.num_heads = int(num_heads)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    @staticmethod
    def _pad_1d(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, tuple[int, int]]:
        hw = x.shape[-1]
        pad = (0, 0) if hw % factor == 0 else (0, (hw // factor + 1) * factor - hw)
        return F.pad(x, pad, "constant", 0), pad

    @staticmethod
    def _unpad_1d(x: torch.Tensor, pad: tuple[int, int]) -> torch.Tensor:
        hw = x.shape[-1]
        return x[..., pad[0] : hw - pad[1]]

    @staticmethod
    def _softmax_1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        logit = x.exp()
        return logit / (logit.sum(dim, keepdim=True) + 1)

    def _reshape_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, if_box: bool) -> torch.Tensor:
        b, c = q.shape[:2]
        q, pad = self._pad_1d(q, self.factor)
        k, _ = self._pad_1d(k, self.factor)
        v, _ = self._pad_1d(v, self.factor)
        n = q.shape[-1]
        hw = n // self.factor

        heads = self.num_heads
        if c % heads != 0:
            raise ValueError(f"AttentionHistogram expects C divisible by heads, got C={c}, heads={heads}")
        c_per_head = c // heads

        if if_box:
            # [B, (H*C), (F*HW)] -> [B, H, C, F, HW] -> [B, H, (C*F), HW]
            qh = q.view(b, heads, c_per_head, self.factor, hw).reshape(b, heads, c_per_head * self.factor, hw)
            kh = k.view(b, heads, c_per_head, self.factor, hw).reshape(b, heads, c_per_head * self.factor, hw)
            vh = v.view(b, heads, c_per_head, self.factor, hw).reshape(b, heads, c_per_head * self.factor, hw)
        else:
            # [B, (H*C), (HW*F)] -> [B, H, C, HW, F] -> [B, H, (C*F), HW]
            qh = q.view(b, heads, c_per_head, hw, self.factor).permute(0, 1, 2, 4, 3).reshape(b, heads, c_per_head * self.factor, hw)
            kh = k.view(b, heads, c_per_head, hw, self.factor).permute(0, 1, 2, 4, 3).reshape(b, heads, c_per_head * self.factor, hw)
            vh = v.view(b, heads, c_per_head, hw, self.factor).permute(0, 1, 2, 4, 3).reshape(b, heads, c_per_head * self.factor, hw)

        qh = F.normalize(qh, dim=-1)
        kh = F.normalize(kh, dim=-1)
        attn = (qh @ kh.transpose(-2, -1)) * self.temperature  # [B, H, CF, CF]
        attn = self._softmax_1(attn, dim=-1)
        out = attn @ vh  # [B, H, CF, HW]

        if if_box:
            out = out.view(b, heads, c_per_head, self.factor, hw).reshape(b, heads * c_per_head, self.factor * hw)
        else:
            out = out.view(b, heads, c_per_head, self.factor, hw).permute(0, 1, 2, 4, 3).reshape(b, heads * c_per_head, hw * self.factor)

        return self._unpad_1d(out, pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # sort first half channels spatially (as in RTDETR-main)
        x_sort, idx_h = x[:, : c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x = x.clone()
        x[:, : c // 2] = x_sort

        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        v_flat, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self._reshape_attn(q1, k1, v_flat, True)
        out2 = self._reshape_attn(q2, k2, v_flat, False)

        # scatter back to original index and restore spatial sort
        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = self.project_out(out1 * out2)

        out_replace = out[:, : c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, : c // 2] = out_replace
        return out


class _ConvFFNEncoder2D(nn.Module):
    """Common residual + norm + conv-ffn skeleton for 2D attention modules."""

    def __init__(self, c1: int, cm: int = 2048, dropout: float = 0.0, act: nn.Module = nn.GELU()):
        super().__init__()
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)
        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    def _ffn(self, src: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(src))))


class TransformerEncoderLayer_LocalWindowAttention(_ConvFFNEncoder2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), window_resolution: int = 14):
        super().__init__(c1, cm, dropout, act)
        c2psa_base = _lazy_import_c2psa_base()
        self.local_windows_attention = c2psa_base.LocalWindowAttention(
            dim=c1, num_heads=num_heads, resolution=20, window_resolution=window_resolution
        )

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src2 = self.local_windows_attention(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self._ffn(src))
        return self.norm2(src)


class TransformerEncoderLayer_DAttention(_ConvFFNEncoder2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), q_size: tuple[int, int] = (20, 20)):
        super().__init__(c1, cm, dropout, act)
        c2psa_base = _lazy_import_c2psa_base()
        self.dattention = c2psa_base.DAttention(channel=c1, q_size=q_size, n_heads=num_heads)

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src2 = self.dattention(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self._ffn(src))
        return self.norm2(src)


class TransformerEncoderLayer_HiLo(_ConvFFNEncoder2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU()):
        super().__init__(c1, cm, dropout, act)
        self.hilo = HiLo(dim=c1, num_heads=num_heads)

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src2 = self.hilo(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self._ffn(src))
        return self.norm2(src)


class TransformerEncoderLayer_EfficientAdditiveAttnetion(_ConvFFNEncoder2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU()):
        super().__init__(c1, cm, dropout, act)
        self.effaddattention = EfficientAdditiveAttnetion(in_dims=c1, num_heads=1)

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src2 = self.effaddattention(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self._ffn(src))
        return self.norm2(src)


class TransformerEncoderLayer_AdditiveTokenMixer(_ConvFFNEncoder2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU()):
        super().__init__(c1, cm, dropout, act)
        common_base = _lazy_import_common_base()
        self.additivetoken = common_base.AdditiveTokenMixer(dim=c1)

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src2 = self.additivetoken(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self._ffn(src))
        return self.norm2(src)


class _MutilScal(nn.Module):
    def __init__(self, dim: int = 512, fc_ratio: int = 4, dilation: list[int] | tuple[int, ...] = (3, 5, 7), pool_ratio: int = 16):
        super().__init__()
        d = list(dilation)
        self.conv0_1 = Conv(dim, dim // fc_ratio)
        self.conv0_2 = Conv(dim // fc_ratio, dim // fc_ratio, 3, d=d[-3], g=dim // fc_ratio)
        self.conv0_3 = Conv(dim // fc_ratio, dim, 1)
        self.conv1_2 = Conv(dim // fc_ratio, dim // fc_ratio, 3, d=d[-2], g=dim // fc_ratio)
        self.conv1_3 = Conv(dim // fc_ratio, dim, 1)
        self.conv2_2 = Conv(dim // fc_ratio, dim // fc_ratio, 3, d=d[-1], g=dim // fc_ratio)
        self.conv2_3 = Conv(dim // fc_ratio, dim, 1)
        self.conv3 = Conv(dim, dim, 1)
        self.avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        attn0_1 = self.conv0_1(x)
        attn0_2 = self.conv0_2(attn0_1)
        attn0_3 = self.conv0_3(attn0_2)
        attn1_2 = self.conv1_2(attn0_1)
        attn1_3 = self.conv1_3(attn1_2)
        attn2_2 = self.conv2_2(attn0_1)
        attn2_3 = self.conv2_3(attn2_2)
        attn = self.conv3(attn0_3 + attn1_3 + attn2_3)
        attn = attn * u
        return self.avg(attn)


class _Mutilscal_MHSA(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, atten_drop: float = 0.0, proj_drop: float = 0.0, dilation=(3, 5, 7), fc_ratio: int = 4, pool_ratio: int = 16):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.msc = _MutilScal(dim=dim, fc_ratio=fc_ratio, dilation=list(dilation), pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim // fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.kv = Conv(dim, 2 * dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        b, c, h, w = x.shape
        kv = self.kv(self.msc(x))  # [B, 2C, H1, W1]
        _, _, h1, w1 = kv.shape
        n = h * w
        n1 = h1 * w1

        # q: [B, heads, N, d]
        d = c // self.num_heads
        q = x.view(b, self.num_heads, d, n).permute(0, 1, 3, 2).contiguous()
        kv = kv.view(b, 2, self.num_heads, d, n1).permute(1, 0, 2, 4, 3).contiguous()
        k, v = kv[0], kv[1]

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.atten_drop(dots.softmax(dim=-1))
        attn = attn @ v  # [B, heads, N, d]
        attn = attn.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)

        c_attn = self.fc(self.avgpool(x)) * u
        return attn + c_attn


class TransformerEncoderLayer_MSMHSA(_ConvFFNEncoder2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__(c1, cm, dropout, act)
        self.msmhsa = _Mutilscal_MHSA(c1, num_heads=num_heads)
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src2 = self.msmhsa(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self._ffn(src))
        return self.norm2(src)


class TransformerEncoderLayer_DHSA(_ConvFFNEncoder2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__(c1, cm, dropout, act)
        self.dhsa = AttentionHistogram(c1, num_heads=num_heads)
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src2 = self.dhsa(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self._ffn(src))
        return self.norm2(src)


class TransformerEncoderLayer_DPB(_ConvFFNEncoder2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__(c1, cm, dropout, act)
        c2psa_base = _lazy_import_c2psa_base()
        self.dpb_attention = c2psa_base.DPB_Attention(c1, (20, 20), num_heads=num_heads)
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _require_fixed_hw(src, module_name=self.__class__.__name__)
        b, c, h, w = src.size()
        src2 = self.dpb_attention(src.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self._ffn(src))
        return self.norm2(src)


class TransformerEncoderLayer_Pola(_ConvFFNEncoder2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__(c1, cm, dropout, act)
        c2psa_base = _lazy_import_c2psa_base()
        self.pola_attention = c2psa_base.PolaLinearAttention(c1, (20, 20), num_heads=num_heads)
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _require_fixed_hw(src, module_name=self.__class__.__name__)
        b, c, h, w = src.size()
        src2 = self.pola_attention(src.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self._ffn(src))
        return self.norm2(src)


class TransformerEncoderLayer_TSSA(_ConvFFNEncoder2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__(c1, cm, dropout, act)
        self.tssa = AttentionTSSA(c1, num_heads=num_heads)
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        b, c, h, w = src.size()
        src2 = self.tssa(src.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self._ffn(src))
        return self.norm2(src)


class TransformerEncoderLayer_ASSA(_ConvFFNEncoder2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__(c1, cm, dropout, act)
        c2psa_base = _lazy_import_c2psa_base()
        self.assa = c2psa_base.AdaptiveSparseSA(c1, num_heads=num_heads, sparseAtt=True)
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        b, c, h, w = src.size()
        src2 = self.assa(src).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self._ffn(src))
        return self.norm2(src)


class TransformerEncoderLayer_MSLA(_ConvFFNEncoder2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__(c1, cm, dropout, act)
        c2psa_base = _lazy_import_c2psa_base()
        self.msla = c2psa_base.MSLA(c1, num_heads=num_heads)
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        b, c, h, w = src.size()
        src2 = self.msla(src.flatten(2).permute(0, 2, 1), hw=(h, w)).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self._ffn(src))
        return self.norm2(src)

class TransformerEncoderLayer_Pola_SEFN(nn.Module):
    """Pola attention + SEFN 2D-FFN (ported from RTDETR-main)."""

    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__()
        c2psa_base = _lazy_import_c2psa_base()
        self.pola_attention = c2psa_base.PolaLinearAttention(c1, (20, 20), num_heads=num_heads)
        self.ffn = SEFN(c1, 2.0, False)
        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        _require_fixed_hw(src, module_name=self.__class__.__name__)
        b, c, h, w = src.size()
        src_spatial = src
        src2 = self.pola_attention(src.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(src, src_spatial)
        src = src + self.dropout2(src2)
        return self.norm2(src)


class TransformerEncoderLayer_ASSA_SEFN(nn.Module):
    """ASSA attention + SEFN 2D-FFN (ported from RTDETR-main)."""

    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__()
        c2psa_base = _lazy_import_c2psa_base()
        self.assa = c2psa_base.AdaptiveSparseSA(c1, num_heads=num_heads, sparseAtt=True)
        self.ffn = SEFN(c1, 2.0, False)
        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        b, c, h, w = src.size()
        src_spatial = src
        src2 = self.assa(src).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(src, src_spatial)
        src = src + self.dropout2(src2)
        return self.norm2(src)


class TransformerEncoderLayer_ASSA_SEFN_Mona(nn.Module):
    """ASSA + SEFN + Mona (ported from RTDETR-main)."""

    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__()
        c2psa_base = _lazy_import_c2psa_base()
        self.assa = c2psa_base.AdaptiveSparseSA(c1, num_heads=num_heads, sparseAtt=True)
        self.ffn = SEFN(c1, 2.0, False)
        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mona1 = Mona(c1)
        self.mona2 = Mona(c1)
        self.act = act
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        b, c, h, w = src.size()
        src_spatial = src
        src2 = self.assa(src).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.mona1(src)
        src2 = self.ffn(src, src_spatial)
        src = src + self.dropout2(src2)
        return self.mona2(self.norm2(src))


class TransformerEncoderLayer_Pola_SEFN_Mona(nn.Module):
    """Pola + SEFN + Mona (ported from RTDETR-main)."""

    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__()
        c2psa_base = _lazy_import_c2psa_base()
        self.pola_attention = c2psa_base.PolaLinearAttention(c1, (20, 20), num_heads=num_heads)
        self.ffn = SEFN(c1, 2.0, False)
        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mona1 = Mona(c1)
        self.mona2 = Mona(c1)
        self.act = act
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        _require_fixed_hw(src, module_name=self.__class__.__name__)
        b, c, h, w = src.size()
        src_spatial = src
        src2 = self.pola_attention(src.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.mona1(src)
        src2 = self.ffn(src, src_spatial)
        src = src + self.dropout2(src2)
        return self.mona2(self.norm2(src))


class TransformerEncoderLayer_ASSA_SEFN_Mona_DyT(nn.Module):
    """ASSA + SEFN + Mona + DyT norm (ported from RTDETR-main)."""

    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__()
        c2psa_base = _lazy_import_c2psa_base()
        self.assa = c2psa_base.AdaptiveSparseSA(c1, num_heads=num_heads, sparseAtt=True)
        self.ffn = SEFN(c1, 2.0, False)
        self.norm1 = DynamicTanh(c1, channels_last=False)
        self.norm2 = DynamicTanh(c1, channels_last=False)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mona1 = Mona(c1)
        self.mona2 = Mona(c1)
        self.act = act
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        b, c, h, w = src.size()
        src_spatial = src
        src2 = self.assa(src).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.mona1(src)
        src2 = self.ffn(src, src_spatial)
        src = src + self.dropout2(src2)
        return self.mona2(self.norm2(src))


class TransformerEncoderLayer_Pola_SEFN_Mona_DyT(nn.Module):
    """Pola + SEFN + Mona + DyT norm (ported from RTDETR-main)."""

    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__()
        c2psa_base = _lazy_import_c2psa_base()
        self.pola_attention = c2psa_base.PolaLinearAttention(c1, (20, 20), num_heads=num_heads)
        self.ffn = SEFN(c1, 2.0, False)
        self.norm1 = DynamicTanh(c1, channels_last=False)
        self.norm2 = DynamicTanh(c1, channels_last=False)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mona1 = Mona(c1)
        self.mona2 = Mona(c1)
        self.act = act
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        _require_fixed_hw(src, module_name=self.__class__.__name__)
        b, c, h, w = src.size()
        src_spatial = src
        src2 = self.pola_attention(src.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.mona1(src)
        src2 = self.ffn(src, src_spatial)
        src = src + self.dropout2(src2)
        return self.mona2(self.norm2(src))


class TransformerEncoderLayer_Pola_SEFFN_Mona_DyT(nn.Module):
    """Pola + SpectralEnhancedFFN + Mona + DyT norm (ported from RTDETR-main)."""

    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__()
        c2psa_base = _lazy_import_c2psa_base()
        self.pola_attention = c2psa_base.PolaLinearAttention(c1, (20, 20), num_heads=num_heads)
        self.ffn = SpectralEnhancedFFN(c1, 2.0, False)
        self.norm1 = DynamicTanh(c1, channels_last=False)
        self.norm2 = DynamicTanh(c1, channels_last=False)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mona1 = Mona(c1)
        self.mona2 = Mona(c1)
        self.act = act
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        _require_fixed_hw(src, module_name=self.__class__.__name__)
        b, c, h, w = src.size()
        src2 = self.pola_attention(src.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.mona1(src)
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        return self.mona2(self.norm2(src))


class TransformerEncoderLayer_Pola_EDFFN_Mona_DyT(nn.Module):
    """Pola + EDFFN + Mona + DyT norm (ported from RTDETR-main)."""

    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act: nn.Module = nn.GELU(), normalize_before: bool = False):
        super().__init__()
        c2psa_base = _lazy_import_c2psa_base()
        self.pola_attention = c2psa_base.PolaLinearAttention(c1, (20, 20), num_heads=num_heads)
        self.ffn = EDFFN(c1, 2.0, False)
        self.norm1 = DynamicTanh(c1, channels_last=False)
        self.norm2 = DynamicTanh(c1, channels_last=False)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mona1 = Mona(c1)
        self.mona2 = Mona(c1)
        self.act = act
        self.normalize_before = normalize_before

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        _require_fixed_hw(src, module_name=self.__class__.__name__)
        b, c, h, w = src.size()
        src2 = self.pola_attention(src.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.mona1(src)
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        return self.mona2(self.norm2(src))


__all__ = [
    "TransformerEncoderLayer_LocalWindowAttention",
    "TransformerEncoderLayer_DAttention",
    "TransformerEncoderLayer_HiLo",
    "TransformerEncoderLayer_EfficientAdditiveAttnetion",
    "TransformerEncoderLayer_AdditiveTokenMixer",
    "TransformerEncoderLayer_MSMHSA",
    "TransformerEncoderLayer_DHSA",
    "TransformerEncoderLayer_DPB",
    "TransformerEncoderLayer_Pola",
    "TransformerEncoderLayer_TSSA",
    "TransformerEncoderLayer_ASSA",
    "TransformerEncoderLayer_MSLA",
    "TransformerEncoderLayer_Pola_SEFN",
    "TransformerEncoderLayer_ASSA_SEFN",
    "TransformerEncoderLayer_ASSA_SEFN_Mona",
    "TransformerEncoderLayer_Pola_SEFN_Mona",
    "TransformerEncoderLayer_ASSA_SEFN_Mona_DyT",
    "TransformerEncoderLayer_Pola_SEFN_Mona_DyT",
    "TransformerEncoderLayer_Pola_SEFFN_Mona_DyT",
    "TransformerEncoderLayer_Pola_EDFFN_Mona_DyT",
]
