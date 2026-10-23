"""AIFI enhancement variants (ported from RTDETR-main)."""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from ..public.transformer_encoder_layer import TransformerEncoderLayer
from .positional import LearnedPositionalEncoding, build_2d_sincos_position_embedding
from .deps.prepbn import LinearNorm, RepBN
from .deps.sefn import SEFN
from .deps.mona import Mona
from .deps.spectral_enhanced_ffn import SpectralEnhancedFFN
from .deps.edffn import EDFFN


_linear_norm_repbn = partial(LinearNorm, norm1=nn.LayerNorm, norm2=RepBN, step=60000)


class AIFI_LPE(TransformerEncoderLayer):
    """AIFI with learned positional encoding."""

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        fmap_size: int = 20 * 20,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.lpe = LearnedPositionalEncoding(int(fmap_size), c1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape[1:]
        pos_embed = self.lpe(h, w, device=x.device).to(dtype=x.dtype)
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()


class TransformerEncoderLayer_RepBN(TransformerEncoderLayer):
    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.norm1 = _linear_norm_repbn(c1)
        self.norm2 = _linear_norm_repbn(c1)


class AIFI_RepBN(TransformerEncoderLayer_RepBN):
    """AIFI with RepBN/LinearNorm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape[1:]
        pos_embed = build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()


class TransformerEncoderLayer_SEFN(nn.Module):
    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError("TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).")
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        self.ffn = SEFN(c1, 2.0, False)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = src.size()
        x_spatial = src
        src = src.flatten(2).permute(0, 2, 1)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(src2.permute(0, 2, 1).view([b, c, h, w]).contiguous(), x_spatial).flatten(2).permute(0, 2, 1)
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = src.size()
        x_spatial = src
        src2 = self.norm1(src.flatten(2).permute(0, 2, 1))
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2n = self.norm2(src)
        src2o = self.ffn(src2n.permute(0, 2, 1).view([b, c, h, w]).contiguous(), x_spatial).flatten(2).permute(0, 2, 1)
        return src + self.dropout2(src2o)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI_SEFN(TransformerEncoderLayer_SEFN):
    """AIFI with SEMNet(SEFN) feed-forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape[1:]
        pos_embed = build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        x = super().forward(x, pos=pos_embed)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()


class TransformerEncoderLayer_Mona(nn.Module):
    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError("TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).")
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.mona1 = Mona(c1)
        self.mona2 = Mona(c1)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        b, c, h, w = src.size()
        src = src.flatten(2).permute(0, 2, 1)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.mona1(src.permute(0, 2, 1).view([-1, c, h, w]).contiguous()).flatten(2).permute(0, 2, 1)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        out = self.norm2(src).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        return self.mona2(out).flatten(2).permute(0, 2, 1)

    def forward_pre(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        b, c, h, w = src.size()
        src = src.flatten(2).permute(0, 2, 1)
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.mona1(src.permute(0, 2, 1).view([-1, c, h, w]).contiguous()).flatten(2).permute(0, 2, 1)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        out = (src + self.dropout2(src2)).permute(0, 2, 1).view([-1, c, h, w]).contiguous()
        return self.mona2(out).flatten(2).permute(0, 2, 1)

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI_Mona(TransformerEncoderLayer_Mona):
    """AIFI with Mona block."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape[1:]
        pos_embed = build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        x = super().forward(x, pos=pos_embed)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape: int, channels_last: bool, alpha_init_value: float = 0.5):
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.alpha_init_value = float(alpha_init_value)
        self.channels_last = bool(channels_last)
        self.alpha = nn.Parameter(torch.ones(1) * self.alpha_init_value)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            return x * self.weight + self.bias
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class TransformerEncoderLayer_DyT(TransformerEncoderLayer):
    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.norm1 = DynamicTanh(normalized_shape=c1, channels_last=True)
        self.norm2 = DynamicTanh(normalized_shape=c1, channels_last=True)


class AIFI_DyT(TransformerEncoderLayer_DyT):
    """AIFI with DynamicTanh normalization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape[1:]
        pos_embed = build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()


class _TransformerEncoderLayer_FFN2D(nn.Module):
    """Shared encoder layer for AIFI_SEFFN/AIFI_EDFFN (attention over tokens + 2D FFN)."""

    def __init__(self, c1: int, ffn_2d: nn.Module, num_heads: int = 8, dropout: float = 0.0, normalize_before: bool = False):
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError("TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).")
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        self.ffn = ffn_2d
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        b, c, h, w = src.size()
        src = src.flatten(2).permute(0, 2, 1)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(src2.permute(0, 2, 1).view([b, c, h, w]).contiguous()).flatten(2).permute(0, 2, 1)
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        b, c, h, w = src.size()
        src2 = self.norm1(src.flatten(2).permute(0, 2, 1))
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src.flatten(2).permute(0, 2, 1) + self.dropout1(src2)
        src2n = self.norm2(src)
        src2o = self.ffn(src2n.permute(0, 2, 1).view([b, c, h, w]).contiguous()).flatten(2).permute(0, 2, 1)
        return src + self.dropout2(src2o)

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, pos=None) -> torch.Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoderLayer_SEFFN(_TransformerEncoderLayer_FFN2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act=nn.GELU(), normalize_before: bool = False):
        super().__init__(c1=c1, ffn_2d=SpectralEnhancedFFN(c1, 2.0, False), num_heads=num_heads, dropout=dropout, normalize_before=normalize_before)


class AIFI_SEFFN(TransformerEncoderLayer_SEFFN):
    """AIFI with SpectralEnhancedFFN."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape[1:]
        pos_embed = build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        x = super().forward(x, pos=pos_embed)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()


class TransformerEncoderLayer_EDFFN(_TransformerEncoderLayer_FFN2D):
    def __init__(self, c1: int, cm: int = 2048, num_heads: int = 8, dropout: float = 0.0, act=nn.GELU(), normalize_before: bool = False):
        super().__init__(c1=c1, ffn_2d=EDFFN(c1, 2.0, False), num_heads=num_heads, dropout=dropout, normalize_before=normalize_before)


class AIFI_EDFFN(TransformerEncoderLayer_EDFFN):
    """AIFI with EDFFN."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape[1:]
        pos_embed = build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        x = super().forward(x, pos=pos_embed)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

