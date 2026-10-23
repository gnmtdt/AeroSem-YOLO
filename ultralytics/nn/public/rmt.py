"""
Retention-based Transformer blocks（源自 upstream ultralytics-yolo11-main/ultralytics/nn/backbone/rmt.py）
仅保留 C3k2_RetBlock 所需的核心组件，未包含模型注册等额外内容。
"""

import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

__all__ = ["DWConv2d", "MaSA", "MaSAd", "FeedForwardNetwork", "RetBlock", "RelPos2d"]


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x


class MaSA(nn.Module):
    """
    全局 Retention (类似注意力) 模块。
    输入/输出形状: (b, h, w, c)
    """

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        bsz, h, w, _ = x.size()
        mask_h, mask_w = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

        qr_w = qr.transpose(1, 2)  # (b h n w d1)
        kr_w = kr.transpose(1, 2)
        v_w = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = torch.softmax(qk_mat_w + mask_w, -1)
        v_w = torch.matmul(qk_mat_w, v_w)

        qr_h = qr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v_h = v_w.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = torch.softmax(qk_mat_h + mask_h, -1)
        output = torch.matmul(qk_mat_h, v_h)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output


class MaSAd(nn.Module):
    """分块 Retention，形状同 MaSA。"""

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        mask_h, mask_w = rel_pos
        bsz, h, w, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 1, 3, 2, 4)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 1, 3, 2, 4)

        qr_h = qr.transpose(1, 2)  # (b w n h d1)
        kr_h = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = torch.softmax(qk_mat_h + mask_h, -1)
        v = torch.matmul(qk_mat_h, v)  # (b w n h d2)

        qr_w = qr  # (b h n w d1)
        kr_w = kr
        v = v.transpose(1, 2).contiguous()  # (b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = torch.softmax(qk_mat_w + mask_w, -1)
        output = torch.matmul(qk_mat_w, v)  # (b h n w d2)

        output = output.permute(0, 1, 3, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class RetBlock(nn.Module):
    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0.0, layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ["chunk", "whole"]
        self.retention = MaSAd(embed_dim, num_heads) if retention == "chunk" else MaSA(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
        else:
            self.gamma_1 = self.gamma_2 = None

    def forward(self, x: torch.Tensor, incremental_state=None, chunkwise_recurrent=False, retention_rel_pos=None):
        x = x + self.pos(x)
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x


class RelPos2d(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_1d_decay(self, l: int):
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        h, w = slen
        if activate_recurrent:
            return self.decay.exp()
        if chunkwise_recurrent:
            mask_h = self.generate_1d_decay(h)
            mask_w = self.generate_1d_decay(w)
            return mask_h, mask_w
        # whole retention
        mask_h = self.generate_1d_decay(h)
        mask_w = self.generate_1d_decay(w)
        return mask_h, mask_w
