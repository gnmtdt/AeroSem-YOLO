"""
CTrans - 跨尺度通道 Transformer (AAAI 2022)
论文: Channel Transformer for Visual Recognition (AAAI 2022)
论文链接: https://ojs.aaai.org/index.php/AAAI/article/view/20144

通过跨尺度通道注意力机制实现多尺度特征的交互融合，
适用于替代 FPN/PAN 中的标准特征金字塔结构。
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Dropout, LayerNorm, Softmax
from torch.nn.modules.utils import _pair


class Channel_Embeddings(nn.Module):
    """Patch embedding used by ChannelTransformer for each input scale."""

    def __init__(self, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.grid_size = int(math.sqrt(n_patches))
        self.patch_embeddings = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=5),
            Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=patchsize // 5,
                stride=patchsize // 5,
            ),
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        pool_kernel_h, pool_kernel_w = _pair(self.patch_embeddings[0].kernel_size)
        kernel_h, kernel_w = self.patch_embeddings[1].kernel_size
        if x.shape[-2] < pool_kernel_h or x.shape[-1] < pool_kernel_w:
            pooled = F.adaptive_avg_pool2d(x, (self.grid_size * kernel_h, self.grid_size * kernel_w))
        else:
            pooled = self.patch_embeddings[0](x)
        if pooled.shape[-2] < kernel_h or pooled.shape[-1] < kernel_w:
            x = F.adaptive_avg_pool2d(pooled, (self.grid_size, self.grid_size))
        else:
            x = self.patch_embeddings[1](pooled)
        x = x.flatten(2).transpose(-1, -2)
        if x.shape[1] != self.position_embeddings.shape[1]:
            pos = self.position_embeddings.transpose(1, 2)
            pos = F.interpolate(pos, size=x.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        else:
            pos = self.position_embeddings
        return self.dropout(x + pos)


class Reconstruct(nn.Module):
    """Project token sequences back to feature maps at the original pyramid scale."""

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x, output_size):
        if x is None:
            return None
        batch, n_patch, hidden = x.size()
        h = w = int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1).contiguous().view(batch, hidden, h, w)
        x = F.interpolate(x, size=output_size, mode="nearest")
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class Attention_org(nn.Module):
    """Cross-scale channel attention core used by ChannelTransformer."""

    def __init__(self, vis, channel_num):
        super().__init__()
        self.vis = vis
        self.kv_size = sum(channel_num)
        self.num_attention_heads = 2

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.query4 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(self.num_attention_heads):
            self.query1.append(copy.deepcopy(nn.Linear(channel_num[0], channel_num[0], bias=False)))
            self.query2.append(copy.deepcopy(nn.Linear(channel_num[1], channel_num[1], bias=False)))
            self.query3.append(copy.deepcopy(nn.Linear(channel_num[2], channel_num[2], bias=False)))
            self.query4.append(
                copy.deepcopy(nn.Linear(channel_num[3], channel_num[3], bias=False))
                if len(channel_num) == 4
                else nn.Identity()
            )
            self.key.append(copy.deepcopy(nn.Linear(self.kv_size, self.kv_size, bias=False)))
            self.value.append(copy.deepcopy(nn.Linear(self.kv_size, self.kv_size, bias=False)))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False) if len(channel_num) == 4 else nn.Identity()
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)

    def _stack_queries(self, emb, query_layers):
        if emb is None:
            return None
        multi_head_q = torch.stack([query(emb) for query in query_layers], dim=1)
        return multi_head_q.transpose(-1, -2)

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        multi_head_q1 = self._stack_queries(emb1, self.query1)
        multi_head_q2 = self._stack_queries(emb2, self.query2)
        multi_head_q3 = self._stack_queries(emb3, self.query3)
        multi_head_q4 = self._stack_queries(emb4, self.query4)

        multi_head_k = torch.stack([key(emb_all) for key in self.key], dim=1)
        multi_head_v = torch.stack([value(emb_all) for value in self.value], dim=1)

        def attend(multi_head_q):
            if multi_head_q is None:
                return None
            attention_scores = torch.matmul(multi_head_q, multi_head_k) / math.sqrt(self.kv_size)
            attention_probs = self.softmax(self.psi(attention_scores))
            attention_probs = self.attn_dropout(attention_probs)
            context = torch.matmul(attention_probs, multi_head_v.transpose(-1, -2))
            context = context.permute(0, 3, 2, 1).contiguous().mean(dim=3)
            return context, attention_probs

        ctx1 = attend(multi_head_q1)
        ctx2 = attend(multi_head_q2)
        ctx3 = attend(multi_head_q3)
        ctx4 = attend(multi_head_q4)

        weights = None
        if self.vis:
            weights = [
                None if ctx1 is None else ctx1[1].mean(1),
                None if ctx2 is None else ctx2[1].mean(1),
                None if ctx3 is None else ctx3[1].mean(1),
                None if ctx4 is None else ctx4[1].mean(1),
            ]

        out1 = None if ctx1 is None else self.proj_dropout(self.out1(ctx1[0]))
        out2 = None if ctx2 is None else self.proj_dropout(self.out2(ctx2[0]))
        out3 = None if ctx3 is None else self.proj_dropout(self.out3(ctx3[0]))
        out4 = None if ctx4 is None else self.proj_dropout(self.out4(ctx4[0]))
        return out1, out2, out3, out4, weights


class Mlp(nn.Module):
    """Feed-forward block inside each ChannelTransformer layer."""

    def __init__(self, in_channel, mlp_channel):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class Block_ViT(nn.Module):
    """Single cross-scale transformer block."""

    def __init__(self, vis, channel_num):
        super().__init__()
        expand_ratio = 4
        self.attn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3], eps=1e-6) if len(channel_num) == 4 else nn.Identity()
        self.attn_norm = LayerNorm(sum(channel_num), eps=1e-6)
        self.channel_attn = Attention_org(vis, channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3], eps=1e-6) if len(channel_num) == 4 else nn.Identity()
        self.ffn1 = Mlp(channel_num[0], channel_num[0] * expand_ratio)
        self.ffn2 = Mlp(channel_num[1], channel_num[1] * expand_ratio)
        self.ffn3 = Mlp(channel_num[2], channel_num[2] * expand_ratio)
        self.ffn4 = Mlp(channel_num[3], channel_num[3] * expand_ratio) if len(channel_num) == 4 else nn.Identity()

    def forward(self, emb1, emb2, emb3, emb4):
        embcat = [emb for emb in (emb1, emb2, emb3, emb4) if emb is not None]
        emb_all = self.attn_norm(torch.cat(embcat, dim=2))

        cx1 = None if emb1 is None else self.attn_norm1(emb1)
        cx2 = None if emb2 is None else self.attn_norm2(emb2)
        cx3 = None if emb3 is None else self.attn_norm3(emb3)
        cx4 = None if emb4 is None else self.attn_norm4(emb4)

        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)
        cx1 = None if emb1 is None else emb1 + cx1
        cx2 = None if emb2 is None else emb2 + cx2
        cx3 = None if emb3 is None else emb3 + cx3
        cx4 = None if emb4 is None else emb4 + cx4

        x1 = None if cx1 is None else self.ffn1(self.ffn_norm1(cx1)) + cx1
        x2 = None if cx2 is None else self.ffn2(self.ffn_norm2(cx2)) + cx2
        x3 = None if cx3 is None else self.ffn3(self.ffn_norm3(cx3)) + cx3
        x4 = None if cx4 is None else self.ffn4(self.ffn_norm4(cx4)) + cx4
        return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    """Stacked ChannelTransformer encoder."""

    def __init__(self, vis, channel_num):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList([copy.deepcopy(Block_ViT(vis, channel_num)) for _ in range(1)])
        self.encoder_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3], eps=1e-6) if len(channel_num) == 4 else nn.Identity()

    def forward(self, emb1, emb2, emb3, emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, weights = layer_block(emb1, emb2, emb3, emb4)
            if self.vis:
                attn_weights.append(weights)

        emb1 = None if emb1 is None else self.encoder_norm1(emb1)
        emb2 = None if emb2 is None else self.encoder_norm2(emb2)
        emb3 = None if emb3 is None else self.encoder_norm3(emb3)
        emb4 = None if emb4 is None else self.encoder_norm4(emb4)
        return emb1, emb2, emb3, emb4, attn_weights


class ChannelTransformer(nn.Module):
    """跨尺度通道 Transformer，用于多尺度特征的通道级交互融合。

    接收 3 或 4 个尺度的特征图列表，通过 patch embedding 将空间特征转为序列，
    利用多头注意力实现跨尺度通道信息交互，最后通过 Reconstruct 模块恢复空间维度。

    Args:
        channel_num: 各尺度通道数元组，如 (64, 128, 256, 512)
        img_size: 输入图像尺寸，默认 640
        vis: 是否可视化注意力权重，默认 False
        patchSize: 各尺度的 patch 大小元组，默认 (40, 20, 10, 5)
    """

    def __init__(self, channel_num=(64, 128, 256, 512), img_size=640, vis=False, patchSize=(40, 20, 10, 5)):
        super().__init__()
        channel_num = list(channel_num)
        patchSize = list(patchSize)
        if len(channel_num) not in (3, 4):
            raise ValueError(f"ChannelTransformer expects 3 or 4 scales, got channel_num={channel_num}")
        if len(patchSize) < len(channel_num):
            raise ValueError(
                f"ChannelTransformer requires at least {len(channel_num)} patch sizes, got patchSize={patchSize}"
            )
        patchSize = patchSize[:len(channel_num)]
        self.num_scales = len(channel_num)
        self.out_channels = tuple(channel_num)

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3] if self.num_scales == 4 else None
        self.embeddings_1 = Channel_Embeddings(self.patchSize_1, img_size=img_size // 8, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(self.patchSize_2, img_size=img_size // 16, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(self.patchSize_3, img_size=img_size // 32, in_channels=channel_num[2])
        self.embeddings_4 = (
            Channel_Embeddings(self.patchSize_4, img_size=img_size // 64, in_channels=channel_num[3])
            if len(channel_num) == 4
            else nn.Identity()
        )
        self.encoder = Encoder(vis, channel_num)
        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1, scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1, scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1, scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_4 = (
            Reconstruct(channel_num[3], channel_num[3], kernel_size=1, scale_factor=(self.patchSize_4, self.patchSize_4))
            if len(channel_num) == 4
            else nn.Identity()
        )

    def forward(self, en):
        if len(en) != self.num_scales:
            raise ValueError(f"ChannelTransformer expects {self.num_scales} input features, got {len(en)}")
        if len(en) == 3:
            en1, en2, en3 = en
            en4 = None
        else:
            en1, en2, en3, en4 = en

        emb1 = None if en1 is None else self.embeddings_1(en1)
        emb2 = None if en2 is None else self.embeddings_2(en2)
        emb3 = None if en3 is None else self.embeddings_3(en3)
        emb4 = None if en4 is None else self.embeddings_4(en4)

        encoded1, encoded2, encoded3, encoded4, _ = self.encoder(emb1, emb2, emb3, emb4)
        x1 = None if en1 is None else self.reconstruct_1(encoded1, en1.shape[2:]) + en1
        x2 = None if en2 is None else self.reconstruct_2(encoded2, en2.shape[2:]) + en2
        x3 = None if en3 is None else self.reconstruct_3(encoded3, en3.shape[2:]) + en3
        x4 = None if en4 is None else self.reconstruct_4(encoded4, en4.shape[2:]) + en4
        return [x1, x2, x3] if self.num_scales == 3 else [x1, x2, x3, x4]


__all__ = ("ChannelTransformer",)
