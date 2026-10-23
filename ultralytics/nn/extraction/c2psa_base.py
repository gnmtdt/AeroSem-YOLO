"""
C2PSA Extraction - Base Components
存放所有C2PSA变体所需的基础组件和注意力模块

包含内容：
- PSABlock: 位置感知注意力基础块
- PSA: PSA单层封装
- 注意力机制模块 (BiLevelRoutingAttention, LocalWindowAttention, DAttention等)
- TopkRouting: 可微分topk路由
- 辅助函数

迁移自 ultralytics-yolo11-main/ultralytics/nn/
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from typing import Tuple, Optional
import einops
from einops import rearrange, repeat
from timm.layers import DropPath, to_2tuple, trunc_normal_

from ..modules.conv import Conv, autopad
from ..modules.block import Attention

__all__ = [
    # 基础PSA模块
    'PSABlock', 'PSA',
    # 辅助模块
    'TopkRouting', 'LayerNormProxy', 'DynamicPosBias', 'DynamicTanh',
    # 注意力机制模块
    'BiLevelRoutingAttention_nchw', 'LocalWindowAttention', 'DAttention',
    'DPB_Attention', 'PolaLinearAttention', 'AttentionTSSA',
    'AdaptiveSparseSA', 'MSLA',
    # ASSA辅助模块
    'LinearProjection', 'WindowAttention_sparse', 'Mlp', 'LeFF', 'FRFN',
    # MSLA辅助模块
    'LinearAttention', 'DepthwiseConv',
    # Batch 4: FFN增强模块
    'FMFFN', 'ConvolutionalGLU', 'SEFN', 'SpectralEnhancedFFN', 'EDFFN',
    # Batch 5: Mona模块化注意力归一化
    'LayerNorm2d', 'MonaOp', 'Mona',
    # 辅助函数
    '_grid2seq', '_seq2grid', 'regional_routing_attention_torch',
    'window_partition', 'window_reverse',
]


# ================================ 基础PSA模块 ================================

class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


# ================================ 辅助模块 ================================

class LayerNormProxy(nn.Module):
    """Layer normalization proxy for NCHW tensors."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class TopkRouting(nn.Module):
    """
    Differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        param_routing: bool, whether incorporate learnable params in routing unit
        diff_routing: bool, whether make routing differentiable
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)

        return r_weight, topk_index


# ================================ BiLevel Routing Attention相关 ================================

def _grid2seq(x: Tensor, region_size: Tuple[int], num_heads: int):
    """
    Args:
        x: BCHW tensor
        region_size: int (region_h, region_w)
        num_heads: number of attention heads
    Return:
        out: rearranged x, has a shape of (bs, nhead, nregion, reg_size, head_dim)
        region_h, region_w: number of regions per col and row
    """
    B, C, H, W = x.size()
    region_h, region_w = H // region_size[0], W // region_size[1]
    x = x.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1])
    x = torch.einsum('bmdhpwq->bmhwpqd', x).flatten(2, 3).flatten(-3, -2)  # (bs, nhead, nregion, reg_size, head_dim)
    return x, region_h, region_w


def _seq2grid(x: Tensor, region_h: int, region_w: int, region_size: Tuple[int]):
    """
    Args:
        x: (bs, nhead, nregion, reg_size^2, head_dim)
    Return:
        x: (bs, nhead*head_dim, H, W)
    """
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    x = x.view(bs, nhead, region_h, region_w, region_size[0], region_size[1], head_dim)
    x = torch.einsum('bmhwpqd->bmdhpwq', x).reshape(bs, nhead * head_dim,
        region_h * region_size[0], region_w * region_size[1])
    return x


def regional_routing_attention_torch(
    query: Tensor, key: Tensor, value: Tensor, scale: float,
    region_graph: LongTensor, region_size: Tuple[int],
    kv_region_size: Optional[Tuple[int]] = None,
    auto_pad=True) -> Tensor:
    """
    Args:
        query, key, value: (B, C, H, W) tensor
        scale: the scale/temperature for dot product attention
        region_graph: (B, nhead, h_q*w_q, topk) tensor, topk <= h_k*w_k
        region_size: region/window size for queries, (rh, rw)
        key_region_size: optional, if None, key_region_size=region_size
        auto_pad: required to be true if the input sizes are not divisible by the region_size
    Return:
        output: (B, C, H, W) tensor
        attn: (bs, nhead, q_nregion, reg_size, topk*kv_region_size) attention matrix
    """
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()

    # Auto pad to deal with any input size
    q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    if auto_pad:
        _, _, Hq, Wq = query.size()
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        if (q_pad_b > 0 or q_pad_r > 0):
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b))  # zero padding

        _, _, Hk, Wk = key.size()
        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        if (kv_pad_r > 0 or kv_pad_b > 0):
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b))  # zero padding
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b))  # zero padding

    # to sequence format, i.e. (bs, nhead, nregion, reg_size, head_dim)
    query, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    # gather key and values.
    bs, nhead, kv_nregion, kv_region_size, head_dim = key.size()
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1).\
        expand(-1, -1, -1, -1, kv_region_size, head_dim)
    key_g = torch.gather(key.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).\
        expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
        index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    value_g = torch.gather(value.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).\
        expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
        index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)

    # token-to-token attention
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    output = attn @ value_g.flatten(-3, -2)

    # to BCHW format
    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    # remove paddings if needed
    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        output = output[:, :, :Hq, :Wq]

    return output, attn


class BiLevelRoutingAttention_nchw(nn.Module):
    """
    Bi-Level Routing Attention that takes nchw input

    Compared to legacy version, this implementation:
    * removes unused args and components
    * uses nchw input format to avoid frequent permutation

    When the size of inputs is not divisible by the region size, there is also a numerical difference
    than legacy implementation, due to:
    * different way to pad the input feature map (padding after linear projection)
    * different pooling behavior (count_include_pad=False)

    Current implementation is more reasonable, hence we do not keep backward numerical compatiability
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_scale=None, topk=4, side_dwconv=3, auto_pad=False, attn_backend='torch'):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5  # NOTE: to be consistent with old models.

        # side_dwconv (i.e. LCE in Shunted Transformer)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)

        # regional routing setting
        self.topk = topk
        self.n_win = n_win  # number of windows per row/col

        self.qkv_linear = nn.Conv2d(self.dim, 3 * self.dim, kernel_size=1)
        self.output_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)

        if attn_backend == 'torch':
            self.attn_fn = regional_routing_attention_torch
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    def forward(self, x: Tensor, ret_attn_mask=False):
        """
        Args:
            x: NCHW tensor, better to be channel_last (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        Return:
            NCHW tensor
        """
        N, C, H, W = x.size()
        region_size = (H // self.n_win, W // self.n_win)

        # STEP 1: linear projection
        qkv = self.qkv_linear.forward(x)  # ncHW
        q, k, v = qkv.chunk(3, dim=1)  # ncHW

        # STEP 2: region-to-region routing
        # NOTE: ceil_mode=True, count_include_pad=False = auto padding
        # NOTE: gradients backward through token-to-token attention. See Appendix A for the intuition.
        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)  # nchw
        q_r: Tensor = q_r.permute(0, 2, 3, 1).flatten(1, 2)  # n(hw)c
        k_r: Tensor = k_r.flatten(2, 3)  # nc(hw)
        a_r = q_r @ k_r  # n(hw)(hw), adj matrix of regional graph
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)  # n(hw)k long tensor
        idx_r: LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1)

        # STEP 3: token to token attention (non-parametric function)
        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size)

        output = output + self.lepe(v)  # ncHW
        output = self.output_linear(output)  # ncHW

        if ret_attn_mask:
            return output, attn_mat

        return output


# ================================ Local Window Attention ================================

class CascadedGroupAttention(nn.Module):
    """Cascaded Group Attention for LocalWindowAttention."""

    def __init__(self, dim, key_dim=16, num_heads=4, attn_ratio=4, resolution=14, kernels=[5, 5, 5, 5]):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(nn.Conv2d(dim // num_heads, self.key_dim * 2 + self.d, 1))
            dws.append(nn.Conv2d(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim))
        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)
        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.d * num_heads, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # Attention bias
        points = [(i, j) for i in range(resolution) for j in range(resolution)]
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1)  # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(nn.Module):
    r""" Local Window Attention. CVPR2023-EfficientViT

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim=16, num_heads=4,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution

        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio,
                                resolution=window_resolution,
                                kernels=kernels)

    def forward(self, x):
        B, C, H, W = x.shape

        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            x = self.attn(x)
            # window reverse
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.permute(0, 3, 1, 2)
        return x


# ================================ Deformable Attention ================================

class DAttention(nn.Module):
    """
    Vision Transformer with Deformable Attention CVPR2022
    Fixed_pe=True need adjust for 640x640 resolution
    """
    def __init__(
        self, channel, q_size, n_heads=8, n_groups=4,
        attn_drop=0.0, proj_drop=0.0, stride=1,
        offset_range_factor=4, use_pe=True, dwc_pe=True,
        no_off=False, fixed_pe=False, ksize=3, log_cpb=False, kv_size=None
    ):
        super().__init__()
        n_head_channels = channel // n_heads
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w))
                torch.nn.init.trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1))
                torch.nn.init.trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
        return ref

    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.offset_range_factor >= 0 and not self.no_off:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (self.rpe_table is not None):
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            else:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))
        return y


# ================================ CrossFormer DPB Attention ================================

class DynamicPosBias(nn.Module):
    """
    DPB module - Use a MLP to predict position bias used in attention.
    From CrossFormer (https://arxiv.org/abs/2108.00154)
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class DPB_Attention(nn.Module):
    """
    Multi-head self attention module with dynamic position bias.
    From CrossFormer (ICLR 2022)

    Args:
        dim (int): Number of input channels.
        group_size (tuple[int]): The height and width of the group.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, group_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):
        super().__init__()
        self.dim = dim
        self.group_size = group_size  # Wh, Ww — used as initial/fallback
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias

        if position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            self._build_position_bias(group_size[0], group_size[1])

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _build_position_bias(self, h, w):
        """(Re)build relative position index and biases buffers for given spatial size."""
        device = next(self.pos.parameters()).device
        position_bias_h = torch.arange(1 - h, h, device=device)
        position_bias_w = torch.arange(1 - w, w, device=device)
        biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij'))
        biases = biases.flatten(1).transpose(0, 1).float()
        self.register_buffer("biases", biases, persistent=False)

        coords_h = torch.arange(h, device=device)
        coords_w = torch.arange(w, device=device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += h - 1
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)
        self.group_size = [h, w]

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        if self.position_bias and self.group_size[0] * self.group_size[1] != N:
            h = w = int(N ** 0.5)
            self._build_position_bias(h, w)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.position_bias:
            pos = self.pos(self.biases)
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.group_size[0] * self.group_size[1], self.group_size[0] * self.group_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ================================ PolaFormer Linear Attention ================================

class PolaLinearAttention(nn.Module):
    """
    Polarized Linear Attention from PolaFormer (ICLR 2025)
    Reduces complexity to O(n) while maintaining performance
    """
    def __init__(self, dim, hw, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.h = hw[0]
        self.w = hw[1]

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, (self.w * self.h) // (sr_ratio * sr_ratio), dim)))

    def forward(self, x):
        B, N, C = x.shape

        # Derive actual spatial dims from token count, handling non-square feature maps
        actual_hw = int(N ** 0.5)
        if actual_hw * actual_hw == N:
            actual_h = actual_w = actual_hw
        else:
            # Find the factor pair closest to being square
            for h in range(actual_hw, 0, -1):
                if N % h == 0:
                    actual_h, actual_w = h, N // h
                    break
            else:
                actual_h = actual_w = actual_hw

        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, actual_h, actual_w)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]

        pe = self.positional_encoding
        if n != pe.shape[1]:
            pe = F.interpolate(pe.transpose(1, 2), size=n, mode='linear', align_corners=False).transpose(1, 2)
        k = k + pe
        kernel_function = nn.ReLU()

        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * nn.functional.sigmoid(self.power)

        q = q / scale
        k = k / scale
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        q_pos = kernel_function(q) ** power
        q_neg = kernel_function(-q) ** power
        k_pos = kernel_function(k) ** power
        k_neg = kernel_function(-k) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)

        v = v.reshape(B * self.num_heads, actual_h, actual_w, -1).permute(0, 3, 1, 2)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        x = x + v
        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# ================================ Token Statistics Self-Attention ================================

class AttentionTSSA(nn.Module):
    """
    Token Statistics Self-Attention from ToST (ICLR 2025)
    https://github.com/RobinWu218/ToST
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()

        self.heads = num_heads

        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        b, h, N, d = w.shape

        w_normed = torch.nn.functional.normalize(w, dim=-2)
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # b * h * n

        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# ================================ Batch 3: 归一化增强模块 ================================

class DynamicTanh(nn.Module):
    """
    Dynamic Tanh normalization module.
    来源: CVPR2025

    动态Tanh激活归一化，通过可学习的alpha参数实现自适应激活。

    Args:
        normalized_shape: 归一化的形状（通道数）
        channels_last: 是否为channels_last格式
        alpha_init_value: alpha参数初始值，默认0.5
    """
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


# ================================ Batch 3: ASSA相关模块 ================================

class LinearProjection(nn.Module):
    """
    Linear projection for Q, K, V in attention mechanism.
    用于ASSA模块的线性投影层。
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


class WindowAttention_sparse(nn.Module):
    """
    Sparse window-based self-attention with adaptive sparse mechanism.
    来源: CVPR2024 Adaptive Sparse Transformer

    自适应稀疏窗口注意力，结合softmax和ReLU²两种注意力机制。
    """
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2  # b,h,w,c
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0 * w1 + attn1 * w2
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'


class Mlp(nn.Module):
    """
    Feed-forward network (MLP) module.
    标准的前馈神经网络，用于ASSA模块。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LeFF(nn.Module):
    """
    Locally-enhanced Feed-Forward network.
    局部增强的前馈网络，结合深度卷积。
    """
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = nn.Identity()

    def forward(self, x):
        bs, hw, c = x.size()
        # 优先使用上游提供的真实空间尺寸，避免非方形场景下的开方取整误差
        spatial_hw = getattr(self, "_spatial_hw", None)
        if spatial_hw is not None and spatial_hw[0] * spatial_hw[1] == hw:
            hh, ww = int(spatial_hw[0]), int(spatial_hw[1])
        else:
            # 兼容旧路径（默认按方形恢复）
            hh = int(math.sqrt(hw))
            ww = hh

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=ww)

        x = self.dwconv(x)

        # flatten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=ww)

        x = self.linear2(x)
        x = self.eca(x)

        return x


class FRFN(nn.Module):
    """
    Frequency-enhanced Feed-Forward Network with gating mechanism.
    频率增强的前馈网络，带有门控机制。
    """
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        bs, hw, c = x.size()
        # 优先使用上游提供的真实空间尺寸，避免非方形场景下的开方取整误差
        spatial_hw = getattr(self, "_spatial_hw", None)
        if spatial_hw is not None and spatial_hw[0] * spatial_hw[1] == hw:
            hh, ww = int(spatial_hw[0]), int(spatial_hw[1])
        else:
            # 兼容旧路径（默认按方形恢复）
            hh = int(math.sqrt(hw))
            ww = hh

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=ww)

        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flatten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=ww)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h=hh, w=ww)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h=hh, w=ww)
        x = x_1 * x_2

        x = self.linear2(x)

        return x


def window_partition(x, win_size, dilation_rate=1):
    """
    Window partition function for window-based attention.
    将特征图分割成窗口用于窗口注意力计算。

    Args:
        x: Input tensor (B, H, W, C)
        win_size: Window size
        dilation_rate: Dilation rate for dilated windows

    Returns:
        windows: Partitioned windows (B', Wh, Ww, C)
    """
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    """
    Reverse window partition.
    将窗口还原为完整特征图。

    Args:
        windows: Windowed tensor (B', Wh, Ww, C)
        win_size: Window size
        H: Height of feature map
        W: Width of feature map
        dilation_rate: Dilation rate

    Returns:
        x: Restored tensor (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class AdaptiveSparseSA(nn.Module):
    """
    Adaptive Sparse Self-Attention module.
    来源: CVPR2024 - Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration
    论文链接: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf

    自适应稀疏自注意力机制，结合窗口注意力和稀疏机制，适用于图像恢复任务。

    Args:
        dim: 输入通道维度
        num_heads: 注意力头数
        sparseAtt: 是否使用稀疏注意力
        win_size: 窗口大小，默认4
        shift_size: 窗口移位大小，默认2
        mlp_ratio: MLP扩展比例，默认4
        token_mlp: FFN类型，可选'frfn', 'leff', 'mlp'
    """
    def __init__(self, dim, num_heads, sparseAtt=False, win_size=4, shift_size=2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='frfn', att=True):
        super().__init__()

        self.att = att
        self.sparseAtt = sparseAtt

        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if self.att:
            self.norm1 = norm_layer(dim)
            if self.sparseAtt:
                self.attn = WindowAttention_sparse(
                    dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                    token_projection=token_projection)
            else:
                raise NotImplementedError("Non-sparse attention not implemented in this extraction")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn', 'mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'leff':
            self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'frfn':
            self.mlp = FRFN(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            raise Exception("FFN error!")

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, mask=None):
        B, C, H, W = x.size()
        # 原始分辨率下的token表示（用于残差）
        x = x.flatten(2).permute(0, 2, 1).contiguous()  # B, H*W, C

        # input mask
        if mask is not None:
            # 将mask对齐到特征分辨率，并在需要时做窗口对齐补齐（稍后与特征一致补齐）
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)  # B, H, W, 1
            # 注意：实际分块前会根据Hp/Wp进行pad
            # 在下面计算Hp/Wp后再执行window_partition，保证整除
        else:
            input_mask = None
        
        # 计算窗口对齐后的尺寸（只在win_size整除不满足时进行补齐）
        Hp = (H + self.win_size - 1) // self.win_size * self.win_size
        Wp = (W + self.win_size - 1) // self.win_size * self.win_size
        pad_h = Hp - H
        pad_w = Wp - W
        need_pad = (pad_h > 0) or (pad_w > 0)

        # 依据窗口补齐后的尺寸构建注意力mask（若存在）
        if input_mask is not None:
            if need_pad:
                # BHWC -> BCHW 以使用F.pad对空间维进行补齐
                im_bchw = input_mask.permute(0, 3, 1, 2)
                im_bchw = F.pad(im_bchw, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
                input_mask = im_bchw.permute(0, 2, 3, 1).contiguous()  # 回到BHWC
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # shift mask
        if self.shift_size > 0:
            # 在窗口补齐后的尺寸上构建移位mask，保证分块整除
            shift_mask = x.new_zeros((1, Hp, Wp, 1))
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        shortcut = x

        if self.att:
            x = self.norm1(x)
            x = x.view(B, H, W, C)
            if need_pad:
                # 将特征在BCHW下做空间补齐，再回到BHWC
                x_bchw = x.permute(0, 3, 1, 2)
                x_bchw = F.pad(x_bchw, (0, pad_w, 0, pad_h))
                x = x_bchw.permute(0, 2, 3, 1).contiguous()

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C
            x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
            # 使用补齐后的Hp/Wp进行还原
            shifted_x = window_reverse(attn_windows, self.win_size, Hp, Wp)  # B Hp Wp C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            # 裁剪回原始空间尺寸
            if need_pad:
                x = x[:, :H, :W, :].contiguous()
            x = x.view(B, H * W, C)
            x = shortcut + self.drop_path(x)

        # FFN：在进入MLP前告知真实空间尺寸，避免MLP内部按方形恢复引发形状不匹配
        if hasattr(self, "mlp") and hasattr(self.mlp, "__dict__"):
            setattr(self.mlp, "_spatial_hw", (H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if hasattr(self, "mlp") and hasattr(self.mlp, "__dict__"):
            # 清理临时尺寸信息，避免跨次调用污染
            if hasattr(self.mlp, "_spatial_hw"):
                delattr(self.mlp, "_spatial_hw")
        del attn_mask
        return x


# ================================ Batch 3: MSLA相关模块 ================================

class LinearAttention(nn.Module):
    """
    Linear Attention module with O(n) complexity.
    线性注意力模块，具有O(n)的计算复杂度。
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.view(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)

        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        key = F.softmax(k, dim=-1)
        query = F.softmax(q, dim=-2)
        context = key.transpose(-2, -1) @ v
        x = (query @ context).reshape(b, h * w, c)

        x = self.proj(x)

        x = x.permute(0, 2, 1).view(b, c, h, w)

        return x


class DepthwiseConv(nn.Module):
    """
    Depthwise Convolution with residual connection.
    深度卷积模块，带有残差连接。
    """
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels,
                                   padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x + residual
        x = self.relu(x)
        return x


class MSLA(nn.Module):
    """
    Multi-Scale Linear Attention module.
    多尺度线性注意力模块，结合不同尺度的深度卷积和线性注意力。

    通过3x3, 5x5, 7x7, 9x9四种不同尺度的深度卷积捕获多尺度特征，
    并使用线性注意力进行特征融合。

    Args:
        dim: 输入通道维度
        num_heads: 注意力头数
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.dw_conv_3x3 = DepthwiseConv(dim // 4, kernel_size=3)
        self.dw_conv_5x5 = DepthwiseConv(dim // 4, kernel_size=5)
        self.dw_conv_7x7 = DepthwiseConv(dim // 4, kernel_size=7)
        self.dw_conv_9x9 = DepthwiseConv(dim // 4, kernel_size=9)

        self.linear_attention = LinearAttention(dim=dim // 4, num_heads=num_heads)

        self.final_conv = nn.Conv2d(dim, dim, 1)

        self.scale_weights = nn.Parameter(torch.ones(4), requires_grad=True)

    def forward(self, input_, hw=None):
        b, n, c = input_.shape
        if hw is not None:
            h, w = hw
        else:
            h = int(n ** 0.5)
            w = int(n ** 0.5)

        input_reshaped = input_.reshape([b, c, h, w])

        split_size = c // 4
        x_3x3 = input_reshaped[:, :split_size, :, :]
        x_5x5 = input_reshaped[:, split_size:2 * split_size, :, :]
        x_7x7 = input_reshaped[:, 2 * split_size:3 * split_size:, :, :]
        x_9x9 = input_reshaped[:, 3 * split_size:, :, :]

        x_3x3 = self.dw_conv_3x3(x_3x3)
        x_5x5 = self.dw_conv_5x5(x_5x5)
        x_7x7 = self.dw_conv_7x7(x_7x7)
        x_9x9 = self.dw_conv_9x9(x_9x9)

        att_3x3 = self.linear_attention(x_3x3)
        att_5x5 = self.linear_attention(x_5x5)
        att_7x7 = self.linear_attention(x_7x7)
        att_9x9 = self.linear_attention(x_9x9)

        processed_input = torch.cat([
            att_3x3 * self.scale_weights[0],
            att_5x5 * self.scale_weights[1],
            att_7x7 * self.scale_weights[2],
            att_9x9 * self.scale_weights[3]
        ], dim=1)

        final_output = self.final_conv(processed_input)

        output_reshaped = final_output.reshape(b, n, self.dim)

        return output_reshaped


# ================================ Batch 4: FFN增强模块 ================================

class FMFFN(nn.Module):
    """
    Frequency-Modulated Feed-Forward Network.
    来源: ICLR2024 FTIC (Frequency-modulated Transformer for Image Colorization)

    频率调制的前馈网络，通过FFT进行频域调制。

    Args:
        in_features: 输入通道数
        hidden_features: 隐藏层通道数
        out_features: 输出通道数
        window_size: 窗口大小用于频率调制，默认4
        act_layer: 激活函数，默认GELU
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, window_size=4, act_layer=nn.GELU) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.window_size = window_size

        self.ffn = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            act_layer(),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        # 窗口频率调制参数
        self.complex_weight = nn.Parameter(
            torch.cat((
                torch.ones(window_size, window_size // 2 + 1, out_features, 1, dtype=torch.float32),
                torch.zeros(window_size, window_size // 2 + 1, out_features, 1, dtype=torch.float32)
            ), dim=-1)
        )

    def forward(self, x):
        # 前馈网络
        x = self.ffn(x)

        # 频率调制
        b, c, h, w = x.shape
        # 将特征图分割成窗口
        x = rearrange(x, 'b c (w1 p1) (w2 p2) -> b w1 w2 p1 p2 c',
                      p1=self.window_size, p2=self.window_size)

        x_dtype = x.dtype
        x = x.to(torch.float32)

        # FFT变换
        x_fft = torch.fft.rfft2(x, dim=(3, 4), norm='ortho')

        # 频率调制
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight

        # 逆FFT
        x = torch.fft.irfft2(x_fft, s=(self.window_size, self.window_size), dim=(3, 4), norm='ortho')

        # 恢复形状
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b c (w1 p1) (w2 p2)')

        return x.to(x_dtype)


class ConvolutionalGLU(nn.Module):
    """
    Convolutional Gated Linear Unit.
    来源: CVPR2024 TransNext

    卷积门控线性单元，结合深度卷积和门控机制。

    Args:
        in_features: 输入通道数
        hidden_features: 隐藏层通道数
        out_features: 输出通道数
        act_layer: 激活函数，默认GELU
        drop: Dropout比例，默认0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1,
                     padding=1, bias=True, groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shortcut = x
        # 门控机制：分割成两部分
        x, v = self.fc1(x).chunk(2, dim=1)
        # 深度卷积
        x = self.dwconv(x) * v  # 门控
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x  # 残差连接


class SEFN(nn.Module):
    """
    Spatial-Enhanced Feed-Forward Network.
    来源: WACV2025 SEMNet (Snake Bi-directional Selective State Space and Mamba for Lightweight Image Restoration)

    空间增强的前馈网络，结合空间分支进行特征增强。

    Args:
        dim: 输入通道数
        ffn_expansion_factor: FFN扩展因子，默认2
        bias: 是否使用偏置，默认False
    """
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(SEFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 融合模块
        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
            groups=hidden_features, bias=bias
        )

        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # 空间分支
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, spatial=None):
        """
        Args:
            x: 主输入特征 [B, C, H, W]
            spatial: 空间输入特征（可选）。为 None 时默认使用 x（满足 C2PSA 单输入 FFN 契约）。
        """
        if spatial is None:
            spatial = x
        if spatial.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"SEFN 要求 x 与 spatial 的空间尺寸一致以完成融合，got x={tuple(x.shape)} vs spatial={tuple(spatial.shape)}"
            )
        x = self.project_in(x)

        # 空间分支
        y = self.avg_pool(spatial)
        y = self.spatial_conv(y)
        # 重要：奇数尺寸 H/W 下，pool(//2) 再 scale_factor=2 会产生 1 像素误差（例如 21->10->20），
        # 必须显式对齐回 spatial 的目标尺寸，避免后续 torch.cat 维度不一致。
        y = F.interpolate(y, size=spatial.shape[-2:], mode="bilinear", align_corners=False)

        # 主分支
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.fusion(torch.cat((x1, y), dim=1))
        x1 = self.dwconv_afterfusion(x1)

        # 门控
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SpectralEnhancedFFN(nn.Module):
    """
    Spectral-Enhanced Feed-Forward Network.
    来源: TransMamba (Transformer with Mamba)

    频谱增强的前馈网络，使用FFT进行频域处理。

    Args:
        dim: 输入通道数
        ffn_expansion_factor: FFN扩展因子，默认2
        bias: 是否使用偏置，默认False
    """
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(SpectralEnhancedFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1,
            padding=2, groups=hidden_features * 2, bias=bias, dilation=2
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # 频域调制参数
        self.fft_channel_weight = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))
        self.fft_channel_bias = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[..., t_pad[0]:hw - t_pad[1]]

    def forward(self, x):
        x_dtype = x.dtype
        x = self.project_in(x)
        x = self.dwconv(x)

        # FFT频域处理
        x, pad_w = self.pad(x, 2)
        x_fft = torch.fft.rfft2(x.float())
        x_fft = self.fft_channel_weight * x_fft + self.fft_channel_bias
        x = torch.fft.irfft2(x_fft)
        x = self.unpad(x, pad_w)

        # 门控
        x1, x2 = x.chunk(2, dim=1)
        x = F.silu(x1) * x2
        x = self.project_out(x.to(x_dtype))
        return x


class EDFFN(nn.Module):
    """
    Enhanced Dynamic Feed-Forward Network.
    来源: CVPR2025 EVSSM (Efficient Visual State Space Model)

    增强动态前馈网络，基于patch的FFT处理。

    Args:
        dim: 输入通道数
        ffn_expansion_factor: FFN扩展因子，默认2
        bias: 是否使用偏置，默认False
    """
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1,
            padding=1, groups=hidden_features * 2, bias=bias
        )

        # Patch级别的FFT参数
        self.fft = nn.Parameter(
            torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x_dtype = x.dtype
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        # Patch级别的FFT处理
        b, c, h, w = x.shape
        # 填充到patch_size的倍数
        h_n = (self.patch_size - h % self.patch_size) % self.patch_size
        w_n = (self.patch_size - w % self.patch_size) % self.patch_size

        x = F.pad(x, (0, w_n, 0, h_n), mode='reflect')

        # 分割成patch
        x_patch = rearrange(
            x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
            patch1=self.patch_size, patch2=self.patch_size
        )

        # FFT处理
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))

        # 恢复形状
        x = rearrange(
            x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
            patch1=self.patch_size, patch2=self.patch_size
        )

        # 去除填充
        x = x[:, :, :h, :w]

        return x.to(x_dtype)


# ================================ Batch 5: Mona模块化注意力归一化 ================================

class LayerNorm2d(nn.LayerNorm):
    """
    2D Layer Normalization
    用于2D特征图的归一化，自动处理通道维度
    """
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor with shape (B, C, H, W)
        Returns:
            Normalized tensor with same shape
        """
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, C, H, W) -> (B, H, W, C)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, H, W, C) -> (B, C, H, W)
        return x


class MonaOp(nn.Module):
    """
    Mona Operation Module
    使用多尺度深度卷积的模块化操作

    来源: CVPR2025 Mona
    特点: 3x3, 5x5, 7x7三种尺度的深度卷积 + 投影
    """
    def __init__(self, in_features):
        """
        Args:
            in_features: Input feature dimension
        """
        super().__init__()
        # 多尺度深度卷积
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        # 1x1投影层
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (B, C, H, W)
        Returns:
            Output tensor with shape (B, C, H, W)
        """
        identity = x

        # 多尺度深度卷积
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        # 平均融合 + 残差连接
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        # 投影 + 残差连接
        x = self.projector(x)

        return identity + x


class Mona(nn.Module):
    """
    Modular Attention Normalization (Mona)
    模块化注意力归一化模块

    来源: CVPR2025
    特点:
    - LayerNorm + 可学习的gamma/gammax参数
    - 降维-MonaOp-升维结构
    - 残差连接

    用途: 增强归一化能力，提升模型性能
    """
    def __init__(self, in_dim):
        """
        Args:
            in_dim: Input dimension (number of channels)
        """
        super().__init__()

        # 降维投影 (降到64维)
        self.project1 = nn.Conv2d(in_dim, 64, 1)

        # 非线性激活
        self.nonlinear = F.gelu

        # 升维投影 (恢复到in_dim)
        self.project2 = nn.Conv2d(64, in_dim, 1)

        # Dropout
        self.dropout = nn.Dropout(p=0.1)

        # Mona操作模块
        self.adapter_conv = MonaOp(64)

        # LayerNorm归一化
        self.norm = LayerNorm2d(in_dim)

        # 可学习的缩放参数
        self.gamma = nn.Parameter(torch.ones(in_dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim, 1, 1))

    def forward(self, x, hw_shapes=None):
        """
        Args:
            x: Input tensor with shape (B, C, H, W)
            hw_shapes: Optional height and width shapes (unused, for compatibility)
        Returns:
            Output tensor with shape (B, C, H, W)
        """
        identity = x

        # 归一化 + 可学习的缩放
        x = self.norm(x) * self.gamma + x * self.gammax

        # 降维投影
        project1 = self.project1(x)

        # Mona操作（多尺度深度卷积）
        project1 = self.adapter_conv(project1)

        # 非线性激活 + Dropout
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)

        # 升维投影
        project2 = self.project2(nonlinear)

        # 残差连接
        return identity + project2
