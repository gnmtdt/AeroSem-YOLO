"""
DPWA - 可变形位置窗口注意力机制 (Deformable Parallel Window Attention)

论文: Dynamic Parallel Window Attention
期刊/会议: IEEE TGRS (2025)
论文链接: https://ieeexplore.ieee.org/document/11146454
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_

__all__ = ['DPWA']


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qk_scale (float | None): Override default qk scale
        attn_drop (float): Dropout ratio of attention weight.
    """

    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: queries with shape of (num_windows*B, N, C)
            k: keys with shape of (num_windows*B, N, C)
            v: values with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
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
        return x


class DPWA(nn.Module):
    """Dynamic Parallel Window Attention

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        alternate (int): alternate parameter controlling attention type reversal.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        qk_scale (float | None): Override default qk scale.
        drop (float): Dropout rate.
        attn_drop (float): Attention dropout rate.
        drop_path (float): Stochastic depth rate.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
    """

    def __init__(self, dim, num_heads=8, window_size=4, shift_size=2, alternate=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.alternate = alternate
        self.attn = nn.ModuleList([
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
        ])

    def forward(self, x):
        B, C, H, W = x.size()
        L = H * W

        x = x.flatten(2).permute(0, 2, 1)

        attn_mask1 = None
        attn_mask2 = None

        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask2 = attn_mask2.masked_fill(
                attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0)).to(
                x.device)

        x = self.norm1(x)

        # double attn
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        if self.alternate == 0:
            qkv_1 = qkv[:, :, :, :C // 2].reshape(3, B, H, W, C // 2)

            if self.shift_size > 0:
                qkv_2 = torch.roll(qkv[:, :, :, C // 2:], shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2)).reshape(3, B, H, W, C // 2)
            else:
                qkv_2 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)
        else:
            qkv_1 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)

            if self.shift_size > 0:
                qkv_2 = torch.roll(qkv[:, :, :, :C // 2], shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2)).reshape(3, B, H, W, C // 2)
            else:
                qkv_2 = qkv[:, :, :, :C // 2].reshape(3, B, H, W, C // 2)

        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_2)

        x1 = self.attn[0](q1_windows, k1_windows, v1_windows, attn_mask1)
        x2 = self.attn[1](q2_windows, k2_windows, v2_windows, attn_mask2)

        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C // 2),
                            self.window_size, H, W)
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C // 2),
                            self.window_size, H, W)

        if self.shift_size > 0:
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if self.alternate == 0:
            x = torch.cat([x1.reshape(B, H * W, C // 2), x2.reshape(B, H * W, C // 2)], dim=2)
        else:
            x = torch.cat([x2.reshape(B, H * W, C // 2), x1.reshape(B, H * W, C // 2)], dim=2)
        x = self.proj(x)

        return x.permute(0, 2, 1).reshape((B, C, H, W))

    def get_window_qkv(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]
        C = q.shape[-1]
        q_windows = window_partition(q, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        k_windows = window_partition(k, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        v_windows = window_partition(v, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        return q_windows, k_windows, v_windows
