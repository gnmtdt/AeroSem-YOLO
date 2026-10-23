"""
CTA - Channel Transposed Attention (通道转置注意力)

论文: Channel Transposed Attention
期刊/会议: IJCAI 2024
论文链接: https://www.ijcai.org/proceedings/2024/0081.pdf
依赖: einops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einops import reduce
except ImportError:
    reduce = None

__all__ = ['ChannelProjection', 'SpatialProjection', 'CTA']


class ChannelProjection(nn.Module):
    """通道投影模块，结合全局平均池化和多尺度深度卷积进行通道信息交互。

    Args:
        dim (int): 输入通道数。
    """

    def __init__(self, dim):
        super().__init__()
        self.pro_in = nn.Conv2d(dim, dim // 6, 1, 1, 0)
        self.CI1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim // 6, dim // 6, kernel_size=1)
        )
        self.CI2 = nn.Sequential(
            nn.Conv2d(dim // 6, dim // 6, kernel_size=3, stride=1, padding=1, groups=dim // 6),
            nn.Conv2d(dim // 6, dim // 6, 7, stride=1, padding=9, groups=dim // 6, dilation=3),
            nn.Conv2d(dim // 6, dim // 6, kernel_size=1)
        )
        self.pro_out = nn.Conv2d(dim // 6, dim, kernel_size=1)

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        x = self.pro_in(x)
        res = x
        ci1 = self.CI1(x)
        ci2 = self.CI2(x)
        out = self.pro_out(res * ci1 * ci2)
        return out


class SpatialProjection(nn.Module):
    """空间投影模块，利用深度卷积和 GELU 门控机制提取空间信息。

    Args:
        dim (int): 输入通道数。
    """

    def __init__(self, dim):
        super().__init__()
        self.pro_in = nn.Conv2d(dim, dim // 2, 1, 1, 0)
        self.dwconv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1, groups=dim // 2)
        self.pro_out = nn.Conv2d(dim // 4, dim, kernel_size=1)

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        x = self.pro_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.pro_out(x)
        return x


class CTA(nn.Module):
    """Channel Transposed Attention (通道转置注意力)。

    基于 XCiT 的通道转置自注意力机制，结合通道投影和空间投影
    实现高效的通道-空间信息交互。

    参考: https://github.com/facebookresearch/xcit

    Args:
        dim (int): 输入通道数。
        num_heads (int): 注意力头数。默认 8。
        qkv_bias (bool): 是否为 qkv 添加偏置。默认 False。
        qk_scale (float | None): 覆盖默认的 qk 缩放因子。默认 None。
        attn_drop (float): 注意力 dropout 率。默认 0.。
        proj_drop (float): 投影 dropout 率。默认 0.。
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.channel_projection = ChannelProjection(dim)
        self.spatial_projection = SpatialProjection(dim)
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
        )

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3 B num_heads N D
        q, k, v = qkv[0], qkv[1], qkv[2]

        # B num_heads D N
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # convolution output
        conv_x = self.dwconv(v_)

        # C-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        channel_map = self.channel_projection(attention_reshape)
        attened_x = attened_x + channel_map.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        if reduce is not None:
            channel_map = reduce(channel_map, 'b c h w -> b c 1 1', 'mean')
        else:
            channel_map = channel_map.mean(dim=[2, 3], keepdim=True)

        # S-Map (before sigmoid)
        spatial_map = self.spatial_projection(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, C)

        # S-I
        attened_x = attened_x * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x

        x = self.proj(x)

        x = self.proj_drop(x)

        return x.permute(0, 2, 1).view(B, C, H, W).contiguous()
