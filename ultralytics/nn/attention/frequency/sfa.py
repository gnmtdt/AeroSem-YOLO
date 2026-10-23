"""
SFA - Spatial Frequency Attention (空间频率注意力)

论文: Spatial Frequency Attention
期刊/会议: IJCAI 2024
论文链接: https://www.ijcai.org/proceedings/2024/0081.pdf
依赖: einops
注意: 此模块不支持多尺度训练，reso 参数为当前特征图的 (height, width) 元组。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einops import reduce
except ImportError:
    reduce = None

__all__ = ['FrequencyProjection', 'SpatialProjection', 'SFA']


def img2windows(img, H_sp, W_sp):
    """将图像切分为窗口。

    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """将窗口合并为图像。

    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class FrequencyProjection(nn.Module):
    """频率投影模块，结合 GELU 激活和残差池化进行频域信息投影。

    Args:
        dim (int): 输入通道数。
    """

    def __init__(self, dim):
        super().__init__()
        self.conv_1 = nn.Conv2d(dim, dim // 2, 1, 1, 0)
        self.act = nn.GELU()
        self.res_2 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.GELU()
        )
        self.conv_out = nn.Conv2d(dim // 2, dim, 1, 1, 0)

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        res = x
        x = self.conv_1(x)
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((self.act(x1), self.res_2(x2)), dim=1)
        out = self.conv_out(out)
        return out + res


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


class DynamicPosBias(nn.Module):
    """动态相对位置偏置。

    参考: https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

    Args:
        dim (int): 输入通道数。
        num_heads (int): 注意力头数。
        residual (bool): 是否使用残差策略连接卷积。
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


class Spatial_Attention(nn.Module):
    """空间自注意力，支持矩形窗口。

    Args:
        dim (int): 输入通道数。
        idx (int): 窗口索引 (0 或 1)。
        split_size (list): 空间窗口的 [高, 宽]。
        dim_out (int | None): 注意力输出维度。默认 None。
        num_heads (int): 注意力头数。默认 6。
        attn_drop (float): 注意力 dropout 率。默认 0.。
        proj_drop (float): 输出 dropout 率。默认 0.。
        qk_scale (float | None): 覆盖默认的 qk 缩放因子。
        position_bias (bool): 是否使用动态相对位置偏置。默认 True。
    """

    def __init__(self, dim, idx, split_size=[8, 8], dim_out=None, num_heads=6,
                 attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij'))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x


class SFA(nn.Module):
    """Spatial Frequency Attention (空间频率注意力)。

    参考 CAT: https://github.com/Zhengchen1999/CAT

    结合空间窗口注意力和频率投影，通过通道/空间映射实现多维度信息融合。
    不支持多尺度训练，reso 参数需指定当前特征图的 (height, width)。

    Args:
        dim (int): 输入通道数。
        reso (tuple): 特征图空间尺寸 (H, W)。
        num_heads (int): 注意力头数。默认 8。
        split_size (list): 空间窗口大小。默认 [8, 8]。
        shift_size (list): 移动窗口偏移。默认 [1, 2]。
        qkv_bias (bool): 是否为 qkv 添加偏置。默认 False。
        qk_scale (float | None): 覆盖默认的 qk 缩放因子。
        drop (float): Dropout 率。默认 0.。
        attn_drop (float): 注意力 dropout 率。默认 0.。
        b_idx (int): Block 索引。默认 0。
    """

    def __init__(self, dim, reso=(20, 20), num_heads=8,
                 split_size=[8, 8], shift_size=[1, 2], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., b_idx=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx = b_idx
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.hf = nn.Linear(dim, dim, bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.dw_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        )

        self.attns = nn.ModuleList([
            Spatial_Attention(
                dim // 2, idx=i,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
            for i in range(self.branch_num)])
        if self.b_idx > 0 and (self.b_idx - 2) % 4 == 0:
            attn_mask = self.calculate_mask(*self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.channel_projection = ChannelProjection(dim)
        self.spatial_projection = SpatialProjection(dim)
        self.frequency_projection = FrequencyProjection(dim)

    def calculate_mask(self, H, W):
        """计算移动窗口注意力掩码。

        参考: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        """
        img_mask_0 = torch.zeros((1, H, W, 1))
        img_mask_1 = torch.zeros((1, H, W, 1))
        h_slices_0 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for window-0
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0],
                                     W // self.split_size[1], self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            -1, self.split_size[0], self.split_size[1], 1)
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(
            attn_mask_0 == 0, float(0.0))

        # calculate mask for window-1
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1],
                                     W // self.split_size[0], self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            -1, self.split_size[1], self.split_size[0], 1)
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(
            attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        x_flat = x.flatten(2).permute(0, 2, 1).contiguous()

        hf = self.hf(x_flat).transpose(-2, -1).contiguous().view(B, C, H, W)

        hf = self.frequency_projection(hf)

        qkv = self.qkv(x_flat).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, HW, C
        v = qkv[2].transpose(-2, -1).contiguous().view(B, C, H, W)

        # image padding
        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3 * B, H, W, C).permute(0, 3, 1, 2)  # 3B C H W
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1)
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        if self.b_idx > 0 and (self.b_idx - 2) % 4 == 0:
            qkv = qkv.view(3, B, _H, _W, C)
            qkv_0 = torch.roll(qkv[:, :, :, :, :C // 2],
                               shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, _L, C // 2)
            qkv_1 = torch.roll(qkv[:, :, :, :, C // 2:],
                               shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, _L, C // 2)

            if self.patches_resolution[0] != _H or self.patches_resolution[1] != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            x1 = self.attns[0](qkv[:, :, :, :C // 2], _H, _W)[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = self.attns[1](qkv[:, :, :, C // 2:], _H, _W)[:, :H, :W, :].reshape(B, L, C // 2)
            attened_x = torch.cat([x1, x2], dim=2)

        conv_x = self.dw_block(v)

        # C-Map (before sigmoid)
        channel_map = self.channel_projection(conv_x)
        conv_x = conv_x + channel_map
        # high_fre info mix channel
        hf = hf + channel_map
        if reduce is not None:
            channel_map = reduce(channel_map, 'b c h w -> b c 1 1', 'mean').permute(
                0, 2, 3, 1).contiguous().view(B, 1, C)
        else:
            channel_map = channel_map.mean(dim=[2, 3], keepdim=True).permute(
                0, 2, 3, 1).contiguous().view(B, 1, C)

        # S-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_projection(attention_reshape)
        # high_fre info mix spatial
        hf = hf + attention_reshape

        # C-I
        if reduce is not None:
            hf_channel = reduce(hf, 'b c h w -> b c 1 1', 'mean').permute(
                0, 2, 3, 1).contiguous().view(B, 1, C)
        else:
            hf_channel = hf.mean(dim=[2, 3], keepdim=True).permute(
                0, 2, 3, 1).contiguous().view(B, 1, C)
        attened_x = attened_x * torch.sigmoid(channel_map) * torch.sigmoid(hf_channel)
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x * torch.sigmoid(hf)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x_out = attened_x + conv_x + hf.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x_out = self.proj(x_out)

        x_out = self.proj_drop(x_out)

        return x_out.permute(0, 2, 1).view(B, C, H, W).contiguous()
