"""
PatchSA - 补丁自注意力机制 (Patch Self-Attention)

论文: Shifted Window Self-Attention
期刊/会议: ACM MM (2025)
论文链接: https://dl.acm.org/doi/epdf/10.1145/3746027.3755657
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_

__all__ = ['PatchSA']


class PatchSA(nn.Module):
    """Patch Self-Attention

    Args:
        dim: input feature dimension
        heads: number of attention heads
        patch_size: size of each patch
        stride: stride for patch sliding
    """

    def __init__(self, dim, heads=8, patch_size=8, stride=8):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.patch_size = patch_size
        self.stride = stride

        self.to_qkv = nn.Conv2d(dim * 3, dim * 3, 1, groups=dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

        self.pos_encode = nn.Parameter(torch.zeros((2 * patch_size - 1) ** 2, heads))
        trunc_normal_(self.pos_encode, std=0.02)
        coord = torch.arange(patch_size)
        coords = torch.stack(torch.meshgrid([coord, coord], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += patch_size - 1
        relative_coords[:, :, 1] += patch_size - 1
        relative_coords[:, :, 0] *= 2 * patch_size - 1
        pos_index = relative_coords.sum(-1)
        self.register_buffer('pos_index', pos_index)

    def _forward1(self, x):
        B, C, H, W = x.shape
        assert H == W
        res = torch.empty_like(x)
        pad_num = self.patch_size - self.stride
        expan_x = F.pad(x, (0, pad_num, 0, pad_num), mode='replicate')
        repeat_x = [expan_x] * 3
        expan_x = torch.cat(repeat_x, dim=1)
        qkv = self.to_qkv(expan_x)

        for i in range(0, H, self.stride):
            for j in range(0, W, self.stride):
                patch = qkv[:, :, i: i + self.patch_size, j: j + self.patch_size]
                patch = patch.reshape(B, 3, self.heads, -1,
                                      self.patch_size ** 2).permute(1, 0, 2, 4, 3)
                q, k, v = patch[0], patch[1], patch[2]
                q = q * self.scale
                attn = (q @ k.transpose(-2, -1))

                pos_encode = self.pos_encode[self.pos_index.view(-1)].view(
                    self.patch_size ** 2, self.patch_size ** 2, -1)
                pos_encode = pos_encode.permute(2, 0, 1).contiguous()
                attn = attn + pos_encode.unsqueeze(0)

                attn = self.softmax(attn)
                _res = (attn @ v)
                _res = _res.transpose(-2, -1).reshape(B, -1, self.patch_size, self.patch_size)

                res[:, :, i: i + self.stride, j: j + self.stride] = \
                    _res[:, :, :self.stride, :self.stride]

        return self.to_out(res)

    def _forward2(self, x):
        B, C, H, W = x.shape
        assert H == W
        pad_num = self.patch_size - self.stride
        patch_num = ((H + pad_num - self.patch_size) // self.stride + 1) ** 2
        expan_x = F.pad(x, (0, pad_num, 0, pad_num), mode='replicate')
        repeat_x = [expan_x] * 3
        expan_x = torch.cat(repeat_x, dim=1)
        qkv = self.to_qkv(expan_x)

        qkv_patches = F.unfold(qkv, kernel_size=self.patch_size, stride=self.stride)
        qkv_patches = qkv_patches.view(
            B, 3, self.heads, -1, self.patch_size ** 2, patch_num
        ).permute(1, 0, 2, 5, 4, 3)
        q, k, v = qkv_patches[0], qkv_patches[1], qkv_patches[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        pos_encode = self.pos_encode[self.pos_index.view(-1)].view(
            self.patch_size ** 2, self.patch_size ** 2, -1)
        pos_encode = pos_encode.permute(2, 0, 1).contiguous().unsqueeze(1).repeat(1, patch_num, 1, 1)
        attn = attn + pos_encode.unsqueeze(0)

        attn = self.softmax(attn)
        _res = (attn @ v)

        _res = _res.view(B, self.heads, patch_num, self.patch_size, self.patch_size, -1)[
               :, :, :, :self.stride, :self.stride]
        _res = _res.transpose(2, 5).contiguous().view(B, -1, patch_num)
        res = F.fold(_res, output_size=(H, W), kernel_size=self.stride, stride=self.stride)
        return self.to_out(res)

    def forward(self, x):
        return self._forward2(x)
