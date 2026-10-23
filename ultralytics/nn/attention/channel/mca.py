"""
MCA - Multi-scale Strip Cross Attention

论文: Multi-scale Strip Cross Attention for Remote Sensing Image Change Detection
期刊: IEEE Transactions on Geoscience and Remote Sensing (TGRS 2025)
论文链接: https://arxiv.org/abs/2402.19289

利用多尺度条带卷积(水平+垂直)捕获长距离依赖，通过水平-垂直交叉注意力机制
建模空间关系。支持多头注意力并行计算。forward 返回 x * sigmoid(h_out + w_out)。
"""

import torch
import torch.nn as nn


class MCA(nn.Module):
    def __init__(self, chn, kernel_sizes=[7, 11, 21], num_heads=8) -> None:
        super().__init__()

        self.num_heads = num_heads

        self.batch_norm1 = nn.BatchNorm2d(chn)
        self.batch_norm2 = nn.BatchNorm2d(chn)

        self.strip_conv_branch1, self.strip_conv_branch2 = nn.ModuleList([]), nn.ModuleList([])
        for k in kernel_sizes:
            self.strip_conv_branch1.append(nn.Conv2d(chn, chn, kernel_size=(k, 1), padding=(k // 2, 0)))
            self.strip_conv_branch2.append(nn.Conv2d(chn, chn, kernel_size=(1, k), padding=(0, k // 2)))

        self.h_qkv = nn.Conv2d(chn, chn * 3, 1)
        self.w_qkv = nn.Conv2d(chn, chn * 3, 1)

        self.h_out = nn.Conv2d(chn, chn, 1)
        self.w_out = nn.Conv2d(chn, chn, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        x_h, x_w = self.batch_norm1(x), self.batch_norm2(x)

        x_h_strip_conv = torch.sum(torch.stack([layer(x_h) for layer in self.strip_conv_branch1], dim=0), dim=0)
        x_w_strip_conv = torch.sum(torch.stack([layer(x_w) for layer in self.strip_conv_branch2], dim=0), dim=0)

        x_h_q, x_h_k, x_h_v = torch.chunk(self.h_qkv(x_h_strip_conv).reshape((B, self.num_heads, C * 3 // self.num_heads, H, W)), chunks=3, dim=2)  # B head C1 H W
        x_w_q, x_w_k, x_w_v = torch.chunk(self.w_qkv(x_w_strip_conv).reshape((B, self.num_heads, C * 3 // self.num_heads, H, W)), chunks=3, dim=2)  # B head C1 H W

        # ----------------------
        x_w_v = x_w_v.permute(0, 1, 4, 2, 3).flatten(3)  # B,Head,C,H,W -> B,Head,W,C,H -> B,Head,W,CH
        x_w_k = x_w_k.reshape((B, self.num_heads, (C // self.num_heads) * H, W))  # B,Head,C,H,W -> B,Head,CH,W
        x_h_q = x_h_q.permute(0, 1, 4, 2, 3).flatten(3)  # B,Head,C,H,W -> B,Head,W,C,H -> B,Head,W,CH
        x_w_attn = (x_w_k @ x_h_q) * ((C // self.num_heads) * H ** -0.5)  # B,Head,CH,CH
        x_w_out = x_w_v @ x_w_attn.permute(0, 1, 3, 2)  # B,Head,W,CH
        x_w_out = x_w_out.reshape((B, self.num_heads, W, C // self.num_heads, H)).permute(0, 1, 3, 4, 2).reshape((B, C, H, W))  # B,Head,W,CH -> B,Head,W,C,H -> B,Head,C,H,W -> B,C,H,W
        x_w_out = self.w_out(x_w_out)

        # ----------------------
        x_h_v = x_h_v.permute(0, 1, 4, 2, 3).flatten(3)  # B,Head,C,H,W -> B,Head,W,C,H -> B,Head,W,CH
        x_h_k = x_h_k.reshape((B, self.num_heads, (C // self.num_heads) * H, W))  # B,Head,C,H,W -> B,Head,CH,W
        x_w_q = x_w_q.permute(0, 1, 4, 2, 3).flatten(3)  # B,Head,C,H,W -> B,Head,W,C,H -> B,Head,W,CH
        x_h_attn = (x_h_k @ x_w_q) * ((C // self.num_heads) * H ** -0.5)  # B,Head,CH,CH
        x_h_out = x_h_v @ x_h_attn.permute(0, 1, 3, 2)  # B,Head,W,CH
        x_h_out = x_h_out.reshape((B, self.num_heads, W, C // self.num_heads, H)).permute(0, 1, 3, 4, 2).reshape((B, C, H, W))  # B,Head,W,CH -> B,Head,W,C,H -> B,Head,C,H,W -> B,C,H,W
        x_h_out = self.h_out(x_h_out)

        return x * torch.sigmoid(x_w_out + x_h_out)


__all__ = ['MCA']
