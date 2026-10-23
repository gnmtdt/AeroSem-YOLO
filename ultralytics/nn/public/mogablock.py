import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

__all__ = [
    "ElementScale",
    "ChannelAggregationFFN",
    "MultiOrderDWConv",
    "MultiOrderGatedAggregation",
    "MogaBlock",
]


class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0.0, requires_grad=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=requires_grad)

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, kernel_size=3, act_type="GELU", ffn_drop=0.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.fc1 = nn.Conv2d(embed_dims, feedforward_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(feedforward_channels, feedforward_channels, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2, bias=True, groups=feedforward_channels)
        self.act = nn.GELU() if act_type == "GELU" else nn.ReLU()
        self.fc2 = nn.Conv2d(feedforward_channels, embed_dims, kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)
        self.decompose = nn.Conv2d(feedforward_channels, 1, kernel_size=1)
        self.sigma = ElementScale(feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = nn.GELU()

    def feat_decompose(self, x):
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    def __init__(self, embed_dims, dw_dilation=[1, 2, 3], channel_split=[1, 3, 4]):
        super().__init__()
        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        self.DW_conv0 = nn.Conv2d(embed_dims, embed_dims, kernel_size=5,
                                  padding=(1 + 4 * dw_dilation[0]) // 2, groups=embed_dims, stride=1, dilation=dw_dilation[0])
        self.DW_conv1 = nn.Conv2d(self.embed_dims_1, self.embed_dims_1, kernel_size=5,
                                  padding=(1 + 4 * dw_dilation[1]) // 2, groups=self.embed_dims_1, stride=1, dilation=dw_dilation[1])
        self.DW_conv2 = nn.Conv2d(self.embed_dims_2, self.embed_dims_2, kernel_size=7,
                                  padding=(1 + 6 * dw_dilation[2]) // 2, groups=self.embed_dims_2, stride=1, dilation=dw_dilation[2])
        self.PW_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0:self.embed_dims_0 + self.embed_dims_1, ...])
        x_2 = self.DW_conv2(x_0[:, self.embed_dims - self.embed_dims_2:, ...])
        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    def __init__(self, embed_dims, attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4],
                 attn_act_type="SiLU", attn_force_fp32=False):
        super().__init__()
        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(embed_dims, attn_dw_dilation, attn_channel_split)
        self.proj_2 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()
        self.sigma = ElementScale(embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        g = g.to(torch.float32)
        v = v.to(torch.float32)
        return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x
        x = self.feat_decompose(x)
        g = self.gate(x)
        v = self.value(x)
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x


class MogaBlock(nn.Module):
    def __init__(self, embed_dims, ffn_ratio=4.0, drop_rate=0.0, drop_path_rate=0.0, act_type="GELU",
                 norm_type="BN", init_value=1e-5, attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4],
                 attn_act_type="SiLU", attn_force_fp32=False):
        super().__init__()
        self.out_channels = embed_dims
        self.norm1 = nn.BatchNorm2d(embed_dims) if norm_type == "BN" else nn.GroupNorm(1, embed_dims)
        self.attn = MultiOrderGatedAggregation(embed_dims, attn_dw_dilation, attn_channel_split, attn_act_type, attn_force_fp32)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(embed_dims) if norm_type == "BN" else nn.GroupNorm(1, embed_dims)
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(embed_dims, mlp_hidden_dim, act_type=act_type, ffn_drop=drop_rate)
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x
