"""
Heat 模块相关（源自 upstream block.py Heat2D / HeatBlock）。
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.layers import LayerNorm2d

__all__ = ["Heat2D", "HeatBlock"]


class Mlp_Heat(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, channels_first=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear = nn.Conv2d if channels_first else nn.Linear
        self.fc1 = linear(in_features, hidden_features, kernel_size=1) if channels_first else linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = linear(hidden_features, out_features, kernel_size=1) if channels_first else linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Heat2D(nn.Module):
    def __init__(self, res=14, dim=0, hidden_dim=0, kernel_dim=14, infer_mode=False):
        super().__init__()
        self.dim = dim
        self.infer_mode = infer_mode
        self.res = res
        self.k = nn.Linear(dim, dim)
        self.q = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.k_norm = LayerNorm2d(dim)
        self.q_norm = LayerNorm2d(dim)
        self.v_norm = LayerNorm2d(dim)
        self.k_linear = nn.Linear(dim, kernel_dim)
        self.q_linear = nn.Linear(dim, kernel_dim)
        self.v_linear = nn.Linear(dim, kernel_dim)
        self.to_k = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.out_norm = nn.LayerNorm(dim)
        self.out_linear = nn.Linear(dim, dim)
        self.kernel_dim = kernel_dim
        self.hidden_dim = hidden_dim
        self.k_exp = None
        self.weight_cosn = None
        self.weight_cosm = None
        self.weight_exp = None
        self.__RES__ = None

    def get_cos_map(self, input_dim, device):
        x = torch.arange(input_dim).to(device)[:, None]
        y = torch.arange(self.kernel_dim).to(device)[None, :]
        weight = torch.cos(x * y * torch.pi / input_dim)
        return weight / self.kernel_dim

    def get_decay_map(self, input_dim, device):
        x = torch.arange(input_dim[0]).to(device)[:, None]
        y = torch.arange(input_dim[1]).to(device)[None, :]
        weight = torch.exp(-(x + y) / max(input_dim))
        return weight / self.kernel_dim

    def forward(self, x, freq_embed):
        # x: B H W C (channels_last)
        if x.ndim == 4 and x.shape[1] == self.dim:
            # 支持 channels_first
            x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        freq_embed = freq_embed.permute(2, 0, 1)  # (C, H, W)
        freq_embed = freq_embed.unsqueeze(0).repeat(B, 1, 1, 1)

        k = self.k_linear(self.k_norm(x + freq_embed))
        q = self.q_linear(self.q_norm(x + freq_embed))
        v = self.v_linear(self.v_norm(x + freq_embed))

        z = self.to_k(freq_embed.permute(0, 2, 3, 1))

        if self.infer_mode and self.__RES__ == (H, W) and self.weight_cosn is not None:
            weight_cosn, weight_cosm, weight_exp = self.weight_cosn, self.weight_cosm, self.weight_exp
        else:
            weight_cosn = self.get_cos_map(H, device=x.device)
            weight_cosm = self.get_cos_map(W, device=x.device)
            weight_exp = self.get_decay_map((H, W), device=x.device)
            self.__RES__ = (H, W)
            self.weight_cosn = weight_cosn
            self.weight_cosm = weight_cosm
            self.weight_exp = weight_exp

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]

        x = F.conv1d(k.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1).type_as(k))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1).type_as(k)).contiguous().view(B, N, M, -1)

        if self.infer_mode:
            k_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed.permute(0, 2, 3, 1))).type_as(k)
            x = torch.einsum("bnmc,nmc -> bnmc", x, k_exp)
        else:
            weight_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed.permute(0, 2, 3, 1)))
            x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp)

        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1).type_as(k))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1).type_as(k)).contiguous().view(B, H, W, -1)

        x = self.out_norm(x)
        x = x * torch.nn.functional.silu(z)
        x = self.out_linear(x)
        return x.permute(0, 3, 1, 2).contiguous()  # channels_first


class HeatBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        res: int = 14,
        infer_mode=False,
        drop_path: float = 0,
        norm_layer: nn.Module = partial(LayerNorm2d, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm=True,
        layer_scale=None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = Heat2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp_Heat(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None

        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)

        self.freq_embed = nn.Parameter(torch.zeros(res, res, hidden_dim), requires_grad=True)
        trunc_normal_(self.freq_embed, std=0.02)
        self.op.infer_init_heat2d = getattr(self.op, "infer_init_heat2d", lambda freq_embed: None)
        self.op.infer_init_heat2d(self.freq_embed)

    def _forward(self, x: torch.Tensor):
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, self.freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.op(self.norm1(x), self.freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.post_norm:
                x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, self.freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1[:, None, None] * self.op(self.norm1(x), self.freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.gamma2[:, None, None] * self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if not self.training:
            self.op.infer_init_heat2d(self.freq_embed)
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, input)
        return self._forward(input)
