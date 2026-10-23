import math
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import LayerNorm2d
from ultralytics.nn.public.common_glu import ConvolutionalGLU

__all__ = ["Modulator", "SMA", "E_MLP", "SMAFormerBlock", "SMAFormerBlock_CGLU"]


class Modulator(nn.Module):
    def __init__(self, in_ch, out_ch, with_pos=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.rate = [1, 6, 12, 18]
        self.with_pos = with_pos
        self.patch_size = 2
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA_fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // 16, in_ch, bias=False),
            nn.Sigmoid(),
        )

        self.PA_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.PA_bn = nn.BatchNorm2d(in_ch)
        self.sigmoid = nn.Sigmoid()

        self.SA_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=rate, dilation=rate),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_ch),
                )
                for rate in self.rate
            ]
        )
        self.SA_out_conv = nn.Conv2d(len(self.rate) * out_ch, out_ch, 1)

        self.output_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self._init_weights()

        self.pj_conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.patch_size + 1, stride=self.patch_size, padding=self.patch_size // 2)
        self.pos_conv = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1, groups=self.out_ch, bias=True)
        self.layernorm = nn.LayerNorm(self.out_ch, eps=1e-6)

    def forward(self, x):
        res = x
        pa = self.PA(x)
        ca = self.CA(x)
        pa_ca = torch.softmax(pa @ ca, dim=-1)
        sa = self.SA(x)
        out = pa_ca @ sa
        out = self.norm(self.output_conv(out))
        out = out + self.bias
        return out + res

    def PE(self, x):
        proj = self.pj_conv(x)
        if self.with_pos:
            pos = proj * self.sigmoid(self.pos_conv(proj))
        pos = pos.flatten(2).transpose(1, 2)
        embedded_pos = self.layernorm(pos)
        return embedded_pos

    def PA(self, x):
        attn = self.PA_conv(x)
        attn = self.PA_bn(attn)
        attn = self.sigmoid(attn)
        return x * attn

    def CA(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.CA_fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def SA(self, x):
        sa_outs = [block(x) for block in self.SA_blocks]
        sa_out = torch.cat(sa_outs, dim=1)
        sa_out = self.SA_out_conv(sa_out)
        return sa_out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SMA(nn.Module):
    def __init__(self, feature_size, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads, dropout=dropout)
        self.combined_modulator = Modulator(feature_size, feature_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, value, key, query):
        msa = self.attention(query, key, value)[0]
        b, seq_len, c = msa.shape
        msa = msa.permute(0, 2, 1).view(b, c, int(seq_len ** 0.5), int(seq_len ** 0.5))
        synergistic_attn = self.combined_modulator(msa)
        x = synergistic_attn.view(b, c, -1).permute(0, 2, 1)
        return x


class E_MLP(nn.Module):
    def __init__(self, feature_size, forward_expansion, dropout):
        super().__init__()
        self.linear1 = nn.Linear(feature_size, forward_expansion * feature_size)
        self.act = nn.GELU()
        self.depthwise_conv = nn.Conv2d(forward_expansion * feature_size, forward_expansion * feature_size, 3, padding=1)
        self.pixelwise_conv = nn.Conv2d(forward_expansion * feature_size, forward_expansion * feature_size, 3, padding=1)
        self.linear2 = nn.Linear(forward_expansion * feature_size, feature_size)

    def forward(self, x):
        b, hw, c = x.size()
        feat = int(math.sqrt(hw))
        x = self.act(self.linear1(x))
        x = rearrange(x, "b (h w) c -> b c h w", h=feat, w=feat)
        x = self.depthwise_conv(x)
        x = self.pixelwise_conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return self.linear2(x)


class SMAFormerBlock(nn.Module):
    def __init__(self, ch_out, heads=8, dropout=0.1, forward_expansion=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(ch_out)
        self.norm2 = nn.LayerNorm(ch_out)
        self.synergistic_multi_attention = SMA(ch_out, heads, dropout)
        self.e_mlp = E_MLP(ch_out, forward_expansion, dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.flatten(2).permute(0, 2, 1)
        value = key = query = res = x
        attention = self.synergistic_multi_attention(query, key, value)
        query = self.dropout(self.norm1(attention + res))
        feed_forward = self.e_mlp(query)
        out = self.dropout(self.norm2(feed_forward + query))
        return out.permute(0, 2, 1).reshape(b, c, h, w)


class SMAFormerBlock_CGLU(nn.Module):
    def __init__(self, ch_out, heads=8, dropout=0.1, forward_expansion=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(ch_out)
        self.norm2 = LayerNorm2d(ch_out)
        self.synergistic_multi_attention = SMA(ch_out, heads, dropout)
        self.e_mlp = ConvolutionalGLU(ch_out, forward_expansion, drop=dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.flatten(2).permute(0, 2, 1)
        value = key = query = res = x
        attention = self.synergistic_multi_attention(query, key, value)
        query = self.dropout(self.norm1(attention + res))
        feed_forward = self.e_mlp(query.permute(0, 2, 1).reshape(b, c, h, w))
        out = self.dropout(self.norm2(feed_forward))
        return out
