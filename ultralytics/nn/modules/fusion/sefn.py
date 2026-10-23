# 论文: SEM-Net: Efficient Pixel Modelling for image inpainting with Spatially Enhanced SSM (WACV 2025)
# 链接: https://arxiv.org/abs/2411.06318
# 模块作用: 双路增强的前馈网络，以辅路空间上下文引导主路门控，使单路输出显式注入跨模态/跨层信息。

import torch
import torch.nn as nn


class SEFN(nn.Module):
    def __init__(self, dim: int | None = None, ffn_expansion_factor: float = 2.0, bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.ffn_expansion_factor = float(ffn_expansion_factor)
        self.bias = bool(bias)
        self._built = False
        self._c = None
        self._hidden = None
        self.project_in: nn.Module | None = None
        self.dwconv: nn.Module | None = None
        self.fusion: nn.Module | None = None
        self.dwconv_afterfusion: nn.Module | None = None
        self.project_out: nn.Module | None = None
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.spatial_conv: nn.Module | None = None
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _build_if_needed(self, c: int) -> None:
        if self._built and self._c == c:
            return
        hidden = max(int(c * self.ffn_expansion_factor), 1)
        self.project_in = nn.Conv2d(c, hidden * 2, kernel_size=1, bias=self.bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1, groups=hidden * 2,
                                 bias=self.bias)
        self.fusion = nn.Conv2d(hidden + c, hidden, kernel_size=1, bias=self.bias)
        self.dwconv_afterfusion = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden,
                                            bias=self.bias)
        self.project_out = nn.Conv2d(hidden, c, kernel_size=1, bias=self.bias)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self._built = True
        self._c = c
        self._hidden = hidden

    def forward(self, x1, x2=None):
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1
        if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
            raise TypeError("SEFN 需要两路输入张量")
        if x1.shape != x2.shape:
            raise ValueError(f"SEFN 要求两路输入形状一致，got {x1.shape} vs {x2.shape}")
        B, C, H, W = x1.shape
        self._build_if_needed(C)
        x = self.project_in(x1)
        x1_main, x2_gate = self.dwconv(x).chunk(2, dim=1)
        y = self.avg_pool(x2)
        y = self.spatial_conv(y)
        y = self.upsample(y)
        x1_main = self.fusion(torch.cat([x1_main, y], dim=1))
        x1_main = self.dwconv_afterfusion(x1_main)
        x = torch.nn.functional.gelu(x1_main) * x2_gate
        x = self.project_out(x)
        return x
