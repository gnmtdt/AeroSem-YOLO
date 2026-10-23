"""
轻量 RepVGGBlock 实现（仅供 SPPF 扩展模块内部使用）

设计目标：
- 提供与常见 RepVGG 训练形态兼容的接口（3x3 分支 + 1x1 分支 + 可选恒等分支），满足 IFM 等模块的依赖；
- 不导出到顶层包，仅作为 extraction 子包内部依赖，避免在项目内多点导出；
- 简化：不实现 deploy 时的卷积等效融合，训练/推理阶段均以多分支显式计算为主；
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RepVGGBlock(nn.Module):
    """简化版 RepVGGBlock。

    结构：
    - 3x3 Conv-BN 分支
    - 1x1 Conv-BN 分支
    - Identity-BN 分支（当 in==out 且 stride==1 时）
    - ReLU 激活

    说明：
    - 为满足 SPPF 扩展中的 IFM/注入模块依赖而提供；
    - 不包含 re-parameterize 融合为单卷积分支的部署逻辑，保持实现简单稳定；
    - 接口尽量与常见实现对齐，便于后续替换为完整版本。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        deploy: bool = False,
    ) -> None:
        super().__init__()

        assert groups == 1, "当前精简实现不支持 groups != 1"
        assert stride in (1, 2), "stride 仅支持 1 或 2"

        padding3 = 1
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding3, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.rbr_identity = (
            nn.BatchNorm2d(in_channels) if (out_channels == in_channels and stride == 1) else None
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.act(out)


__all__ = ["RepVGGBlock"]

