"""多模态输入适配 + 单模块多输出主干（timm features_only）。

设计目标
- 满足 RTDETRMM 的多模态规范：主干入口先用 Conv 吸收指定模态输入（RGB/X/Dual 由 router 决定）
- 主干本体保持“单模态、多输出”形态：输出多尺度特征 list[Tensor]
- Fail-Fast：输入通道/输出类型不符合预期时明确报错（中文）
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER

from ..modules.conv import Conv


class TimmBackbone(nn.Module):
    """多模态输入投影 + timm features_only 主干（单模块多输出）。"""

    def __init__(
        self,
        in_chans: int,
        model_name: str,
        pretrained: bool = False,
        out_indices: Sequence[int] = (1, 2, 3, 4),
        proj_out_chans: int = 3,
        proj_act: bool = False,
    ) -> None:
        super().__init__()
        self.in_chans = int(in_chans)
        self.model_name = str(model_name)
        self.pretrained = bool(pretrained)
        self.out_indices = tuple(int(i) for i in out_indices)
        self.proj_out_chans = int(proj_out_chans)

        if self.in_chans <= 0:
            raise ValueError(f"TimmBackbone: in_chans 必须为正整数，当前={self.in_chans}")
        if self.proj_out_chans <= 0:
            raise ValueError(f"TimmBackbone: proj_out_chans 必须为正整数，当前={self.proj_out_chans}")
        if not self.out_indices:
            raise ValueError("TimmBackbone: out_indices 不能为空")

        # 入口投影：把多模态输入（如 Dual=6ch）投影到主干期望的单模态通道数（默认 3ch）
        # 注意：Conv 内含 BN；该行为是“显式设计选择”，不是自动降级。
        self.input_proj = Conv(self.in_chans, self.proj_out_chans, k=1, s=1, act=proj_act)

        try:
            import timm  # noqa: PLC0415
        except Exception as e:
            raise ModuleNotFoundError(
                "TimmBackbone 需要安装 timm（用于 create_model(features_only=True)）。"
                f"当前导入失败：{type(e).__name__}: {e}"
            ) from e

        self.net = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            features_only=True,
            out_indices=self.out_indices,
            in_chans=self.proj_out_chans,
        )

        # 按 RTDETR-main 约定：单模块多输出主干应提供 .channel（list[int]）
        try:
            channels = list(self.net.feature_info.channels())
        except Exception as e:
            raise RuntimeError(
                "TimmBackbone: 无法从 timm features_only 模型读取 feature_info.channels()；"
                f"model_name={self.model_name}，错误={type(e).__name__}: {e}"
            ) from e

        if not channels or not all(isinstance(c, int) and c > 0 for c in channels):
            raise RuntimeError(f"TimmBackbone: 非法 channel 列表：{channels}")

        self.channel = channels

        # 供 tasks.py 检测：hasattr(m, 'backbone') 即视为多输出主干
        self.backbone = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"TimmBackbone: 输入必须为 torch.Tensor，当前={type(x).__name__}")
        if x.ndim != 4:
            raise ValueError(f"TimmBackbone: 输入必须为 4D BCHW，当前 shape={tuple(x.shape)}")
        if x.shape[1] != self.in_chans:
            msg = (
                f"TimmBackbone: 输入通道不匹配，期望 {self.in_chans}ch，实际 {int(x.shape[1])}ch。"
                "请检查 YAML 的模态路由标记（RGB/X/Dual）与实际输入是否一致。"
            )
            LOGGER.error(msg)
            raise RuntimeError(msg)

        x = self.input_proj(x)
        outs = self.net(x)

        if not isinstance(outs, (list, tuple)):
            raise RuntimeError(f"TimmBackbone: 主干输出类型必须为 list/tuple，当前={type(outs).__name__}")
        outs = list(outs)
        if not outs or not all(isinstance(t, torch.Tensor) for t in outs):
            raise RuntimeError("TimmBackbone: 主干输出必须为 Tensor 列表")

        return outs
