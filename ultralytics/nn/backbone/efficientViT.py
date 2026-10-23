# NOTE(ultralyticsmm):
# - 为了“尽量保持原汁原味 + 便于长期维护”，本项目对这类主干采用 timm 的 features_only 实现。
# - 保持上游 RTDETR-main 的**模块/函数命名**（EfficientViT_M0..M5），便于 YAML 直接迁移。
# - 满足本项目规范：主干入口先用 Conv 接收模态输入（RGB/X/Dual），再进入单模态主干（3ch）。

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER

from ..modules.conv import Conv

__all__ = ["EfficientViT_M0", "EfficientViT_M1", "EfficientViT_M2", "EfficientViT_M3", "EfficientViT_M4", "EfficientViT_M5"]


class _TimmFeaturesBackbone(nn.Module):
    def __init__(self, in_chans: int, model_name: str, pretrained: bool = False) -> None:
        super().__init__()
        self.in_chans = int(in_chans)
        self.model_name = str(model_name)
        self.pretrained = bool(pretrained)

        if self.in_chans <= 0:
            raise ValueError(f"{self.model_name}: in_chans 必须为正整数，当前={self.in_chans}")

        self.input_proj = Conv(self.in_chans, 3, k=1, s=1, act=False)

        try:
            import timm  # noqa: PLC0415
        except Exception as e:
            raise ModuleNotFoundError(f"{self.model_name}: 需要安装 timm，当前导入失败：{type(e).__name__}: {e}") from e

        self.net = timm.create_model(self.model_name, pretrained=self.pretrained, features_only=True, in_chans=3)
        self.channel = list(self.net.feature_info.channels())
        self.backbone = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"{self.model_name}: 输入必须为 4D BCHW，当前 shape={tuple(x.shape)}")
        if int(x.shape[1]) != self.in_chans:
            msg = (
                f"{self.model_name}: 输入通道不匹配，期望 {self.in_chans}ch，实际 {int(x.shape[1])}ch。"
                "请检查 YAML 的模态路由标记（RGB/X/Dual）与实际输入是否一致。"
            )
            LOGGER.error(msg)
            raise RuntimeError(msg)
        x = self.input_proj(x)
        feats = self.net(x)
        if not isinstance(feats, (list, tuple)):
            raise TypeError(f"{self.model_name}: features_only 输出必须为 list/tuple，当前={type(feats).__name__}")
        feats = list(feats)
        if not feats or not all(isinstance(t, torch.Tensor) for t in feats):
            raise RuntimeError(f"{self.model_name}: features_only 输出必须为 Tensor 列表")
        return feats


def _load_weights_if_needed(model: nn.Module, weights: str) -> None:
    if not weights:
        return
    ckpt = torch.load(weights, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        raise ValueError(f"weights 文件格式不支持：{weights}")
    model.load_state_dict(state, strict=False)


def EfficientViT_M0(in_chans: int = 3, weights: str = "", pretrained: bool = False, **kwargs) -> nn.Module:
    if kwargs:
        raise ValueError(f"EfficientViT_M0 不支持额外 kwargs：{sorted(kwargs.keys())}")
    m = _TimmFeaturesBackbone(in_chans=in_chans, model_name="efficientvit_m0", pretrained=pretrained)
    _load_weights_if_needed(m, weights)
    return m


def EfficientViT_M1(in_chans: int = 3, weights: str = "", pretrained: bool = False, **kwargs) -> nn.Module:
    if kwargs:
        raise ValueError(f"EfficientViT_M1 不支持额外 kwargs：{sorted(kwargs.keys())}")
    m = _TimmFeaturesBackbone(in_chans=in_chans, model_name="efficientvit_m1", pretrained=pretrained)
    _load_weights_if_needed(m, weights)
    return m


def EfficientViT_M2(in_chans: int = 3, weights: str = "", pretrained: bool = False, **kwargs) -> nn.Module:
    if kwargs:
        raise ValueError(f"EfficientViT_M2 不支持额外 kwargs：{sorted(kwargs.keys())}")
    m = _TimmFeaturesBackbone(in_chans=in_chans, model_name="efficientvit_m2", pretrained=pretrained)
    _load_weights_if_needed(m, weights)
    return m


def EfficientViT_M3(in_chans: int = 3, weights: str = "", pretrained: bool = False, **kwargs) -> nn.Module:
    if kwargs:
        raise ValueError(f"EfficientViT_M3 不支持额外 kwargs：{sorted(kwargs.keys())}")
    m = _TimmFeaturesBackbone(in_chans=in_chans, model_name="efficientvit_m3", pretrained=pretrained)
    _load_weights_if_needed(m, weights)
    return m


def EfficientViT_M4(in_chans: int = 3, weights: str = "", pretrained: bool = False, **kwargs) -> nn.Module:
    if kwargs:
        raise ValueError(f"EfficientViT_M4 不支持额外 kwargs：{sorted(kwargs.keys())}")
    m = _TimmFeaturesBackbone(in_chans=in_chans, model_name="efficientvit_m4", pretrained=pretrained)
    _load_weights_if_needed(m, weights)
    return m


def EfficientViT_M5(in_chans: int = 3, weights: str = "", pretrained: bool = False, **kwargs) -> nn.Module:
    if kwargs:
        raise ValueError(f"EfficientViT_M5 不支持额外 kwargs：{sorted(kwargs.keys())}")
    m = _TimmFeaturesBackbone(in_chans=in_chans, model_name="efficientvit_m5", pretrained=pretrained)
    _load_weights_if_needed(m, weights)
    return m
