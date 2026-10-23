# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ultralytics.nn.mm.utils import resolve_input_source_role
from ultralytics.utils import LOGGER
from ultralytics.utils.patches import torch_load


MM_ROUTE_TOKEN_HINT = "RGB/X/Dual/A/B/Fuse"


@dataclass(frozen=True)
class MMSpecEvidence:
    """多模态判据证据。

    目标：用于 Fail-Fast 报错时给出“为什么判定为/不为多模态”的可解释信息。
    """

    is_multimodal: bool
    reason: str
    routing_layers: Tuple[Tuple[str, int, str], ...] = ()  # (section, index, tag)
    source: str = ""
    extra: Tuple[Tuple[str, str], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_multimodal": self.is_multimodal,
            "reason": self.reason,
            "routing_layers": [
                {"section": s, "index": i, "tag": t} for (s, i, t) in self.routing_layers
            ],
            "source": self.source,
            "extra": dict(self.extra),
        }


def _iter_layers(cfg_dict: Dict[str, Any]) -> Sequence[Tuple[str, int, Any]]:
    for section in ("backbone", "head"):
        layers = cfg_dict.get(section, [])
        if isinstance(layers, list):
            for i, layer in enumerate(layers):
                yield section, i, layer


def detect_mm_from_cfg_dict(cfg_dict: Dict[str, Any], *, source: str = "cfg_dict") -> MMSpecEvidence:
    routing: List[Tuple[str, int, str]] = []
    for section, i, layer in _iter_layers(cfg_dict):
        if not isinstance(layer, list) or len(layer) < 5:
            continue
        tag = layer[4]
        role = resolve_input_source_role(tag)
        if role is not None:
            tag_s = str(tag).strip()
            routing.append((section, i, tag_s))

    if routing:
        return MMSpecEvidence(
            is_multimodal=True,
            reason=f"检测到 YAML 层定义包含第 5 列多模态路由标记({MM_ROUTE_TOKEN_HINT})",
            routing_layers=tuple(routing),
            source=source,
        )

    return MMSpecEvidence(
        is_multimodal=False,
        reason=f"未检测到任何第 5 列多模态路由标记({MM_ROUTE_TOKEN_HINT})",
        source=source,
    )


def detect_mm_from_yaml(yaml_ref: Union[str, Path, Dict[str, Any]]) -> MMSpecEvidence:
    if isinstance(yaml_ref, dict):
        return detect_mm_from_cfg_dict(yaml_ref, source="yaml_dict")

    from ultralytics.nn.tasks import yaml_model_load

    p = Path(str(yaml_ref))
    cfg_dict = yaml_model_load(str(p))
    return detect_mm_from_cfg_dict(cfg_dict, source=f"yaml:{p.name}")


def detect_mm_from_checkpoint(pt_path: Union[str, Path]) -> MMSpecEvidence:
    p = Path(str(pt_path))
    ckpt = torch_load(str(p), map_location="cpu")
    if not isinstance(ckpt, dict):
        return MMSpecEvidence(False, "checkpoint 不是 dict，无法读取多模态元信息", source=f"pt:{p.name}")

    # 1) 显式标记优先
    if ckpt.get("is_multimodal") is True:
        extra = (("ckpt_key", "is_multimodal"),)
        return MMSpecEvidence(True, "checkpoint 显式标记 is_multimodal=True", source=f"pt:{p.name}", extra=extra)

    if "multimodal_config" in ckpt and ckpt.get("multimodal_config"):
        extra = (("ckpt_key", "multimodal_config"),)
        return MMSpecEvidence(True, "checkpoint 包含 multimodal_config 元信息", source=f"pt:{p.name}", extra=extra)

    # 2) 尝试从模型对象的 yaml 里恢复结构判据
    model_obj = ckpt.get("model")
    model_yaml = getattr(model_obj, "yaml", None) if model_obj is not None else None
    if isinstance(model_yaml, dict) and model_yaml:
        ev = detect_mm_from_cfg_dict(model_yaml, source=f"pt:{p.name}:model.yaml")
        if ev.is_multimodal:
            return ev

    # 3) 尝试从 ckpt 直接携带的 yaml/config 字段恢复
    for key in ("yaml", "cfg", "model_yaml"):
        v = ckpt.get(key)
        if isinstance(v, dict) and v:
            ev = detect_mm_from_cfg_dict(v, source=f"pt:{p.name}:{key}")
            if ev.is_multimodal:
                extra = (("ckpt_key", key),)
                return MMSpecEvidence(True, ev.reason, ev.routing_layers, ev.source, extra=extra)

    return MMSpecEvidence(False, "checkpoint 未包含可识别的多模态结构/元信息", source=f"pt:{p.name}")


def detect_mm(model_ref: Union[str, Path, Dict[str, Any]]) -> MMSpecEvidence:
    """统一入口：根据 model_ref 类型判定是否为 RTDETRMM 多模态结构。"""
    if isinstance(model_ref, dict):
        return detect_mm_from_cfg_dict(model_ref, source="cfg_dict")

    p = Path(str(model_ref))
    suf = p.suffix.lower()
    if suf in {".yaml", ".yml"}:
        return detect_mm_from_yaml(p)
    if suf == ".pt":
        return detect_mm_from_checkpoint(p)

    return MMSpecEvidence(False, f"不支持从该格式判定多模态：{suf or 'unknown'}", source=f"ref:{p.name}")


def require_multimodal(model_ref: Union[str, Path, Dict[str, Any]], *, who: str) -> MMSpecEvidence:
    """Fail-Fast：要求模型引用必须可判定为多模态结构。"""
    ev = detect_mm(model_ref)
    if not ev.is_multimodal:
        LOGGER.error(f"{who}: RTDETRMM 需要多模态结构，但判定失败：{ev.to_dict()}")
        raise ValueError(
            f"{who}: 目标模型/配置不满足 RTDETRMM 多模态判据。"
            f"原因：{ev.reason}；source={ev.source}。"
            "\n修复建议："
            "\n- 若使用 YAML：请确保 backbone/head 层定义包含第 5 列路由标记（RGB/X/Dual/A/B/Fuse）。"
            "\n- 若使用 .pt：请使用 RTDETRMM 训练产物（包含 is_multimodal/multimodal_config），"
            "  或提供对应的多模态 YAML 并从 YAML 重新训练/导出权重。"
        )
    return ev
