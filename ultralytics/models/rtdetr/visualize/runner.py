"""RTDETRMM family visualization runner.

This runner delegates to visualize_core Pipeline with family='rtdetr'.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ultralytics.models.utils.multimodal.visualize_core import Pipeline


class RTDETRMMVisualizationRunner:
    @staticmethod
    def run(
        *,
        model: Any,
        rgb_source: Optional[Any] = None,
        x_source: Optional[Any] = None,
        method: str = 'heat',
        layers: Optional[List[int]] = None,
        modality: Optional[str] = None,
        save: bool = True,
        overlay: Optional[str] = None,
        # 新：使用 project/name 控制保存目录（推荐）
        project: Optional[str] = None,
        name: Optional[str] = None,
        # 旧：out_dir 已废弃，仅保留兼容
        out_dir: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        if layers is None:
            raise ValueError("layers 参数必填，需为 List[int]")
        pipe = Pipeline(model=model, family='rtdetr')
        return pipe.run(
            rgb_source=rgb_source,
            x_source=x_source,
            method=method,
            layers=layers,
            modality=modality,
            save=save,
            overlay=overlay,
            project=project,
            name=name,
            out_dir=out_dir,
            device=device,
            **kwargs,
        )
