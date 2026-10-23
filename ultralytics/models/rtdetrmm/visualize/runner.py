"""RTDETRMM family visualization runner.

该 Runner 委托给通用 visualize_core Pipeline。
注意：此处 family 仍使用 'rtdetr' 以复用既有插件与渲染逻辑。
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
        method: str = "heat",
        layers: Optional[List[int]] = None,
        modality: Optional[str] = None,
        save: bool = True,
        overlay: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        out_dir: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        if layers is None:
            raise ValueError("layers 参数必填，需为 List[int]")
        pipe = Pipeline(model=model, family="rtdetr")
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
