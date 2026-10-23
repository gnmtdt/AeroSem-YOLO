"""Core visualization exceptions (Fail-Fast)."""

from typing import List, Tuple, Optional


class VisualizationError(RuntimeError):
    """Base error for visualization pipeline."""


class MethodNotRegisteredError(VisualizationError):
    def __init__(self, method: str, available: Optional[List[str]] = None):
        msg = (
            f"可视化方法未注册: '{method}'.\n"
            f"请实现并注册对应插件，或选择已实现的方法。"
        )
        if available:
            msg += f"\n已注册方法: {available}"
        super().__init__(msg)


class InputValidationError(VisualizationError):
    pass


class ModalityConflictError(VisualizationError):
    pass


class LayerResolutionError(VisualizationError):
    def __init__(
        self,
        invalid: List[int],
        valid_range: Tuple[int, int],
        suggestions: Optional[List[int]] = None,
        overview: Optional[List[Tuple[int, str]]] = None,
    ):
        msg = f"无效的层索引: {invalid}，有效范围为 [{valid_range[0]}, {valid_range[1]}]。"
        if suggestions:
            msg += f"\n建议可视化的层（优先 backbone/stage 输出）：{suggestions[:10]}"
        if overview:
            head = "\n可用层概览（idx -> 模块）：\n" + "\n".join(
                [f"  - {i:>3}: {name}" for i, name in overview[:20]]
            )
            if len(overview) > 20:
                head += f"\n  ... （共 {len(overview)} 层）"
            msg += head
        super().__init__(msg)


class DeviceMismatchError(VisualizationError):
    def __init__(self, current: str, requested: str):
        super().__init__(
            f"device参数与当前模型设备不一致：当前为 {current}，收到 {requested}。请先执行 model.to('{requested}') 再调用 vis。"
        )
