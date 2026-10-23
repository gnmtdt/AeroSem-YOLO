"""
离线模态生成基础类与注册表。

设计目标：
- 完全离线：仅负责读取源数据、生成目标模态并保存，不与训练/推理管线耦合。
- 不自动降级：缺权重/设备/依赖即抛错，由调用方显式处理。
- 可插拔：不同生成方法通过注册表选择，统一 run 接口。
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device


# --------------------------
# 通用配置
# --------------------------

@dataclass
class SaveOptions:
    """保存相关可选项。"""

    enable_save: bool = True           # 是否落盘
    save_dir: Optional[Path | str] = None  # 基础输出目录；None 表示由子类自行决定
    keep_structure: bool = True        # 是否保持输入目录结构（仅在 save_dir 非 None 时使用）
    overwrite: bool = False            # 已存在时是否覆盖


@dataclass
class GeneratorRunStats:
    """运行统计信息。"""

    total: int = 0
    success: int = 0
    failed: int = 0
    failures: List[Tuple[str, str]] = field(default_factory=list)  # (path, err)


# --------------------------
# 注册表
# --------------------------

class GeneratorRegistry:
    """生成器注册表，用于按名称创建实例。"""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, generator_cls: type):
        if not issubclass(generator_cls, ModalGeneratorBase):
            raise TypeError(f"{generator_cls} 不是 ModalGeneratorBase 的子类")
        cls._registry[name] = generator_cls

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise KeyError(f"未注册的生成方法: {name}")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_methods(cls) -> List[str]:
        return sorted(cls._registry.keys())


# --------------------------
# 抽象基类
# --------------------------

class ModalGeneratorBase(abc.ABC):
    """
    离线模态生成器基类。

    子类需实现：
        - load_model(): 负责权重加载与模型初始化
        - preprocess(item): 读取单个样本并返回 (input, meta)
        - infer(batch_inputs): 前向推理
        - postprocess(outputs, metas): 可选后处理
        - save(outputs, metas): 按需落盘
    """

    def __init__(
        self,
        method: str,
        device: str | torch.device | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        save_options: Optional[SaveOptions | Dict[str, Any]] = None,
        method_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.method = method
        self.device = select_device(device or "cpu", verbose=False)
        if batch_size < 1:
            raise ValueError("batch_size 必须 >= 1")
        self.batch_size = batch_size
        self.num_workers = max(0, num_workers)
        self.method_cfg = method_cfg or {}

        if isinstance(save_options, dict):
            self.save_options = SaveOptions(**save_options)
        elif save_options is None:
            self.save_options = SaveOptions()
        else:
            self.save_options = save_options

        self.model = None
        self._loaded = False

    # ---------- 子类必须实现 ----------

    @abc.abstractmethod
    def load_model(self):
        """加载模型与权重，设置到 self.model。"""

    @abc.abstractmethod
    def preprocess(self, item: str):
        """读取单个输入并返回 (input, meta)。meta 至少含源路径。"""

    @abc.abstractmethod
    def infer(self, batch_inputs: List[Any]) -> List[Any]:
        """对预处理后的批次进行前向推理。"""

    @abc.abstractmethod
    def postprocess(self, outputs: List[Any], metas: List[Dict[str, Any]]) -> List[Any]:
        """对推理结果进行后处理。"""

    @abc.abstractmethod
    def save(self, outputs: List[Any], metas: List[Dict[str, Any]]) -> List[str]:
        """保存结果，返回保存路径列表（或空列表）。"""

    # ---------- 通用运行逻辑 ----------

    def _ensure_loaded(self):
        if not self._loaded:
            self.load_model()
            if self.model is None:
                raise RuntimeError("模型未加载成功")
            self._loaded = True

    def _gather_sources(self, source: str | Path | Iterable[str | Path]) -> List[str]:
        """
        收集输入样本路径。

        具体解析逻辑由子类重载（例如 DepthGen 支持 data.yaml）；默认处理目录/文件列表。
        """
        paths: List[str] = []
        if isinstance(source, (str, Path)):
            src = Path(source)
            if src.is_dir():
                paths = [str(p) for p in src.rglob("*") if p.is_file()]
            elif src.is_file():
                paths = [str(src)]
            else:
                raise FileNotFoundError(f"找不到输入: {source}")
        else:
            for item in source:
                p = Path(item)
                if p.exists():
                    paths.append(str(p))
                else:
                    raise FileNotFoundError(f"找不到输入: {item}")

        if not paths:
            raise RuntimeError("未找到任何可处理的输入文件")
        return sorted(paths)

    def run(self, source: str | Path | Iterable[str | Path]) -> GeneratorRunStats:
        """
        运行离线生成流程。
        """
        self._ensure_loaded()
        files = self._gather_sources(source)
        self._source_root = os.path.commonpath(files) if files else None

        stats = GeneratorRunStats(total=len(files))
        LOGGER.info(f"[{self.method}] 离线生成开始，样本数: {len(files)}, 设备: {self.device}")

        # 简单批处理循环
        for start in range(0, len(files), self.batch_size):
            batch_paths = files[start : start + self.batch_size]
            try:
                batch_inputs, metas = [], []
                for p in batch_paths:
                    inp, meta = self.preprocess(p)
                    batch_inputs.append(inp)
                    metas.append(meta)

                outputs = self.infer(batch_inputs)
                outputs = self.postprocess(outputs, metas)
                saved = self.save(outputs, metas) if self.save_options.enable_save else []

                stats.success += len(batch_paths)
                processed = stats.success + stats.failed
                if processed % 100 == 0 or processed == stats.total:
                    LOGGER.info(
                        f"[{self.method}] 进度 {processed}/{stats.total} | 成功 {stats.success} | 失败 {stats.failed}"
                    )
                if saved and processed % 500 == 0:
                    LOGGER.debug(f"[{self.method}] 示例已保存: {saved[:1]}")
            except Exception as e:  # noqa: BLE001
                for p in batch_paths:
                    stats.failures.append((p, str(e)))
                stats.failed += len(batch_paths)
                LOGGER.error(f"[{self.method}] 处理失败 ({batch_paths}): {e}")

        LOGGER.info(
            f"[{self.method}] 结束: 成功 {stats.success}/{stats.total}, 失败 {stats.failed}"
        )
        return stats


__all__ = [
    "ModalGeneratorBase",
    "GeneratorRegistry",
    "SaveOptions",
    "GeneratorRunStats",
]
