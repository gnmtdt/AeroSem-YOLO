"""
C2f Extraction - Base Components

职责：
- 提供 C2f 变体迁移时可复用的基础封装，避免每个 C2f_* 变体重复样板代码。

约束：
- 这里仅提供“构造工具/基类”，不在此导出具体 C2f_* 变体类。
- 具体 C2f_* 变体请在 `c2f_variants.py` 中定义与导出。
"""

from __future__ import annotations

import torch.nn as nn

from ultralytics.nn.modules.block import C2f


class C2fVariantBase(C2f):
    """
    C2f 变体基类：继承标准 C2f，并提供统一的内部 block 替换入口。

    C2f 的核心结构是：
    - `cv1` 将输入映射为 2*c 的隐层
    - `m` 为长度 n 的 block 列表（默认是 Bottleneck）
    - `cv2` 将 (2+n)*c 拼接结果映射回 c2

    迁移 C2f 变体时，只需要替换 `self.m` 的 block 类型即可。
    """

    def _build_blocks(self, repeats: int, block_factory) -> None:
        """
        使用 block_factory 构造 ModuleList 并替换 self.m。

        Args:
            repeats: 对应 C2f 的 n（重复次数）
            block_factory: 0 参 callable，每次调用返回一个 nn.Module
        """
        self.m = nn.ModuleList(block_factory() for _ in range(repeats))


__all__ = ["C2fVariantBase"]

