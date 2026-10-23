"""万物皆可融 Block 体系 - C3/C2f/C3k2 万物皆可融模块

本模块实现"万物皆可融"（Omni-Fusible）Block 架构，允许通过 YAML 配置动态注入任意子模块到 CSP 结构中。

=== 设计理念 ===

传统的 C3k2/C2f/C3 变体（如 C3k2_EMA、C2f_CAMixer）每种组合都是独立的类，导致大量重复代码。
本 Block 体系将 CSP 容器（C3/C2f/C3k2）与子模块（注意力/卷积/骨干等）解耦：
- 容器（本文件）：提供 C3/C2f/C3k2 三种 CSP 结构，通过 `module` 参数接受任意子模块
- 子模块：任何已注册到 tasks.py 的模块类均可作为子模块注入

=== selfatt 参数使用指南 ===

selfatt 控制子模块实例化时接收的参数数量：

| selfatt | 子模块实例化     | 适用场景                       | 示例子模块           |
|--------|-----------------|------------------------------|---------------------|
| False  | module(c_, c_)  | 卷积类模块（需要 c1/c2 两个通道参数） | Bottleneck, DRG, Faster_Block |
| True   | module(c_)      | 注意力类模块（仅需 dim 一个参数）   | CoordAtt, EMA, LSKBlock, SimAM |

=== YAML 配置示例 ===

1. 使用默认 Bottleneck（等同于标准 C2f/C3/C3k2）：
   [-1, 1, C2f_Block, [64]]

2. 指定子模块（注意力类，selfatt=True）：
   [-1, 1, C3k2_Block, [64, {'module': 'CoordAtt', 'selfatt': True}]]

3. 指定子模块（卷积类，selfatt=False 可省略）：
   [-1, 1, C2f_Block, [64, {'module': 'Faster_Block'}]]

4. 指定子模块 + 自定义参数：
   [-1, 1, C3k2_Block, [64, {'module': 'Star_Block', 'selfatt': False}]]

=== 新增子模块适配指南 ===

要让一个新模块兼容 Block 体系，需要满足以下条件之一：
1. 双参数接口：__init__(self, c1, c2, ...) — 对应 selfatt=False
2. 单参数接口：__init__(self, dim, ...) — 对应 selfatt=True

模块无需任何修改，只需确保已在 tasks.py 中导入即可。YAML 中直接引用类名即可。
"""

from functools import partial

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck


class C3k_Block(nn.Module):
    """C3k 内部嵌套 CSP 单元 — 作为 C3k2_Block 的内部组件。

    结构：cv1 分流 + n个module串联 + cv2 输出。
    当 selfatt=True 时，子模块接收单参数 (c_)；否则接收双参数 (c_, c_)。
    """

    def __init__(self, c1, c2, module=partial(Bottleneck, k=(3, 3), shortcut=True, e=0.5), n=2, e=0.5, selfatt=False):
        """初始化 C3k 内部 Block。

        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            module (callable): 子模块工厂，默认为 Bottleneck。
            n (int): 子模块重复次数。
            e (float): 扩展比例，控制隐藏通道数 c_ = int(c2 * e)。
            selfatt (bool): True 时子模块接收单参数(dim)，False 时接收双参数(c1,c2)。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 1, 1)
        if selfatt:
            self.m = nn.Sequential(*(module(c_) for _ in range(n)))
        else:
            self.m = nn.Sequential(*(module(c_, c_) for _ in range(n)))

    def forward(self, x):
        """前向传播：cv1 -> m -> cv2，残差连接。"""
        return self.cv2(self.m(self.cv1(x))) + x


class C3_Block(nn.Module):
    """C3 万物皆可融 — CSP 结构，支持任意子模块注入。

    结构：cv1 -> m(n个module) 与 cv2 并行 -> cat -> cv3 输出。
    默认 k=(1,3)，与标准 C3 一致。
    """

    def __init__(self, c1, c2, module=partial(Bottleneck, k=(1, 3), shortcut=True, e=0.5), n=1, e=0.5, selfatt=False):
        """初始化 C3 万物皆可融 Block。

        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            module (callable): 子模块工厂，默认为 Bottleneck(k=(1,3))。
            n (int): 子模块重复次数。
            e (float): 扩展比例。
            selfatt (bool): True 时子模块接收单参数(dim)，False 时接收双参数(c1,c2)。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        if selfatt:
            self.m = nn.Sequential(*(module(c_) for _ in range(n)))
        else:
            self.m = nn.Sequential(*(module(c_, c_) for _ in range(n)))

    def forward(self, x):
        """前向传播：m(cv1(x)) 与 cv2(x) 拼接后经 cv3 输出。"""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C2f_Block(nn.Module):
    """C2f 万物皆可融 — C2f 结构，支持任意子模块注入。

    结构：cv1 分两半 + n个module逐步串联 -> concat -> cv2 输出。
    默认 k=(3,3)，与标准 C2f 一致。
    """

    def __init__(self, c1, c2, module=partial(Bottleneck, k=(3, 3), shortcut=True, e=0.5), n=1, e=0.5, selfatt=False):
        """初始化 C2f 万物皆可融 Block。

        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            module (callable): 子模块工厂，默认为 Bottleneck(k=(3,3))。
            n (int): 子模块重复次数。
            e (float): 扩展比例。
            selfatt (bool): True 时子模块接收单参数(dim)，False 时接收双参数(c1,c2)。
        """
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        if selfatt:
            self.m = nn.ModuleList(module(self.c) for _ in range(n))
        else:
            self.m = nn.ModuleList(module(self.c, self.c) for _ in range(n))

    def forward(self, x):
        """前向传播：cv1 分两半，逐步串联 n 个子模块，全部拼接后经 cv2 输出。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_Block(nn.Module):
    """C3k2 万物皆可融 — C3k2 结构，支持 c3k 开关和任意子模块注入。

    结构：
    - c3k=True（默认）：C2f 式分流 + 每个分支内部嵌套 C3k_Block 形成双层 CSP
    - c3k=False：退化为 C2f_Block 行为

    默认 k=(3,3)，与标准 C3k2 一致。
    """

    def __init__(self, c1, c2, module=partial(Bottleneck, k=(3, 3), shortcut=True, e=0.5), n=1, c3k=True, e=0.5, selfatt=False):
        """初始化 C3k2 万物皆可融 Block。

        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            module (callable): 子模块工厂，默认为 Bottleneck(k=(3,3))。
            n (int): 子模块重复次数。
            c3k (bool): True 时内部嵌套 C3k_Block 形成双层 CSP，False 时退化为 C2f 行为。
            e (float): 扩展比例。
            selfatt (bool): True 时子模块接收单参数(dim)，False 时接收双参数(c1,c2)。
        """
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        if selfatt:
            self.m = nn.ModuleList(
                C3k_Block(self.c, self.c, module, 2, selfatt=selfatt) if c3k else module(self.c)
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                C3k_Block(self.c, self.c, module, 2) if c3k else module(self.c, self.c)
                for _ in range(n)
            )

    def forward(self, x):
        """前向传播：与 C2f_Block 相同的分流串联结构。"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
