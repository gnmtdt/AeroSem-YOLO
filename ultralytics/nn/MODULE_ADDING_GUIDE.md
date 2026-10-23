# YOLOMM 模块添加规范指南

本文档总结了在 YOLOMM 项目中添加新模块的规范流程，基于 `module-add` 分支添加 MROD-YOLO 模块（MJRNet、RFEM、MSIA）的实践经验。

## 目录

- [五步规范流程](#五步规范流程)
- [完整检查清单](#完整检查清单)
- [代码模板](#代码模板)
- [常见问题及解决](#常见问题及解决)

---

## 五步规范流程

### 第一步：编写模块代码

**位置选择**：根据模块功能放置到对应目录：

| 模块类型 | 目录路径 | 说明 |
|----------|----------|------|
| Fusion 模块 | `ultralytics/nn/modules/fusion/` | 多模态融合模块 |
| Extraction 模块 | `ultralytics/nn/extraction/` | 特征提取模块 |
| Neck 模块 | `ultralytics/nn/Neck/` | 颈部连接模块 |
| Block 模块 | `ultralytics/nn/modules/block.py` | 基础构建块 |

**代码规范要求**：

```python
"""
模块名称
论文: 论文标题
来源: 期刊/会议 年份

包含模块:
- ModuleA: 功能描述
- ModuleB: 功能描述
"""

import torch
import torch.nn as nn

__all__ = ['ModuleA', 'ModuleB']  # 必须声明导出列表


class ModuleA(nn.Module):
    """模块描述

    【核心功能】
    描述模块的主要作用

    【工作机制】
    1. 步骤一
    2. 步骤二

    Args:
        in_channels (int): 输入通道数
        reduction (int): 压缩比例，默认16
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 模块实现
        ...

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W] 或列表 [feat1, feat2]
        Returns:
            输出特征图 [B, C, H, W]
        """
        ...
```

### 第二步：更新包的 `__init__.py`

在对应目录的 `__init__.py` 中添加导入和导出：

```python
# 1. 添加导入语句
from .your_module import ModuleA, ModuleB

# 2. 添加到 __all__ 元组
__all__ = (
    # 现有模块...
    'ModuleA',
    'ModuleB',
)
```

### 第三步：在 `tasks.py` 中注册模块

位置：`ultralytics/nn/tasks.py`

**方式一：通过包的通配符导入（推荐用于 fusion 等已有通配符导入的包）**

如果目标包已经使用通配符导入，只需确保模块在 `__init__.py` 的 `__all__` 中即可：

```python
# tasks.py 中已有
from ultralytics.nn.modules.fusion import *  # 自动导入 __all__ 中的模块
```

**方式二：显式导入（独立模块或新包）**

```python
# 在 tasks.py 的导入区添加
_YOUR_IMPORT_ERROR = None
try:
    from ultralytics.nn.extraction.your_module import (
        ModuleA,
        ModuleB,
    )
    YOUR_MODULE_AVAILABLE = True
except ImportError as _err:
    _YOUR_IMPORT_ERROR = _err
    YOUR_MODULE_AVAILABLE = False
    LOGGER.warning(f"Your module import failed: {_err}")

    def _missing_your_module(*args, **kwargs):
        raise RuntimeError(f"Your module unavailable: {_err}")

    for _name in ['ModuleA', 'ModuleB']:
        globals()[_name] = _missing_your_module
```

### 第四步：在 `parse_model()` 中添加通道处理逻辑

这是**最关键的一步**，位置：`ultralytics/nn/tasks.py` 的 `parse_model()` 函数

**单输入模块示例**：

```python
elif m is ModuleA:
    # 单输入模块：输出通道等于输入通道
    c1 = ch[f]
    # args[0] 是配置文件中的通道数占位符，用实际值替换
    args = [c1] + list(args[1:]) if len(args) > 1 else [c1]
    c2 = c1  # 输出通道数
```

**双输入模块示例**：

```python
elif m is ModuleB:
    # 双输入融合模块
    if isinstance(f, int) or len(f) != 2:
        raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
    c_left, c_right = ch[f[0]], ch[f[1]]

    # 验证通道一致性（如果模块要求）
    if c_left != c_right:
        raise ValueError(f"{m.__name__} expects equal input channels, got {c_left} vs {c_right} at layer {i}")

    # 构建参数列表
    reduction = args[1] if len(args) > 1 else 16
    args = [c_left, reduction]
    c2 = c_left  # 输出通道数
```

**多输入模块示例（通道不等）**：

```python
elif m is ModuleC:
    # 多输入模块，支持不同通道数
    if isinstance(f, int):
        f = [f]
    c_inputs = [ch[idx] for idx in f]
    c2 = sum(c_inputs)  # 例如 Concat 操作
    args = [c_inputs] + list(args[1:])
```

**关键原则**：
- 必须从 `ch[]` 获取实际输入通道数
- 必须正确设置 `c2`（输出通道），供后续层使用
- 必须验证多输入模块的通道一致性（如果模块有此要求）

### 第五步：编写 YAML ���置文件

位置：`ultralytics/cfg/models/mm/` 下的对应子目录

**YAML 层定义格式**：

```yaml
# [from, repeats, module, args, input_routing]
# 第1字段：输入层索引（-1 表示上一层，[0, 1] 表示多输入）
# 第2字段：重复次数
# 第3字段：模块名称
# 第4字段：模块参数列表
# 第5字段：多模态路由标识（可选，'RGB', 'X', 'Dual'）
```

**配置文件示例**：

```yaml
# Ultralytics YOLOMM - 模块示例配置
nc: 80
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]

backbone:
  # ========== 双输入模块示例（多模态路由）==========
  - [-1, 1, Conv, [64, 3, 2], 'RGB']     # 0 RGB 路径
  - [-1, 1, Conv, [64, 3, 2], 'X']       # 1 X 模态路径
  - [[0, 1], 1, MJRNet, [64, 16]]        # 2 双输入融合

  # ========== 单输入模块示例 ==========
  - [-1, 1, Conv, [128, 3, 2]]           # 3
  - [-1, 1, RFEM, [128]]                 # 4 单输入感受野扩展

  # ========== Neck 模块示例 ==========
  - [[3, 5], 1, MSIA, [256]]             # 双输入聚合

head:
  - [[...], 1, Detect, [nc]]
```

---

## 完整检查清单

| 步骤 | 文件位置 | 操作内容 | 验证方法 |
|------|----------|----------|----------|
| 1 | `nn/xxx/your_module.py` | 编写模块类，包含 `__all__` | 单独 import 测试 |
| 2 | `nn/xxx/__init__.py` | 添加 import 和 `__all__` 导出 | `from package import *` 测试 |
| 3 | `nn/tasks.py` 导入区 | 注册模块（显式或通配符） | 检查 `globals()` 中是否存在 |
| 4 | `nn/tasks.py` parse_model() | 添加通道处理 elif 分支 | 构建模型测试 |
| 5 | `cfg/models/mm/*.yaml` | 编写配置文件 | 加载配置测试 |
| 6 | 测试验证 | 完整模型构建和前向传播 | 运行推理测试 |

---

## 代码模板

### 模块文件模板

```python
"""
模块名称 - 简短描述
论文: 完整论文标题
来源: 期刊/会议名称 年份
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['YourModule']


class YourModule(nn.Module):
    """模块完整名称

    【核心功能】
    描述模块解决什么问题

    【工作机制】
    1. 第一步处理
    2. 第二步处理
    3. 输出结果

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数，默认等于输入
        reduction (int): 瓶颈压缩比例，默认16
    """

    def __init__(self, in_channels, out_channels=None, reduction=16):
        super().__init__()
        out_channels = out_channels or in_channels

        # 定义层
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [B, C, H, W] 或列表 [x1, x2]
        Returns:
            输出张量 [B, C_out, H, W]
        """
        # 处理多输入情况
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, dim=1)

        return self.act(self.bn(self.conv(x)))
```

### parse_model 分支模板

```python
# 在 parse_model() 函数的 elif 链中添加

elif m is YourModule:
    # YourModule: 描述模块类型和输入输出关系
    if isinstance(f, int):
        # 单输入情况
        c1 = ch[f]
        args = [c1] + list(args[1:])
        c2 = c1  # 或根据模块逻辑计算
    else:
        # 多输入情况
        if len(f) != 2:
            raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
        c_left, c_right = ch[f[0]], ch[f[1]]
        # 通道验证（如需要）
        if c_left != c_right:
            raise ValueError(f"{m.__name__} channel mismatch: {c_left} vs {c_right}")
        args = [c_left] + list(args[1:])
        c2 = c_left
```

---

## 常见问题及解决

### 1. 通道数不匹配错误

**错误信息**：`expects equal input channels, got X vs Y`

**解决方案**：
- 在 YAML 中添加 1x1 Conv 层对齐通道
- 或修改模块支持不同通道输入

```yaml
# 通道对齐示例
- [5, 1, nn.Upsample, [None, 2, "nearest"]]  # 上采样
- [-1, 1, Conv, [256, 1, 1]]                  # 通道对齐到 256
- [[3, -1], 1, MSIA, [256]]                   # 现在通道匹配
```

### 2. 模块未找到错误

**错误信息**：`NameError: name 'YourModule' is not defined`

**排查步骤**：
1. 检查模块文件的 `__all__` 是否包含模块名
2. 检查 `__init__.py` 是否正确导入
3. 检查 `tasks.py` 是否正确注册
4. 检查是否有 ImportError 被静默处理

### 3. AMP 兼容性问题

**现象**：使用混合精度训练时出现 NaN 或精度下降

**解决方案**：
- 模块包含 LayerNorm 等对精度敏感的层时，建议在配置文件添加警告
- 或在训练时禁用 AMP：`model.train(amp=False)`

```yaml
# 在 YAML 文件头部添加注释
# 注意: 本配置包含 LayerNorm 层，建议禁用 AMP 训练
# 使用 amp=False 参数进行训练
```

### 4. Width Scaling 问题

**现象**：使用不同 scale（n/s/m/l/x）时通道数错误

**原因**：`parse_model` 中的通道处理未���虑 width 缩放

**解决方案**：确保从 `ch[]` 获取通道数，而非使用 YAML 中的原始值

```python
# 错误做法
c1 = args[0]  # 直接使用 YAML 值

# 正确做法
c1 = ch[f]    # 从通道历史获取实际值
```

### 5. 输入索引错误

**错误信息**：`IndexError: list index out of range`

**原因**：YAML 中的层索引超出范围

**排查**：
- 检查 `from` 字段的索引是否正确
- 负索引 `-1` 表示上一层，`-2` 表示上两层
- 确保引用的层已经定义

## 万物皆可融 Block 体系

### 概述

万物皆可融 Block 体系将 CSP 容器（C3/C2f/C3k2）与子模块解耦，允许通过 YAML 配置动态注入任意子模块，无需为每种组合创建独立类。

**何时使用 Block 而不是传统变体：**
- 新模块需要快速验证在 C2f/C3k2 中的效果 — 用 Block
- 已有稳定变体且需要特殊处理逻辑 — 继续用传统变体
- 两种方式可共存（双轨并行）

### 可用的 Block 容器

| Block 类 | 对应标准类 | 默认 Bottleneck k | 特点 |
|----------|----------|------------------|------|
| `C3_Block` | C3 | (1,3) | CSP 三卷积结构 |
| `C2f_Block` | C2f | (3,3) | C2f 分流串联结构 |
| `C3k2_Block` | C3k2 | (3,3) | C3k2 双层 CSP（c3k 开关） |

### selfatt 参数说明

`selfatt` 控制子模块实例化时接收的参数数量：

| selfatt | 实例化方式 | 适用场景 | 示例子模块 |
|---------|-----------|---------|-----------|
| `False`（默认） | `module(c1, c2)` | 卷积/瓶颈类模块 | Bottleneck, Faster_Block, Star_Block, iRMB |
| `True` | `module(dim)` | 注意力类模块 | CoordAtt, EMA, LSKBlock, SimAM, DeformableLKA |

### YAML 配置示例

**1. 使用默认 Bottleneck（等同于标准 C2f）：**
```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, C2f_Block, [64]]
```

**2. 指定注意力子模块（selfatt=True）：**
```yaml
backbone:
  - [-1, 1, C3k2_Block, [64, {'module': 'CoordAtt', 'selfatt': True}]]
```

**3. 指定卷积子模块（selfatt=False 可省略）：**
```yaml
backbone:
  - [-1, 1, C2f_Block, [64, {'module': 'Faster_Block'}]]
```

**4. C3_Block 使用：**
```yaml
backbone:
  - [-1, 1, C3_Block, [128, {'module': 'Star_Block'}]]
```

### 新增子模块适配指南

要让一个新模块兼容 Block 体系，需要满足以下条件之一：
1. **双参数接口**（selfatt=False）：`__init__(self, c1, c2, ...)` — 如 Bottleneck、Faster_Block
2. **单参数接口**（selfatt=True）：`__init__(self, dim, ...)` — 如 CoordAtt、EMA、SimAM

模块无需任何修改，只需确保已在 tasks.py 中导入即可。YAML 中直接引用类名。

### 注册位置

Block 相关代码位置：
- 模块实现：`ultralytics/nn/extraction/block_fusion.py`
- 注册集合：`ultralytics/nn/tasks.py` 中的 `block_repeat_modules`
- parse_model 分支：`m in block_repeat_modules` 的 elif 分支（在 base_modules 之前）

---

*文档版本: 1.1*
*最后更新: 2026-03-31*
*基于: module-add 分支 MROD-YOLO 模块添加实践 + 万物皆可融 Block 体系*
