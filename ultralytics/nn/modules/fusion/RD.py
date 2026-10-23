"""
RD modules for YOLO-RD style dictionary injection.

Exports:
- DConv: Dictionary-injection block (CG -> GIE -> PONO -> D) with residual mix.
- RepNCSPELAND: ELAN-style block followed by DConv (composed on RepNCSPELAN4).

Purpose:
- Provide the minimal necessary components to bring YOLO-RD's核心“检索-字典”思想
  到 Ultralytics 代码库，便于在 YAML 中被引用或在多模态结构中复用。

Notes:
- 依赖现有模块 Conv 与 RepNCSPELAN4，无外部依赖。
- 若需在 YAML 中直接使用，请确保在 ultralytics.nn.modules/__init__.py 和
  ultralytics.nn.tasks 的构建映射中完成注册（本仓库已按 FFN 模块模式注册）。
"""

from __future__ import annotations

import torch
from torch import nn

# Reuse Ultralytics modules
from ..conv import Conv  # standard Conv-BN-Activation wrapper
from ..block import RepNCSPELAN4  # GELAN/ELAN block used in YOLOv9 family

__all__ = ("DConv", "RepNCSPELAND")


class DConv(nn.Module):
    """
    【核心创新】
    Dictionary injection convolution block 实现了基于字典检索的特征增强机制。该模块将输入特征通过"编码-检索-归一化-解码"四阶段流水线，
    模拟字典查询过程实现特征的自适应增强。核心思想是将特征映射到高维字典空间进行信息检索，然后通过位置归一化确保特征分布稳定，
    最后通过残差混合保持原始信息的同时注入检索到的增强信息。

    【解决的问题】
    - 传统卷积难以捕获长距离依赖和全局上下文信息
    - 特征表示缺乏自适应性，无法根据输入内容动态调整
    - 深层网络中特征分布不稳定，影响模型训练和泛化
    - 缺乏有效的特征检索和重用机制

    【工作机制】
    输入特征 r 首先通过 CG (Codebook Generator) 1x1卷积映射到atoms维度的字典空间，生成查询向量。
    然后通过 GIE (Global Information Extraction) 深度可分离5x5卷积在空间维度进行信息交互，无激活函数保持线性变换。
    接着应用 PONO (Position-wise Normalization) 进行通道维度的位置归一化，稳定特征分布。
    最后通过 D (Dictionary) 1x1卷积将字典特征映射回原始维度，与输入特征进行alpha加权残差融合。

    【设计优势】
    - 字典机制：通过高维字典空间实现特征检索和增强，提升表示能力
    - 位置归一化：PONO确保特征在通道维度的分布稳定，提高训练效率
    - 残差混合：alpha参数控制原始信息与增强信息的平衡，保持特征完整性
    - 轻量设计：仅使用1x1和深度卷积，计算开销小且易于集成

    Dictionary injection convolution block used by YOLO-RD.

    Pipeline:
        r --(CG 1x1)--> x --(GIE depthwise 5x5, no act)--> x --(PONO)--> x --(D 1x1, no act)-->
        x --(alpha mix with residual r)--> out

    Args:
        c1 (int): Input channels.
        alpha (float): Residual mixing factor, out = alpha * x + (1 - alpha) * r. Default: 0.8
        atoms (int): Size of dictionary atoms (intermediate channels). Default: 512
    """

    def __init__(self, c1: int, alpha: float = 0.8, atoms: int = 512) -> None:
        super().__init__()
        self.alpha = float(alpha)

        # 1x1 conv to generate atoms (codebook-like channels)
        self.CG = Conv(c1, atoms, 1)
        # depthwise 5x5 conv with no activation
        self.GIE = Conv(atoms, atoms, 5, g=atoms, act=False)
        # projection back to input channels with no activation
        self.D = Conv(atoms, c1, 1, act=False)

    @staticmethod
    def _pono(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Position-wise Normalization (channel-wise mean/std)."""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + eps)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = self.CG(r)
        x = self.GIE(x)
        x = self._pono(x)
        x = self.D(x)
        return self.alpha * x + (1.0 - self.alpha) * r


class RepNCSPELAND(RepNCSPELAN4):
    """
    【核心创新】
    RepNCSPELAND 将 ELAN 高效结构化特征提取与字典注入机制相结合，实现了结构感知的特征增强。该模块首先通过RepNCSPELAN4
    进行多尺度特征聚合和结构化表示学习，然后通过DConv字典注入模块对提取的结构特征进行自适应增强。
    这种"结构先导，字典增强"的设计哲学兼顾了特征的层次性和自适应性。

    【解决的问题】
    - 单一特征提取策略难以同时处理结构化和语义化信息
    - ELAN结构缺乏对特征的自适应增强能力
    - 传统残差连接无法进行特征的动态重构
    - 多尺度特征融合后仍需要进一步的表示优化

    【工作机制】
    输入特征首先经过继承的RepNCSPELAN4模块进行结构化特征提取，该过程包括多分支并行卷积、跨尺度特征融合和高效信息聚合。
    然后将ELAN输出的结构化特征送入DConv字典注入模块，通过编码-检索-归一化-解码的四阶段流水线进行特征增强。
    整体数据流为：输入 -> RepNCSPELAN4结构提取 -> DConv字典增强 -> 输出增强特征。

    【设计优势】
    - 双阶段增强：结构化提取与字典增强的有机结合，提升特征表达力
    - 继承优化：完全复用RepNCSPELAN4的高效实现，降低开发和维护成本  
    - 模块化设计：DConv作为可配置插件，支持灵活的超参数调整
    - 计算高效：利用现有优化模块，避免重复计算和冗余操作

    RepNCSPELAN4 followed by DConv (YOLO-RD style).

    This mirrors the idea of yolo_rd's RepNCSPELAND, adapted to Ultralytics:
    we compose the existing RepNCSPELAN4 block and append an RD DConv.

    Args:
        c1 (int): Input channels (to RepNCSPELAN4).
        c2 (int): Output channels (from RepNCSPELAN4), also the DConv input.
        c3 (int): Intermediate channels for RepNCSPELAN4.
        c4 (int): Intermediate channels for RepCSP inside RepNCSPELAN4.
        n (int): Number of RepCSP blocks in RepNCSPELAN4.
        atoms (int): DConv dictionary atoms.
        alpha (float): DConv residual mixing factor.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
        n: int = 1,
        *,
        atoms: int = 512,
        alpha: float = 0.8,
    ) -> None:
        super().__init__(c1, c2, c3, c4, n)
        self.dconv = DConv(c2, alpha=alpha, atoms=atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return self.dconv(x)
