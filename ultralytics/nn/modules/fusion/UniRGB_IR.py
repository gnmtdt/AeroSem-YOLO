"""
UniRGB-IR: simplified fusion modules for RGB-IR (PyTorch-only).

导出类（对外公开使用）：
- SpatialPriorModuleLite：IR侧轻量金字塔，输出 8x/16x/32x 多尺度特征（B,C,H,W）
- ConvMixFusion：分组局部卷积 + 共享门控的同尺度融合（输入/输出形状一致）
- ScalarGate / ChannelGate：简化门控（标量门/通道门）控制 IR 注入强度
- ncc：融合诊断的归一化互相关（建议仅用于训练/验证诊断，不进入推理路径）

说明：本文件仅使用 PyTorch 原生组件，避免序列/注意力路径，便于在 YOLO 主干/Neck 的
P3/P4/P5 等尺度以最小改动接入。
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv2d + (optional) BatchNorm2d + (optional) ReLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1,
                 bn: bool = True, act: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialPriorModuleLite(nn.Module):
    """
    【核心创新】
    SpatialPriorModuleLite 实现了轻量化的IR侧多尺度特征金字塔，专门为红外模态设计的空间先验提取器。
    该模块通过渐进式下采样构建与YOLO标准尺度(P3/P4/P5)完全对齐的多层级特征表示，避免了复杂的注意力机制，
    仅使用标准卷积操作实现高效的红外空间信息编码。设计哲学强调"轻量而不失效能"。

    【解决的问题】
    - 红外图像的空间结构信息提取和多尺度表示学习
    - RGB-IR双模态融合中IR侧特征的尺度对齐问题
    - 计算资源约束下的高效红外特征编码需求
    - 标准YOLO架构的无缝集成和模态兼容性

    【工作机制】
    输入红外图像通过stem_ir模块进行4倍下采样和初步特征提取，包含3个连续卷积层和最大池化。
    然后依次通过conv2、conv3、conv4进行8倍、16倍、32倍渐进式下采样，每个阶段提取不同感受野的空间特征。
    最终通过out_8、out_16、out_32三个投影层将特征维度调整为与YOLO P3/P4/P5尺度匹配的目标维度，
    输出三元组(T8, T16, T32)用于后续的多模态特征融合。

    【设计优势】  
    - 尺度对齐：输出尺度完全匹配YOLO标准，无需额外适配层
    - 轻量高效：纯卷积实现，无注意力机制，计算开销小
    - 渐进提取：多阶段下采样保持细节到语义的平滑过渡
    - 模块化设计：可选BatchNorm，支持不同部署环境的灵活配置

    IR-side lightweight CNN pyramid (8x/16x/32x downsample) that outputs
    2D feature maps aligned to YOLO scales (P3/P4/P5).

    - Input: IR image tensor (B, C_ir, H, W)
    - Output: tuple(T8, T16, T32) with shapes (B, C8, H/8, W/8), (B, C16, H/16, W/16), (B, C32, H/32, W/32)

    This is a simplified variant of UniRGB-IR's SpatialPriorModule keeping
    only standard PyTorch ops and (optionally) BatchNorm2d.
    """

    def __init__(
        self,
        inplanes: int = 64,
        embed_dims: Tuple[int, int, int] = (256, 512, 1024),
        in_chans: int = 3,
        use_bn: bool = True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        bn = use_bn

        # stem_ir: 4x downsample
        self.stem_ir = nn.Sequential(
            ConvBNAct(in_chans, inplanes, k=3, s=2, p=1, bn=bn, act=True),
            ConvBNAct(inplanes, inplanes, k=3, s=1, p=1, bn=bn, act=True),
            ConvBNAct(inplanes, inplanes, k=3, s=1, p=1, bn=bn, act=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # -> 4x
        )

        # 8x, 16x, 32x
        self.conv2 = ConvBNAct(inplanes, 2 * inplanes, k=3, s=2, p=1, bn=bn, act=True)   # -> 8x
        self.conv3 = ConvBNAct(2 * inplanes, 4 * inplanes, k=3, s=2, p=1, bn=bn, act=True)  # -> 16x
        self.conv4 = ConvBNAct(4 * inplanes, 4 * inplanes, k=3, s=2, p=1, bn=bn, act=True)  # -> 32x

        # project to desired dims (typically match YOLO P3/P4/P5)
        c8, c16, c32 = embed_dims
        self.out_8 = nn.Conv2d(2 * inplanes, c8, kernel_size=1, stride=1, padding=0, bias=True)
        self.out_16 = nn.Conv2d(4 * inplanes, c16, kernel_size=1, stride=1, padding=0, bias=True)
        self.out_32 = nn.Conv2d(4 * inplanes, c32, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, ir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x4 = self.stem_ir(ir)
        x8 = self.conv2(x4)
        x16 = self.conv3(x8)
        x32 = self.conv4(x16)

        t8 = self.out_8(x8)
        t16 = self.out_16(x16)
        t32 = self.out_32(x32)
        return t8, t16, t32


class ConvMixFusion(nn.Module):
    """
    【核心创新】
    ConvMixFusion 实现了分组局部卷积与共享门控的同尺度多模态融合机制。该模块将通道维度分组，每组使用不同尺寸的卷积核进行RGB/IR特征的独立处理，
    然后通过组内共享的1x1门控网络自适应调节两个模态的融合权重。这种"分组处理，门控融合"的设计实现了多尺度感受野的并行处理和精细化的模态平衡控制。

    【解决的问题】
    - RGB和IR模态特征的有效融合，保持输入输出形状一致性
    - 不同感受野下的多尺度特征交互和互补信息提取
    - 模态间贡献度的自适应调节和动态权重分配
    - 计算效率与融合效果的平衡优化

    【工作机制】
    将输入的RGB和IR特征在通道维度按groups分组，每组通道数为channels//groups。
    为每个分组配置不同kernel size的局部卷积(默认3,3,5,7)，分别处理RGB和IR对应的通道组。
    处理后的特征进行元素级相加得到混合特征，然后通过共享的1x1门控卷积生成sigmoid激活的门控权重alpha。
    最终输出为rgb_i * alpha + ir_i * (1 - alpha)，实现自适应的分组级别融合。

    【设计优势】
    - 多尺度感受野：不同kernel size的分组卷积捕获多层次的空间特征
    - 精细化融合：组级别的独立门控实现细粒度的模态权重控制
    - 形状保持：融合前后特征图尺寸完全一致，便于集成到现有架构
    - 计算高效：分组处理减少参数量，门控机制计算开销小

    Grouped local convs on RGB/IR partial channels + shared 1x1 gate per group.
    Fuses two same-shape tensors (B, C, H, W) into (B, C, H, W).

    Derived from UniRGB-IR ConvMixFusion idea, adapted to pure PyTorch.
    """

    def __init__(self, channels: int, kernels: Tuple[int, ...] = (3, 3, 5, 7), groups: int = 4):
        super().__init__()
        assert channels % groups == 0, "channels must be divisible by groups"
        assert len(kernels) == groups, "len(kernels) must equal groups"

        self.groups = groups
        self.channels = channels
        self.channel_per_group = channels // groups

        convs_rgb, convs_ir = [], []
        for ks in kernels:
            pad = (ks - 1) // 2
            convs_rgb.append(nn.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=ks, stride=1, padding=pad, bias=True))
            convs_ir.append(nn.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=ks, stride=1, padding=pad, bias=True))

        self.convs_rgb = nn.ModuleList(convs_rgb)
        self.convs_ir = nn.ModuleList(convs_ir)
        # shared gate (per-group): 1x1 conv on partial channels -> 1 map
        self.gate = nn.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x) -> torch.Tensor:
        # 仅支持列表/元组传参的双输入模式
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("ConvMixFusion 需要以 [rgb, ir] 列表/元组形式传入两路特征")
        rgb, ir = x
        if not (isinstance(rgb, torch.Tensor) and isinstance(ir, torch.Tensor)):
            raise TypeError("ConvMixFusion 的两路输入必须为张量")
        if rgb.shape != ir.shape:
            raise ValueError(f"两路输入形状需一致，got {rgb.shape} vs {ir.shape}")
        B, C, H, W = rgb.shape
        outs = []
        for i in range(self.groups):
            sl = slice(i * self.channel_per_group, (i + 1) * self.channel_per_group)
            rgb_i = self.convs_rgb[i](rgb[:, sl, :, :])
            ir_i = self.convs_ir[i](ir[:, sl, :, :])
            mix = rgb_i + ir_i
            alpha = torch.sigmoid(self.gate(mix))  # gate in [0,1]
            outs.append(rgb_i * alpha + ir_i * (1 - alpha))
        return torch.cat(outs, dim=1)


class ScalarGate(nn.Module):
    """
    【核心创新】
    ScalarGate 实现了全局标量门控的多模态融合机制，通过单一标量权重控制整个特征图的RGB-IR融合比例。
    该模块利用全局平均池化提取两个模态的全局统计信息，然后通过简单的全连接层学习全局级别的模态权重，
    实现计算高效且全局感知的模态自适应融合。设计理念强调"全局统筹，简洁高效"。

    【解决的问题】  
    - 多模态特征融合中的全局权重分配和模态贡献度调节
    - 计算资源约束下的高效融合机制设计需求
    - 全局上下文信息在模态融合中的有效利用
    - 简化融合控制，避免复杂的空间或通道级门控计算

    【工作机制】
    分别对RGB和IR特征图进行全局平均池化(GAP)，得到每个模态的全局描述符(B,C,1,1)。
    将两个全局描述符在通道维度拼接，形成联合全局特征(B,2C,1,1)。
    通过1x1卷积(等价于全连接)将联合特征映射为单一标量权重，经sigmoid激活得到[0,1]范围的门控系数z。
    最终融合输出为rgb * (1-z) + ir * z，其中z全局控制IR模态的贡献度。

    【设计优势】
    - 全局感知：GAP提取全局统计信息，融合决策基于全局上下文
    - 计算高效：仅一个1x1卷积层，参数量和计算量极小
    - 直观控制：单一标量权重，融合逻辑简单易理解
    - 广播机制：标量权重自动广播到整个特征图，实现全局一致控制

    Global scalar gate per feature map:
    z = sigmoid(FC([GAP(rgb), GAP(ir)])) -> (B, 1, 1, 1)
    out = (1 - z) * rgb + z * ir
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(2 * channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("ScalarGate 需要以 [rgb, ir] 列表/元组形式传入两路特征")
        rgb, ir = x
        if not (isinstance(rgb, torch.Tensor) and isinstance(ir, torch.Tensor)):
            raise TypeError("ScalarGate 的两路输入必须为张量")
        if rgb.shape != ir.shape:
            raise ValueError(f"两路输入形状需一致，got {rgb.shape} vs {ir.shape}")
        gap_rgb = F.adaptive_avg_pool2d(rgb, output_size=1)
        gap_ir = F.adaptive_avg_pool2d(ir, output_size=1)
        g = torch.cat([gap_rgb, gap_ir], dim=1)  # (B, 2C, 1, 1)
        z = torch.sigmoid(self.fc(g))  # (B, 1, 1, 1)
        return rgb * (1 - z) + ir * z


class ChannelGate(nn.Module):
    """
    【核心创新】
    ChannelGate 实现了基于Squeeze-and-Excite风格的通道级门控多模态融合机制。该模块为每个通道独立学习模态融合权重，
    通过通道级的精细化控制实现RGB-IR特征的自适应融合。相比全局标量门控，ChannelGate提供更细粒度的通道级别调节能力，
    能够根据不同通道的语义内容自适应调整模态贡献度。

    【解决的问题】
    - 不同通道携带语义信息的差异化和模态贡献度的异质性
    - 全局门控无法处理通道级别的精细化融合控制需求
    - 多模态特征在通道维度上的互补性和冗余性的平衡
    - SE注意力机制在多模态融合中的有效应用和扩展

    【工作机制】
    分别对RGB和IR特征进行全局平均池化(GAP)获得通道级全局描述符，然后在通道维度拼接形成联合特征(B,2C,1,1)。
    通过两层MLP网络进行特征变换：第一层将2C维特征压缩到hidden维度并应用ReLU激活，第二层扩展回C维度。
    最后通过sigmoid激活得到每个通道的门控权重z(B,C,1,1)，实现通道级的模态权重分配。
    融合输出为rgb * (1-z) + ir * z，其中z的每个通道分量独立控制该通道的IR模态贡献度。

    【设计优势】
    - 通道级精细控制：每个通道独立的门控权重，实现细粒度融合调节
    - SE注意力扩展：借鉴SE机制的通道注意力思想，适配多模态融合场景
    - 可配置压缩比：通过hidden_ratio调节中间层维度，平衡表达力与计算效率
    - 语义感知能力：通道级权重能够反映不同语义特征的模态偏好

    Per-channel gate via squeeze-and-excite style MLP:
    z = sigmoid(Conv1x1([GAP(rgb), GAP(ir)])_{2C->C}) -> (B, C, 1, 1)
    out = (1 - z) * rgb + z * ir
    """

    def __init__(self, channels: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(1, int(channels * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("ChannelGate 需要以 [rgb, ir] 列表/元组形式传入两路特征")
        rgb, ir = x
        if not (isinstance(rgb, torch.Tensor) and isinstance(ir, torch.Tensor)):
            raise TypeError("ChannelGate 的两路输入必须为张量")
        if rgb.shape != ir.shape:
            raise ValueError(f"两路输入形状需一致，got {rgb.shape} vs {ir.shape}")
        gap_rgb = F.adaptive_avg_pool2d(rgb, output_size=1)
        gap_ir = F.adaptive_avg_pool2d(ir, output_size=1)
        g = torch.cat([gap_rgb, gap_ir], dim=1)  # (B, 2C, 1, 1)
        z = torch.sigmoid(self.mlp(g))           # (B, C, 1, 1)
        return rgb * (1 - z) + ir * z


def ncc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute normalized cross-correlation over feature maps for diagnostics.
    Returns a scalar per batch (B,) representing NCC between a and b.
    """
    assert a.shape == b.shape, "Inputs must have the same shape"
    B = a.shape[0]
    a_flat = a.view(B, -1)
    b_flat = b.view(B, -1)
    a_mean = a_flat.mean(dim=1, keepdim=True)
    b_mean = b_flat.mean(dim=1, keepdim=True)
    num = ((a_flat - a_mean) * (b_flat - b_mean)).sum(dim=1)
    den = torch.sqrt(((a_flat - a_mean) ** 2).sum(dim=1) * ((b_flat - b_mean) ** 2).sum(dim=1) + 1e-12)
    return num / den


__all__ = [
    "SpatialPriorModuleLite",
    "ConvMixFusion",
    "ScalarGate",
    "ChannelGate",
    "ncc",
]
