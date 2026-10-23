"""
DEYOLO fusion modules: DEA (DECA + DEPA) and BiFocus (with C2f_BiFocus wrapper).

论文地址：https://arxiv.org/abs/2412.04931
论文题目：DEYOLO: Dual-Feature-Enhancement YOLO for Multi-Modal Road Defect Detection
中文题目：DEYOLO：用于多模态道路缺陷检测的双特征增强YOLO

This file consolidates DEYOLO's cross-modality Dual Enhancement Attention (DEA)
and Bi-directional Decoupled Focus (BiFocus) for usage within ultralyticsmm.
"""

import torch
import torch.nn as nn

# Use relative imports to the core Ultralytics modules in this package
from ..conv import Conv
from ..block import Bottleneck


class DEA(nn.Module):
    """Dual Enhancement Attention - 双增强注意力机制
    
    【核心创新】：
    DEA是DEYOLO的核心创新模块，通过"双重增强"策略解决多模态特征融合中的关键问题。
    不同于传统的简单拼接或相加融合方式，DEA采用了通道-空间两阶段渐进式增强架构。
    
    【解决的问题】：
    1. 多模态信息利用不充分：传统方法往往无法充分挖掘不同模态间的互补信息
    2. 特征融合粒度粗糙：缺乏对通道维度和空间维度的精细化建模
    3. 跨模态依赖关系建模不足：RGB和红外等模态间存在复杂的相互依赖关系
    
    【工作机制】：
    1. 第一阶段 - 通道增强(DECA)：专注于"什么信息重要"的问题
       - 分析RGB和红外两个模态在通道维度上的重要性分布
       - 通过全局上下文建模，让RGB特征学习红外的通道权重，红外特征学习RGB的通道权重
       - 实现跨模态的通道级信息交换和增强
       
    2. 第二阶段 - 空间增强(DEPA)：专注于"哪里的信息重要"的问题  
       - 在DECA增强后的特征基础上，进一步建模空间维度的重要性
       - 为每个模态生成空间注意力掩码，并设计全局门控机制
       - 同样采用跨模态策略：RGB用红外的空间权重，红外用RGB的空间权重
       
    3. 最终融合：通过sigmoid激活函数将两个增强后的模态特征融合为统一表示
    
    【设计优势】：
    - 渐进式增强：先通道后空间，符合人类视觉认知的层次化处理机制
    - 跨模态互补：每个模态都用另一个模态的注意力权重，最大化互补信息利用
    - 端到端优化：整个双增强流程可微分，支持端到端的联合训练

    x[0] -> RGB (or visible) feature map
    x[1] -> IR (or another modality) feature map
    """

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        super().__init__()
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)
        self.act = nn.Sigmoid()

    def forward(self, x):
        result_vi, result_ir = self.depa(self.deca(x))
        return self.act(result_vi + result_ir)


class DECA(nn.Module):
    """Dual Enhancement Channel Attention - 双增强通道注意力机制
    
    【核心创新】：
    DECA专门负责通道维度的跨模态增强，创新性地将通道注意力与全局上下文建模相结合。
    通过"交叉赋权"策略，让每个模态都能从另一个模态的视角重新审视自己的通道重要性。
    
    【解决的问题】：
    1. 通道重要性建模单一：传统方法只考虑单模态内部的通道关系
    2. 全局上下文信息利用不足：缺乏对特征图全局语义信息的有效整合
    3. 跨模态通道交互缺失：RGB和红外模态的通道间缺乏有效的信息交换机制
    
    【工作机制】：
    1. 全局特征提取：使用自适应平均池化将空间维度的信息聚合为全局描述符
       - 将[B,C,H,W]的特征图压缩为[B,C,1,1]的全局特征
       - 保留通道维度的完整信息，去除空间维度的干扰
       
    2. 通道重要性建模：通过两层MLP网络学习通道间的非线性依赖关系
       - 第一层：通道维度压缩，学习通道间的相关性
       - 第二层：恢复通道维度，并通过Sigmoid生成[0,1]范围的权重
       
    3. 全局上下文增强：设计卷积金字塔结构捕获多尺度全局信息
       - conv_c1、conv_c2、conv_c3形成三级金字塔，逐步扩大感受野
       - 当特征图尺寸足够大时使用金字塔，否则直接使用全局平均池化
       
    4. 跨模态交叉赋权：核心创新点
       - RGB特征使用红外模态学习到的通道权重进行重加权
       - 红外特征使用RGB模态学习到的通道权重进行重加权  
       - 让每个模态都能从"对方的眼光"看待自己的通道重要性
    
    【设计优势】：
    - 互补性最大化：跨模态权重赋值充分利用了模态间的互补特性
    - 多尺度建模：卷积金字塔能够适应不同尺度的全局上下文信息
    - 计算高效：全局池化大幅降低了计算复杂度，MLP网络参数量小
    """

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )
        self.act = nn.Sigmoid()
        self.compress = Conv(channel * 2, channel, 3)

        # Convolution pyramid for global context
        if p_kernel is None:
            p_kernel = [5, 4]
        # 不做隐式容错：若类型不符，让下游显式报错更早更清晰
        kernel1, kernel2 = p_kernel
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=channel),
            nn.SiLU(),
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=channel),
            nn.SiLU(),
        )
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(
                channel,
                channel,
                int(self.kernel_size / kernel1 / kernel2),
                int(self.kernel_size / kernel1 / kernel2),
                0,
                groups=channel,
            ),
            nn.SiLU(),
        )

    def forward(self, x):
        b, c, h, w = x[0].size()
        w_vi = self.avg_pool(x[0]).view(b, c)
        w_ir = self.avg_pool(x[1]).view(b, c)
        w_vi = self.fc(w_vi).view(b, c, 1, 1)
        w_ir = self.fc(w_ir).view(b, c, 1, 1)

        glob_t = self.compress(torch.cat([x[0], x[1]], 1))
        glob = (
            self.conv_c3(self.conv_c2(self.conv_c1(glob_t)))
            if min(h, w) >= self.kernel_size
            else torch.mean(glob_t, dim=[2, 3], keepdim=True)
        )
        result_vi = x[0] * (self.act(w_ir * glob)).expand_as(x[0])
        result_ir = x[1] * (self.act(w_vi * glob)).expand_as(x[1])

        return result_vi, result_ir


class DEPA(nn.Module):
    """Dual Enhancement Position Attention - 双增强位置注意力机制
    
    【核心创新】：
    DEPA专门负责空间维度的跨模态增强，通过精细化的空间建模和全局门控机制，
    实现了空间位置级别的跨模态信息交换和增强。这是继DECA通道增强后的第二阶段精细化处理。
    
    【解决的问题】：
    1. 空间注意力建模不足：传统方法难以精确建模空间位置的重要性分布
    2. 跨模态空间信息利用缺失：不同模态在空间维度上的互补信息未被充分利用
    3. 全局与局部信息平衡问题：缺乏有效机制平衡全局语义和局部细节信息
    
    【工作机制】：
    1. 多尺度空间特征提取：为每个模态设计双分支卷积结构
       - cv_v1/cv_i1：使用较小卷积核(如3x3)捕获局部空间模式
       - cv_v2/cv_i2：使用较大卷积核(如7x7)捕获更大范围的空间上下文
       - 通过不同尺度的卷积核组合，丰富空间表征能力
       
    2. 空间注意力掩码生成：为每个模态学习专门的空间权重
       - 将双分支特征concatenate后通过卷积融合
       - 生成对应模态的空间注意力掩码[B,1,H,W]
       - 每个位置的权重反映该位置对当前模态的重要程度
       
    3. 全局门控机制：设计共享的全局上下文调节器
       - compress1和compress2分别将两个模态的特征压缩为全局描述符
       - 通过相加融合生成统一的全局门控信号
       - 该信号作为全局上下文信息，调节局部空间注意力的激活强度
       
    4. 跨模态交叉注意力：核心创新的交叉赋权策略
       - RGB特征使用红外模态的空间注意力权重进行重加权
       - 红外特征使用RGB模态的空间注意力权重进行重加权
       - 让每个模态都能从另一个模态的空间视角重新定位重要区域
    
    【设计优势】：
    - 多尺度空间建模：双分支结构能够捕获不同尺度的空间模式
    - 全局-局部平衡：全局门控与局部空间注意力的结合实现了更好的平衡
    - 跨模态空间互补：充分利用了不同模态在空间维度上的互补特性
    - 端到端学习：整个空间增强过程完全可学习，无需手工设计的先验知识
    """

    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        self.conv1 = Conv(2, 1, 5)
        self.conv2 = Conv(2, 1, 5)
        self.compress1 = Conv(channel, 1, 3)
        self.compress2 = Conv(channel, 1, 3)
        self.act = nn.Sigmoid()

        # Convolution merge with different kernel sizes
        if m_kernel is None:
            m_kernel = [3, 7]
        # 不做隐式容错：若类型不符，让下游显式报错更早更清晰
        self.cv_v1 = Conv(channel, 1, m_kernel[0])
        self.cv_v2 = Conv(channel, 1, m_kernel[1])
        self.cv_i1 = Conv(channel, 1, m_kernel[0])
        self.cv_i2 = Conv(channel, 1, m_kernel[1])

    def forward(self, x):
        w_vi = self.conv1(torch.cat([self.cv_v1(x[0]), self.cv_v2(x[0])], 1))
        w_ir = self.conv2(torch.cat([self.cv_i1(x[1]), self.cv_i2(x[1])], 1))
        glob = self.act(self.compress1(x[0]) + self.compress2(x[1]))
        w_vi = self.act(glob + w_vi)
        w_ir = self.act(glob + w_ir)
        result_vi = x[0] * w_ir.expand_as(x[0])
        result_ir = x[1] * w_vi.expand_as(x[1])

        return result_vi, result_ir


class C2f_BiFocus(nn.Module):
    """C2f with integrated BiFocus - 集成双向焦点的C2f模块
    
    【核心创新】：
    C2f_BiFocus将经典的C2f架构与创新的BiFocus机制相结合，在保持C2f高效特征提取能力的基础上，
    增强了模型对方向性特征的捕获能力。这种设计实现了计算效率与表征能力的完美平衡。
    
    【解决的问题】：
    1. 感受野各向同性问题：传统卷积的感受野在各个方向上是均匀的，难以适应目标的方向性特征
    2. C2f模块表征能力限制：标准C2f虽然计算高效，但在复杂场景下表征能力仍有提升空间
    3. 方向性信息建模不足：道路缺陷等目标往往具有明显的方向性特征，需要专门的建模机制
    
    【工作机制】：
    1. 标准C2f特征提取：保持原有C2f的高效设计
       - cv1：输入通道分割，生成两路并行特征流
       - Bottleneck序列：通过n个瓶颈模块进行特征变换和深度学习
       - cv2：特征融合，将分割的特征流重新整合
       
    2. BiFocus方向性增强：在C2f输出基础上进行方向性特征增强
       - 不改变C2f的内在逻辑，而是作为后处理模块
       - 专门针对水平和垂直方向进行感受野扩展
       - 保持了模块的可插拔性和通用性
       
    3. 渐进式特征构建：
       - 第一步：通过C2f构建基础的多层级特征表示
       - 第二步：通过BiFocus增强方向性特征表达能力
       - 实现了从通用特征到专门化方向特征的渐进式构建
    
    【设计优势】：
    - 模块化设计：BiFocus与C2f的独立性保证了模块的可复用性
    - 计算高效：在C2f高效基础上只增加少量的方向性计算开销
    - 兼容性强：可以轻松集成到现有的YOLO架构中
    - 性能提升：针对方向性目标检测任务有显著的性能提升
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

        self.bifocus = BiFocus(c2, c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        return self.bifocus(y)

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class BiFocus(nn.Module):
    """Bi-directional Decoupled Focus - 双向解耦焦点机制
    
    【核心创新】：
    BiFocus是DEYOLO的另一个重要创新，通过"解耦焦点"策略实现对方向性特征的精细化建模。
    不同于传统的各向同性卷积，BiFocus专门设计了水平和垂直两个独立的焦点分支，
    能够更好地捕获具有明显方向性的目标特征，如道路裂缝、车道线等。
    
    【解决的问题】：
    1. 方向性特征捕获不足：传统卷积对水平和垂直方向的特征处理是耦合的，缺乏针对性
    2. 感受野设计缺陷：标准卷积的感受野形状固定，难以适应不同方向的目标形状
    3. 特征表达冗余：所有方向使用相同的卷积核，存在参数和计算的冗余
    
    【工作机制】：
    1. 双向解耦设计：将方向性特征建模解耦为两个独立分支
       - FocusH分支：专门负责水平方向的特征增强
       - FocusV分支：专门负责垂直方向的特征增强  
       - 两个分支使用不同的参数，能够学习到方向特定的特征模式
       
    2. 多特征融合策略：采用三路特征拼接的设计
       - 原始特征：保留输入的完整信息，作为基础表示
       - 水平焦点特征：从FocusH获得的水平方向增强特征
       - 垂直焦点特征：从FocusV获得的垂直方向增强特征
       - 三路特征在通道维度拼接，形成3倍通道数的丰富表示
       
    3. 深度可分离融合：使用高效的深度可分离卷积进行最终融合
       - 先深度卷积：在每个通道独立进行空间特征建模
       - 后逐点卷积：跨通道进行信息整合和维度调整
       - 既保证了融合效果，又控制了计算复杂度
    
    【设计优势】：
    - 方向性建模：针对水平和垂直方向的专门化处理
    - 信息保留：通过三路拼接保留了原始特征和增强特征
    - 计算高效：深度可分离卷积大幅降低了参数量和计算量
    - 灵活性强：可以独立调整水平和垂直方向的增强强度
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.focus_h = FocusH(c1, c1, 3, 1)
        self.focus_v = FocusV(c1, c1, 3, 1)
        self.depth_wise = DepthWiseConv(3 * c1, c2, 3)

    def forward(self, x):
        return self.depth_wise(torch.cat([x, self.focus_h(x), self.focus_v(x)], dim=1))


class FocusH(nn.Module):
    """Horizontal Decoupled Focus - 水平解耦焦点机制
    
    【核心创新】：
    FocusH实现了专门针对水平方向的特征焦点增强，通过巧妙的像素重排和双分支卷积设计，
    能够增强模型对水平方向特征（如水平裂缝、车道线等）的感知能力。
    
    【解决的问题】：
    1. 水平特征提取不充分：传统卷积对水平方向的特征提取缺乏针对性
    2. 感受野利用不均匀：在水平方向上的感受野扩展受限
    3. 计算资源分配不合理：所有方向使用相同的计算资源，缺乏重点
    
    【工作机制】：
    1. 像素重排策略：将输入特征按照特定规律重新组织
       - 将原始[B,C,H,W]特征重排为两个[B,C,H,W//2]的子特征
       - x1捕获棋盘模式的像素组合：(::2,::2)和(1::2,1::2)
       - x2捕获交错模式的像素组合：(::2,1::2)和(1::2,::2)
       - 通过重排增强水平方向的信息密度
       
    2. 双分支卷积处理：对重排后的特征分别进行卷积处理
       - conv1和conv2使用独立参数，能够学习不同的特征模式
       - 每个分支专门处理一种像素组合模式
       - 实现了水平方向的并行特征提取
       
    3. 像素重构：将处理后的特征重新组织回原始空间布局
       - 按照与重排相反的规律将特征放回原始位置
       - 确保空间对应关系的正确性
       - 最终输出与输入具有相同的空间尺寸
    
    【设计优势】：
    - 方向特化：专门针对水平方向的特征增强
    - 像素级精细化：通过像素重排实现更精细的特征建模
    - 并行处理：双分支设计提高了处理效率
    - 空间保持：输出保持原始的空间尺寸，便于后续处理
    """

    def __init__(self, c1, c2, kernel=3, stride=1):
        super().__init__()
        self.c2 = c2
        self.conv1 = Conv(c1, c2, kernel, stride)
        self.conv2 = Conv(c1, c2, kernel, stride)

    def forward(self, x):
        b, _, h, w = x.shape
        result = torch.zeros(size=[b, self.c2, h, w], device=x.device, dtype=x.dtype)
        x1 = torch.zeros(size=[b, self.c2, h, w // 2], device=x.device, dtype=x.dtype)
        x2 = torch.zeros(size=[b, self.c2, h, w // 2], device=x.device, dtype=x.dtype)

        x1[..., ::2, :], x1[..., 1::2, :] = x[..., ::2, ::2], x[..., 1::2, 1::2]
        x2[..., ::2, :], x2[..., 1::2, :] = x[..., ::2, 1::2], x[..., 1::2, ::2]

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        result[..., ::2, ::2] = x1[..., ::2, :]
        result[..., 1::2, 1::2] = x1[..., 1::2, :]
        result[..., ::2, 1::2] = x2[..., ::2, :]
        result[..., 1::2, ::2] = x2[..., 1::2, :]

        return result


class FocusV(nn.Module):
    """Vertical Decoupled Focus - 垂直解耦焦点机制
    
    【核心创新】：
    FocusV实现了专门针对垂直方向的特征焦点增强，与FocusH形成互补的双向焦点系统。
    通过垂直方向的像素重排和特征处理，能够增强模型对垂直特征（如垂直裂缝、柱状目标等）的检测能力。
    
    【解决的问题】：
    1. 垂直特征提取局限：传统卷积对垂直方向特征的针对性建模不足
    2. 垂直感受野扩展困难：在垂直方向上缺乏有效的感受野扩展机制
    3. 方向性特征不平衡：水平和垂直方向的特征处理缺乏平衡性设计
    
    【工作机制】：
    1. 垂直像素重排：按照垂直方向的特定规律重新组织像素
       - 将原始[B,C,H,W]特征重排为两个[B,C,H//2,W]的子特征
       - x1捕获垂直棋盘模式：(::2,::2)和(1::2,1::2)
       - x2捕获垂直交错模式：(1::2,::2)和(::2,1::2)
       - 重排策略专门优化垂直方向的信息组织
       
    2. 垂直分支处理：使用独立的卷积分支处理重排后的特征
       - conv1和conv2针对不同的垂直像素模式进行学习
       - 每个分支捕获特定的垂直方向特征模式
       - 实现垂直方向的并行特征增强
       
    3. 垂直重构：将处理后的特征重新组装为原始空间结构
       - 按照垂直重排的逆过程恢复空间布局
       - 保证垂直方向特征增强的有效性
       - 维持与输入相同的空间维度
    
    【设计优势】：
    - 垂直特化：专门针对垂直方向的特征建模和增强
    - 互补性设计：与FocusH形成水平-垂直的完整方向覆盖
    - 对称性架构：与FocusH保持一致的设计哲学，便于理解和维护
    - 平衡性处理：确保水平和垂直方向得到均衡的特征增强
    """

    def __init__(self, c1, c2, kernel=3, stride=1):
        super().__init__()
        self.c2 = c2
        self.conv1 = Conv(c1, c2, kernel, stride)
        self.conv2 = Conv(c1, c2, kernel, stride)

    def forward(self, x):
        b, _, h, w = x.shape
        result = torch.zeros(size=[b, self.c2, h, w], device=x.device, dtype=x.dtype)
        x1 = torch.zeros(size=[b, self.c2, h // 2, w], device=x.device, dtype=x.dtype)
        x2 = torch.zeros(size=[b, self.c2, h // 2, w], device=x.device, dtype=x.dtype)

        x1[..., ::2], x1[..., 1::2] = x[..., ::2, ::2], x[..., 1::2, 1::2]
        x2[..., ::2], x2[..., 1::2] = x[..., 1::2, ::2], x[..., ::2, 1::2]

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        result[..., ::2, ::2] = x1[..., ::2]
        result[..., 1::2, 1::2] = x1[..., 1::2]
        result[..., 1::2, ::2] = x2[..., ::2]
        result[..., ::2, 1::2] = x2[..., 1::2]

        return result


class DepthWiseConv(nn.Module):
    """Depthwise Separable Convolution - 深度可分离卷积融合组件
    
    【核心创新】：
    DepthWiseConv实现了高效的深度可分离卷积融合策略，专门为BiFocus的多特征融合而设计。
    通过将传统卷积分解为深度卷积和逐点卷积两个步骤，在保证融合效果的同时大幅降低了计算复杂度。
    
    【解决的问题】：
    1. 多特征融合计算开销大：BiFocus产生3倍通道的特征需要高效融合方法
    2. 参数量爆炸：直接使用标准卷积会导致参数量急剧增加
    3. 空间-通道耦合处理：传统卷积同时处理空间和通道维度，缺乏针对性
    
    【工作机制】：
    1. 深度卷积阶段(Depthwise Convolution)：
       - 在每个通道独立进行空间卷积操作
       - groups参数设为输入通道数，实现通道间的完全分离
       - 专门负责空间维度的特征建模，不涉及通道间信息交换
       - 大幅减少参数量：从C_in × C_out × K × K减少到C_in × K × K
       
    2. 逐点卷积阶段(Pointwise Convolution)：  
       - 使用1×1卷积核进行跨通道信息融合
       - 专门负责通道维度的特征整合和维度调整
       - 将3倍通道数的特征融合为目标通道数
       - 实现高效的通道间信息交换
       
    3. 两阶段协同：
       - 深度卷积处理空间相关性，逐点卷积处理通道相关性
       - 分解的处理方式提高了模型的表达效率
       - 在BiFocus中起到关键的特征整合作用
    
    【设计优势】：
    - 计算高效：参数量和计算量相比标准卷积大幅降低
    - 功能专一：深度和逐点卷积分别专注于空间和通道处理
    - 融合效果好：两阶段设计能够充分融合BiFocus的多路特征
    - 可扩展性强：可以轻松适应不同的输入输出通道配置
    """

    def __init__(self, in_channel, out_channel, kernel):
        super().__init__()
        self.depth_conv = Conv(in_channel, in_channel, kernel, 1, 1, in_channel)
        self.point_conv = Conv(in_channel, out_channel, 1, 1, 0, 1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out
