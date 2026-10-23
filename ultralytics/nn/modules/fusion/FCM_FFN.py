import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_  # FCM 初始化依赖

"""
    论文地址：https://ieeexplore.ieee.org/abstract/document/10786275
    论文题目：CFFormer: A Cross-Fusion Transformer Framework for the Semantic Segmentation of Multisource Remote Sensing Images （TGRS 2025）
    中文题目：CFFormer：一种用于多源遥感图像语义分割的交叉融合Transformer框架（IEEE TGRS 2025）
    多源特征融合模块（Feature Fusion Module，FFM）：
        实际意义：①跨模态信息的全局交互不足：多模态遥感图像（如光学与 SAR/DSM）的互补信息需要通过全局建模。传统方法（简单相加或拼接）仅能实现局部或浅层的特征交互，无法捕捉不同模态间的长距离依赖关系。
                ②特征冗余与噪声干扰问题：多模态数据可能存在特征冗余（如重复的背景信息）或因传感器差异的噪声，直接融合会导致模型性能下降。
        实现方式：多头交叉注意力机制 + 特征增强与融合
"""

# 交叉注意力模块（核心特征交互组件）
class CrossAttention(nn.Module):
    """Cross Attention - 交叉注意力机制
    
    【核心创新】：
    CrossAttention实现了双向交叉注意力机制，是CFFormer框架中特征交互的核心组件。
    不同于传统的自注意力机制，该模块专门设计用于两个不同模态特征间的交叉交互，
    通过"你关注我，我关注你"的策略实现互补信息的充分挖掘。
    
    【解决的问题】：
    1. 跨模态信息交互不充分：传统融合方法难以建立模态间的长距离依赖关系
    2. 计算复杂度过高：标准注意力机制在高分辨率特征图上计算开销巨大
    3. 特征维度不匹配：不同模态的特征可能具有不同的表示空间
    
    【工作机制】：
    1. 双向查询设计：为两个输入模态分别构建独立的查询(Q)矩阵
       - q1：第一个模态的查询向量，用于关注第二个模态
       - q2：第二个模态的查询向量，用于关注第一个模态
       - 实现真正的双向交互，而非单向注意力
       
    2. 交叉键值对构建：每个模态为对方提供键值对(K,V)
       - kv1：第一个模态为第二个模态提供的键值对
       - kv2：第二个模态为第一个模态提供的键值对
       - 确保交叉注意力的对称性和互补性
       
    3. 空间缩减策略：通过sr_ratio参数控制计算复杂度
       - 当sr_ratio>1时，使用深度可分离卷积对K,V进行空间下采样
       - 大幅降低注意力计算的复杂度，从O(N²)降低到O(N×N')
       - N'=N/sr_ratio²，有效平衡精度与效率
       
    4. 多头并行处理：将注意力分解为多个头进行并行计算
       - 每个头关注不同的表示子空间
       - 增强模型的表达能力和特征多样性
       - 通过头数分解降低单头的计算负担
    
    【设计优势】：
    - 交叉对称性：双向交互设计确保两个模态都能从对方获得增益
    - 计算可控：通过空间缩减有效控制计算复杂度
    - 表示丰富：多头设计增强特征表示的多样性
    - 长距离建模：全局注意力机制能够建立长距离特征依赖
    """
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=False, qk_scale=None):
        """
        :param dim: 输入特征的维度（通道数）
        :param num_heads: 多头注意力的头数
        :param sr_ratio: 空间缩减比例（用于降低计算量）
        :param qkv_bias: Q/K/V线性层是否使用偏置
        :param qk_scale: QK缩放因子（默认使用头维度的平方根倒数）
        """
        super(CrossAttention, self).__init__()  # 调用父类初始化

        # 维度必须能被头数整除（多头注意力的基本要求）
        assert dim % num_heads == 0, f"dim {dim} 必须能被头数 {num_heads} 整除"

        self.dim = dim  # 保存输入维度
        self.num_heads = num_heads  # 保存头数
        head_dim = dim // num_heads  # 每个头的维度（总维度/头数）
        self.scale = qk_scale or head_dim ** -0.5  # 注意力缩放因子（默认√(d_k)的倒数）

        # 定义Q/K/V线性层（注意这里的交叉注意力设计：两组Q/KV）
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)  # 第一组查询Q的线性层
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 第一组K/V的线性层（合并输出）

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)  # 第二组查询Q的线性层
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 第二组K/V的线性层（合并输出）

        self.sr_ratio = sr_ratio  # 保存空间缩减比例

        # 当sr_ratio>1时，添加空间缩减模块（降低空间分辨率减少计算量）
        if sr_ratio > 1:
            # 第一组特征的空间缩减卷积（深度可分离卷积）
            self.sr1 = nn.Conv2d(
                dim, dim,
                kernel_size=sr_ratio + 1,  # 卷积核大小（比步长多1）
                stride=sr_ratio,  # 步长等于sr_ratio（下采样）
                padding=sr_ratio // 2,  # 填充保持尺寸对齐
                groups=dim  # 深度可分离卷积（每组处理一个通道）
            )
            self.norm1 = nn.LayerNorm(dim)  # 层归一化

            # 第二组特征的空间缩减卷积（与第一组对称）
            self.sr2 = nn.Conv2d(
                dim, dim,
                kernel_size=sr_ratio + 1,
                stride=sr_ratio,
                padding=sr_ratio // 2,
                groups=dim
            )
            self.norm2 = nn.LayerNorm(dim)

    # 前向传播函数（核心计算逻辑）
    def forward(self, x1, x2, H, W):
        """
        :param x1: 输入特征1（形状：[B, N, C]）B-批次，N-序列长度，C-通道数
        :param x2: 输入特征2（形状同x1）
        :param H: 特征图高度（用于空间还原）
        :param W: 特征图宽度（用于空间还原）
        """
        B, N, C = x1.shape  # 获取输入张量形状（B-批次，N-序列长度，C-通道数）

        # 计算查询Q（两组特征分别计算）
        # 形状变换：[B, N, C] -> [B, N, num_heads, C//num_heads] -> [B, num_heads, N, head_dim]
        q1 = self.q1(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = self.q2(x2).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # 处理K/V的空间缩减（当sr_ratio>1时）
        if self.sr_ratio > 1:
            # 特征1的空间缩减处理
            x_1 = x1.permute(0, 2, 1).reshape(B, C, H, W)  # 序列转特征图：[B, N, C] -> [B, C, H, W]
            x_1 = self.sr1(x_1)  # 空间缩减卷积（下采样）
            x_1 = x_1.reshape(B, C, -1).permute(0, 2, 1)  # 特征图转序列：[B, C, H', W'] -> [B, N', C]
            x_1 = self.norm1(x_1)  # 层归一化
            # 计算K1/V1（合并输出后拆分）
            # 形状变换：[B, N', 2*C] -> [B, N', 2, num_heads, head_dim] -> [2, B, num_heads, N', head_dim]
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            # 特征2的空间缩减处理（与特征1对称）
            x_2 = x2.permute(0, 2, 1).reshape(B, C, H, W)
            x_2 = self.sr2(x_2)
            x_2 = x_2.reshape(B, C, -1).permute(0, 2, 1)
            x_2 = self.norm2(x_2)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            # 不做空间缩减时直接计算K/V
            kv1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # 拆分K和V（kv的第0维度是[K, V]）
        k1, v1 = kv1[0], kv1[1]  # K1形状：[B, num_heads, N', head_dim]，V1同
        k2, v2 = kv2[0], kv2[1]  # K2形状：[B, num_heads, N', head_dim]，V2同

        # 计算交叉注意力（q1关注k2，q2关注k1）
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale  # Q1*K2^T 并缩放（形状：[B, num_heads, N, N']）
        attn1 = attn1.softmax(dim=-1)  # 对最后一维做softmax得到注意力分数

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale  # Q2*K1^T 并缩放（形状：[B, num_heads, N, N']）
        attn2 = attn2.softmax(dim=-1)  # 注意力分数归一化

        # 应用注意力到V并恢复形状
        # 形状变换：[B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim] -> [B, N, C]
        main_out = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)  # 主输出（q1关注v2）
        aux_out = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)  # 辅助输出（q2关注v1）

        return main_out, aux_out  # 返回两组交互后的特征


# 特征交互模块（整合交叉注意力和通道变换）
class FeatureInteraction(nn.Module):
    """Feature Interaction - 特征交互模块
    
    【核心创新】：
    FeatureInteraction是CFFormer的核心特征交互引擎，将交叉注意力与通道变换巧妙结合，
    实现了高效的跨模态特征增强。通过"分而治之"的设计哲学，将特征分为直通和交互两个分支，
    既保留了原始信息又增强了跨模态交互能力。
    
    【解决的问题】：
    1. 特征交互与信息保留的平衡：如何在增强交互的同时避免原始信息的丢失
    2. 计算资源的合理分配：避免所有特征都参与昂贵的注意力计算
    3. 梯度流的优化：确保交互后的特征具有良好的梯度传播特性
    
    【工作机制】：
    1. 通道投影与分割：智能的特征分配策略
       - 将输入特征投影到缩减维度的2倍空间
       - 使用chunk操作将特征分为两个等大的部分：y和z
       - y部分：直通路径，保留原始特征信息
       - z部分：交互路径，用于跨模态注意力计算
       
    2. 分支处理策略：并行的直通与交互机制
       - 直通分支(y)：保持原始特征的完整性，提供稳定的基础表示
       - 交互分支(z)：通过CrossAttention进行跨模态信息交换
       - 两个分支独立工作，最后在通道维度融合
       
    3. 交叉注意力增强：专门的跨模态信息交换
       - z1和z2通过CrossAttention进行双向交互
       - 获得c1和c2两个交互增强的特征表示
       - 交互过程中保持特征的空间结构不变
       
    4. 特征重组与投影：信息整合与维度恢复
       - 将直通分支y与交互分支c在通道维度拼接
       - 通过end_proj进行维度恢复和特征整合
       - 残差连接确保梯度流和信息保留
       - LayerNorm进行特征标准化
    
    【设计优势】：
    - 信息保留：直通分支确保原始信息不被破坏
    - 交互增强：交叉注意力有效捕获跨模态依赖
    - 计算高效：只有部分特征参与注意力计算
    - 梯度友好：残差连接和标准化确保稳定训练
    """
    def __init__(self, dim, reduction=1, num_heads=None, sr_ratio=None, norm_layer=nn.LayerNorm):
        """
        :param dim: 输入特征维度
        :param reduction: 通道缩减比例（用于降低计算量）
        :param num_heads: 交叉注意力头数
        :param sr_ratio: 空间缩减比例
        :param norm_layer: 归一化层类型
        """
        super().__init__()

        # 通道投影层（将特征投影到缩减后的维度）
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)  # 特征1的通道投影（输出2倍缩减维度）
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)  # 特征2的通道投影（对称设计）

        # 激活函数
        self.act1 = nn.ReLU(inplace=True)  # 特征1的激活函数（inplace节省内存）
        self.act2 = nn.ReLU(inplace=True)  # 特征2的激活函数

        # 交叉注意力模块（输入维度为缩减后的维度）
        self.cross_attn = CrossAttention(
            dim // reduction,
            num_heads=num_heads,
            sr_ratio=sr_ratio
        )

        # 最终投影层（恢复原始维度）
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)  # 特征1的最终投影
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)  # 特征2的最终投影

        # 归一化层（用于残差连接后）
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2, H, W):
        # 通道投影并激活（分割为两部分：y用于直接连接，z用于交叉注意力）
        # chunk(2, dim=-1)：将最后一维分成两部分（形状：[B, N, C//reduction] * 2）
        y1, z1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)  # 特征1的投影和分割
        y2, z2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)  # 特征2的投影和分割

        # 交叉注意力交互（z1和z2作为输入）
        c1, c2 = self.cross_attn(z1, z2, H, W)  # c1: z1与z2交互结果，c2: z2与z1交互结果

        # 拼接交互结果（y保留原始信息，c添加交互信息）
        y1 = torch.cat((y1, c1), dim=-1)  # 特征1的信息拼接（形状：[B, N, 2*(C//reduction)]）
        y2 = torch.cat((y2, c2), dim=-1)  # 特征2的信息拼接

        # 最终投影并残差连接（输入特征x与投影后的y相加）
        main_out = self.norm1(x1 + self.end_proj1(y1))  # 主输出（特征1增强）
        aux_out = self.norm2(x2 + self.end_proj2(y2))  # 辅助输出（特征2增强）

        return main_out, aux_out  # 返回增强后的两组特征

# 通道嵌入模块（调整特征维度并融合）
class ChannelEmbed(nn.Module):
    """Channel Embed - 通道嵌入模块
    
    【核心创新】：
    ChannelEmbed实现了高效的特征维度调整和融合策略，是CFFormer中连接序列特征与卷积特征的桥梁。
    通过残差路径与深度嵌入路径的并行设计，既保证了信息的完整传递，又实现了特征的深度变换。
    
    【解决的问题】：
    1. 序列到卷积的转换：Transformer输出的序列特征需要转换为卷积网络可用的2D特征图
    2. 维度不匹配问题：交叉注意力输出的拼接特征(2C)需要映射到目标维度(C)
    3. 信息损失风险：简单的维度变换可能导致重要信息的丢失
    
    【工作机制】：
    1. 双路径架构：残差路径与嵌入路径的并行处理
       - 残差路径：通过1x1卷积直接映射，保证信息的快速传递
       - 嵌入路径：通过多级卷积进行深度特征变换
       - 两路径最终相加，实现信息的充分利用
       
    2. 序列到特征图的转换：灵活的维度重组
       - 接收[B,N,2C]的序列特征作为输入
       - 通过permute和reshape操作转换为[B,2C,H,W]的特征图
       - 为后续的卷积操作提供标准的输入格式
       
    3. 深度可分离嵌入：高效的特征变换策略
       - 第一步：1x1卷积进行通道数降维(2C→C/r)
       - 第二步：3x3深度可分离卷积增强空间特征表达
       - 第三步：1x1卷积进行通道数恢复(C/r→C)
       - 倒残差结构确保特征表达能力的最大化
       
    4. 特征融合与标准化：稳定的输出保证
       - 残差路径与嵌入路径的特征相加融合
       - 通过BatchNorm进行特征标准化
       - 确保输出特征的分布稳定性
    
    【设计优势】：
    - 信息保持：残差连接确保原始信息不丢失
    - 特征增强：深度嵌入路径提供丰富的特征变换
    - 计算高效：深度可分离卷积控制参数量和计算量
    - 结构灵活：支持任意的输入输出通道数配置
    """
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param reduction: 通道缩减比例
        :param norm_layer: 归一化层类型（默认批量归一化）
        """
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels  # 保存输出通道数

        # 残差连接（1x1卷积调整通道）
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # 通道嵌入序列（多级卷积调整特征）
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),  # 1x1卷积降维
            # 深度可分离卷积（保持通道数不变，增强空间特征）
            nn.Conv2d(
                out_channels // reduction,
                out_channels // reduction,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=out_channels // reduction  # 分组卷积=通道数（深度可分离）
            ),
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),  # 1x1卷积升维
            norm_layer(out_channels)  # 归一化层
        )
        self.norm = norm_layer(out_channels)  # 最终归一化层（用于残差和）

    def forward(self, x, H, W):
        """
        :param x: 输入特征（形状：[B, N, C]）
        :param H: 特征图高度
        :param W: 特征图宽度
        """
        B, N, _C = x.shape  # 获取输入形状（B-批次，N-序列长度，C-通道数）

        # 序列转特征图（用于卷积操作）
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()  # 形状：[B, C, H, W]

        # 残差路径（直接1x1卷积）
        residual = self.residual(x)  # 形状：[B, out_channels, H, W]

        # 通道嵌入路径（多级卷积）
        x = self.channel_embed(x)  # 形状：[B, out_channels, H, W]

        # 残差相加并归一化
        out = self.norm(residual + x)  # 形状：[B, out_channels, H, W]

        return out  # 返回融合后的特征

# 特征融合模块（完整流程整合）
class FeatureFusion(nn.Module):
    """Feature Fusion - 特征融合模块
    
    【核心创新】：
    FeatureFusion是CFFormer的顶层特征融合组件，将交叉注意力交互与通道嵌入完整整合，
    实现了从双模态输入到单模态融合输出的端到端处理。该模块体现了"交互-融合-输出"的
    完整特征处理流程，是多模态遥感图像处理的核心引擎。
    
    【解决的问题】：
    1. 端到端融合流程：需要一个完整的管道将多模态特征处理为统一表示
    2. 特征维度的灵活适配：支持不同输入输出通道配置的通用性需求
    3. 计算与精度的平衡：在保证融合效果的前提下控制计算复杂度
    
    【工作机制】：
    1. 输入预处理：二维特征图到序列的转换
       - 将输入的[B,C,H,W]特征图展平为[B,H×W,C]的序列
       - 通过flatten(2)和transpose(1,2)实现维度重组
       - 为后续的Transformer类操作准备标准输入格式
       
    2. 交叉交互处理：基于FeatureInteraction的双向增强
       - 调用FeatureInteraction模块进行跨模态特征交互
       - 两个模态的特征经过交叉注意力机制相互增强
       - 输出增强后的双路特征，保持序列格式
       
    3. 特征拼接融合：多模态信息的整合
       - 将交互增强后的两路特征在通道维度拼接
       - 形成[B,H×W,2C]的融合特征表示
       - 包含了两个模态的互补信息和交互增强信息
       
    4. 通道嵌入映射：序列到特征图的最终转换
       - 通过ChannelEmbed模块将融合序列转换为2D特征图
       - 同时完成维度调整：从2C映射到C
       - 输出标准的[B,C,H,W]特征图供后续网络使用
    
    【设计优势】：
    - 端到端处理：提供完整的多模态融合解决方案
    - 模块化设计：内部组件可独立优化和替换
    - 通用适配：支持不同的输入输出通道配置
    - 性能可控：通过reduction和sr_ratio参数灵活调节计算量
    """
    def __init__(self, dim, reduction=1, sr_ratio=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        """
        :param dim: 输入特征维度
        :param reduction: 通道缩减比例
        :param sr_ratio: 空间缩减比例
        :param num_heads: 交叉注意力头数
        :param norm_layer: 归一化层类型
        """
        super().__init__()

        # 交叉交互模块（特征交互的核心）
        self.cross = FeatureInteraction(
            dim=dim,
            reduction=reduction,
            num_heads=num_heads,
            sr_ratio=sr_ratio
        )

        # 通道嵌入模块（融合后调整维度）
        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2,  # 输入是两组特征的拼接（维度2*dim）
            out_channels=dim,  # 输出维度恢复为dim
            reduction=reduction,
            norm_layer=norm_layer
        )

        # 初始化权重（调用自定义初始化函数）
        self.apply(self._init_weights)

    # 权重初始化函数（遵循常见的深度学习初始化策略）
    @classmethod
    def _init_weights(cls, m):
        """
        :param m: 网络模块
        """
        if isinstance(m, nn.Linear):
            # 截断正态分布初始化（防止梯度消失/爆炸）
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # 偏置初始化为0
            nn.init.constant_(m.weight, 1.0)  # 权重初始化为1（单位缩放）
        elif isinstance(m, nn.Conv2d):
            # 卷积核初始化（根据扇出计算方差）
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups  # 分组卷积时调整扇出
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))  # 正态分布初始化
            if m.bias is not None:
                m.bias.data.zero_()  # 偏置初始化为0

    def forward(self, x1, x2=None):
        # 支持 parse_model 传入的列表输入形式 [x1, x2]
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1
        """
        :param x1: 输入特征1（形状：[B, C, H, W]）
        :param x2: 输入特征2（形状同x1）
        """
        B, C, H, W = x1.shape  # 获取输入形状（B-批次，C-通道数，H-高度，W-宽度）

        # 特征展平（二维特征转序列，用于Transformer类操作）
        # flatten(2): 将H和W维度展平为一维（形状：[B, C, H*W]）
        # transpose(1, 2): 交换通道和序列维度（形状：[B, H*W, C]）
        x1 = x1.flatten(2).transpose(1, 2)  # 特征1展平为序列
        x2 = x2.flatten(2).transpose(1, 2)  # 特征2展平为序列

        # 交叉交互（输出增强后的两组特征）
        x1, x2 = self.cross(x1, x2, H, W)  # 形状：[B, H*W, C]（每组特征）

        # 特征拼接（合并两组增强后的特征）
        fuse = torch.cat((x1, x2), dim=-1)  # 形状：[B, H*W, 2*C]

        # 通道嵌入（调整维度并融合）
        fuse = self.channel_emb(fuse, H, W)  # 形状：[B, C, H, W]（恢复二维特征图）

        return fuse  # 返回最终融合后的特征

if __name__ == "__main__":
    x1 = torch.randn(1, 32, 50, 50)  # 形状：[B=1, C=32, H=50, W=50]
    x2 = torch.randn(1, 32, 50, 50)  # 形状：[B=1, C=32, H=50, W=50]
    fusion_module = FeatureFusion(dim=32,  reduction=1,  sr_ratio=4, num_heads=8)
    output = fusion_module(x1, x2)
    print(f"输入张量1形状: {x1.shape}")
    print(f"输入张量2形状: {x2.shape}")
    print(f"输出张量形状: {output.shape}")

# ------------------------------
# FCM: Feature Correction Module
# ------------------------------

class ChannelWeights(nn.Module):
    """Channel Weights - 通道权重生成模块
    
    【核心创新】：
    ChannelWeights实现了基于多统计特征的通道重要性建模，是FCM特征校正机制的重要组成部分。
    通过整合平均值、标准差、最大值三种统计特性，全面捕获通道级的特征分布特征，
    为跨模态通道校正提供精准的权重指导。
    
    【解决的问题】：
    1. 通道重要性评估不全面：单一统计量无法完整描述通道特征的重要性
    2. 跨模态通道校正缺乏指导：需要准确的权重来指导通道级的特征校正
    3. 特征分布信息利用不足：通道的统计特性包含丰富的语义信息
    
    【工作机制】：
    1. 多维统计特征提取：全面的通道特征描述
       - 平均池化(avg_pool)：提取通道的均值信息，反映通道的整体激活水平
       - 最大池化(max_pool)：提取通道的峰值信息，反映通道的最强响应
       - 标准差计算(std)：提取通道的变异性信息，反映特征分布的离散程度
       
    2. 特征融合与扩展：多模态统计信息整合
       - 将两个模态的特征拼接后统一计算统计量
       - 三种统计特征的拼接形成6倍通道的丰富描述符
       - 为每种统计特征提供独立的表示空间
       
    3. 非线性映射与权重生成：智能的重要性建模
       - 通过两层MLP进行非线性特征变换
       - 第一层：特征压缩和模式识别
       - 第二层：权重生成和激活控制
       - Sigmoid激活确保权重在[0,1]范围内
       
    4. 权重重构与分配：双模态权重的组织
       - 将生成的权重重构为[2, B, C, 1, 1]的形状
       - 第一维对应两个模态的权重
       - 为每个模态的每个通道提供独立的校正权重
    
    【设计优势】：
    - 统计全面：三种统计量提供完整的通道特征描述
    - 模态协同：统一处理两个模态的统计信息
    - 权重精准：非线性映射提供精确的重要性评估
    - 结构灵活：支持任意通道数的权重生成
    """
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim * 6 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 6 // reduction, self.dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        std = torch.std(x, dim=(2, 3), keepdim=True).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, std, max), dim=1)
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights


class SpatialWeights(nn.Module):
    """Spatial Weights - 空间权重生成模块
    
    【核心创新】：
    SpatialWeights实现了精细化的空间位置重要性建模，专门用于FCM的空间维度特征校正。
    与ChannelWeights形成互补，该模块关注"哪个位置重要"的问题，为跨模态空间校正
    提供位置级的精确权重指导。
    
    【解决的问题】：
    1. 空间重要性建模缺失：传统方法忽略了不同空间位置的重要性差异
    2. 跨模态空间不一致性：不同模态在空间位置上可能存在语义不对齐
    3. 局部空间信息利用不足：需要充分利用空间邻域的上下文信息
    
    【工作机制】：
    1. 多模态空间特征整合：统一的空间上下文建模
       - 将两个模态的特征在通道维度拼接
       - 形成包含双模态空间信息的联合表示
       - 为统一的空间权重生成提供全面的上下文
       
    2. 卷积空间权重生成：基于CNN的空间建模
       - 第一层1x1卷积：特征降维和初步空间特征提取
       - ReLU激活：引入非线性，增强表达能力  
       - 第二层1x1卷积：生成双模态的空间权重图
       - Sigmoid激活：确保权重在[0,1]范围内
       
    3. 权重重构与分配：双模态空间权重的组织
       - 将输出重构为[2, B, 1, H, W]的形状
       - 第一维对应两个模态的空间权重
       - 每个空间位置都有独立的校正权重
       - 保持原始特征的空间分辨率
       
    4. 空间语义保持：位置级精细化处理
       - 空间权重图保持与输入相同的空间尺寸
       - 每个像素位置都有对应的重要性权重
       - 支持精细化的空间级特征校正
    
    【设计优势】：
    - 空间精细：提供像素级的空间重要性建模
    - 上下文感知：利用卷积的局部感受野捕获空间上下文
    - 计算高效：基于卷积的实现计算量可控
    - 分辨率保持：输出权重图与输入特征空间尺寸一致
    """
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x)
        spatial_weights = spatial_weights.reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights


class FCM(nn.Module):
    """FCM: Feature Correction Module - 特征校正模块
    
    【核心创新】：
    FCM是CFFormer的关键创新组件，实现了两阶段的跨模态特征校正机制。
    通过"空间校正→通道校正"的递进式策略，系统性地解决多模态特征融合中的
    特征不一致和信息冗余问题，为高质量的特征融合奠定基础。
    
    【解决的问题】：
    1. 多模态特征不一致性：不同传感器获取的特征存在空间和通道维度的不对齐
    2. 特征冗余与噪声：多模态数据中存在重复信息和传感器特有的噪声干扰
    3. 融合权重学习困难：传统方法难以学习到最优的多模态融合权重
    4. 特征校正缺乏系统性：现有方法缺乏针对性的特征校正机制
    
    【工作机制】：
    1. 自适应融合权重学习：可学习的模态重要性权重
       - weights参数：可学习的双模态权重向量
       - ReLU激活确保权重非负性
       - 归一化处理确保权重和为1，避免梯度不稳定
       - 为两阶段校正提供全局的模态平衡控制
       
    2. 第一阶段-空间校正：位置级的跨模态特征校正
       - 通过SpatialWeights生成精细化的空间权重图
       - 实现跨模态空间校正：x1用x2的空间权重校正，x2用x1的空间权重校正
       - 利用融合权重fuse_weights[0]控制空间校正的强度
       - 输出空间校正后的特征x1_1和x2_1
       
    3. 第二阶段-通道校正：通道级的跨模态特征校正  
       - 基于空间校正后的特征计算通道权重
       - 通过ChannelWeights生成通道级的重要性权重
       - 实现跨模态通道校正：x1_1用x2_1的通道权重校正，x2_1用x1_1的通道权重校正
       - 利用融合权重fuse_weights[1]控制通道校正的强度
       
    4. 输出双路增强特征：保持模态独立性的同时实现互补增强
       - main_out：第一个模态的最终校正结果
       - aux_out：第二个模态的最终校正结果
       - 两个输出都经过了双阶段的跨模态校正增强
    
    【设计优势】：
    - 两阶段递进：空间校正为通道校正提供更好的基础
    - 跨模态互补：每个模态都能从另一个模态获得校正指导
    - 权重自适应：可学习的融合权重实现最优的校正强度控制
    - 特征增强：输出的双路特征都得到了有效的质量提升
    """
    def __init__(self, dim, reduction=1, eps=1e-8):
        super(FCM, self).__init__()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2=None):
        # 支持 parse_model 传入的列表输入形式 [x1, x2]
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        spatial_weights = self.spatial_weights(x1, x2)
        x1_1 = x1 + fuse_weights[0] * spatial_weights[1] * x2
        x2_1 = x2 + fuse_weights[0] * spatial_weights[0] * x1
        channel_weights = self.channel_weights(x1_1, x2_1)
        main_out = x1_1 + fuse_weights[1] * channel_weights[1] * x2_1
        aux_out = x2_1 + fuse_weights[1] * channel_weights[0] * x1_1
        return main_out, aux_out


if __name__ == "__main__":
    # FCM quick test
    x1 = torch.randn(1, 32, 50, 50)
    x2 = torch.randn(1, 32, 50, 50)
    fcm = FCM(dim=32)
    main_out, aux_out = fcm(x1, x2)
    print(f"FCM 输出张量1形状: {main_out.shape}")
    print(f"FCM 输出张量2形状: {aux_out.shape}")


# ==============================================================
# 二创模块：FCM + FeatureFusion 串联封装（无额外改动，顺序链接）
# 说明：
# - 输入：两路特征 [B, C, H, W]（通道一致）
# - 流程：先经 FCM 得到 (main, aux)，再送入 FeatureFusion 得到单路融合特征
# - 输出：单路 [B, C, H, W]
# - 目的：提供端到端的“纠偏→融合”便捷模块，便于在代码中直接调用
# 注意：本模块不修改 FCM/FeatureFusion 内部逻辑，仅做顺序封装
# ==============================================================
class FCMFeatureFusion(nn.Module):
    """顺序串联 FCM 与 FeatureFusion 的便捷块。

    参数：
        dim (int | None): 通道数；为 None 时在首次 forward 时按输入通道自动推断
        reduction (int): 通道压缩比，传入 FCM 与 FeatureFusion
        sr_ratio (int): FeatureFusion 内交叉注意力的空间缩减比
        num_heads (int | None): FeatureFusion 交叉注意力头数
        norm_layer (nn.Module): FeatureFusion 通道嵌入使用的归一化层（默认 BatchNorm2d）
        detach_fcm (bool): 若为 True，则在进入 FeatureFusion 前对 FCM 输出断开梯度（用于消融）
        return_pair (bool): 若为 True，则返回 (fused, (main, aux)) 便于调试观察
    """

    def __init__(
        self,
        dim: int | None = None,
        reduction: int = 1,
        sr_ratio: int = 1,
        num_heads: int | None = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
        detach_fcm: bool = False,
        return_pair: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.sr_ratio = sr_ratio
        self.num_heads = num_heads
        self.norm_layer = norm_layer
        self.detach_fcm = detach_fcm
        self.return_pair = return_pair

        # 延迟构建：允许 dim=None，首次 forward 时根据输入通道数构建子模块
        self.fcm: FCM | None = None
        self.ffm: FeatureFusion | None = None

    def _build_if_needed(self, c: int) -> None:
        if self.fcm is None or self.ffm is None:
            dim = c if self.dim is None else self.dim
            if dim != c:
                raise AssertionError(f"FCMFeatureFusion: dim={self.dim} 与输入通道 {c} 不一致")
            # 直接使用当前文件中定义的 FCM 与 FeatureFusion
            self.fcm = FCM(dim=dim, reduction=self.reduction)
            self.ffm = FeatureFusion(
                dim=dim,
                reduction=self.reduction,
                sr_ratio=self.sr_ratio,
                num_heads=self.num_heads,
                norm_layer=self.norm_layer,
            )

    def forward(self, x1, x2=None):
        # 兼容 parse_model 风格的列表/元组输入 [x1, x2]
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1

        if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
            raise TypeError("FCMFeatureFusion 需要两路输入张量")
        if x1.shape != x2.shape:
            raise ValueError(f"两路输入形状需一致，got {x1.shape} vs {x2.shape}")

        _, c, _, _ = x1.shape
        self._build_if_needed(c)

        # FCM 纠偏
        main, aux = self.fcm(x1, x2)
        if self.detach_fcm:
            main, aux = main.detach(), aux.detach()

        # FeatureFusion 融合
        fused = self.ffm(main, aux)
        return (fused, (main, aux)) if self.return_pair else fused


# ==============================================================
# 二创模块：ConvFFN-GLU（可替换 ChannelEmbed 的卷积版 FFN）
# 说明：
# - I/O 与 ChannelEmbed 保持一致：forward(x: [B, N, 2C], H, W) → [B, C, H, W]
# - 结构：1x1降维到 rC → DWConv → GLU 门控（SwiGLU/GEGLU）→ 1x1升维 → 残差(1x1) → Norm
# - 目标：以 GLU 提升非线性和梯度流，DWConv 维持局部混合，成本可控
# ==============================================================
class ConvFFN_GLU(nn.Module):
    """基于卷积与 GLU 门控的 FFN 变体。

    参数:
        in_channels (int): 输入通道，一般为 2C
        out_channels (int): 输出通道，一般为 C
        expand (int): 扩展比，隐层通道 rC = expand * out_channels
        dwk (int): 深度可分离卷积核大小（3/5）
        act (str): 门控激活，'swiglu' | 'geglu' | 'sigmoid'
        norm_layer (nn.Module): 归一化层，默认 BatchNorm2d
        share_dw (bool): 是否对 gate 分支也应用同一个 DWConv
        drop_path (float): 随机深度概率，0 表示不启用
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand: int = 4,
        dwk: int = 3,
        act: str = 'swiglu',
        norm_layer: nn.Module = nn.BatchNorm2d,
        share_dw: bool = False,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        assert dwk in (3, 5, 7), f"dwk {dwk} 不支持，推荐 3/5/7"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = max(out_channels * expand, 1)
        self.act = act.lower()
        self.share_dw = share_dw

        # 主路径: 1x1 降维（两条）+ DWConv（可共享到 gate）+ GLU 门控 + 1x1 升维
        self.proj_u = nn.Conv2d(in_channels, self.hidden, kernel_size=1, bias=True)
        self.proj_g = nn.Conv2d(in_channels, self.hidden, kernel_size=1, bias=True)
        self.dw = nn.Conv2d(
            self.hidden, self.hidden, kernel_size=dwk, stride=1, padding=dwk // 2, groups=self.hidden, bias=True
        )
        self.proj_out = nn.Conv2d(self.hidden, out_channels, kernel_size=1, bias=True)

        # 残差路径: 1x1 投影到 out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = norm_layer(out_channels)
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0.0 else nn.Identity()

        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _gate(self, g: torch.Tensor) -> torch.Tensor:
        if self.act == 'swiglu':
            return F.silu(g)
        if self.act == 'geglu':
            return F.gelu(g)
        if self.act == 'sigmoid':
            return torch.sigmoid(g)
        # 默认退化为 SiLU
        return F.silu(g)

    def forward(self, x, H: int | None = None, W: int | None = None):
        """支持两种输入：
        - x: [B, N, 2C] 且提供 H, W
        - x: [B, 2C, H, W]（无需提供 H, W）
        输出: [B, C, H, W]
        """
        if x.dim() == 3:  # [B, N, 2C]
            assert H is not None and W is not None, "当输入为 [B, N, 2C] 时需提供 H 与 W"
            B, N, C = x.shape
            x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        elif x.dim() == 4:
            B, C, H, W = x.shape
        else:
            raise ValueError(f"不支持的输入维度: {x.shape}")

        u = self.proj_u(x)
        g = self.proj_g(x)

        u = self.dw(u)
        if self.share_dw:
            g = self.dw(g)

        y = self._gate(g) * u  # GLU 门控
        y = self.proj_out(y)

        out = self.residual(x)
        out = self.norm(out + self.drop_path(y))
        return out


class DropPath(nn.Module):
    """Stochastic Depth per sample (当 drop_prob>0 且训练模式生效)。"""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor
