import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Depend on base conv/concat ops from core modules
from ultralytics.nn.modules.conv import Conv, Concat


class NiNfusion(nn.Module):
    """NiN Fusion - Network-in-Network风格轻量级融合模块
    
    【核心创新】：
    NiNfusion实现了基于Network-in-Network理念的轻量级多模态融合策略，采用"拼接-投影"的
    简洁设计哲学。该模块体现了ICAFusion的核心思想：通过最小的计算开销实现有效的特征融合，
    为复杂的Transformer融合机制提供高效的基础融合能力。
    
    【解决的问题】：
    1. 计算复杂度过高：复杂的融合机制带来巨大的计算开销
    2. 参数量爆炸：多模态融合容易导致参数量的急剧增加
    3. 基础融合能力缺失：需要一个简单高效的基础融合组件
    
    【工作机制】：
    1. 特征拼接：直接的多模态信息整合
       - 使用Concat操作在通道维度拼接多模态特征
       - 保持空间维度不变，仅在通道维度扩展
       - 简单直接地将多模态信息组合在一起
       
    2. 1x1卷积投影：高效的维度调整和特征整合
       - 1x1卷积实现跨通道的特征线性组合
       - 将拼接后的多通道特征映射到目标通道数
       - 相当于为每个输出通道学习一个多模态特征的加权组合
       
    3. 非线性激活：增强表达能力
       - SiLU激活函数引入非线性变换
       - 增强模型的表达能力和特征区分度
       - 为后续复杂处理提供丰富的特征表示
    
    【设计优势】：
    - 计算高效：仅包含拼接和1x1卷积，计算量极小
    - 参数可控：参数量仅与输入输出通道数相关
    - 通用适配：可适应任意数量和类型的模态输入
    - 即插即用：可作为基础组件集成到任何多模态架构中
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """ICAFusion-style lightweight fusion: concat then 1x1 projection."""
        super(NiNfusion, self).__init__()
        self.concat = Concat(dimension=1)
        self.conv = nn.Conv2d(c1, c2, k, s, padding=(k // 2 if p is None else p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.concat(x)
        y = self.act(self.conv(y))
        return y


class LearnableCoefficient(nn.Module):
    """Learnable Coefficient - 可学习系数模块
    
    【核心创新】：
    LearnableCoefficient实现了最简化的可学习权重控制机制，为ICAFusion提供精细化的特征调节能力。
    通过单一的可学习标量参数，实现对特征响应强度的自适应控制，体现了"简单有效"的设计理念。
    
    【解决的问题】：
    1. 特征响应强度固定：传统方法缺乏对特征响应强度的自适应调节
    2. 梯度流控制困难：需要可学习的机制来控制梯度的传播强度  
    3. 残差连接权重不平衡：不同分支的贡献权重需要动态平衡
    
    【工作机制】：
    1. 标量权重学习：最简化的自适应调节
       - bias参数：单一的可学习标量，初始化为1.0
       - 通过梯度反向传播自动学习最优的缩放系数
       - 避免复杂的权重设计，直接学习全局的响应强度
       
    2. 元素级缩放：统一的特征调节
       - 对输入特征的每个元素进行相同比例的缩放
       - 保持特征的相对关系不变，仅调节整体强度
       - 实现简单但有效的特征调制
    
    【设计优势】：
    - 极简设计：仅包含一个可学习参数，几乎不增加模型复杂度
    - 自适应学习：权重通过训练自动学习到最优值
    - 通用适用：可以应用于任意形状的特征张量
    - 梯度友好：简单的乘法操作保持良好的梯度传播特性
    """
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        return x * self.bias


class LearnableWeights(nn.Module):
    """Learnable Weights - 可学习权重融合模块
    
    【核心创新】：
    LearnableWeights实现了双分支特征的可学习加权融合机制，专门用于平衡不同处理分支（如avg pooling和max pooling）
    的贡献。该模块体现了自适应融合的核心理念：让模型自动学习最优的分支权重分配。
    
    【解决的问题】：
    1. 固定权重融合的局限性：传统的固定权重(0.5:0.5)无法适应不同数据的特性
    2. 分支贡献不均衡：不同的处理分支可能对最终结果有不同程度的贡献
    3. 手工设计权重的主观性：需要自动化的权重学习机制
    
    【工作机制】：
    1. 双权重参数设计：为两个分支分别设置可学习权重
       - w1参数：第一个分支的权重，初始化为0.5
       - w2参数：第二个分支的权重，初始化为0.5  
       - 通过训练自动调整两个权重的相对重要性
       
    2. 加权线性组合：简单而有效的融合策略
       - 对两个输入特征进行加权相加
       - 每个特征都有独立的学习权重进行调制
       - 最终输出为两个加权特征的线性组合
       
    3. 自适应权重学习：数据驱动的权重优化
       - 权重通过反向传播自动学习
       - 能够根据任务和数据特性自适应调整
       - 无需人工调优，实现端到端学习
    
    【设计优势】：
    - 自适应性强：权重能够根据数据特性自动调整
    - 参数精简：仅包含两个标量参数，开销极小
    - 通用适用：可用于任意两个同形状特征的融合
    - 可解释性：权重值直接反映了分支的相对重要性
    """
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        return x1 * self.w1 + x2 * self.w2


class CrossAttention(nn.Module):
    """Cross Attention - 交叉注意力机制（ICAFusion版本）
    
    【核心创新】：
    CrossAttention实现了ICAFusion的核心交叉注意力机制，专门设计用于RGB和红外两个模态间的
    深度交互。与传统自注意力不同，该模块实现了"你的查询关注我的键值"的交叉配对策略，
    充分挖掘跨模态的互补信息和长距离依赖关系。
    
    【解决的问题】：
    1. 跨模态交互深度不足：传统方法难以建立深层的跨模态特征依赖
    2. 注意力机制单模态局限：自注意力只能捕获单模态内部的关系
    3. 多模态信息整合缺失：缺乏有效机制将两个模态的注意力信息进行整合
    
    【工作机制】：
    1. 独立的QKV投影：为每个模态构建专门的查询-键-值体系
       - RGB模态：que_proj_vis, key_proj_vis, val_proj_vis
       - IR模态：que_proj_ir, key_proj_ir, val_proj_ir
       - 每个模态都有完整的QKV投影，保持模态特异性
       
    2. 多头注意力分解：增强表示能力的并行处理
       - 将d_model维度分解为h个头，每头d_k维度
       - 多头并行计算不同子空间的注意力模式
       - 增强模型对不同类型特征关系的捕获能力
       
    3. 交叉注意力计算：核心的跨模态交互机制
       - RGB查询关注IR的键值：q_vis @ k_ir → att_ir → att_ir @ v_ir
       - IR查询关注RGB的键值：q_ir @ k_vis → att_vis → att_vis @ v_vis  
       - 实现真正的跨模态信息交换和增强
       
    4. 残差连接与标准化：稳定的训练和信息保持
       - LayerNorm对输入进行标准化处理
       - Dropout进行正则化防止过拟合
       - 独立的输出投影确保维度一致性
    
    【设计优势】：
    - 交叉对称：两个模态都能从对方获得注意力增强
    - 多头并行：不同注意力头捕获多样化的交互模式
    - 深度交互：建立跨模态的长距离依赖关系
    - 训练稳定：标准化和正则化确保训练的稳定性
    """
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        """
        Cross attention between two sequences (RGB and IR), each with its own Q/K/V projections.
        """
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention (per modality LN)
        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis))
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir))

        return [out_vis, out_ir]


class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        self.mlp_ir = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

        # Learnable Coefficient
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        bs, nx, c = rgb_fea_flat.size()

        for _ in range(self.loops):
            rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

        return [rgb_fea_flat, ir_fea_flat]


class TransformerFusionBlock(nn.Module):
    """Transformer Fusion Block - Transformer融合块
    
    【核心创新】：
    TransformerFusionBlock是ICAFusion的顶层融合组件，实现了完整的"下采样-交叉注意力-上采样-融合"流程。
    该模块创新性地将自适应池化、位置编码、交叉Transformer和残差连接整合为统一框架，
    为多模态特征融合提供了端到端的解决方案。
    
    【解决的问题】：
    1. 高分辨率特征的计算复杂度：直接在原分辨率上计算注意力开销巨大
    2. 位置信息缺失：池化操作后需要重新引入位置感知能力
    3. 融合后信息丢失：需要有效机制保持原始特征信息
    4. 训练推理一致性：需要适应训练和推理阶段的不同需求
    
    【工作机制】：
    1. 自适应下采样：计算高效的特征压缩
       - AdaptivePool2d将特征下采样到固定尺寸(vert_anchors × horz_anchors)
       - LearnableWeights融合avg pooling和max pooling的优势
       - 大幅降低后续Transformer的计算复杂度
       
    2. 位置编码增强：空间感知能力恢复
       - pos_emb_vis和pos_emb_ir为两个模态提供独立的位置编码
       - 可学习的位置编码能够适应不同的空间分布模式
       - 弥补池化操作造成的位置信息损失
       
    3. 交叉Transformer处理：深度跨模态交互
       - 将2D特征重塑为序列格式供Transformer处理
       - 通过CrossTransformerBlock进行多层跨模态注意力计算
       - 实现深层的跨模态特征交互和增强
       
    4. 上采样与融合：信息整合和恢复
       - 将Transformer输出重塑回2D格式
       - 训练时使用nearest插值，推理时使用bilinear插值
       - 通过残差连接保持原始特征信息
       - 最终通过concat和1x1卷积生成融合特征
    
    【设计优势】：
    - 计算可控：通过下采样有效控制Transformer的计算量
    - 位置感知：独立的位置编码保持空间结构信息
    - 深度融合：多层交叉注意力实现深度特征交互
    - 信息保持：残差连接确保原始信息不丢失
    - 适应性强：训练和推理使用不同的插值策略优化性能
    """
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlock, self).__init__()
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        # downsampling via adaptive pooling (avg/max)
        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # Learnable weights for avg/max pooling fusion
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # cross transformer
        self.crosstransformer = nn.Sequential(
            *[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for _ in range(n_layer)]
        )

        # Concat and 1x1 projection
        self.concat = Concat(dimension=1)
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # cross-modal feature fusion at downsampled grid
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis

        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        new_rgb_fea = rgb_fea_CFE + rgb_fea

        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea

        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_fea)
        return new_fea


class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()
        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape
        if (input_h > self.output_h) or (input_w > self.output_w):
            stride_h = input_h // self.output_h
            stride_w = input_w // self.output_w
            kernel_size = (input_h - (self.output_h - 1) * stride_h,
                           input_w - (self.output_w - 1) * stride_w)
            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=kernel_size, stride=(stride_h, stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=kernel_size, stride=(stride_h, stride_w), padding=0)(x)
        else:
            y = x
        return y

