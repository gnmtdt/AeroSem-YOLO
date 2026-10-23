"""
C3k2 Extraction - Base Components
存放所有C3k2变体所需的辅助类和基础组件

包含内容：
- DBB辅助函数和变换
- 基础层（IdentityBasedConv1x1, BNAndPadLayer）
- 注意力机制（EMA, OD_Attention）
- 卷积变体（Partial_conv3, ODConv2d）
- 重参数化模块（DiverseBranchBlock系列）
- Bottleneck变体
- C3k变体
- Block变体（Faster_Block等）

迁移自 ultralytics-yolo11-main/ultralytics/nn/extra_modules/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import math
from ultralytics.nn.modules.conv import Conv
from timm.models.layers import DropPath
from timm.layers import CondConv2d
from einops import rearrange
from ultralytics.nn.public import (
    RetBlock,
    RelPos2d,
    Heat2D,
    HeatBlock,
    WTConv2d,
    DMlp,
    SMFA,
    PCFN,
    FMB,
    MutilScal,
    Mutilscal_MHSA,
    MSMHSA_CGLU,
    ElementScale,
    ChannelAggregationFFN,
    MultiOrderDWConv,
    MultiOrderGatedAggregation,
    MogaBlock,
    SHSA_GroupNorm,
    SHSABlock_FFN,
    SHSA,
    SHSABlock,
    SHSABlock_CGLU,
    EdgeEnhancer,
    MutilScaleEdgeInformationEnhance,
    MutilScaleEdgeInformationSelect,
    FourierUnit,
    Freq_Fusion,
    Fused_Fourier_Conv_Mixer,
    Modulator,
    SMA,
    E_MLP,
    SMAFormerBlock,
    SMAFormerBlock_CGLU,
    DeepPoolLayer,
    MSMBlock,
    CAB,
    HDRAB,
    RAB,
    ShiftConv2d0,
    ShiftConv2d1,
    LFE,
    InceptionDWConv2d,
    MetaNeXtBlock,
    CAMixer,
)

# 确保在定义派生类前可用的基础类
from ultralytics.nn.modules.block import Bottleneck, C3k

# 公共基础实现（单一来源）：供本文件内相关别名指向，消除重复实现
from .common_base import (
    ConvolutionalGLU as _CommonConvolutionalGLU,
    Mlp_CASVIT as _CommonMlp_CASVIT,
    SpatialOperation as _CommonSpatialOperation,
    ChannelOperation as _CommonChannelOperation,
    LocalIntegration as _CommonLocalIntegration,
    AdditiveTokenMixer as _CommonAdditiveTokenMixer,
    AdditiveBlock as _CommonAdditiveBlock,
    AdditiveBlock_CGLU as _CommonAdditiveBlock_CGLU,
)

__all__ = [
    # 辅助函数
    'autopad', 'conv_bn',
    # DBB变换函数
    'transI_fusebn', 'transII_addbranch', 'transIII_1x1_kxk',
    'transIV_depthconcat', 'transV_avg', 'transVI_multiscale',
    # 基础层
    'IdentityBasedConv1x1', 'BNAndPadLayer',
    # 注意力机制
    'EMA', 'OD_Attention',
    # 卷积变体
    'Partial_conv3', 'ODConv2d', 'fuse_conv_bn',
    # 重参数化模块
    'DiverseBranchBlock', 'DiverseBranchBlockNOAct',
    'WideDiverseBranchBlock', 'DeepDiverseBranchBlock',
    # Bottleneck变体 (Batch 1)
    'Bottleneck_PConv', 'Bottleneck_ODConv',
    'Bottleneck_DBB', 'Bottleneck_WDBB', 'Bottleneck_DeepDBB',
    # C3k变体 (Batch 1)
    'C3k_Faster', 'C3k_PConv', 'C3k_ODConv',
    'C3k_Faster_EMA', 'C3k_DBB', 'C3k_WDBB', 'C3k_DeepDBB',
    # Block变体
    'Faster_Block', 'Faster_Block_EMA',
    # Batch 2 - 注意力和卷积模块
    'MemoryEfficientSwish', 'AttnMap', 'EfficientAttention',
    'SCConv', 'GroupBatchnorm2d', 'SRU', 'CRU', 'ScConv',
    'EMSConv', 'EMSConvP',
    # Batch 2 - Bottleneck变体
    'Bottleneck_CloAtt', 'Bottleneck_SCConv', 'Bottleneck_ScConv',
    'Bottleneck_EMSC', 'Bottleneck_EMSCP',
    # Batch 2 - C3k变体
    'C3k_CloAtt', 'C3k_SCConv', 'C3k_ScConv',
    'C3k_EMSC', 'C3k_EMSCP',
    # Batch 3 - ContextGuided模块
    'FGlo', 'ContextGuidedBlock', 'C3k_ContextGuided',
    # Batch 3 - MSBlock模块
    'MSBlockLayer', 'MSBlock', 'C3k_MSBlock',
    # Batch 3 - MBConv模块
    'EffectiveSEModule', 'MBConv', 'C3k_EMBC',
    # Batch 3 - EMA模块
    'Bottleneck_EMA', 'C3k_EMA',
    # Batch 4 - deformable_LKA模块
    'DeformConv', 'deformable_LKA', 'Bottleneck_DLKA', 'C3k_DLKA',
    # Batch 4 - DAttention模块
    'LayerNormProxy', 'DAttention', 'Bottleneck_DAttention', 'C3k_DAttention',
    # Batch 4 - ParC模块
    'ParC_operator', 'ParConv', 'Bottleneck_ParC', 'C3k_Parc',
    # Batch 4 - DWR模块
    'DWR', 'C3k_DWR',
    # Batch 4 - RFAConv系列
    'h_sigmoid', 'h_swish', 'SE', 'RFAConv', 'RFCBAMConv', 'RFCAConv',
    'Bottleneck_RFAConv', 'Bottleneck_RFCBAMConv', 'Bottleneck_RFCAConv',
    'C3k_RFAConv', 'C3k_RFCBAMConv', 'C3k_RFCAConv',
    # Batch 5 - FocusedLinearAttention模块
    'img2windows', 'windows2img', 'FocusedLinearAttention',
    'Bottleneck_FocusedLinearAttention', 'C3k_FocusedLinearAttention',
    # Batch 5 - MLCA模块
    'MLCA', 'Bottleneck_MLCA', 'C3k_MLCA',
    # Batch 5 - AKConv模块
    'AKConv', 'Bottleneck_AKConv', 'C3k_AKConv',
    # Batch 6 - UniRepLKNet支持函数和类
    'GRNwithNHWC', 'NCHWtoNHWC', 'NHWCtoNCHW',
    'get_conv2d_unirepLK', 'get_bn_unirepLK', 'SEBlock_UniRep',
    'fuse_bn_unirepLK', 'convert_dilated_to_nondilated', 'merge_dilated_into_large_kernel',
    # Batch 6 - DilatedReparamBlock和UniRepLKNetBlock
    'DilatedReparamBlock', 'UniRepLKNetBlock', 'C3k_UniRepLKNetBlock',
    # Batch 6 - DRB模块
    'Bottleneck_DRB', 'C3k_DRB',
    # Batch 6 - DWR_DRB模块
    'DWR_DRB', 'C3k_DWR_DRB',
    # Batch 6 - AggregatedAttention模块
    'Bottleneck_AggregatedAttention', 'C3k_AggregatedAtt',
    # Batch 6 - SWC模块
    'ReparamLargeKernelConv', 'Bottleneck_SWC', 'C3k_SWC',
    # Batch 6 - RetBlock 模块支持
    'RetBlock', 'RelPos2d', 'C3k_RetBlock',
    # Heat 模块
    'Heat2D', 'HeatBlock', 'C3k_Heat',
    # WTConv 模块
    'WTConv2d', 'C3k_WTConv',
    # FMB 模块
    'DMlp', 'SMFA', 'PCFN', 'FMB', 'C3k_FMB',
    # MSMHSA_CGLU
    'MutilScal', 'Mutilscal_MHSA', 'MSMHSA_CGLU', 'C3k_MSMHSA_CGLU',
    # MogaBlock
    'ElementScale', 'ChannelAggregationFFN', 'MultiOrderDWConv', 'MultiOrderGatedAggregation', 'MogaBlock', 'C3k_MogaBlock',
    # SHSA
    'SHSA_GroupNorm', 'SHSABlock_FFN', 'SHSA', 'SHSABlock', 'SHSABlock_CGLU', 'C3k_SHSA', 'C3k_SHSA_CGLU',
    # Edge enhancement/select
    'EdgeEnhancer', 'MutilScaleEdgeInformationEnhance', 'MutilScaleEdgeInformationSelect', 'C3k_MutilScaleEdgeInformationEnhance', 'C3k_MutilScaleEdgeInformationSelect',
    # FFCM
    'FourierUnit', 'Freq_Fusion', 'Fused_Fourier_Conv_Mixer', 'C3k_FFCM',
    # SMAFormer
    'Modulator', 'SMA', 'E_MLP', 'SMAFormerBlock', 'SMAFormerBlock_CGLU', 'C3k_SMAFB', 'C3k_SMAFB_CGLU',
    # MSM / HDRAB / RAB / LFE
    'DeepPoolLayer', 'MSMBlock', 'C3k_MSM',
    'CAB', 'HDRAB', 'C3k_HDRAB',
    'RAB', 'C3k_RAB',
    'ShiftConv2d0', 'ShiftConv2d1', 'LFE', 'C3k_LFE',
    # IDW / CAMixer
    'InceptionDWConv2d', 'MetaNeXtBlock', 'Bottleneck_IDWC', 'C3k_IDWC', 'C3k_IDWB',
    'CAMixer', 'C3k_CAMixer',
]


# ================================ 辅助函数 ================================
def autopad(k, p=None, d=1):
    """自动padding以保持'same'形状输出."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
    """创建卷积+BN层."""
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


def fuse_conv_bn(conv, bn):
    """融合卷积和BN层."""
    fusedconv = nn.Conv2d(
        conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
        stride=conv.stride, padding=conv.padding, groups=conv.groups, bias=True
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


# ================================ DBB变换函数 ================================
def transI_fusebn(kernel, bn):
    """融合BN到卷积核."""
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std


def transII_addbranch(kernels, biases):
    """合并多个分支."""
    return sum(kernels), sum(biases)


def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    """1x1和kxk卷积融合."""
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


def transIV_depthconcat(kernels, biases):
    """深度拼接."""
    return torch.cat(kernels, dim=0), torch.cat(biases)


def transV_avg(channels, kernel_size, groups):
    """平均池化等效卷积."""
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k


def transVI_multiscale(kernel, target_kernel_size):
    """多尺度卷积核padding."""
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])


# ================================ 基础层 ================================
class IdentityBasedConv1x1(nn.Module):
    """基于恒等映射的1x1卷积."""
    def __init__(self, channels, groups=1):
        super().__init__()
        assert channels % groups == 0
        input_dim = channels // groups
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, groups=groups, bias=False)

        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value)
        nn.init.zeros_(self.conv.weight)
        self.groups = groups

    def forward(self, input):
        kernel = self.conv.weight + self.id_tensor.to(self.conv.weight.device).type_as(self.conv.weight)
        result = F.conv2d(input, kernel, None, stride=1, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.conv.weight + self.id_tensor.to(self.conv.weight.device).type_as(self.conv.weight)


class BNAndPadLayer(nn.Module):
    """BN+Padding层."""
    def __init__(self, pad_pixels, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


# ================================ 注意力机制 ================================
class EMA(nn.Module):
    """高效多尺度注意力机制 (Efficient Multi-scale Attention)."""
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class OD_Attention(nn.Module):
    """全维度动态注意力 (Omni-Dimensional Dynamic Attention)."""
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(OD_Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        pass

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

    def switch_to_deploy(self):
        self.fc = fuse_conv_bn(self.fc, self.bn)
        del self.bn


# ================================ 卷积变体 ================================
class Partial_conv3(nn.Module):
    """部分卷积层 (Partial Convolution)."""
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class ODConv2d(nn.Module):
    """全维度动态卷积 (Omni-Dimensional Dynamic Convolution)."""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=None, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=1):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = autopad(kernel_size, padding, dilation)
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = OD_Attention(in_planes, out_planes, kernel_size, groups=groups,
                                      reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        pass

    def _forward_impl_common(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)


# ================================ 重参数化模块 ================================
class DiverseBranchBlock(nn.Module):
    """多样化分支模块 (Diverse Branch Block) - 重参数化."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False):
        super(DiverseBranchBlock, self).__init__()
        self.deploy = deploy
        from ultralytics.nn.modules.conv import Conv
        self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                      stride=stride, padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                              kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                          kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        if single_init:
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device), self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels, out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation,
                                     groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


class DiverseBranchBlockNOAct(nn.Module):
    """多样化分支模块(无激活函数版本)."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False):
        super(DiverseBranchBlockNOAct, self).__init__()
        self.deploy = deploy

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups

        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                          stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                              kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                          kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        if single_init:
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device), self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels, out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation,
                                     groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.dbb_reparam(inputs)

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return out

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


class DeepDiverseBranchBlock(nn.Module):
    """深度多样化分支模块 (Deep Diverse Branch Block)."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False, conv_orgin=DiverseBranchBlockNOAct):
        super(DeepDiverseBranchBlock, self).__init__()
        self.deploy = deploy
        from ultralytics.nn.modules.conv import Conv
        self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups

        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.dbb_origin = DiverseBranchBlockNOAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                          stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                              kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                          kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        if single_init:
            self.single_init()

    def get_equivalent_kernel_bias(self):
        self.dbb_origin.switch_to_deploy()
        k_origin, b_origin = self.dbb_origin.dbb_reparam.weight, self.dbb_origin.dbb_reparam.bias

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device), self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.dbb_reparam.in_channels,
                                     out_channels=self.dbb_origin.dbb_reparam.out_channels,
                                     kernel_size=self.dbb_origin.dbb_reparam.kernel_size,
                                     stride=self.dbb_origin.dbb_reparam.stride,
                                     padding=self.dbb_origin.dbb_reparam.padding,
                                     dilation=self.dbb_origin.dbb_reparam.dilation,
                                     groups=self.dbb_origin.dbb_reparam.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


class WideDiverseBranchBlock(nn.Module):
    """宽度多样化分支模块 (Wide Diverse Branch Block) - 包含水平和垂直卷积."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, single_init=False):
        super(WideDiverseBranchBlock, self).__init__()
        self.deploy = deploy
        from ultralytics.nn.modules.conv import Conv
        self.nonlinear = Conv.default_act

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups

        if padding is None:
            padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                          stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                              kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                          kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        if single_init:
            self.single_init()

        if padding - kernel_size // 2 >= 0:
            self.crop = 0
            hor_padding = [padding - kernel_size // 2, padding]
            ver_padding = [padding, padding - kernel_size // 2]
        else:
            self.crop = kernel_size // 2 - padding
            hor_padding = [0, padding]
            ver_padding = [padding, 0]

        # 垂直和水平卷积
        self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=(kernel_size, 1), stride=stride, padding=ver_padding,
                                  dilation=dilation, groups=groups, bias=False)
        self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=(1, kernel_size), stride=stride, padding=hor_padding,
                                  dilation=dilation, groups=groups, bias=False)
        self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)
        self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        """将非对称卷积核添加到方形卷积核中心."""
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :,
                      square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                      square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias_1xk_kx1_kxk(self):
        """融合1xk, kx1和kxk卷积."""
        hor_k, hor_b = transI_fusebn(self.hor_conv.weight, self.hor_bn)
        ver_k, ver_b = transI_fusebn(self.ver_conv.weight, self.ver_bn)
        square_k, square_b = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        return square_k, hor_b + ver_b + square_b

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = self.get_equivalent_kernel_bias_1xk_kx1_kxk()

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device), self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels, out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation,
                                     groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')
        self.__delattr__('hor_conv')
        self.__delattr__('hor_bn')
        self.__delattr__('ver_conv')
        self.__delattr__('ver_bn')

    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)

        if self.crop > 0:
            ver_input = inputs[:, :, :, self.crop:-self.crop]
            hor_input = inputs[:, :, self.crop:-self.crop, :]
        else:
            ver_input = inputs
            hor_input = inputs
        vertical_outputs = self.ver_bn(self.ver_conv(ver_input))
        horizontal_outputs = self.hor_bn(self.hor_conv(hor_input))
        result = out + vertical_outputs + horizontal_outputs

        return self.nonlinear(result)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


# ================================ Bottleneck变体 ================================
class Bottleneck_PConv(nn.Module):
    """使用部分卷积的Bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        self.cv1 = Partial_conv3(c1)
        self.cv2 = Partial_conv3(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_ODConv(nn.Module):
    """使用ODConv的Bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ODConv2d(c1, c_, k[0], 1)
        self.cv2 = ODConv2d(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_DBB(nn.Module):
    """使用DiverseBranchBlock的Bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = DiverseBranchBlock(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_WDBB(nn.Module):
    """使用WideDiverseBranchBlock的Bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = WideDiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = WideDiverseBranchBlock(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_DeepDBB(nn.Module):
    """使用DeepDiverseBranchBlock的Bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DeepDiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = DeepDiverseBranchBlock(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ================================ C3k变体 ================================
class C3k_Faster(nn.Module):
    """使用Faster Block的C3k."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Faster_Block(c_, c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k_PConv(nn.Module):
    """使用PConv的C3k."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck_PConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k_ODConv(nn.Module):
    """使用ODConv的C3k."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck_ODConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k_Faster_EMA(nn.Module):
    """使用Faster EMA Block的C3k."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Faster_Block_EMA(c_, c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k_DBB(nn.Module):
    """使用DBB的C3k."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck_DBB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k_WDBB(nn.Module):
    """使用WDBB的C3k."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck_WDBB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k_DeepDBB(nn.Module):
    """使用DeepDBB的C3k."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck_DeepDBB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# ================================ Block变体 ================================
class Faster_Block(nn.Module):
    """FasterNet Block."""
    def __init__(self, inc, dim, n_div=4, mlp_ratio=2, drop_path=0.1,
                 layer_scale_init_value=0.0, pconv_fw_type='split_cat'):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)
        from ultralytics.nn.modules.conv import Conv

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class Faster_Block_EMA(nn.Module):
    """FasterNet Block with EMA attention."""
    def __init__(self, inc, dim, n_div=4, mlp_ratio=2, drop_path=0.1,
                 layer_scale_init_value=0.0, pconv_fw_type='split_cat'):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)
        from ultralytics.nn.modules.conv import Conv

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)
        self.attention = EMA(dim)

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.attention(self.drop_path(self.mlp(x)))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


# ================================ Batch 2: 第二批辅助类 ================================

# ========== CloAtt 相关 (EfficientAttention) ==========
try:
    from efficientnet_pytorch.model import MemoryEfficientSwish
except ImportError:
    class MemoryEfficientSwish(nn.Module):
        """内存高效的Swish激活函数（如果efficientnet_pytorch不可用则使用回退实现）"""
        def forward(self, x):
            return x * torch.sigmoid(x)


class AttnMap(nn.Module):
    """注意力映射模块

    用于EfficientAttention的注意力权重映射。
    源代码位置: attention.py:762
    """
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            MemoryEfficientSwish(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.act_block(x)


class EfficientAttention(nn.Module):
    """高效注意力机制

    结合高频和低频注意力的高效实现。
    源代码位置: attention.py:773
    """
    def __init__(self, dim, num_heads=8, group_split=[4, 4], kernel_sizes=[5], window_size=4,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3*self.dim_head*group_head, 3*self.dim_head*group_head, kernel_size,
                         1, kernel_size//2, groups=3*self.dim_head*group_head))
            act_blocks.append(AttnMap(self.dim_head*group_head))
            qkvs.append(nn.Conv2d(dim, 3*group_head*self.dim_head, 1, 1, 0, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1]*self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1]*self.dim_head*2, 1, 1, 0, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size!=1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        """高频注意力"""
        b, c, h, w = x.size()
        qkv = to_qkv(x)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()
        q, k, v = qkv
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)
        return res

    def low_fre_attention(self, x : torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        """低频注意力"""
        b, c, h, w = x.size()
        q = to_q(x).reshape(b, -1, self.dim_head, h*w).transpose(-1, -2).contiguous()
        kv = avgpool(x)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h*w)//(self.window_size**2)).permute(1, 0, 2, 4, 3).contiguous()
        k, v = kv
        attn = self.scalor * q @ k.transpose(-1, -2)
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))


class Bottleneck_CloAtt(Bottleneck):
    """使用EfficientAttention的Bottleneck

    源代码位置: block.py:1093
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.attention = EfficientAttention(c2)

    def forward(self, x):
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C3k_CloAtt(C3k):
    """使用Bottleneck_CloAtt的C3k模块

    源代码位置: block.py:1105
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_CloAtt(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ========== SCConv (CVPR 2020) ==========
class SCConv(nn.Module):
    """Spatial and Channel Convolution

    论文: http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf
    源代码位置: block.py:1121
    """
    def __init__(self, c1, c2, s=1, d=1, g=1, pooling_r=4):
        super(SCConv, self).__init__()
        from ultralytics.nn.modules.conv import Conv
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            Conv(c1, c2, k=3, d=d, g=g, act=False)
        )
        self.k3 = Conv(c1, c2, k=3, d=d, g=g, act=False)
        self.k4 = Conv(c1, c2, k=3, s=s, d=d, g=g, act=False)

    def forward(self, x):
        import torch.nn.functional as F
        identity = x
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))
        out = torch.mul(self.k3(x), out)
        out = self.k4(out)
        return out


class Bottleneck_SCConv(Bottleneck):
    """使用SCConv的Bottleneck

    源代码位置: block.py:1141
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        from ultralytics.nn.modules.conv import Conv
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = SCConv(c_, c2, g=g)


class C3k_SCConv(C3k):
    """使用Bottleneck_SCConv的C3k模块

    源代码位置: block.py:1148
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_SCConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ========== ScConv (CVPR 2023 - Spatial and Channel Reconstruction Convolution) ==========
class GroupBatchnorm2d(nn.Module):
    """分组批归一化

    源代码位置: block.py:1164
    """
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    """Spatial Reconstruction Unit - 空间重构单元

    源代码位置: block.py:1185
    """
    def __init__(self, oup_channels: int, group_num: int = 16, gate_treshold: float = 0.5):
        super().__init__()
        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([x_11+x_22, x_12+x_21], dim=1)


class CRU(nn.Module):
    """Channel Reconstruction Unit - 通道重构单元

    alpha: 0<alpha<1
    源代码位置: block.py:1215
    """
    def __init__(self, op_channel: int, alpha: float = 1/2, squeeze_radio: int = 2,
                 group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()
        self.up_channel = up_channel = int(alpha*op_channel)
        self.low_channel = low_channel = op_channel-up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel//squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel//squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel//squeeze_radio, op_channel, kernel_size=group_kernel_size,
                            stride=1, padding=group_kernel_size//2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel//squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,
                             kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        import torch.nn.functional as F
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1)//2, dim=1)
        return out1+out2


class ScConv(nn.Module):
    """Spatial and Channel Reconstruction Convolution

    论文: CVPR2023 https://openaccess.thecvf.com/content/CVPR2023/papers/...
    源代码位置: block.py:1252
    """
    def __init__(self, op_channel: int, group_num: int = 16, gate_treshold: float = 0.5,
                 alpha: float = 1/2, squeeze_radio: int = 2, group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__()
        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio,
                      group_size=group_size, group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


class Bottleneck_ScConv(Bottleneck):
    """使用ScConv的Bottleneck

    源代码位置: block.py:1278
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        from ultralytics.nn.modules.conv import Conv
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ScConv(c2)


class C3k_ScConv(C3k):
    """使用Bottleneck_ScConv的C3k模块

    源代码位置: block.py:1285
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_ScConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ========== EMSConv (Efficient Multi-Scale Conv) ==========
class EMSConv(nn.Module):
    """Efficient Multi-Scale Convolution

    源代码位置: block.py:1328
    """
    def __init__(self, channel=256, kernels=[3, 5]):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        try:
            from einops import rearrange
            self.rearrange = rearrange
        except ImportError:
            raise ImportError("einops is required for EMSConv. Install it with: pip install einops")

        self.groups = len(kernels)
        min_ch = channel // 4
        assert min_ch >= 16, f'channel must Greater than {64}, but {channel}'

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        _, c, _, _ = x.size()
        x_cheap, x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = self.rearrange(x_group, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_group = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_group = self.rearrange(x_group, 'g bs ch h w -> bs (g ch) h w')
        x = torch.cat([x_cheap, x_group], dim=1)
        x = self.conv_1x1(x)
        return x


class EMSConvP(nn.Module):
    """Efficient Multi-Scale Convolution Plus

    源代码位置: block.py:1352
    """
    def __init__(self, channel=256, kernels=[1, 3, 5, 7]):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        try:
            from einops import rearrange
            self.rearrange = rearrange
        except ImportError:
            raise ImportError("einops is required for EMSConvP. Install it with: pip install einops")

        self.groups = len(kernels)
        min_ch = channel // self.groups
        assert min_ch >= 16, f'channel must Greater than {16 * self.groups}, but {channel}'

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(Conv(c1=min_ch, c2=min_ch, k=ks))
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        x_group = self.rearrange(x, 'bs (g ch) h w -> bs ch h w g', g=self.groups)
        x_convs = torch.stack([self.convs[i](x_group[..., i]) for i in range(len(self.convs))])
        x_convs = self.rearrange(x_convs, 'g bs ch h w -> bs (g ch) h w')
        x_convs = self.conv_1x1(x_convs)
        return x_convs


class Bottleneck_EMSC(Bottleneck):
    """使用EMSConv的Bottleneck

    源代码位置: block.py:1373
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        from ultralytics.nn.modules.conv import Conv
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = EMSConv(c2)


class C3k_EMSC(C3k):
    """使用Bottleneck_EMSC的C3k模块

    源代码位置: block.py:1380
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_EMSC(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class Bottleneck_EMSCP(Bottleneck):
    """使用EMSConvP的Bottleneck

    源代码位置: block.py:1391
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        from ultralytics.nn.modules.conv import Conv
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = EMSConvP(c2)


class C3k_EMSCP(C3k):
    """使用Bottleneck_EMSCP的C3k模块

    源代码位置: block.py:1398
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_EMSCP(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ================================ Batch 3: 第三批辅助类 ================================

# ========== ContextGuided 相关 ==========
class FGlo(nn.Module):
    """全局特征提取模块

    用于精炼局部特征和周围上下文的联合特征。
    源代码位置: block.py:2298
    """
    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ContextGuidedBlock(nn.Module):
    """上下文引导块

    源代码位置: block.py:2318
    """
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        n = int(nOut/2)
        self.conv1x1 = Conv(nIn, n, 1, 1)
        self.F_loc = nn.Conv2d(n, n, 3, padding=1, groups=n)
        self.F_sur = nn.Conv2d(n, n, 3, padding=autopad(3, None, dilation_rate),
                              dilation=dilation_rate, groups=n)
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(nOut),
            Conv.default_act
        )
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.bn_act(joi_feat)
        output = self.F_glo(joi_feat)
        if self.add:
            output = input + output
        return output


class C3k_ContextGuided(C3k):
    """使用ContextGuidedBlock的C3k模块

    源代码位置: block.py:2390
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(ContextGuidedBlock(c_, c_) for _ in range(n)))


# ========== MSBlock 相关 ==========
class MSBlockLayer(nn.Module):
    """多尺度块层

    源代码位置: block.py:2405
    """
    def __init__(self, inc, ouc, k):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.in_conv = Conv(inc, ouc, 1)
        self.mid_conv = Conv(ouc, ouc, k, g=ouc)
        self.out_conv = Conv(ouc, inc, 1)

    def forward(self, x):
        return self.out_conv(self.mid_conv(self.in_conv(x)))


class MSBlock(nn.Module):
    """多尺度块

    源代码位置: block.py:2416
    """
    def __init__(self, inc, ouc, kernel_sizes, in_expand_ratio=3., mid_expand_ratio=2.,
                 layers_num=3, in_down_ratio=2.):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        in_channel = int(inc * in_expand_ratio // in_down_ratio)
        self.mid_channel = in_channel // len(kernel_sizes)
        groups = int(self.mid_channel * mid_expand_ratio)
        self.in_conv = Conv(inc, in_channel)

        self.mid_convs = []
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [MSBlockLayer(self.mid_channel, groups, k=kernel_size)
                        for _ in range(int(layers_num))]
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = Conv(in_channel, ouc, 1)
        self.attention = None

    def forward(self, x):
        out = self.in_conv(x)
        channels = []
        for i, mid_conv in enumerate(self.mid_convs):
            channel = out[:, i * self.mid_channel:(i+1) * self.mid_channel, ...]
            if i >= 1:
                channel = channel + channels[i-1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        if self.attention is not None:
            out = self.attention(out)
        return out


class C3k_MSBlock(C3k):
    """使用MSBlock的C3k模块

    源代码位置: block.py:2452
    """
    def __init__(self, c1, c2, n=1, kernel_sizes=[1, 3, 3], in_expand_ratio=3.,
                 mid_expand_ratio=2., layers_num=3, in_down_ratio=2.,
                 shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(MSBlock(c_, c_, kernel_sizes, in_expand_ratio,
                                        mid_expand_ratio, layers_num, in_down_ratio)
                                for _ in range(n)))


# ========== MBConv 相关 (EfficientNet) ==========
class EffectiveSEModule(nn.Module):
    """Effective Squeeze-and-Excitation模块

    源代码位置: attention.py:1052
    """
    def __init__(self, channels, add_maxpool=False):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


class MBConv(nn.Module):
    """MobileNet Block (EfficientNet风格)

    源代码位置: block.py:2688
    """
    def __init__(self, inc, ouc, shortcut=True, e=4, dropout=0.1):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        midc = inc * e
        self.conv_pw_1 = Conv(inc, midc, 1)
        self.conv_dw_1 = Conv(midc, midc, 3, g=midc)
        self.effective_se = EffectiveSEModule(midc)
        self.conv1 = Conv(midc, ouc, 1, act=False)
        self.dropout = nn.Dropout2d(p=dropout)
        self.add = shortcut and inc == ouc

    def forward(self, x):
        return (x + self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x)))))
                if self.add else
                self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x))))))


class C3k_EMBC(C3k):
    """使用MBConv的C3k模块

    源代码位置: block.py:2702
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(MBConv(c_, c_, shortcut) for _ in range(n)))


# ========== EMA 相关 (使用已有的EMA类) ==========
class Bottleneck_EMA(nn.Module):
    """使用EMA注意力的Bottleneck

    源代码位置: block.py:2765
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.attention = EMA(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return (x + self.attention(self.cv2(self.cv1(x))) if self.add
                else self.attention(self.cv2(self.cv1(x))))


class C3k_EMA(C3k):
    """使用Bottleneck_EMA的C3k模块

    源代码位置: block.py:2781
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_EMA(c_, c_, shortcut, g, k=(k, k), e=1.0)
                                for _ in range(n)))


# ================================ Batch 4: 第四批辅助类 ================================

# ========== deformable_LKA 相关 ==========
class DeformConv(nn.Module):
    """可变形卷积包装器

    源代码位置: attention.py:1011
    """
    def __init__(self, in_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()
        import torchvision

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class deformable_LKA(nn.Module):
    """可变形大核注意力

    源代码位置: attention.py:1038
    """
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class Bottleneck_DLKA(nn.Module):
    """使用可变形LKA的Bottleneck

    源代码位置: block.py:2467
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = deformable_LKA(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_DLKA(C3k):
    """使用Bottleneck_DLKA的C3k模块

    源代码位置: block.py:2474
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DLKA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ========== DAttention 相关 ==========
class LayerNormProxy(nn.Module):
    """LayerNorm代理，用于处理NCHW格式

    源代码位置: attention.py:1151
    """
    def __init__(self, dim):
        super().__init__()
        import einops
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        import einops
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DAttention(nn.Module):
    """可变形注意力 (CVPR2022)

    Vision Transformer with Deformable Attention
    源代码位置: attention.py:1161
    """
    def __init__(
        self, channel, q_size, n_heads=8, n_groups=4,
        attn_drop=0.0, proj_drop=0.0, stride=1,
        offset_range_factor=4, use_pe=True, dwc_pe=True,
        no_off=False, fixed_pe=False, ksize=3, log_cpb=False, kv_size=None
    ):
        super().__init__()
        from timm.models.layers import trunc_normal_

        n_head_channels = channel // n_heads
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        # 允许 q_size 为空/布尔（兼容迁移 YAML 未显式提供的情况），仅在需要固定位置编码时才用到
        if isinstance(q_size, (tuple, list)) and len(q_size) == 2:
            self.q_h, self.q_w = int(q_size[0]), int(q_size[1])
            self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        else:
            # 延迟到 forward 通过输入形状确定；当使用 dwc_pe=True（默认）时不依赖 q_size
            self.q_h = self.q_w = None
            self.kv_h = self.kv_w = None
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 0.5).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off)

        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        offset = rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        pos = (offset + reference).clamp(-1., +1.)

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],
            mode='bilinear', align_corners=True)

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)

        if self.rpe_table is not None:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            else:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.rpe_table is not None and self.dwc_pe:
            out = out + residual_lepe

        out = out.reshape(B, C, H, W)
        y = self.proj_drop(self.proj_out(out))

        return y


class Bottleneck_DAttention(nn.Module):
    """使用DAttention的Bottleneck

    源代码位置: block.py:2739
    """
    def __init__(self, c1, c2, fmapsize, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.attention = DAttention(c2, fmapsize)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C3k_DAttention(C3k):
    """使用Bottleneck_DAttention的C3k模块

    源代码位置: block.py:2750
    """
    def __init__(self, c1, c2, n=1, fmapsize=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DAttention(c_, c_, fmapsize, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ========== ParC (Parallel Convolution) 相关 ==========
class ParC_operator(nn.Module):
    """并行卷积算子

    源代码位置: block.py:2796
    """
    def __init__(self, dim, type, global_kernel_size, use_pe=True, groups=1):
        super().__init__()
        from timm.models.layers import trunc_normal_

        self.type = type  # H or W
        self.dim = dim
        self.use_pe = use_pe
        self.global_kernel_size = global_kernel_size
        self.kernel_size = (global_kernel_size, 1) if self.type == 'H' else (1, global_kernel_size)
        self.gcc_conv = nn.Conv2d(dim, dim, kernel_size=self.kernel_size, groups=dim)
        if use_pe:
            if self.type=='H':
                self.pe = nn.Parameter(torch.randn(1, dim, self.global_kernel_size, 1))
            elif self.type=='W':
                self.pe = nn.Parameter(torch.randn(1, dim, 1, self.global_kernel_size))
            trunc_normal_(self.pe, std=.02)

    def forward(self, x):
        if self.use_pe:
            x = x + self.pe.expand(1, self.dim, self.global_kernel_size, self.global_kernel_size)

        x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2) if self.type == 'H' else torch.cat((x, x[:, :, :, :-1]), dim=3)
        x = self.gcc_conv(x_cat)

        return x


class ParConv(nn.Module):
    """并行卷积

    源代码位置: block.py:2821
    """
    def __init__(self, dim, fmapsize, use_pe=True, groups=1):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        self.parc_H = ParC_operator(dim // 2, 'H', fmapsize[0], use_pe, groups = groups)
        self.parc_W = ParC_operator(dim // 2, 'W', fmapsize[1], use_pe, groups = groups)
        self.bn = nn.BatchNorm2d(dim)
        self.act = Conv.default_act

    def forward(self, x):
        out_H, out_W = torch.chunk(x, 2, dim=1)
        out_H, out_W = self.parc_H(out_H), self.parc_W(out_W)
        out = torch.cat((out_H, out_W), dim=1)
        out = self.bn(out)
        out = self.act(out)
        return out


class Bottleneck_ParC(nn.Module):
    """使用并行卷积的Bottleneck

    源代码位置: block.py:2838
    """
    def __init__(self, c1, c2, fmapsize, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        if c_ == c2:
            self.cv2 = ParConv(c2, fmapsize, groups=g)
        else:
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_Parc(C3k):
    """使用Bottleneck_ParC的C3k模块

    源代码位置: block.py:2858
    """
    def __init__(self, c1, c2, n=1, fmapsize=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_ParC(c_, c_, fmapsize, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ========== DWR (Dilation-wise Residual) 相关 ==========
class DWR(nn.Module):
    """扩张残差模块

    源代码位置: block.py:2871
    """
    def __init__(self, dim):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        self.conv_3x3 = Conv(dim, dim // 2, 3)

        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5)

        self.conv_1x1 = Conv(dim * 2, dim, k=1)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out


class C3k_DWR(C3k):
    """使用DWR的C3k模块

    源代码位置: block.py:2890
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(DWR(c_) for _ in range(n)))


# ========== RFAConv系列 (Receptive-Field Attention) ==========
class h_sigmoid(nn.Module):
    """Hard Sigmoid激活

    源代码位置: RFAconv.py:8
    """
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard Swish激活

    源代码位置: RFAconv.py:16
    """
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SE(nn.Module):
    """Squeeze-and-Excitation模块

    源代码位置: RFAconv.py:53
    """
    def __init__(self, in_channel, ratio=16):
        super(SE, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel, ratio, bias=False),
            nn.ReLU(),
            nn.Linear(ratio, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c= x.shape[0:2]
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class RFAConv(nn.Module):
    """Receptive-Field Attention卷积

    源代码位置: RFAconv.py:24
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        from einops import rearrange
        from ultralytics.nn.modules.conv import Conv

        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1, groups=in_channel,bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size,padding=kernel_size//2,stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        from einops import rearrange
        b, c = x.shape[0:2]
        weight =  self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)
        return self.conv(conv_data)


class RFCBAMConv(nn.Module):
    """RFA + CBAM卷积

    源代码位置: RFAconv.py:70
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        if kernel_size % 2 == 0:
            assert("the kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(in_channel,in_channel * (kernel_size**2),kernel_size,padding=kernel_size//2,
                                                stride=stride,groups=in_channel,bias =False),
                                      nn.BatchNorm2d(in_channel * (kernel_size**2)),
                                      nn.ReLU()
                                      )
        self.get_weight = nn.Sequential(nn.Conv2d(2,1,kernel_size=3,padding=1,bias=False),nn.Sigmoid())
        self.se = SE(in_channel)

        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        from einops import rearrange
        b, c = x.shape[0:2]
        channel_attention =  self.se(x)
        generate_feature = self.generate(x)

        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b,c,self.kernel_size**2,h,w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)

        unfold_feature = generate_feature * channel_attention
        max_feature,_ = torch.max(generate_feature,dim=1,keepdim=True)
        mean_feature = torch.mean(generate_feature,dim=1,keepdim=True)
        receptive_field_attention = self.get_weight(torch.cat((max_feature,mean_feature),dim=1))
        conv_data = unfold_feature  * receptive_field_attention
        return self.conv(conv_data)


class RFCAConv(nn.Module):
    """RFA + Coordinate Attention卷积

    源代码位置: RFAconv.py:105
    """
    def __init__(self, inp, oup, kernel_size, stride=1, reduction=32):
        super(RFCAConv, self).__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(inp,inp * (kernel_size**2),kernel_size,padding=kernel_size//2,
                                                stride=stride,groups=inp,
                                                bias =False),
                                      nn.BatchNorm2d(inp * (kernel_size**2)),
                                      nn.ReLU()
                                      )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp,oup,kernel_size,stride=kernel_size))


    def forward(self, x):
        from einops import rearrange
        b, c = x.shape[0:2]
        generate_feature = self.generate(x)
        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b,c,self.kernel_size**2,h,w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)

        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        h, w = generate_feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return self.conv(generate_feature * a_w * a_h)


class Bottleneck_RFAConv(nn.Module):
    """使用RFAConv的Bottleneck

    源代码位置: block.py:2905
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RFAConv(c_, c2, k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_RFAConv(C3k):
    """使用Bottleneck_RFAConv的C3k模块

    源代码位置: block.py:2914
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_RFAConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class Bottleneck_RFCBAMConv(nn.Module):
    """使用RFCBAMConv的Bottleneck

    源代码位置: block.py:2925
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RFCBAMConv(c_, c2, k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_RFCBAMConv(C3k):
    """使用Bottleneck_RFCBAMConv的C3k模块

    源代码位置: block.py:2934
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_RFCBAMConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class Bottleneck_RFCAConv(nn.Module):
    """使用RFCAConv的Bottleneck

    源代码位置: block.py:2945
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RFCAConv(c_, c2, k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_RFCAConv(C3k):
    """使用Bottleneck_RFCAConv的C3k模块

    源代码位置: block.py:2954
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_RFCAConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ================================ Batch 5 - FocusedLinearAttention模块 ================================
def img2windows(img, H_sp, W_sp):
    """将图像分割为窗口

    源代码位置: attention.py:1366

    Args:
        img: B C H W 格式的图像张量
        H_sp: 窗口高度
        W_sp: 窗口宽度

    Returns:
        窗口张量 B' H_sp*W_sp C
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """将窗口重组为图像

    源代码位置: attention.py:1375

    Args:
        img_splits_hw: B' H W C 格式的窗口张量
        H_sp: 窗口高度
        W_sp: 窗口宽度
        H: 图像高度
        W: 图像宽度

    Returns:
        图像张量 B H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class FocusedLinearAttention(nn.Module):
    """聚焦线性注意力模块

    源代码位置: attention.py:1385
    论文: 实现了高效的线性注意力机制，通过聚焦因子增强注意力的表达能力
    """
    def __init__(self, dim, resolution, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None, focusing_factor=3, kernel_size=5):
        super().__init__()
        import numpy as np
        from einops import rearrange

        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        H_sp, W_sp = self.resolution[0], self.resolution[1]
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.conv_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, self.H_sp * self.W_sp, dim)))

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  # B', C, H', W'

        lepe = func(x)  # B', C, H', W'
        lepe = lepe.reshape(-1, C // self.num_heads, H_sp * W_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, self.H_sp * self.W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B C H W
        """
        from einops import rearrange
        import numpy as np

        qkv = self.conv_qkv(qkv)
        q, k, v = torch.chunk(qkv.flatten(2).transpose(1, 2), 3, dim=-1)

        # Img2Window
        H, W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        feature_map = rearrange(v, "b (h w) c -> b c h w", h=self.H_sp, w=self.W_sp)
        feature_map = rearrange(self.dwc(feature_map), "b c h w -> b (h w) c")
        x = x + feature_map
        x = x + lepe
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = windows2img(x, self.H_sp, self.W_sp, H, W).permute(0, 3, 1, 2)
        return x


class Bottleneck_FocusedLinearAttention(nn.Module):
    """使用FocusedLinearAttention的Bottleneck

    源代码位置: block.py:3059
    """
    def __init__(self, c1, c2, fmapsize, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.attention = FocusedLinearAttention(c2, fmapsize)

    def forward(self, x):
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C3k_FocusedLinearAttention(C3k):
    """使用Bottleneck_FocusedLinearAttention的C3k模块

    源代码位置: block.py:3070
    """
    def __init__(self, c1, c2, n=1, fmapsize=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_FocusedLinearAttention(c_, c_, fmapsize, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))


# ================================ Batch 5 - MLCA模块 ================================
class MLCA(nn.Module):
    """多级通道注意力模块

    源代码位置: attention.py:1484
    论文: 结合局部和全局池化的通道注意力机制
    """
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super(MLCA, self).__init__()
        import math

        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight = local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        import torch.nn.functional as F

        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)

        b, c, m, n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)

        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c, self.local_size, self.local_size)
        y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)

        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global * (1 - self.local_weight) + (att_local * self.local_weight), [m, n])

        x = x * att_all
        return x


class Bottleneck_MLCA(nn.Module):
    """使用MLCA的Bottleneck

    源代码位置: block.py:3085
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.attention = MLCA(c2)

    def forward(self, x):
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C3k_MLCA(C3k):
    """使用Bottleneck_MLCA的C3k模块

    源代码位置: block.py:3095
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_MLCA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ================================ Batch 5 - AKConv模块 ================================
class AKConv(nn.Module):
    """Alterable Kernel Convolution - 可变核卷积

    源代码位置: block.py:3110
    论文: 通过可学习的偏移来调整卷积核的形状和大小
    """
    def __init__(self, inc, outc, num_param=5, stride=1, bias=None):
        super(AKConv, self).__init__()
        import math
        from einops import rearrange

        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU()
        )
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        from einops import rearrange

        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        import math

        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int), indexing='ij')
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number), indexing='ij')

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride), indexing='ij')

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        from einops import rearrange

        b, c, h, w, n = x_offset.size()
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset


class Bottleneck_AKConv(nn.Module):
    """使用AKConv的Bottleneck

    源代码位置: block.py:3245
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1) if k[0] != 3 else AKConv(c1, c_, k[0])
        self.cv2 = AKConv(c_, c2, k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_AKConv(C3k):
    """使用Bottleneck_AKConv的C3k模块

    源代码位置: block.py:3254
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_AKConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ================================ Batch 6 - UniRepLKNet支持类和函数 ================================
class GRNwithNHWC(nn.Module):
    """Global Response Normalization层

    源代码位置: backbone/UniRepLKNet.py:21
    论文: ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    输入格式: (N, H, W, C)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class NCHWtoNHWC(nn.Module):
    """NCHW转NHWC格式

    源代码位置: backbone/UniRepLKNet.py:43
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    """NHWC转NCHW格式

    源代码位置: backbone/UniRepLKNet.py:51
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


def get_conv2d_unirepLK(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                        attempt_use_lk_impl=True):
    """创建卷积层（支持大核优化）

    源代码位置: backbone/UniRepLKNet.py:63
    """
    from timm.models.layers import to_2tuple

    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn_unirepLK(dim, use_sync_bn=False):
    """创建BatchNorm层

    源代码位置: backbone/UniRepLKNet.py:88
    """
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


class SEBlock_UniRep(nn.Module):
    """Squeeze-and-Excitation Block

    源代码位置: backbone/UniRepLKNet.py:94
    论文: SENet (https://arxiv.org/abs/1709.01507)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock_UniRep, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        import torch.nn.functional as F
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)


def fuse_bn_unirepLK(conv, bn):
    """融合卷积和BN层

    源代码位置: backbone/UniRepLKNet.py:116
    """
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std


def convert_dilated_to_nondilated(kernel, dilate_rate):
    """将膨胀卷积核转换为非膨胀卷积核

    源代码位置: backbone/UniRepLKNet.py:121
    """
    import torch.nn.functional as F

    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    """将膨胀卷积核合并到大卷积核中

    源代码位置: backbone/UniRepLKNet.py:135
    """
    import torch.nn.functional as F

    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


# ================================ Batch 6 - DilatedReparamBlock和UniRepLKNetBlock ================================
class DilatedReparamBlock(nn.Module):
    """膨胀重参数化块

    源代码位置: block.py:3270
    论文: UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    输入格式: (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d_unirepLK(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn_unirepLK(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn_unirepLK(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def switch_to_deploy(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn_unirepLK(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn_unirepLK(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d_unirepLK(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class UniRepLKNetBlock(nn.Module):
    """UniRepLKNet Block

    源代码位置: block.py:3347
    """
    def __init__(self, dim, kernel_size, drop_path=0., layer_scale_init_value=1e-6,
                 deploy=False, attempt_use_lk_impl=True, with_cp=False,
                 use_sync_bn=False, ffn_factor=4):
        super().__init__()
        from timm.models.layers import DropPath

        self.with_cp = with_cp
        self.need_contiguous = (not deploy) or kernel_size >= 7

        if kernel_size == 0:
            self.dwconv = nn.Identity()
            self.norm = nn.Identity()
        elif deploy:
            self.dwconv = get_conv2d_unirepLK(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=True,
                                     attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              use_sync_bn=use_sync_bn,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
            self.norm = get_bn_unirepLK(dim, use_sync_bn=use_sync_bn)
        elif kernel_size == 1:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=1, bias=deploy)
            self.norm = get_bn_unirepLK(dim, use_sync_bn=use_sync_bn)
        else:
            assert kernel_size in [3, 5]
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                    dilation=1, groups=dim, bias=deploy)
            self.norm = get_bn_unirepLK(dim, use_sync_bn=use_sync_bn)

        self.se = SEBlock_UniRep(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim),
                NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn_unirepLK(dim, use_sync_bn=use_sync_bn))

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, inputs):
        def _f(x):
            if self.need_contiguous:
                x = x.contiguous()
            y = self.se(self.norm(self.dwconv(x)))
            y = self.pwconv2(self.act(self.pwconv1(y)))
            if self.gamma is not None:
                y = self.gamma.view(1, -1, 1, 1) * y
            return self.drop_path(y) + x

        if self.with_cp and inputs.requires_grad:
            from torch.utils import checkpoint
            return checkpoint.checkpoint(_f, inputs)
        else:
            return _f(inputs)


class C3k_UniRepLKNetBlock(C3k):
    """使用UniRepLKNetBlock的C3k模块

    源代码位置: block.py:3458
    """
    def __init__(self, c1, c2, n=1, k=7, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(UniRepLKNetBlock(c_, k) for _ in range(n)))


# ================================ Batch 6 - DRB模块 ================================
class Bottleneck_DRB(nn.Module):
    """使用DilatedReparamBlock的Bottleneck

    源代码位置: block.py:3469
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DilatedReparamBlock(c2, 7)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_DRB(C3k):
    """使用Bottleneck_DRB的C3k模块

    源代码位置: block.py:3477
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DRB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ================================ Batch 6 - DWR_DRB模块 ================================
class DWR_DRB(nn.Module):
    """Dilation-wise Residual with DilatedReparamBlock

    源代码位置: block.py:3492
    """
    def __init__(self, dim, act=True) -> None:
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        self.conv_3x3 = Conv(dim, dim // 2, 3, act=act)

        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1, act=act)
        self.conv_3x3_d3 = DilatedReparamBlock(dim // 2, 5)
        self.conv_3x3_d5 = DilatedReparamBlock(dim // 2, 7)

        self.conv_1x1 = Conv(dim * 2, dim, k=1, act=act)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out


class C3k_DWR_DRB(C3k):
    """使用DWR_DRB的C3k模块

    源代码位置: block.py:3511
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(DWR_DRB(c_) for _ in range(n)))


# ================================ Batch 6 - ReparamLargeKernelConv和SWC模块 ================================
class ReparamLargeKernelConv(nn.Module):
    """可重参数化大核卷积

    源代码位置: shiftwise_conv.py:219
    """
    def __init__(self, in_channels, out_channels, kernel_size, small_kernel=5,
                 stride=1, groups=1, small_kernel_merged=False, Decom=True, bn=True):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.Decom = Decom
        padding = kernel_size // 2

        if small_kernel_merged:
            self.lkb_reparam = get_conv2d_unirepLK(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            if self.Decom:
                self.LoRA = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size, small_kernel),
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=groups,
                )
            else:
                self.lkb_origin = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=groups,
                )

            if (small_kernel is not None) and small_kernel < kernel_size:
                self.small_conv = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,
                    groups=groups,
                    dilation=1,
                )

        self.bn = get_bn_unirepLK(out_channels)
        self.act = nn.SiLU()

    def forward(self, inputs):
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(inputs)
        elif self.Decom:
            out = self.LoRA(inputs)
            if hasattr(self, "small_conv"):
                out += self.small_conv(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, "small_conv"):
                out += self.small_conv(inputs)
        return self.act(self.bn(out))

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_conv_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = fuse_conv_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def switch_to_deploy(self):
        if hasattr(self, 'lkb_origin'):
            eq_k, eq_b = self.get_equivalent_kernel_bias()
            self.lkb_reparam = get_conv2d_unirepLK(
                in_channels=self.lkb_origin.conv.in_channels,
                out_channels=self.lkb_origin.conv.out_channels,
                kernel_size=self.lkb_origin.conv.kernel_size[0],
                stride=self.lkb_origin.conv.stride,
                padding=self.lkb_origin.conv.padding,
                dilation=self.lkb_origin.conv.dilation,
                groups=self.lkb_origin.conv.groups,
                bias=True,
            )
            self.lkb_reparam.weight.data = eq_k
            self.lkb_reparam.bias.data = eq_b
            self.__delattr__("lkb_origin")
            if hasattr(self, "small_conv"):
                self.__delattr__("small_conv")


class Bottleneck_SWC(nn.Module):
    """使用ReparamLargeKernelConv的Bottleneck

    源代码位置: block.py:4213
    """
    def __init__(self, c1, c2, kernel_size, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ReparamLargeKernelConv(c2, c2, kernel_size, groups=(c2 // 16))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_SWC(C3k):
    """使用Bottleneck_SWC的C3k模块

    源代码位置: block.py:4221
    """
    def __init__(self, c1, c2, n=1, kernel_size=13, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_SWC(c_, c_, kernel_size, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ================================ Batch 6 - AggregatedAttention模块 ================================
# 注意: TransNeXt_AggregatedAttention 已在 Batch 5 的 MLCA 模块之后定义在attention.py中
# 这里简化处理，在实际使用时会从正确位置导入

class Bottleneck_AggregatedAttention(nn.Module):
    """使用AggregatedAttention的Bottleneck

    源代码位置: block.py:3746
    依赖: 需要从attention.py导入TransNeXt_AggregatedAttention
    """
    def __init__(self, c1, c2, input_resolution, sr_ratio, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        # 延迟导入TransNeXt_AggregatedAttention以避免循环依赖
        from ultralytics.nn.extraction.c3k2_base import MLCA  # 验证可以导入

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        # TransNeXt_AggregatedAttention的实际导入在运行时处理
        # 占位符 - 实际使用时需替换
        self.attention = nn.Identity()
        self.input_resolution = input_resolution
        self.sr_ratio = sr_ratio

    def forward(self, x):
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C3k_AggregatedAtt(C3k):
    """使用Bottleneck_AggregatedAttention的C3k模块

    源代码位置: block.py:3757
    """
    def __init__(self, c1, c2, n=1, input_resolution=None, sr_ratio=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_AggregatedAttention(c_, c_, input_resolution, sr_ratio, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ================================ Batch 7 - iRMB和DynamicConv模块 ================================

# drop_path 函数
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample

    源代码位置: block.py:2158
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class SEAttention(nn.Module):
    """Squeeze-and-Excitation注意力模块

    源代码位置: attention.py:896
    """
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Conv2d_BN(torch.nn.Sequential):
    """卷积+BN组合层（用于EfficientViT风格注意力）

    源代码位置: attention.py:1581
    """
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def switch_to_deploy(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class CascadedGroupAttention(torch.nn.Module):
    """级联组注意力（EfficientViT）

    源代码位置: attention.py:1604
    """
    def __init__(self, dim, key_dim, num_heads=4,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5]):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = dim // num_heads
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(torch.nn.Module):
    """局部窗口注意力（EfficientViT）

    源代码位置: attention.py:1683
    """
    def __init__(self, dim, key_dim=16, num_heads=4,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution

        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio,
                                resolution=window_resolution,
                                kernels=kernels)

    def forward(self, x):
        B, C, H, W = x.shape

        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            x = self.attn(x)
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.permute(0, 3, 1, 2)

        return x


# ================================ iRMB系列模块 ================================

class iRMB(nn.Module):
    """Inverted Residual Mobile Block（倒残差移动块）

    源代码位置: block.py:4236
    """
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
                 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=16, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        self.act = Conv.default_act if act else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = nn.Conv2d(dim_in, int(dim_in * 2), 1, bias=qkv_bias)
            self.v = nn.Sequential(
                nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                self.act
            )
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = nn.Sequential(
                    nn.Conv2d(dim_in, dim_mid, kernel_size=1, bias=qkv_bias),
                    self.act
                )
            else:
                self.v = nn.Identity()
        self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid)
        self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj_drop(x)
        x = self.proj(x)
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class iRMB_Cascaded(nn.Module):
    """iRMB with Cascaded Group Attention

    源代码位置: block.py:4323
    """
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
                 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, num_head=16, se_ratio=0.0,
                 attn_s=True, qkv_bias=False, drop=0., drop_path=0., v_group=False):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        self.act = Conv.default_act if act else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        self.num_head = num_head
        if self.attn_s:
            self.attn = LocalWindowAttention(dim_mid)
        else:
            if v_proj:
                self.v = nn.Sequential(
                    nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                    self.act
                )
            else:
                self.v = nn.Identity()
        self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid)
        self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        if self.attn_s:
            x = self.attn(x)
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj_drop(x)
        x = self.proj(x)
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class iRMB_DRB(nn.Module):
    """iRMB with DilatedReparamBlock

    源代码位置: block.py:4368
    """
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
                 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=16, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        self.act = Conv.default_act if act else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = nn.Conv2d(dim_in, int(dim_in * 2), 1, bias=qkv_bias)
            self.v = nn.Sequential(
                nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                self.act
            )
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = nn.Sequential(
                    nn.Conv2d(dim_in, dim_mid, kernel_size=1, bias=qkv_bias),
                    self.act
                )
            else:
                self.v = nn.Identity()
        self.conv_local = DilatedReparamBlock(dim_mid, dim_mid, dw_ks, stride=stride, groups=(dim_mid // 16))
        self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj_drop(x)
        x = self.proj(x)
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class iRMB_SWC(nn.Module):
    """iRMB with Shift-Wise Conv (SWC)

    源代码位置: block.py:4455
    """
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
                 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=16, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        self.act = Conv.default_act if act else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = nn.Conv2d(dim_in, int(dim_in * 2), 1, bias=qkv_bias)
            self.v = nn.Sequential(
                nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                self.act
            )
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = nn.Sequential(
                    nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                    self.act
                )
            else:
                self.v = nn.Identity()
        self.conv_local = ReparamLargeKernelConv(dim_mid, dim_mid, dw_ks, stride=stride, groups=(dim_mid // 16))
        self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj_drop(x)
        x = self.proj(x)
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


# iRMB的C3k包装类
class C3k_iRMB(C3k):
    """使用iRMB模块的C3k

    源代码位置: block.py:4542
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(iRMB(c_, c_) for _ in range(n)))


class C3k_iRMB_Cascaded(C3k):
    """使用iRMB_Cascaded模块的C3k

    源代码位置: block.py:4553
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(iRMB_Cascaded(c_, c_) for _ in range(n)))


class C3k_iRMB_DRB(C3k):
    """使用iRMB_DRB模块的C3k

    源代码位置: block.py:4564
    """
    def __init__(self, c1, c2, n=1, kernel_size=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(iRMB_DRB(c_, c_, dw_ks=kernel_size) for _ in range(n)))


class C3k_iRMB_SWC(C3k):
    """使用iRMB_SWC模块的C3k

    源代码位置: block.py:4575
    """
    def __init__(self, c1, c2, n=1, kernel_size=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(iRMB_SWC(c_, c_, dw_ks=kernel_size) for _ in range(n)))


# ================================ DynamicConv系列模块 ================================

class DynamicConv_Single(nn.Module):
    """Dynamic Convolution Single Layer

    源代码位置: block.py:4890
    """
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding='', dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                 groups, bias, num_experts)

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        x = self.cond_conv(x, routing_weights)
        return x


class DynamicConv(nn.Module):
    """Dynamic Convolution with BatchNorm and Activation

    源代码位置: block.py:4906
    """
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, num_experts=4):
        super().__init__()
        self.conv = nn.Sequential(
            DynamicConv_Single(c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d), dilation=d, groups=g, num_experts=num_experts),
            nn.BatchNorm2d(c2),
            self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class GhostModule(nn.Module):
    """Ghost Module with Dynamic Convolution

    源代码位置: block.py:4919
    """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, act_layer=nn.SiLU, num_experts=4):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = DynamicConv(inp, init_channels, kernel_size, stride, num_experts=num_experts)
        self.cheap_operation = DynamicConv(init_channels, new_channels, dw_size, 1, g=init_channels, num_experts=num_experts)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class Bottleneck_DynamicConv(Bottleneck):
    """Bottleneck with Dynamic Convolution

    源代码位置: block.py:4936
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv2 = DynamicConv(c2, c2, 3)


class C3k_DynamicConv(C3k):
    """使用Bottleneck_DynamicConv的C3k模块

    源代码位置: block.py:4942
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DynamicConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k_GhostDynamicConv(C3k):
    """使用GhostModule的C3k模块

    源代码位置: block.py:4953
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostModule(c_, c_) for _ in range(n)))


# ================================ Batch 8 - RepViT模块 ================================

class RepVGGDW(torch.nn.Module):
    """RepVGG Depthwise卷积块

    源代码位置: modules/block.py:755
    """
    def __init__(self, ed) -> None:
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        from ultralytics.nn.modules.conv import fuse_conv_and_bn
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])
        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        self.conv = conv
        del self.conv1


class RepViTBlock(nn.Module):
    """RepViT Block

    源代码位置: block.py:4968
    注意: SqueezeExcite从timm库导入
    """
    def __init__(self, inp, oup, use_se=True):
        super(RepViTBlock, self).__init__()
        from timm.models.layers import SqueezeExcite

        self.identity = inp == oup
        hidden_dim = 2 * inp

        self.token_mixer = nn.Sequential(
            RepVGGDW(inp),
            SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
        )
        self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViTBlock_EMA(RepViTBlock):
    """RepViT Block with EMA

    源代码位置: block.py:4990
    """
    def __init__(self, inp, oup, use_se=True):
        super().__init__(inp, oup, use_se)

        self.token_mixer = nn.Sequential(
            RepVGGDW(inp),
            EMA(inp) if use_se else nn.Identity(),
        )


class C3k_RVB(C3k):
    """使用RepViTBlock的C3k模块

    源代码位置: block.py:4999
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(RepViTBlock(c_, c_, False) for _ in range(n)))


class C3k_RVB_SE(C3k):
    """使用RepViTBlock（带SE）的C3k模块

    源代码位置: block.py:5010
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(RepViTBlock(c_, c_) for _ in range(n)))


class C3k_RVB_EMA(C3k):
    """使用RepViTBlock_EMA的C3k模块

    源代码位置: block.py:5021
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(RepViTBlock_EMA(c_, c_) for _ in range(n)))


# ===================== Batch 9: PKIModule, PPA, Faster_CGLU, Star =====================

class PKIModule_CAA(nn.Module):
    """Poly Kernel Inception模块的上下文聚合注意力

    源代码位置: block.py:5130-5143
    """
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11) -> None:
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


class PKIModule(nn.Module):
    """Poly Kernel Inception模块

    源代码位置: block.py:5145-5179
    """
    def __init__(self, inc, ouc, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True) -> None:
        super().__init__()
        hidc = make_divisible(int(ouc * expansion), 8)

        self.pre_conv = Conv(inc, hidc)
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc, hidc, kernel_size=k, padding=autopad(k), groups=hidc) for k in kernel_sizes)
        self.pw_conv = Conv(hidc, hidc)
        self.post_conv = Conv(hidc, ouc)

        if with_caa:
            self.caa_factor = PKIModule_CAA(hidc, caa_kernel_size, caa_kernel_size)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and inc == ouc

    def forward(self, x):
        x = self.pre_conv(x)

        y = x
        x = self.dw_conv[0](x)
        x = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv[1:]], dim=0), dim=0)
        x = self.pw_conv(x)

        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x


class C3k_PKIModule(C3k):
    """使用PKIModule的C3k模块

    源代码位置: block.py:5181-5185
    """
    def __init__(self, c1, c2, n=1, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, True, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PKIModule(c_, c_, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) for _ in range(n)))


# PPA相关类（从hcfnet.py迁移）

class SpatialAttentionModule(nn.Module):
    """空间注意力模块

    源代码位置: hcfnet.py:9-20
    """
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x


class LocalGlobalAttention(nn.Module):
    """局部-全局注意力模块

    源代码位置: hcfnet.py:22-62
    """
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class ECA(nn.Module):
    """高效通道注意力模块

    源代码位置: hcfnet.py:64-81
    """
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        import math
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class PPA(nn.Module):
    """并行Patch感知注意力模块

    源代码位置: hcfnet.py:84-114
    """
    def __init__(self, in_features, filters) -> None:
        super().__init__()

        self.skip = Conv(in_features, filters, act=False)
        self.c1 = Conv(filters, filters, 3)
        self.c2 = Conv(filters, filters, 3)
        self.c3 = Conv(filters, filters, 3)
        self.sa = SpatialAttentionModule()
        self.cn = ECA(filters)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        self.drop = nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.silu = nn.SiLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.silu(x)
        return x


class C3k_PPA(C3k):
    """使用PPA的C3k模块

    源代码位置: block.py:5273-5277
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PPA(c_, c_) for _ in range(n)))


# Faster_CGLU相关类

## ConvolutionalGLU 定义移除，使用 common_base 版本（见顶部别名）


class Faster_Block_CGLU(nn.Module):
    """带卷积GLU的Faster块

    源代码位置: block.py:5809-5856
    """
    def __init__(self, inc, dim, n_div=4, mlp_ratio=2, drop_path=0.1, layer_scale_init_value=0.0, pconv_fw_type='split_cat'):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        self.mlp = ConvolutionalGLU(dim)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class C3k_Faster_CGLU(C3k):
    """使用Faster_Block_CGLU的C3k模块

    源代码位置: block.py:5858-5862
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Faster_Block_CGLU(c_, c_) for _ in range(n)))


# Star相关类

class CAA(nn.Module):
    """上下文锚点注意力模块

    源代码位置: attention.py:1765-1778
    """
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11) -> None:
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor * x


class Star_Block(nn.Module):
    """Star操作块

    源代码位置: block.py:6005-6023
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class Star_Block_CAA(Star_Block):
    """带CAA的Star操作块

    源代码位置: block.py:6025-6038
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0):
        super().__init__(dim, mlp_ratio, drop_path)

        self.attention = CAA(mlp_ratio * dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(self.attention(x)))
        x = input + self.drop_path(x)
        return x


class C3k_Star(C3k):
    """使用Star_Block的C3k模块

    源代码位置: block.py:6040-6044
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Star_Block(c_) for _ in range(n)))


# ===================== Batch 10: Star_CAA, EIEM, DEConv =====================

class C3k_Star_CAA(C3k):
    """使用Star_Block_CAA的C3k模块

    源代码位置: block.py:6051-6055
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Star_Block_CAA(c_) for _ in range(n)))


# EIEM相关类

class SobelConv(nn.Module):
    """Sobel边缘检测卷积

    源代码位置: block.py:6101-6119
    """
    def __init__(self, channel) -> None:
        super().__init__()

        import numpy as np
        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)

        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()

        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :, 0]


class EIEM(nn.Module):
    """边缘信息增强模块

    源代码位置: block.py:6141-6156
    """
    def __init__(self, inc, ouc) -> None:
        super().__init__()

        self.sobel_branch = SobelConv(inc)
        self.conv_branch = Conv(inc, inc, 3)
        self.conv1 = Conv(inc * 2, inc, 1)
        self.conv2 = Conv(inc, ouc, 1)

    def forward(self, x):
        x_sobel = self.sobel_branch(x)
        x_conv = self.conv_branch(x)
        x_concat = torch.cat([x_sobel, x_conv], dim=1)
        x_feature = self.conv1(x_concat)
        x = self.conv2(x_feature + x)
        return x


class C3k_EIEM(C3k):
    """使用EIEM的C3k模块

    源代码位置: block.py:6158-6162
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(EIEM(c_, c_) for _ in range(n)))


# DEConv相关类（从deconv.py迁移）

class Conv2d_cd(nn.Module):
    """中心差分卷积

    源代码位置: deconv.py:8-28
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        from einops.layers.torch import Rearrange
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.Rearrange = Rearrange

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = self.Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        if conv_weight.is_cuda:
            conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        else:
            conv_weight_cd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_cd = conv_weight_cd.to(conv_weight.dtype)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = self.Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    """角度差分卷积

    源代码位置: deconv.py:31-45
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_ad, self).__init__()
        from einops.layers.torch import Rearrange
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.Rearrange = Rearrange

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = self.Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = self.Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_hd(nn.Module):
    """水平差分卷积

    源代码位置: deconv.py:79-97
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_hd, self).__init__()
        from einops.layers.torch import Rearrange
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.Rearrange = Rearrange

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        if conv_weight.is_cuda:
            conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        else:
            conv_weight_hd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_hd = conv_weight_hd.to(conv_weight.dtype)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = self.Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    """垂直差分卷积

    源代码位置: deconv.py:100-118
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        from einops.layers.torch import Rearrange
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.Rearrange = Rearrange

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        if conv_weight.is_cuda:
            conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        else:
            conv_weight_vd = torch.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_vd = conv_weight_vd.to(conv_weight.dtype)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = self.Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class DEConv(nn.Module):
    """方向增强卷积

    源代码位置: deconv.py:121-165
    """
    def __init__(self, dim):
        super(DEConv, self).__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

        self.bn = nn.BatchNorm2d(dim)
        self.act = Conv.default_act

    def forward(self, x):
        if hasattr(self, 'conv1_1'):
            w1, b1 = self.conv1_1.get_weight()
            w2, b2 = self.conv1_2.get_weight()
            w3, b3 = self.conv1_3.get_weight()
            w4, b4 = self.conv1_4.get_weight()
            w5, b5 = self.conv1_5.weight, self.conv1_5.bias

            w = w1 + w2 + w3 + w4 + w5
            b = b1 + b2 + b3 + b4 + b5
            res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)
        else:
            res = self.conv1_5(x)

        if hasattr(self, 'bn'):
            res = self.bn(res)

        return self.act(res)


class Bottleneck_DEConv(Bottleneck):
    """使用DEConv的Bottleneck

    源代码位置: block.py:6199-6206
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DEConv(c_)


class C3k_DEConv(C3k):
    """使用Bottleneck_DEConv的C3k模块

    源代码位置: block.py:6208-6212
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DEConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# ===================== Batch 11: gConv和Additive注意力模块 =====================

class gConvBlock(nn.Module):
    """
    门控卷积块 (Gated Convolution Block)
    使用门控机制调制特征，包含值分支和门控分支

    源代码位置: extra_modules/block.py:7423-7462
    """
    def __init__(self, dim, kernel_size=3, gate_act=nn.Sigmoid, net_depth=8):
        super().__init__()
        self.dim = dim
        self.net_depth = net_depth
        self.kernel_size = kernel_size

        self.norm_layer = nn.BatchNorm2d(dim)

        # 值分支：1x1卷积 + 深度可分离卷积
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect')
        )

        # 门控分支：1x1卷积 + 激活函数
        self.Wg = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
        )

        self.proj = nn.Conv2d(dim, dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化，使用截断正态分布"""
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.net_depth) ** (-1/4)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        iden = X
        X = self.norm_layer(X)
        # 门控机制：值 * 门控
        out = self.Wv(X) * self.Wg(X)
        out = self.proj(out)
        return out + iden


class C3k_gConv(C3k):
    """
    使用gConvBlock的C3k模块

    源代码位置: extra_modules/block.py:7464-7468
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(gConvBlock(c_) for _ in range(n)))


## Mlp_CASVIT 定义移除，使用 common_base 版本（见顶部别名）


## SpatialOperation 定义移除，使用 common_base 版本（见顶部别名）


## ChannelOperation 定义移除，使用 common_base 版本（见顶部别名）


## LocalIntegration 定义移除，使用 common_base 版本（见顶部别名）


## AdditiveTokenMixer 定义移除，使用 common_base 版本（见顶部别名）


## AdditiveBlock 定义移除，使用 common_base 版本（见顶部别名）


## AdditiveBlock_CGLU 定义移除，使用 common_base 版本（见顶部别名）


class C3k_AdditiveBlock(C3k):
    """
    使用AdditiveBlock的C3k模块

    源代码位置: extra_modules/block.py:7596-7600
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(AdditiveBlock(c_) for _ in range(n)))


class C3k_AdditiveBlock_CGLU(C3k):
    """
    使用AdditiveBlock_CGLU的C3k模块

    源代码位置: extra_modules/block.py (推断)
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(AdditiveBlock_CGLU(c_) for _ in range(n)))


# ===================== Retention block 支撑 =====================
class C3k_RetBlock(C3k):
    def __init__(self, c1, c2, n=1, retention='chunk', num_heads=8, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.retention = retention
        self.relpos = RelPos2d(c_, num_heads, 2, 4)
        self.m = nn.ModuleList(RetBlock(retention, c_, num_heads, c_) for _ in range(n))

    def forward(self, x):
        b, c, h, w = x.size()
        rel_pos = self.relpos((h, w), chunkwise_recurrent=self.retention == 'chunk')
        y = list(self.cv1(x).chunk(2, 1))
        for layer in self.m:
            y.append(layer(y[-1].permute(0, 2, 3, 1), None, self.retention == 'chunk', rel_pos).permute(0, 3, 1, 2))
        return self.cv2(torch.cat(y, 1))


class C3k_Heat(C3k):
    def __init__(self, c1, c2, n=1, feat_size=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(HeatBlock(c_, res=feat_size or 14) for _ in range(n)))


class C3k_WTConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(WTConv2d(c_, c_) for _ in range(n)))


class C3k_FMB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(FMB(c_) for _ in range(n)))


class C3k_MSMHSA_CGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(MSMHSA_CGLU(c_) for _ in range(n)))


class C3k_MogaBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(MogaBlock(c_) for _ in range(n)))


class C3k_SHSA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(SHSABlock(c_) for _ in range(n)))


class C3k_SHSA_CGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(SHSABlock_CGLU(c_) for _ in range(n)))


class C3k_MutilScaleEdgeInformationEnhance(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(MutilScaleEdgeInformationEnhance(c_, [3, 6, 9, 12]) for _ in range(n)))


class C3k_MutilScaleEdgeInformationSelect(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(MutilScaleEdgeInformationSelect(c_, [3, 6, 9, 12]) for _ in range(n)))


class C3k_FFCM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Fused_Fourier_Conv_Mixer(c_) for _ in range(n)))


class C3k_SMAFB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(SMAFormerBlock(c_) for _ in range(n)))


class C3k_SMAFB_CGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(SMAFormerBlock_CGLU(c_) for _ in range(n)))


class C3k_MSM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(DeepPoolLayer(c_) for _ in range(n)))


class C3k_HDRAB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(HDRAB(c_, c_) for _ in range(n)))


class C3k_RAB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(RAB(c_, c_) for _ in range(n)))


class C3k_LFE(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(LFE(c_, c_) for _ in range(n)))


class Bottleneck_IDWC(Bottleneck):
    """InceptionDWConv2d 版瓶颈块（源自 upstream block.py:10454）。"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.cv1 = InceptionDWConv2d(c1)
        self.cv2 = InceptionDWConv2d(c2)


class C3k_IDWC(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_IDWC(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k_IDWB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(MetaNeXtBlock(c_) for _ in range(n)))


class C3k_CAMixer(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CAMixer(c_) for _ in range(n)))

# ---- 统一导出到公共基础实现，确保单一定义来源 ----
ConvolutionalGLU = _CommonConvolutionalGLU
Mlp_CASVIT = _CommonMlp_CASVIT
SpatialOperation = _CommonSpatialOperation
ChannelOperation = _CommonChannelOperation
LocalIntegration = _CommonLocalIntegration
AdditiveTokenMixer = _CommonAdditiveTokenMixer
AdditiveBlock = _CommonAdditiveBlock
AdditiveBlock_CGLU = _CommonAdditiveBlock_CGLU
