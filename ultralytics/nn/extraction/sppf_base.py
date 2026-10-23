"""
SPPF Extraction - Base Components
存放所有SPPF及空间池化变体所需的基础组件

包含内容：
- 标准 SPP/SPPF 模块
- SPP-ELAN 模块
- LSKA 注意力机制
- SPPF_LSKA 增强模块
- 金字塔池化聚合模块
- 特征融合池化模块
- 小波池化模块

迁移自 ultralytics-yolo11-main/ultralytics/nn/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入基础模块
from ultralytics.nn.modules.conv import Conv
# 依赖改为本目录内的轻量实现，避免跨包导出
from .rep_block import RepVGGBlock

__all__ = [
    # Batch 1 - 标准 SPP/SPPF 模块
    'SPP',
    'SPPF',
    'SPPELAN',
    'SPP_DAMO',
    'LSKA',
    'SPPF_LSKA',
    # Batch 2 - 金字塔池化聚合和融合模块
    'onnx_AdaptiveAvgPool2d',
    'get_avg_pool',
    'get_shape',
    'PyramidPoolAgg',
    'PyramidPoolAgg_PCE',
    'SimFusion_3in',
    'SimFusion_4in',
    'AdvPoolFusion',
    # Batch 3 - 小波池化和注入融合模块
    'h_sigmoid',
    'IFM',
    'InjectionMultiSum_Auto_pool',
    'WaveletPool',
    'WaveletUnPool',
]


# ===================== Batch 1: 标准 SPP/SPPF 模块 =====================

class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729.

    经典SPP实现，使用并行的多尺度MaxPool层。
    """

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 池化核尺寸元组，默认 (5, 9, 13)
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher.

    快速SPP实现，使用串行池化代替并行池化。
    等效于 SPP(k=(5, 9, 13)) 但计算效率更高。
    """

    def __init__(self, c1, c2, k=5):
        """Initializes the SPPF layer with given input/output channels and kernel size.

        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 池化核大小，默认 5

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through SPPF layer."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class SPPELAN(nn.Module):
    """SPP-ELAN.

    结合ELAN架构的SPP模块，提供更丰富的梯度流。
    适用于 YOLOv7/v9 风格网络。
    """

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling.

        Args:
            c1: 输入通道数
            c2: 输出通道数
            c3: 隐藏通道数
            k: 池化核大小，默认 5
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class SPP_DAMO(nn.Module):
    """DAMO-YOLO版本的SPP实现.

    支持自定义池化尺寸配置，用于 DAMO-YOLO GFPN 架构。
    """

    def __init__(self, ch_in, ch_out, k, pool_size):
        """初始化DAMO-SPP模块.

        Args:
            ch_in: 输入通道数
            ch_out: 输出通道数
            k: 卷积核大小
            pool_size: 池化尺寸列表，例如 [5, 9, 13]
        """
        super(SPP_DAMO, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size,
                                stride=1,
                                padding=size // 2,
                                ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = Conv(ch_in, ch_out, k)

    def forward(self, x):
        """前向传播，执行空间金字塔池化."""
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)
        y = self.conv(y)
        return y


class LSKA(nn.Module):
    """Large-Separable-Kernel-Attention.

    大核可分离注意力机制。
    参考: https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention/tree/main
    """

    def __init__(self, dim, k_size=7):
        """初始化LSKA模块.

        Args:
            dim: 输入特征维度
            k_size: 核大小，支持 7/11/23/35/41/53
        """
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """前向传播，应用大核可分离注意力."""
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn


class SPPF_LSKA(nn.Module):
    """Spatial Pyramid Pooling - Fast with LSKA attention.

    集成Large Separable Kernel Attention的SPPF变体。
    结合空间池化和大核注意力，特征表达能力显著提升。
    """

    def __init__(self, c1, c2, k=5):
        """初始化SPPF_LSKA模块.

        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 池化核大小，默认 5

        等效于 SPP(k=(5, 9, 13)) + LSKA注意力增强
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4, k_size=11)

    def forward(self, x):
        """前向传播，执行SPPF + LSKA注意力."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))


# ===================== Batch 2: 金字塔池化聚合和融合模块 =====================

# 辅助函数
def onnx_AdaptiveAvgPool2d(x, output_size):
    """ONNX导出兼容的自适应平均池化.

    Args:
        x: 输入张量
        output_size: 输出尺寸 (H, W)

    Returns:
        池化后的张量
    """
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x


def get_avg_pool():
    """获取自适应平均池化函数，支持ONNX导出.

    Returns:
        池化函数 (torch或onnx版本)
    """
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool


def get_shape(tensor):
    """获取张量形状，支持ONNX导出.

    Args:
        tensor: 输入张量

    Returns:
        张量形状
    """
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class PyramidPoolAgg(nn.Module):
    """金字塔池化聚合模块.

    多尺度输入自适应池化后拼接，用于GOLD-YOLO架构。
    支持ONNX导出优化。
    """

    def __init__(self, inc, ouc, stride, pool_mode='torch'):
        """初始化PyramidPoolAgg模块.

        Args:
            inc: 输入通道数
            ouc: 输出通道数
            stride: 下采样步长
            pool_mode: 池化模式，'torch' 或 'onnx'
        """
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
        self.conv = Conv(inc, ouc)

    def forward(self, inputs):
        """前向传播，执行金字塔池化聚合.

        Args:
            inputs: 多尺度特征列表

        Returns:
            聚合后的特征
        """
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1

        output_size = np.array([H, W])

        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d

        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d

        out = [self.pool(inp, output_size) for inp in inputs]

        return self.conv(torch.cat(out, dim=1))


class PyramidPoolAgg_PCE(nn.Module):
    """金字塔池化聚合PCE版本.

    简化版PyramidPoolAgg，去除卷积层，计算开销更小。
    """

    def __init__(self, stride=2):
        """初始化PyramidPoolAgg_PCE模块.

        Args:
            stride: 下采样步长，默认 2
        """
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        """前向传播，执行轻量化金字塔池化聚合.

        Args:
            inputs: 多尺度特征列表

        Returns:
            聚合后的特征（无卷积融合）
        """
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class SimFusion_3in(nn.Module):
    """3输入简化融合模块.

    自适应池化 + 上采样 + 通道对齐 + 拼接融合。
    用于GOLD-YOLO三尺度特征融合。
    """

    def __init__(self, in_channel_list, out_channels):
        """初始化SimFusion_3in模块.

        Args:
            in_channel_list: 输入通道列表 [c0, c1, c2]
            out_channels: 输出通道数
        """
        super().__init__()
        self.cv1 = Conv(in_channel_list[0], out_channels, act=nn.ReLU()) if in_channel_list[0] != out_channels else nn.Identity()
        self.cv2 = Conv(in_channel_list[1], out_channels, act=nn.ReLU()) if in_channel_list[1] != out_channels else nn.Identity()
        self.cv3 = Conv(in_channel_list[2], out_channels, act=nn.ReLU()) if in_channel_list[2] != out_channels else nn.Identity()
        self.cv_fuse = Conv(out_channels * 3, out_channels, act=nn.ReLU())
        self.downsample = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        """前向传播，执行三尺度特征融合.

        Args:
            x: 输入特征列表 [x0, x1, x2]

        Returns:
            融合后的特征
        """
        N, C, H, W = x[1].shape
        output_size = (H, W)

        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])

        x0 = self.cv1(self.downsample(x[0], output_size))
        x1 = self.cv2(x[1])
        x2 = self.cv3(F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False))
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))


class SimFusion_4in(nn.Module):
    """4输入简化融合模块.

    四尺度特征自适应池化融合。
    用于GOLD-YOLO四尺度特征融合。
    """

    def __init__(self):
        """初始化SimFusion_4in模块."""
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        """前向传播，执行四尺度特征融合.

        Args:
            x: 输入特征列表 [x_l, x_m, x_s, x_n]

        Returns:
            融合后的特征
        """
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])

        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d

        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out


class AdvPoolFusion(nn.Module):
    """高级池化融合模块.

    双输入自适应平均池化后拼接。
    轻量高效的双尺度特征融合。
    """

    def forward(self, x):
        """前向传播，执行双尺度特征融合.

        Args:
            x: 输入特征列表 [x1, x2]

        Returns:
            融合后的特征
        """
        x1, x2 = x
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d

        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)

        return torch.cat([x1, x2], 1)


# ===================== Batch 3: 小波池化和注入融合模块 =====================

class h_sigmoid(nn.Module):
    """Hard-Sigmoid激活函数.

    使用ReLU6实现的硬Sigmoid，计算效率更高。
    """

    def __init__(self, inplace=True):
        """初始化h_sigmoid模块.

        Args:
            inplace: 是否原地操作
        """
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        """前向传播."""
        return self.relu(x + 3) / 6


class IFM(nn.Module):
    """Injection Fusion Module - 注入融合模块.

    RepVGG 块串联 + 多输出分支，可重参数化设计。
    用于GOLD-YOLO注入式特征融合。
    """

    def __init__(self, inc, ouc, embed_dim_p=96, fuse_block_num=3) -> None:
        """初始化IFM模块.

        Args:
            inc: 输入通道数
            ouc: 输出通道数（列表），支持多输出分支
            embed_dim_p: 嵌入维度，默认 96
            fuse_block_num: 融合块数量，默认 3
        """
        super().__init__()

        self.conv = nn.Sequential(
            Conv(inc, embed_dim_p),
            *[RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
            Conv(embed_dim_p, sum(ouc))
        )

    def forward(self, x):
        """前向传播，执行注入融合."""
        return self.conv(x)


class InjectionMultiSum_Auto_pool(nn.Module):
    """自动池化注入多和融合模块.

    局部特征 + 全局池化特征加权融合。
    用于GOLD-YOLO全局-局部特征融合。
    """

    def __init__(
            self,
            inp: int,
            oup: int,
            global_inp: list,
            flag: int
    ) -> None:
        """初始化InjectionMultiSum_Auto_pool模块.

        Args:
            inp: 局部输入通道数
            oup: 输出通道数
            global_inp: 全局输入通道列表
            flag: 标志位，指定使用哪个全局输入
        """
        super().__init__()
        self.global_inp = global_inp
        self.flag = flag
        self.local_embedding = Conv(inp, oup, 1, act=False)
        self.global_embedding = Conv(global_inp[self.flag], oup, 1, act=False)
        self.global_act = Conv(global_inp[self.flag], oup, 1, act=False)
        self.act = h_sigmoid()

    def forward(self, x):
        """前向传播，执行全局-局部特征融合.

        Args:
            x: 输入元组 (x_l, x_g)
                x_l: 局部特征
                x_g: 全局特征

        Returns:
            融合后的特征
        """
        x_l, x_g = x
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        gloabl_info = x_g.split(self.global_inp, dim=1)[self.flag]

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(gloabl_info)
        global_feat = self.global_embedding(gloabl_info)

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


class WaveletPool(nn.Module):
    """小波池化 - Haar小波下采样.

    使用小波变换进行下采样，保留高频信息。
    输出4通道（LL/LH/HL/HH），适合频域分析。
    """

    def __init__(self):
        """初始化WaveletPool模块，使用固定Haar小波核."""
        super(WaveletPool, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                            hl[None,::-1,::-1], hh[None,::-1,::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        """前向传播，执行小波池化下采样.

        Args:
            x: 输入张量 (B, C, H, W)

        Returns:
            小波变换后的张量 (B, 4*C, H/2, W/2)
        """
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y


class WaveletUnPool(nn.Module):
    """小波反池化 - Haar小波上采样.

    使用小波逆变换进行上采样，与WaveletPool配对使用。
    保持频域信息的一致性。
    """

    def __init__(self):
        """初始化WaveletUnPool模块，使用固定Haar小波核."""
        super(WaveletUnPool, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                            hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        """前向传播，执行小波池化上采样.

        Args:
            x: 输入张量 (B, 4*C, H, W)

        Returns:
            小波逆变换后的张量 (B, C, 2*H, 2*W)
        """
        C = torch.floor_divide(x.shape[1], 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        return y
