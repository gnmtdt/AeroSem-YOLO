# Ultralytics YOLOMM - LSPCD Head Module
# LSPCD: Lightweight Shared Partial Convolutional Detection Head
#
# 来源: 参考库 Ultralytics_674595707/ultralytics/nn/extra_modules/head/LSPCD.py
# 与已有的 LSCD 系列（基于 Conv_GN）不同，LSPCD 使用 Partial_Conv 构建共享 stem，
# 设计理念更加轻量。本文件包含 Detect/Segment/Pose/OBB 及其 YOLO26 变体共 7 个头。
#
# 注意：参考库的 LSPCD 继承自 Detect 基类（支持 end2end、forward_share_head 等新接口），
# 但本项目 Detect 基类接口不同，因此 LSPCD 在此独立实现（继承 nn.Module），
# 仅在 parse_model 层面注册为 DETECT_CLASS / SEGMENT_CLASS 等以获得完整的训练/推理支持。

import copy
import math

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, DFL, Proto
from ultralytics.nn.modules.block import Proto26, RealNVP
from ultralytics.nn.modules.conv import autopad
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors

__all__ = [
    'Partial_Conv',
    'Conv_GN',
    'Scale',
    'Detect_LSPCD',
    'Segment_LSPCD',
    'Segment26_LSPCD',
    'OBB_LSPCD',
    'OBB26_LSPCD',
    'Pose_LSPCD',
    'Pose26_LSPCD',
]


class Partial_Conv(nn.Module):
    """部分卷积模块 (Partial Convolution).

    来自 FasterNet (CVPR2023): https://arxiv.org/pdf/2303.03667
    仅对前 dim//n_div 个通道做 3x3 卷积，其余通道直通，最后通过可选的 1x1 卷积适配输出通道。

    Args:
        inc (int): 输入通道数
        ouc (int): 输出通道数
        n_div (int): 通道分割数，仅前 inc//n_div 个通道参与卷积
        forward (str): 前向模式，'split_cat' 或 'slicing'
    """

    def __init__(self, inc, ouc, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = inc // n_div
        self.dim_untouched = inc - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if inc != ouc:
            self.conv1x1 = Conv(inc, ouc, k=1)
        else:
            self.conv1x1 = nn.Identity()

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError(f'Partial_Conv 不支持的前向模式: {forward}')

    def forward_slicing(self, x):
        """推理模式：原地替换部分通道。"""
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return self.conv1x1(x)

    def forward_split_cat(self, x):
        """训练/推理通用模式：split-cat 方式。"""
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return self.conv1x1(x)


class Scale(nn.Module):
    """可学习的缩放参数。

    将输入乘以一个可学习的标量因子。

    Args:
        scale (float): 缩放因子初始值，默认 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class Conv_GN(nn.Module):
    """使用 GroupNorm 替代 BatchNorm 的标准卷积。

    Args:
        c1: 输入通道数
        c2: 输出通道数
        k: 卷积核大小
        s: 步长
        p: 填充（None 为自动填充）
        g: 分组卷积的组数
        d: 膨胀率
        act: 激活函数（True 使用默认 SiLU）
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """依次执行卷积、GroupNorm 和激活函数。"""
        return self.act(self.gn(self.conv(x)))


class Detect_LSPCD(nn.Module):
    """基于 Partial_Conv 的轻量共享检测头 (LSPCD)。

    核心设计：
    - 使用 Partial_Conv 构建共享 stem，大幅减少参数量
    - 所有检测层共享同一套 stem 卷积
    - 每层使用独立的可学习 Scale 参数
    - 支持 end2end 端到端检测模式

    Args:
        nc (int): 类别数
        reg_max (int): DFL 最大通道数
        end2end (bool): 是否启用端到端 NMS-free 检测
        ch (tuple): 各检测层的输入通道数
    """

    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False
    xyxy = False

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=...):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.end2end = bool(end2end)
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.c_hid = max(c2, c3)
        self.conv_adjust = nn.ModuleList(Conv_GN(x, self.c_hid) for x in ch)
        self.stem = nn.Sequential(
            Partial_Conv(self.c_hid, self.c_hid),
            Conv(self.c_hid, self.c_hid, 1),
            Partial_Conv(self.c_hid, self.c_hid),
            Conv(self.c_hid, self.c_hid, 1),
        )
        self.cv2 = nn.Conv2d(self.c_hid, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(self.c_hid, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_scale = copy.deepcopy(self.scale)

    @property
    def one2many(self):
        """返回一对多头组件。"""
        return dict(box_head=self.cv2, cls_head=self.cv3, scale_head=self.scale)

    @property
    def one2one(self):
        """返回一对一头组件。"""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, scale_head=self.one2one_scale)

    def forward_share_head(
        self, x, box_head=None, cls_head=None, scale_head=None
    ):
        """共享 stem 前向推理，返回 boxes/scores/feats 字典。"""
        if box_head is None or cls_head is None or scale_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat(
            [scale_head[i](box_head(x[i])).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1
        )
        scores = torch.cat(
            [cls_head(x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(self, x):
        """前向推理：共享 stem -> 分头预测 -> end2end 后处理。"""
        x = [self.stem(self.conv_adjust[i](x[i])) for i in range(len(self.conv_adjust))]
        preds = self.forward_share_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_share_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x):
        """推理解码：DFL 解码 + anchors + scores。"""
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x):
        """根据 anchors 和 strides 解码 bbox。"""
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """从预测值解码 bbox。"""
        return dist2bbox(bboxes, anchors, xywh=xywh and not self.end2end and not self.xyxy, dim=1)

    def bias_init(self):
        """初始化 Detect 头偏置，注意：需要 stride 已可用。"""
        self.one2many["box_head"].bias.data[:] = 2.0
        self.one2many["cls_head"].bias.data[:self.nc] = math.log(
            5 / self.nc / (640 / torch.mean(self.stride)) ** 2
        )
        if self.end2end:
            self.one2one["box_head"].bias.data[:] = 2.0
            self.one2one["cls_head"].bias.data[:self.nc] = math.log(
                5 / self.nc / (640 / torch.mean(self.stride)) ** 2
            )

    def postprocess(self, preds):
        """Postprocess end2end predictions to [x, y, w/h or x2/y2, score, cls]."""
        boxes, class_scores = preds.split([4, self.nc], dim=-1)
        scores, cls, anchor_idx = self.get_topk_index(class_scores, self.max_det)
        boxes = boxes.gather(dim=1, index=anchor_idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, cls], dim=-1)

    def get_topk_index(self, scores, max_det):
        """Return top-k scores, class ids, and anchor indices."""
        batch_size, anchors, nc = scores.shape
        k = max_det if self.export else min(max_det, anchors)
        anchor_idx = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        selected_scores = scores.gather(dim=1, index=anchor_idx.repeat(1, 1, nc))
        top_scores, flat_index = selected_scores.flatten(1).topk(k)
        top_anchor_idx = anchor_idx[torch.arange(batch_size, device=scores.device)[..., None], flat_index // nc]
        top_cls = (flat_index % nc)[..., None].float()
        return top_scores[..., None], top_cls, top_anchor_idx


class Segment_LSPCD(Detect_LSPCD):
    """LSPCD 分割头。

    在 LSPCD 检测头基础上增加掩码预测能力。

    Args:
        nc (int): 类别数
        nm (int): 掩码数量
        npr (int): 原型数量
        reg_max (int): DFL 最大通道数
        end2end (bool): 是否启用端到端检测
        ch (tuple): 各层通道数
    """

    def __init__(self, nc=80, nm=32, npr=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        self.cv4 = nn.Conv2d(self.c_hid, self.nm, 1)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, scale_head=self.scale, mask_head=self.cv4)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            scale_head=self.one2one_scale,
            mask_head=self.one2one_cv4,
        )

    def forward(self, x):
        """返回模型输出和掩码系数。"""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])
        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x):
        """推理解码，拼接掩码系数。"""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_share_head(self, x, box_head=None, cls_head=None, scale_head=None, mask_head=None):
        """共享 stem 前向推理，增加掩码系数输出。"""
        preds = super().forward_share_head(x, box_head, cls_head, scale_head)
        if mask_head is not None:
            bs = x[0].shape[0]
            preds["mask_coefficient"] = torch.cat(
                [mask_head(x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2
            )
        return preds

    def postprocess(self, preds):
        """后处理分割预测结果。"""
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self):
        """移除 one2many 头以优化推理。"""
        self.cv2 = self.cv3 = self.cv4 = self.scale = None


class Segment26_LSPCD(Segment_LSPCD):
    """LSPCD YOLO26 分割头。

    使用 Proto26 替代 Proto，支持 YOLO26 架构。

    Args:
        nc (int): 类别数
        nm (int): 掩码数量
        npr (int): 原型数量
        reg_max (int): DFL 最大通道数
        end2end (bool): 是否启用端到端检测
        ch (tuple): 各层通道数
    """

    def __init__(self, nc=80, nm=32, npr=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)

    def forward(self, x):
        """返回模型输出和 YOLO26 风格的掩码原型。"""
        outputs = Detect_LSPCD.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)
        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self):
        """移除 one2many 头及 Proto26 额外部分以优化推理。"""
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()


class OBB_LSPCD(Detect_LSPCD):
    """LSPCD 旋转目标检测头。

    在 LSPCD 检测头基础上增加旋转角度预测。

    Args:
        nc (int): 类别数
        ne (int): 额外参数数（角度）
        reg_max (int): DFL 最大通道数
        end2end (bool): 是否启用端到端检测
        ch (tuple): 各层通道数
    """

    def __init__(self, nc=80, ne=1, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne
        self.cv4 = nn.Conv2d(self.c_hid, self.ne, 1)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, scale_head=self.scale, angle_head=self.cv4)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            scale_head=self.one2one_scale,
            angle_head=self.one2one_cv4,
        )

    def _inference(self, x):
        """推理解码，拼接旋转角度。"""
        self.angle = x["angle"]
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def forward_share_head(self, x, box_head=None, cls_head=None, scale_head=None, angle_head=None):
        """共享 stem 前向推理，增加角度预测输出。"""
        preds = super().forward_share_head(x, box_head, cls_head, scale_head)
        if angle_head is not None:
            bs = x[0].shape[0]
            angle = torch.cat(
                [angle_head(x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )
            angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
            preds["angle"] = angle
        return preds

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """解码旋转边界框。"""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def postprocess(self, preds):
        """后处理 OBB 预测结果。"""
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)

    def fuse(self):
        """移除 one2many 头以优化推理。"""
        self.cv2 = self.cv3 = self.cv4 = self.scale = None


class OBB26_LSPCD(OBB_LSPCD):
    """LSPCD YOLO26 旋转目标检测头。

    与 OBB_LSPCD 不同，角度预测不经过 sigmoid 变换，直接输出原始角度值。

    Args:
        nc (int): 类别数
        ne (int): 额外参数数（角度）
        reg_max (int): DFL 最大通道数
        end2end (bool): 是否启用端到端检测
        ch (tuple): 各层通道数
    """

    def forward_share_head(self, x, box_head=None, cls_head=None, scale_head=None, angle_head=None):
        """共享 stem 前向推理，输出原始角度（不经过 sigmoid 变换）。"""
        preds = Detect_LSPCD.forward_share_head(self, x, box_head, cls_head, scale_head)
        if angle_head is not None:
            bs = x[0].shape[0]
            angle = torch.cat(
                [angle_head(x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )
            preds["angle"] = angle
        return preds


class Pose_LSPCD(Detect_LSPCD):
    """LSPCD 姿态估计头。

    在 LSPCD 检测头基础上增加关键点预测能力。

    Args:
        nc (int): 类别数
        kpt_shape (tuple): 关键点形状 (num_keypoints, dims)
        reg_max (int): DFL 最大通道数
        end2end (bool): 是否启用端到端检测
        ch (tuple): 各层通道数
    """

    def __init__(self, nc=80, kpt_shape=(17, 3), reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        self.cv4 = nn.Conv2d(self.c_hid, self.nk, 1)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, scale_head=self.scale, pose_head=self.cv4)

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            scale_head=self.one2one_scale,
            pose_head=self.one2one_cv4,
        )

    def _inference(self, x):
        """推理解码，拼接关键点。"""
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def forward_share_head(self, x, box_head=None, cls_head=None, scale_head=None, pose_head=None):
        """共享 stem 前向推理，增加关键点预测输出。"""
        preds = super().forward_share_head(x, box_head, cls_head, scale_head)
        if pose_head is not None:
            bs = x[0].shape[0]
            preds["kpts"] = torch.cat(
                [pose_head(x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2
            )
        return preds

    def postprocess(self, preds):
        """后处理姿态估计预测结果。"""
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    def fuse(self):
        """移除 one2many 头以优化推理。"""
        self.cv2 = self.cv3 = self.cv4 = self.scale = None

    def kpts_decode(self, kpts):
        """解码关键点预测值。"""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Pose26_LSPCD(Pose_LSPCD):
    """LSPCD YOLO26 姿态估计头。

    在 Pose_LSPCD 基础上增加 RealNVP 流模型和 sigma 预测。

    Args:
        nc (int): 类别数
        kpt_shape (tuple): 关键点形状 (num_keypoints, dims)
        reg_max (int): DFL 最大通道数
        end2end (bool): 是否启用端到端检测
        ch (tuple): 各层通道数
    """

    def __init__(self, nc=80, kpt_shape=(17, 3), reg_max=16, end2end=False, ch=()):
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = Conv(self.c_hid, c4, 3)

        self.cv4_kpts = nn.Conv2d(c4, self.nk, 1)
        self.nk_sigma = kpt_shape[0] * 2  # 每个关键点的 sigma_x, sigma_y
        self.cv4_sigma = nn.Conv2d(c4, self.nk_sigma, 1)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self):
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            scale_head=self.scale,
            pose_head=self.cv4,
            kpts_head=self.cv4_kpts,
            kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            scale_head=self.one2one_scale,
            pose_head=self.one2one_cv4,
            kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,
        )

    def forward_share_head(
        self, x, box_head=None, cls_head=None, scale_head=None,
        pose_head=None, kpts_head=None, kpts_sigma_head=None,
    ):
        """共享 stem 前向推理，增加 YOLO26 风格关键点和 sigma 预测。"""
        preds = Detect_LSPCD.forward_share_head(self, x, box_head, cls_head, scale_head)
        if pose_head is not None:
            bs = x[0].shape[0]
            features = [pose_head(x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat(
                [kpts_head(features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2
            )
            if self.training:
                preds["kpts_sigma"] = torch.cat(
                    [kpts_sigma_head(features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                )
        return preds

    def fuse(self):
        """移除 one2many 头和 flow_model 以优化推理。"""
        super().fuse()
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None

    def kpts_decode(self, kpts):
        """解码 YOLO26 风格关键点（注意：与 Pose_LSPCD 的解码方式不同）。"""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y
