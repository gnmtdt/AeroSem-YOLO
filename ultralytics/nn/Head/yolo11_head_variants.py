# Ultralytics YOLOMM - YOLO11 Head Variants (ported from YOLO11 extra_modules/head.py)
#
# 迁移约束：
# - 仅迁移“不需要编译/不依赖自定义 CUDA 扩展”的检测头家族。
# - 对于需要编译的头（DyHead+DCN 等），在迁移记录文档中单独列出并明确排除。

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ultralytics.nn.Neck import AFPN_P2345, AFPN_P2345_Custom, AFPN_P345, AFPN_P345_Custom
from ultralytics.nn.modules import Conv, DWConv, DFL, Detect, Proto
from ultralytics.nn.modules.conv import DSConv
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors

from ultralytics.nn.extraction.c3k2_base import DEConv, DiverseBranchBlock
from ultralytics.nn.public.seam import MultiSEAM, SEAM

from .lscd import Conv_GN, Detect_LSCD, Scale

__all__ = [
    # AFPN head family
    "Detect_AFPN_P345",
    "Detect_AFPN_P345_Custom",
    "Detect_AFPN_P2345",
    "Detect_AFPN_P2345_Custom",
    # Lightweight / structured heads
    "Detect_Efficient",
    "DetectAux",
    "Segment_Efficient",
    "Detect_SEAM",
    "Detect_MultiSEAM",
    # Alignment / decomposition head (pure PyTorch)
    "Detect_LADH",
    "Segment_LADH",
    "Pose_LADH",
    "OBB_LADH",
    # Lightweight shared-conv head family
    "Detect_LSCSBD",
    "Segment_LSCSBD",
    "Pose_LSCSBD",
    "OBB_LSCSBD",
    "Detect_LSDECD",
    "Segment_LSDECD",
    "Pose_LSDECD",
    "OBB_LSDECD",
    "Detect_RSCD",
    "Segment_RSCD",
    "Pose_RSCD",
    "OBB_RSCD",
    # Quality estimation heads
    "Detect_LQE",
    "Segment_LQE",
    "Pose_LQE",
    "OBB_LQE",
    "Detect_LSCD_LQE",
    "Segment_LSCD_LQE",
    "Pose_LSCD_LQE",
    "OBB_LSCD_LQE",
]


class Detect_AFPN_P345(nn.Module):
    """YOLO Detect head with AFPN(P3/P4/P5) neck inside."""

    dynamic = False
    export = False
    format = None
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc: int = 80, hidc: int = 256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], self.nc)
        self.afpn = AFPN_P345(ch, hidc)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(hidc, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for _ in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(hidc, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        x = self.afpn(x)
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class Detect_AFPN_P345_Custom(Detect_AFPN_P345):
    """YOLO Detect head with AFPN(P3/P4/P5) - custom block_type."""

    def __init__(self, nc: int = 80, hidc: int = 256, block_type=Conv, ch=()):
        super().__init__(nc, hidc, ch)
        self.afpn = AFPN_P345_Custom(ch, hidc, block_type, 4)


class Detect_AFPN_P2345(Detect_AFPN_P345):
    """YOLO Detect head with AFPN(P2/P3/P4/P5) neck inside."""

    def __init__(self, nc: int = 80, hidc: int = 256, ch=()):
        super().__init__(nc, hidc, ch)
        self.afpn = AFPN_P2345(ch, hidc)


class Detect_AFPN_P2345_Custom(Detect_AFPN_P345):
    """YOLO Detect head with AFPN(P2/P3/P4/P5) - custom block_type."""

    def __init__(self, nc: int = 80, hidc: int = 256, block_type=Conv, ch=()):
        super().__init__(nc, hidc, ch)
        self.afpn = AFPN_P2345_Custom(ch, hidc, block_type)


class Detect_Efficient(nn.Module):
    """YOLO Efficient head for detection models (group conv stems)."""

    dynamic = False
    export = False
    format = None
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc: int = 80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        self.stem = nn.ModuleList(
            nn.Sequential(Conv(x, x, 3, g=x // 16), Conv(x, x, 3, g=x // 16)) for x in ch
        )
        self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a.bias.data[:] = 1.0
            b.bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class DetectAux(nn.Module):
    """YOLO Detect head with auxiliary outputs (training-only extra heads)."""

    dynamic = False
    export = False
    format = None
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc: int = 80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch) // 2
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], self.nc)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[: self.nl]
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch[: self.nl]
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch[self.nl :]
        )
        self.cv5 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch[self.nl :]
        )
        self.dfl_aux = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            for i in range(self.nl, 2 * self.nl):
                x[i] = torch.cat((self.cv4[i - self.nl](x[i]), self.cv5[i - self.nl](x[i])), 1)
            return x
        if self.dynamic or self.shape != shape:
            if hasattr(self, "dfl_aux"):
                for i in range(self.nl, 2 * self.nl):
                    x[i] = torch.cat((self.cv4[i - self.nl](x[i]), self.cv5[i - self.nl](x[i])), 1)
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x[: self.nl], self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x[: self.nl]], 2)
        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x[: self.nl])

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
        for a, b, s in zip(self.cv4, self.cv5, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

    def switch_to_deploy(self):
        # 部署时删除辅助分支（与源实现一致）
        del self.cv4, self.cv5, self.dfl_aux


class Detect_SEAM(nn.Module):
    """YOLO Detect head with SEAM blocks."""

    dynamic = False
    export = False
    format = None
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc: int = 80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), SEAM(c2, c2, 1, 16), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                SEAM(c3, c3, 1, 16),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class Detect_MultiSEAM(Detect_SEAM):
    """YOLO Detect head with MultiSEAM blocks."""

    def __init__(self, nc: int = 80, ch=()):
        super().__init__(nc, ch)
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), MultiSEAM(c2, c2, 1), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                MultiSEAM(c3, c3, 1),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


class Segment_Efficient(Detect_Efficient):
    """Segmentation head based on Efficient detect head."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        self.detect = Detect_Efficient.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Detect_LADH(nn.Module):
    """Lightweight Alignment/Decomposition Head (pure PyTorch)."""

    dynamic = False
    export = False
    format = None
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc: int = 80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                DSConv(x, c2, 3),
                DSConv(c2, c2, 3),
                DSConv(c2, c2, 3),
                Conv(c2, c2, 1),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 1), Conv(c3, c3, 1), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


class Segment_LADH(Detect_LADH):
    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        self.detect = Detect_LADH.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(DSConv(x, c4, 3), DSConv(c4, c4, 3), Conv(c4, c4, 1), nn.Conv2d(c4, self.nm, 1)) for x in ch
        )

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose_LADH(Detect_LADH):
    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        self.detect = Detect_LADH.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(
            nn.Sequential(DSConv(x, c4, 3), DSConv(c4, c4, 3), Conv(c4, c4, 1), nn.Conv2d(c4, self.nk, 1)) for x in ch
        )

    def forward(self, x):
        bs = x[0].shape[0]
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        ndim = self.kpt_shape[1]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3] = y[:, 2::3].sigmoid()
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y


class OBB_LADH(Detect_LADH):
    def __init__(self, nc=80, ne=1, ch=()):
        super().__init__(nc, ch)
        self.ne = ne
        self.detect = Detect_LADH.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(DSConv(x, c4, 3), Conv(c4, c4, 1), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes):
        return dist2rbox(self.dfl(bboxes), self.angle, self.anchors.unsqueeze(0), dim=1) * self.strides


class Detect_LSCSBD(nn.Module):
    """
    Lightweight Shared Convolutional Separate BN Detection Head.

    说明：源实现中 `separate_bn` 的索引逻辑存在歧义。这里按“每个尺度、每个共享卷积 stage 各自 BN 分离”的语义实现：
    - share_conv 有 2 个 stage（3x3 + 1x1）
    - separate_bn: [nl][2]，每个尺度各自维护 2 个 BN
    """

    dynamic = False
    export = False
    format = None
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        self.conv = nn.ModuleList(nn.Sequential(Conv(x, hidc, 3)) for x in ch)
        self.share_conv = nn.ModuleList([nn.Conv2d(hidc, hidc, 3, 1, 1), nn.Conv2d(hidc, hidc, 1, 1, 0)])
        self.separate_bn = nn.ModuleList(
            nn.ModuleList([nn.BatchNorm2d(hidc), nn.BatchNorm2d(hidc)]) for _ in ch
        )
        self.act = nn.SiLU()
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            for j, conv in enumerate(self.share_conv):
                x[i] = self.act(self.separate_bn[i][j](conv(x[i])))
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        self.cv2.bias.data[:] = 1.0
        self.cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


class Segment_LSCSBD(Detect_LSCSBD):
    def __init__(self, nc=80, nm=32, npr=256, hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        self.detect = Detect_LSCSBD.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 1), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose_LSCSBD(Detect_LSCSBD):
    def __init__(self, nc=80, kpt_shape=(17, 3), hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        self.detect = Detect_LSCSBD.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 1), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        ndim = self.kpt_shape[1]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3] = y[:, 2::3].sigmoid()
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y


class OBB_LSCSBD(Detect_LSCSBD):
    def __init__(self, nc=80, ne=1, hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.ne = ne
        self.detect = Detect_LSCSBD.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 1), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes):
        return dist2rbox(self.dfl(bboxes), self.angle, self.anchors.unsqueeze(0), dim=1) * self.strides


class _DEConv_GN(DEConv):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.bn = nn.GroupNorm(16, dim)


class Detect_LSDECD(nn.Module):
    """Lightweight Shared Detail Enhanced Convolutional Detection Head (DEConv-enhanced)."""

    dynamic = False
    export = False
    format = None
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 1)) for x in ch)
        self.share_conv = nn.Sequential(_DEConv_GN(hidc), _DEConv_GN(hidc))
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        self.cv2.bias.data[:] = 1.0
        self.cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


class Segment_LSDECD(Detect_LSDECD):
    def __init__(self, nc=80, nm=32, npr=256, hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        self.detect = Detect_LSDECD.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv_GN(x, c4, 1), _DEConv_GN(c4), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose_LSDECD(Detect_LSDECD):
    def __init__(self, nc=80, kpt_shape=(17, 3), hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        self.detect = Detect_LSDECD.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 1), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        ndim = self.kpt_shape[1]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3] = y[:, 2::3].sigmoid()
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y


class OBB_LSDECD(Detect_LSDECD):
    def __init__(self, nc=80, ne=1, hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.ne = ne
        self.detect = Detect_LSDECD.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv_GN(x, c4, 1), _DEConv_GN(c4), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes):
        return dist2rbox(self.dfl(bboxes), self.angle, self.anchors.unsqueeze(0), dim=1) * self.strides


class Detect_RSCD(Detect_LSCD):
    """Variant of LSCD using DiverseBranchBlock for conv/share_conv."""

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.conv = nn.ModuleList(nn.Sequential(DiverseBranchBlock(x, hidc, 3)) for x in ch)
        self.share_conv = nn.Sequential(DiverseBranchBlock(hidc, hidc, 3, groups=hidc), Conv_GN(hidc, hidc, 1))


class Segment_RSCD(Detect_RSCD):
    def __init__(self, nc=80, nm=32, npr=256, hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        self.detect = Detect_RSCD.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv_GN(x, c4, 1), Conv_GN(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose_RSCD(Detect_RSCD):
    def __init__(self, nc=80, kpt_shape=(17, 3), hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        self.detect = Detect_RSCD.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 1), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        ndim = self.kpt_shape[1]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3] = y[:, 2::3].sigmoid()
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y


class OBB_RSCD(Detect_RSCD):
    def __init__(self, nc=80, ne=1, hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.ne = ne
        self.detect = Detect_RSCD.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv_GN(x, c4, 1), Conv_GN(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes):
        return dist2rbox(self.dfl(bboxes), self.angle, self.anchors.unsqueeze(0), dim=1) * self.strides


class LQEConvMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Conv2d(n, k, 1) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LQE(nn.Module):
    """
    位置质量估计器 (Location Quality Estimator, LQE)。
    用于评估和调整边界框预测的质量分数，结合分布统计信息提升精度。
    """

    def __init__(self, k: int, hidden_dim: int, num_layers: int, reg_max: int):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = LQEConvMLP(4 * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores: torch.Tensor, pred_corners: torch.Tensor) -> torch.Tensor:
        b, _, h, w = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(b, self.reg_max, 4, h, w), dim=1)
        prob_topk, _ = prob.topk(self.k, dim=1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=1, keepdim=True)], dim=1)
        quality_score = self.reg_conf(stat.reshape(b, -1, h, w))
        return scores + quality_score


class Detect_LQE(Detect):
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        self.lqe = nn.ModuleList(LQE(4, 64, 2, self.reg_max) for _ in ch)
        if self.end2end:
            self.one2one_lqe = copy.deepcopy(self.lqe)

    def forward(self, x):
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            pred_corners = self.cv2[i](x[i])
            pred_scores = self.lqe[i](self.cv3[i](x[i]), pred_corners)
            x[i] = torch.cat((pred_corners, pred_scores), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        one2one = [None for _ in range(self.nl)]
        for i in range(self.nl):
            pred_corners = self.one2one_cv2[i](x[i])
            pred_scores = self.one2one_lqe[i](self.one2one_cv3[i](x[i]), pred_corners)
            one2one[i] = torch.cat((pred_corners, pred_scores), 1)

        if hasattr(self, "cv2") and hasattr(self, "cv3"):
            for i in range(self.nl):
                pred_corners = self.cv2[i](x[i])
                pred_scores = self.lqe[i](self.cv3[i](x[i]), pred_corners)
                x[i] = torch.cat((pred_corners, pred_scores), 1)

        if self.training:
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})


class Segment_LQE(Detect_LQE):
    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = Detect_LQE.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB_LQE(Detect_LQE):
    def __init__(self, nc=80, ne=1, ch=()):
        super().__init__(nc, ch)
        self.ne = ne
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi
        if not self.training:
            self.angle = angle
        x = Detect_LQE.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose_LQE(Detect_LQE):
    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)
        x = Detect_LQE.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        ndim = self.kpt_shape[1]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3] = y[:, 2::3].sigmoid()
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y


class Detect_LSCD_LQE(nn.Module):
    """LSCD head + LQE quality estimation (pure PyTorch, no compilation)."""

    dynamic = False
    export = False
    format = None
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 3)) for x in ch)
        self.share_conv = nn.Sequential(Conv_GN(hidc, hidc, 3, g=hidc), Conv_GN(hidc, hidc, 1))
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.lqe = nn.ModuleList(LQE(4, 64, 2, self.reg_max) for _ in ch)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])
            pred_corners = self.scale[i](self.cv2(x[i]))
            pred_scores = self.lqe[i](self.cv3(x[i]), pred_corners)
            x[i] = torch.cat((pred_corners, pred_scores), 1)
        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        self.cv2.bias.data[:] = 1.0
        self.cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


class Segment_LSCD_LQE(Detect_LSCD_LQE):
    def __init__(self, nc=80, nm=32, npr=256, hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        self.detect = Detect_LSCD_LQE.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv_GN(x, c4, 1), Conv_GN(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose_LSCD_LQE(Detect_LSCD_LQE):
    def __init__(self, nc=80, kpt_shape=(17, 3), hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        self.detect = Detect_LSCD_LQE.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 1), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        ndim = self.kpt_shape[1]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3] = y[:, 2::3].sigmoid()
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y


class OBB_LSCD_LQE(Detect_LSCD_LQE):
    def __init__(self, nc=80, ne=1, hidc=256, ch=()):
        super().__init__(nc, hidc, ch)
        self.ne = ne
        self.detect = Detect_LSCD_LQE.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv_GN(x, c4, 1), Conv_GN(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes):
        return dist2rbox(self.dfl(bboxes), self.angle, self.anchors.unsqueeze(0), dim=1) * self.strides

