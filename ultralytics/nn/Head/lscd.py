# Ultralytics YOLOMM - LSCD Head Module
# LSCD: Lightweight Shared Convolutional Detection Head

import math
import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, DFL, Proto
from ultralytics.nn.modules.conv import autopad
from ultralytics.utils.tal import dist2bbox, make_anchors, dist2rbox

__all__ = ['Scale', 'Conv_GN', 'Detect_LSCD', 'Segment_LSCD', 'Pose_LSCD', 'OBB_LSCD']


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        """Initialize Scale layer with initial scale value."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scale factor to input tensor."""
        return x * self.scale


class Conv_GN(nn.Module):
    """Standard convolution with GroupNorm instead of BatchNorm.

    Uses GroupNorm which has been shown in FCOS paper to improve
    detection head's localization and classification performance.

    Args:
        c1: Input channels
        c2: Output channels
        k: Kernel size
        s: Stride
        p: Padding (None for autopad)
        g: Groups for grouped convolution
        d: Dilation
        act: Activation function (True for default SiLU)
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with GroupNorm and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)  # Fixed 16 groups
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, group normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))


class Detect_LSCD(nn.Module):
    """YOLOv8 Detect head with Lightweight Shared Convolutional Detection.

    Key features:
    - Shared convolutional layers across detection heads to reduce parameters
    - GroupNorm instead of BatchNorm for better performance
    - Learnable scale parameters for each detection layer
    - Depthwise separable convolution for efficiency

    Args:
        nc: Number of classes
        hidc: Hidden channels for intermediate layers
        ch: Input channels from backbone/neck for each detection layer
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initialize LSCD detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        # Per-layer input convolution
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 3)) for x in ch)

        # Shared convolution across all layers (key innovation for parameter reduction)
        self.share_conv = nn.Sequential(
            Conv_GN(hidc, hidc, 3, g=hidc),  # Depthwise separable conv
            Conv_GN(hidc, hidc, 1)
        )

        # Output heads
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)  # bbox prediction
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)  # class prediction

        # Learnable scale for each detection layer
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)

        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # Detect() module
        m.cv2.bias.data[:] = 1.0  # box
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


class Segment_LSCD(Detect_LSCD):
    """YOLOv8 Segment head with LSCD for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, hidc=256, ch=()):
        """Initialize LSCD segmentation head with masks, prototypes, and convolution layers."""
        super().__init__(nc, hidc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect_LSCD.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv_GN(x, c4, 1), Conv_GN(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch
        )

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Pose_LSCD(Detect_LSCD):
    """YOLOv8 Pose head with LSCD for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), hidc=256, ch=()):
        """Initialize LSCD pose head with keypoint shape and convolutional layers."""
        super().__init__(nc, hidc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect_LSCD.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 1), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch
        )

    def forward(self, x):
        """Perform forward pass through LSCD pose model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class OBB_LSCD(Detect_LSCD):
    """YOLOv8 OBB detection head with LSCD for detection with rotation models."""

    def __init__(self, nc=80, ne=1, hidc=256, ch=()):
        """Initialize LSCD OBB head with number of classes and extra parameters."""
        super().__init__(nc, hidc, ch)
        self.ne = ne  # number of extra parameters
        self.detect = Detect_LSCD.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv_GN(x, c4, 1), Conv_GN(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch
        )

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes):
        """Decode rotated bounding boxes."""
        return dist2rbox(self.dfl(bboxes), self.angle, self.anchors.unsqueeze(0), dim=1) * self.strides
