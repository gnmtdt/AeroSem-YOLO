# Ultralytics Multimodal Inference - Helper Utilities
# Utility functions for multimodal inference preprocessing
# Version: v1.0
# Date: 2026-01-13

import cv2
import numpy as np
import torch
from typing import Tuple
from ultralytics.utils import LOGGER
from .precision_contract import LEGACY


def align_and_validate_x(
    x_img: np.ndarray,
    rgb_img: np.ndarray,
    expected_xch: int
) -> np.ndarray:
    """
    对齐并校验X模态图像的空间尺寸和通道数

    Args:
        x_img: X模态图像（可能是 H×W, H×W×1, H×W×3, H×W×Xch等）
        rgb_img: RGB图像（H×W×3，作为尺寸基准）
        expected_xch: 期望的X模态通道数

    Returns:
        对齐后的X模态图像（H×W×Xch）

    Raises:
        ValueError: 通道数不匹配且无法转换
    """
    # (A) 空间对齐：以RGB为基准，将X resize到RGB的原始尺寸
    rgb_h, rgb_w = rgb_img.shape[:2]
    x_h, x_w = x_img.shape[:2]

    if (x_h, x_w) != (rgb_h, rgb_w):
        if len(x_img.shape) == 2:
            # 灰度图resize
            x_img = cv2.resize(x_img, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)
        else:
            # 多通道图resize
            x_img = cv2.resize(x_img, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)

    # (B) 通道数校验与整形
    # 实际通道数
    if len(x_img.shape) == 2:
        actual_xch = 1
        x_img = x_img[:, :, np.newaxis]  # 转为 H×W×1
    elif len(x_img.shape) == 3:
        actual_xch = x_img.shape[2]
    else:
        raise ValueError(f"X模态图像维度异常: {x_img.shape}（应为 H×W 或 H×W×C）")

    # 严格校验规则
    if expected_xch in {1, 3} and actual_xch in {1, 3}:
        # Xch∈{1,3} 且实际也是{1,3}：允许1↔3显式转换
        if actual_xch == 1 and expected_xch == 3:
            # 1->3: 灰度转三通道
            x_img = np.repeat(x_img, 3, axis=2)
        elif actual_xch == 3 and expected_xch == 1:
            # 3->1: RGB转灰度
            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        # actual == expected: 无需转换

    elif actual_xch == expected_xch:
        # 通道数完全匹配：无需转换
        pass

    else:
        # 不匹配且不在允许转换范围内：报错
        raise ValueError(
            f"X模态通道数不匹配：期望 {expected_xch} 通道，实际 {actual_xch} 通道。\n"
            f"仅当 Xch∈{{1,3}} 时允许 1↔3 显式转换。\n"
            f"对于 Xch>3，请提供严格匹配通道数的多通道文件（推荐 .tif/.npy/.npz）。"
        )

    # 最终验证
    if x_img.shape[2] != expected_xch:
        raise ValueError(
            f"X模态通道数校验失败：期望 {expected_xch}，实际 {x_img.shape[2]}"
        )

    return x_img


def letterbox_with_ratio_pad(
    letterbox_func,
    img: np.ndarray
) -> Tuple[np.ndarray, Tuple[float, Tuple[float, float]]]:
    """
    应用LetterBox并返回ratio_pad信息

    Args:
        letterbox_func: LetterBox函数实例
        img: 输入图像（H×W×C）

    Returns:
        (letterboxed_img, ratio_pad)
        - letterboxed_img: 填充后的图像
        - ratio_pad: (gain, (padw, padh))

    Notes:
        LetterBox返回格式：
        - 如果传入空字典/None：返回np.ndarray（仅图像）
        - 如果传入包含'img'的字典：返回dict（包含'img'等字段）
    """
    h0, w0 = img.shape[:2]

    # 调用LetterBox（传入空字典以避免labels处理）
    result = letterbox_func(labels={}, image=img)

    # LetterBox返回的是图像数组（因为labels为空字典）
    letterboxed_img = result
    h, w = letterboxed_img.shape[:2]

    # 手动计算ratio_pad（与scale_boxes还原坐标所需格式一致）
    # gain: 缩放比例
    gain = min(h / h0, w / w0) if h0 > 0 and w0 > 0 else 1.0

    # padding: 填充量（letterbox居中时为两侧总和的一半）
    # 根据LetterBox源码，center=True时dw/dh已经除以2
    padw = (w - w0 * gain) / 2
    padh = (h - h0 * gain) / 2

    ratio_pad = (gain, (padw, padh))

    return letterboxed_img, ratio_pad


def to_tensor_rgb(img: np.ndarray, precision: str = LEGACY) -> torch.Tensor:
    """RGB HWC -> CHW float32 tensor。

    Legacy: 按 dtype 归一化 (uint8/255, uint16/65535, float 直通)。
    High_precision: 显式高精度归一化路径。
    注意：不做BGR->RGB转换，保持与训练时Format._format_img()一致。
    """
    from .precision_contract import HIGH_PRECISION, normalize_high_precision_tensor

    if img.ndim < 3:
        img = img[..., None]
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
    if precision == LEGACY:
        if t.max() > 1.0:
            t = t / 255.0
    else:
        t = normalize_high_precision_tensor(t, modality="rgb")
    return t


def to_tensor_x(img: np.ndarray, precision: str = LEGACY) -> torch.Tensor:
    """X 模态 HWC -> CHW float32 tensor。

    Legacy: 按 dtype 归一化 (uint8/255, uint16/65535, float 直通)。
    High_precision: 显式高精度归一化路径。
    """
    from .precision_contract import HIGH_PRECISION, normalize_high_precision_tensor

    if img.ndim < 3:
        img = img[..., None]
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
    if precision == LEGACY:
        if img.dtype == np.uint8:
            t = t / 255.0
        elif img.dtype == np.uint16:
            t = t / 65535.0
        # float32 legacy X: 直通（保持原始值域）
    else:
        t = normalize_high_precision_tensor(t, modality="x")
    return t
