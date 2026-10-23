# Ultralytics YOLO, AGPL-3.0 license

"""
多模态图像I/O复用层

本模块提供多模态图像加载、路径查找、对齐等功能的Mixin类，
可被不同任务的数据集类（如检测、分类、分割等）共享复用。

核心功能:
- X模态图像路径查找（同名不同扩展枚举）
- X模态图像加载（支持 npy/npz/tif/标准图像）
- RGB+X 图像空间对齐
- 通道数校验与转换

DType 策略 (precision contract routing):
- legacy 分支: 加载时返回原始数组（uint8/uint16/float 等），不提前归一化。
  归一化统一在 tensorize / preprocess 阶段按 contract 处理。
- high_precision 分支: 显式调用 normalize_modality_array()。
  仅由已经判定为 high_precision 的路径调用，不允许默认自动触发。
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from ultralytics.utils import LOGGER


def load_modality_array_raw(path: Union[str, Path], *, cv2_flag: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """
    纯加载 — 返回原始数组，不做任何归一化。

    这是 legacy 分支的标准加载路径。高精度分支应显式调用
    normalize_modality_array() 而非使用此函数。

    Args:
        path: 图像/数组文件路径
        cv2_flag: OpenCV 读取标志，默认 IMREAD_UNCHANGED

    Returns:
        原始 numpy 数组，保持文件原始 dtype 和值域

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 无法读取或解析文件
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")

    suffix = p.suffix.lower()

    if suffix == ".npy":
        return np.load(p)

    if suffix == ".npz":
        with np.load(p) as npz_file:
            for key in ("image", "arr_0", "array", "data"):
                if key in npz_file.files:
                    return npz_file[key]
            if len(npz_file.files) == 1:
                return npz_file[npz_file.files[0]]
            raise ValueError(
                f"npz文件 {p} 含多个数组 {npz_file.files}，"
                f"无法确定默认键，请使用标准键(image/arr_0)。"
            )

    if suffix in {".tif", ".tiff"}:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    else:
        arr = cv2.imread(str(p), cv2_flag)

    if arr is None:
        raise ValueError(f"无法读取图像: {p}")
    return arr


def normalize_modality_array(arr: np.ndarray, modality: str = "x") -> np.ndarray:
    """
    将模态 numpy 数组归一化为 float32 [0, 1] 范围。

    仅限 high_precision 分支显式调用。legacy 分支不允许使用此函数。

    Args:
        arr: HWC numpy 数组，任意数值 dtype
        modality: 模态标识 ('rgb' 或 'x')，用于日志消息

    Returns:
        float32 数组，范围 [0, 1]
    """
    dt = arr.dtype

    # 已经是 float32 -> 直通
    if dt == np.float32:
        return arr

    # uint8 -> /255
    if dt == np.uint8:
        return arr.astype(np.float32) / 255.0

    # uint16 -> /65535 (16-bit 深度图/热红外典型场景)
    if dt == np.uint16:
        return arr.astype(np.float32) / 65535.0

    # float64 -> 直接 cast（精度损失在模型计算可接受范围内）
    if dt == np.float64:
        return arr.astype(np.float32)

    # int64 / int32 / int16 -> cast to float32, 按数据范围自动归一化
    if np.issubdtype(dt, np.integer):
        arr_f = arr.astype(np.float32)
        max_val = arr_f.max()
        if max_val > 1.0:
            arr_f = arr_f / max_val if max_val > 0 else arr_f
        return arr_f

    # 其他类型（如 bool）-> float32
    return arr.astype(np.float32)


class MultiModalImageIOMixin:
    """
    多模态图像I/O操作的Mixin类

    提供X模态图像的查找、加载、对齐等核心功能，
    可被 YOLOMultiModalImageDataset 和 YOLOMultiModalClassifyDataset 共享使用。

    Attributes:
        x_modality (str): X模态类型，如 'depth', 'thermal', 'ir' 等
        x_modality_dir (str): X模态图像目录名
        x_modality_suffix (str, optional): X模态图像文件后缀

    Methods:
        find_corresponding_x_image: 根据RGB路径查找对应的X模态图像路径
        load_x_modality: 加载X模态图像
        align_x_to_rgb: 将X模态图像对齐到RGB尺寸
        validate_x_channels: 校验并调整X模态通道数
    """

    # 支持的图像扩展名（用于查找X模态图像）
    SUPPORTED_EXTENSIONS = ['.npy', '.npz', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']

    def find_corresponding_x_image(
        self,
        rgb_path: Union[str, Path],
        x_modality_dir: str,
        x_modality_suffix: Optional[str] = None
    ) -> str:
        """
        根据RGB图像路径找到对应的X模态图像路径

        路径映射规则：
        - RGB路径: /dataset/images/train/img.jpg
        - X路径:   /dataset/images_depth/train/img.jpg（同名，可不同扩展）

        Args:
            rgb_path: RGB图像路径
            x_modality_dir: X模态图像目录名（如 'images_depth'）
            x_modality_suffix: X模态文件后缀（可选）

        Returns:
            X模态图像路径

        Examples:
            >>> mixin = MultiModalImageIOMixin()
            >>> x_path = mixin.find_corresponding_x_image(
            ...     "/dataset/images/train/img001.jpg",
            ...     "images_depth"
            ... )
            >>> print(x_path)  # "/dataset/images_depth/train/img001.png"
        """
        rgb_path = Path(rgb_path)

        # 构建X模态图像路径
        # 保持相同的子目录结构 (train/val/test)，只替换基础模态目录
        dataset_root = rgb_path.parent.parent.parent  # 跳过 images 和 train
        split_dir = rgb_path.parent.name  # train/val/test
        x_dir = dataset_root / x_modality_dir / split_dir

        # 处理文件后缀
        if x_modality_suffix:
            x_filename = rgb_path.stem + x_modality_suffix + rgb_path.suffix
        else:
            x_filename = rgb_path.name

        x_path = x_dir / x_filename

        # 如果指定路径不存在，尝试常见的文件扩展名
        if not x_path.exists():
            for ext in self.SUPPORTED_EXTENSIONS:
                test_path = x_dir / (rgb_path.stem + ext)
                if test_path.exists():
                    return str(test_path)

        return str(x_path)

    def load_x_modality(self, x_path: Union[str, Path]) -> np.ndarray:
        """
        加载X模态图像（原始数组，不归一化）

        支持多种文件格式：
        - .npy: NumPy数组文件
        - .npz: NumPy压缩文件
        - .tif/.tiff: TIFF文件（常用于深度图）
        - 其他: 标准图像文件

        Legacy 分支: 返回原始数组，归一化在 tensorize/preprocess 阶段按 contract 处理。
        High-precision 分支: 调用方需显式调用 normalize_modality_array()。

        Args:
            x_path: X模态图像路径

        Returns:
            X模态原始图像数组（保持文件原始 dtype 和值域）

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 无法读取图像
        """
        return load_modality_array_raw(x_path)

    def align_x_to_rgb(
        self,
        x_img: np.ndarray,
        rgb_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        将X模态图像对齐到RGB尺寸

        Args:
            x_img: X模态图像
            rgb_shape: RGB图像尺寸 (height, width)

        Returns:
            对齐后的X模态图像
        """
        target_h, target_w = rgb_shape
        x_h, x_w = x_img.shape[:2]

        if (x_h, x_w) != (target_h, target_w):
            x_img = cv2.resize(x_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return x_img

    def validate_x_channels(
        self,
        x_img: np.ndarray,
        expected_xch: int,
        strict: bool = False,
        x_path_hint: Optional[str] = None
    ) -> np.ndarray:
        """
        校验并调整X模态通道数

        规则：
        - Xch=1: 保持单通道
        - Xch=3: 允许灰度转三通道
        - Xch>3: 必须严格匹配

        Args:
            x_img: X模态图像
            expected_xch: 期望的通道数
            strict: 是否严格模式（禁止自动转换）
            x_path_hint: 图像路径（用于错误信息）

        Returns:
            调整后的X模态图像

        Raises:
            ValueError: 通道数不匹配且无法转换
        """
        # 获取实际通道数
        if len(x_img.shape) == 2:
            actual_xch = 1
        elif len(x_img.shape) == 3:
            actual_xch = x_img.shape[2]
        else:
            raise ValueError(f"X模态图像维度异常: {x_img.shape}")

        # 严格模式检查
        if strict and actual_xch != expected_xch:
            raise ValueError(
                f"X通道不一致: 期望={expected_xch}, 实际={actual_xch}。"
                f" 文件: {x_path_hint or 'unknown'}"
            )

        # 通道转换
        if len(x_img.shape) == 2:
            # 灰度图像
            if expected_xch == 1:
                x_img = x_img[:, :, np.newaxis]
            else:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)

        elif x_img.shape[2] == 1:
            # 单通道图像
            if expected_xch != 1:
                x_img = np.repeat(x_img, 3, axis=2)

        elif x_img.shape[2] == 4:
            # RGBA图像
            x_img = x_img[:, :, :3]
            if expected_xch == 1:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

        elif x_img.shape[2] == 3:
            # RGB/BGR图像
            if expected_xch == 1:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

        # 最终验证
        final_ch = x_img.shape[2] if len(x_img.shape) == 3 else 1
        if final_ch != expected_xch:
            LOGGER.warning(
                f"X模态通道数({final_ch})与期望({expected_xch})不匹配，"
                f"文件: {x_path_hint or 'unknown'}"
            )

        return x_img

    def load_and_align_x_image(
        self,
        rgb_path: Union[str, Path],
        rgb_img: np.ndarray,
        x_modality_dir: str,
        expected_xch: int,
        x_modality_suffix: Optional[str] = None,
        strict: bool = False
    ) -> np.ndarray:
        """
        加载并对齐X模态图像（便捷方法）

        组合了查找、加载、对齐、通道校验的完整流程。

        Args:
            rgb_path: RGB图像路径
            rgb_img: 已加载的RGB图像
            x_modality_dir: X模态目录名
            expected_xch: 期望的X模态通道数
            x_modality_suffix: X模态文件后缀
            strict: 是否严格模式

        Returns:
            处理后的X模态图像
        """
        # 查找X模态路径
        x_path = self.find_corresponding_x_image(rgb_path, x_modality_dir, x_modality_suffix)

        # 加载X模态图像
        x_img = self.load_x_modality(x_path)

        # 对齐到RGB尺寸
        x_img = self.align_x_to_rgb(x_img, rgb_img.shape[:2])

        # 校验并调整通道数
        x_img = self.validate_x_channels(x_img, expected_xch, strict, x_path)

        return x_img

    def concatenate_multimodal(
        self,
        rgb_img: np.ndarray,
        x_img: np.ndarray
    ) -> np.ndarray:
        """
        拼接RGB和X模态图像

        Args:
            rgb_img: RGB图像 (H, W, 3)
            x_img: X模态图像 (H, W, Xch)

        Returns:
            多模态图像 (H, W, 3+Xch)
        """
        return np.concatenate([rgb_img, x_img], axis=2)
