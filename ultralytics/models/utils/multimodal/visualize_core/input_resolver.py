"""Raw input resolution and validation for visualize_core (Fail-Fast).

This module performs original image reading and basic validation only.
No channel-semantic transformations (e.g. GRAY→RGB, 1ch→3ch repeat)
are applied here — all model-expected channel adaptation is deferred
to Preprocessor, which consults the router's PRIMARY/SECONDARY/FUSED
channel contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import cv2

from .exceptions import InputValidationError, ModalityConflictError


class InputResolver:
    """
    Resolve and validate inputs into a standardized dict: {'rgb': np.ndarray?, 'x': np.ndarray?}.

    Notes:
    - 自动模态推断：提供两个源→dual；单源→对应单模态；缺失→报错。
    - 严格校验：dtype 数值型；维度 2/3/4；无 NaN/Inf。
    - 不做通道语义变换：单通道保持原始，不做 GRAY→RGB；所有模型入口通道适配统一由 Preprocessor 处理。
    """

    @staticmethod
    def _load_image(src: Any) -> np.ndarray:
        if isinstance(src, (str, Path)):
            p = Path(src)
            if not p.exists():
                raise InputValidationError(f"输入图像不存在: {p}")
            # .npy/.npz 格式：使用 numpy 直接加载
            if p.suffix.lower() in {'.npy', '.npz'}:
                im = np.load(str(p), allow_pickle=False)
                if im is None:
                    raise InputValidationError(f"无法加载 .npy 文件: {p}")
            else:
                im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if im is None:
                    raise InputValidationError(f"无法加载图像: {p}")
                # BGR->RGB (only for 3-channel images)
                if im.ndim == 3 and im.shape[2] == 3:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                # 2D / 1ch / >3ch: keep raw channels, no GRAY→RGB conversion here
            return im  # preserve original dtype and channels, no implicit cast
        elif isinstance(src, np.ndarray):
            return src
        else:
            raise InputValidationError(f"不支持的输入类型: {type(src)}。仅支持文件路径或 numpy.ndarray。")

    @staticmethod
    def _validate_array(arr: np.ndarray, name: str) -> None:
        shape = arr.shape
        dtype = arr.dtype
        if arr.size == 0:
            raise InputValidationError(f"{name} 为空（shape={shape}, dtype={dtype}）")
        if not np.issubdtype(dtype, np.number):
            raise InputValidationError(f"{name} dtype 应为数值型（收到 {dtype}），建议转换为 float32/uint8 后重试")
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise InputValidationError(f"{name} 含 NaN/Inf（shape={shape}, dtype={dtype}），请先清理数据")
        if arr.ndim not in (2, 3, 4):
            raise InputValidationError(f"{name} 维度应为 2D/3D/4D（收到 {shape}），请检查输入形状")
        if arr.ndim >= 3:
            c = arr.shape[2] if arr.ndim == 3 else arr.shape[1]
            # uint16 is now supported — preprocessor._to_float01() handles it
            # int64/float64 also supported — normalization handles them
            if not np.issubdtype(dtype, np.number):
                raise InputValidationError(
                    f"{name} dtype 应为数值型（收到 {dtype}），建议转换为 float32/uint8 后重试"
                )
            # 通道建议（不强制失败）
            if c not in (1, 3, 4, 6):
                pass

    @staticmethod
    def resolve(rgb_source: Optional[Any], x_source: Optional[Any], modality: Optional[str]) -> Any:
        """
        解析输入：自动识别文件/目录/数组，并按规则返回：
        - 双文件/数组：返回 {'rgb': np.ndarray, 'x': np.ndarray}
        - 双目录：返回 List[{'rgb': np.ndarray, 'x': np.ndarray, 'img_key': str}]
        - 仅一侧（文件或目录）：返回单样本 dict 或样本列表（另一侧由上游做消融补全）
        - 一侧目录一侧文件：直接报错
        """

        def is_dir(v: Any) -> bool:
            return isinstance(v, (str, Path)) and Path(v).exists() and Path(v).is_dir()

        def is_file(v: Any) -> bool:
            return isinstance(v, (str, Path)) and Path(v).exists() and Path(v).is_file()

        def enumerate_images(d: Path) -> List[Path]:
            exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.npy', '.npz'}
            files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]
            return sorted(files)

        def pair_by_stem(rgb_dir: Path, x_dir: Path) -> List[Tuple[Path, Path, str]]:
            rgb_files = enumerate_images(rgb_dir)
            x_files = enumerate_images(x_dir)
            if not rgb_files or not x_files:
                raise InputValidationError(
                    f"目录中没有可用图片，RGB:{len(rgb_files)} X:{len(x_files)}。支持扩展名: jpg/jpeg/png/bmp/tif/tiff"
                )
            rgb_map = {p.stem: p for p in rgb_files}
            x_map = {p.stem: p for p in x_files}
            common = sorted(set(rgb_map.keys()) & set(x_map.keys()))
            if not common:
                # 打印各侧样例 stem，帮助定位
                sample_rgb = ', '.join(list(rgb_map.keys())[:5]) or '空'
                sample_x = ', '.join(list(x_map.keys())[:5]) or '空'
                raise InputValidationError(
                    f"RGB 与 X 目录无同名文件（忽略后缀）。示例 RGB: [{sample_rgb}] | X: [{sample_x}]"
                )
            # 检查未配对
            only_rgb = sorted(set(rgb_map.keys()) - set(common))
            only_x = sorted(set(x_map.keys()) - set(common))
            if only_rgb or only_x:
                msg = []
                if only_rgb:
                    msg.append(f"仅在 RGB 目录出现: {', '.join(only_rgb[:5])}{' ...' if len(only_rgb)>5 else ''}")
                if only_x:
                    msg.append(f"仅在 X 目录出现: {', '.join(only_x[:5])}{' ...' if len(only_x)>5 else ''}")
                raise InputValidationError("目录配对不完整：\n" + "\n".join(msg))
            return [(rgb_map[k], x_map[k], k) for k in common]

        # 0) 完整性校验
        if rgb_source is None and x_source is None:
            raise InputValidationError("必须提供 rgb_source 或 x_source（或二者同时提供）。")

        # 1) 双侧都给了
        if rgb_source is not None and x_source is not None:
            if is_dir(rgb_source) and is_dir(x_source):
                pairs = pair_by_stem(Path(rgb_source), Path(x_source))
                out_list: List[Dict[str, np.ndarray]] = []
                for rp, xp, key in pairs:
                    rgb_img = InputResolver._load_image(rp)
                    InputResolver._validate_array(rgb_img, f"rgb:{rp.name}")
                    x_img = InputResolver._load_image(xp)
                    InputResolver._validate_array(x_img, f"x:{xp.name}")
                    out_list.append(
                        {
                            'rgb': rgb_img,
                            'x': x_img,
                            'img_key': key,
                            '__source_paths__': {'rgb': str(rp), 'x': str(xp)},
                        }
                    )
                return out_list
            if (is_dir(rgb_source) and not is_dir(x_source)) or (is_dir(x_source) and not is_dir(rgb_source)):
                raise InputValidationError("不支持：一侧目录一侧文件。请传两个目录或两个文件，或只传一侧执行消融。")

            # 两个都是文件或数组：单对
            out: Dict[str, np.ndarray] = {}
            rgb_img = InputResolver._load_image(rgb_source)
            InputResolver._validate_array(rgb_img, 'rgb_source')
            x_img = InputResolver._load_image(x_source)
            InputResolver._validate_array(x_img, 'x_source')
            out['rgb'] = rgb_img
            out['x'] = x_img
            out['__source_paths__'] = {
                'rgb': str(rgb_source) if is_file(rgb_source) else None,
                'x': str(x_source) if is_file(x_source) else None,
            }
            return out

        # 2) 仅一侧（自动视为单模态消融）
        #    - 如果是目录：生成样本列表；如果是文件/数组：返回单样本 dict
        if rgb_source is not None and x_source is None:
            if is_dir(rgb_source):
                files = enumerate_images(Path(rgb_source))
                if not files:
                    raise InputValidationError("RGB 目录为空或无可用图片")
                lst: List[Dict[str, np.ndarray]] = []
                for p in files:
                    img = InputResolver._load_image(p)
                    InputResolver._validate_array(img, f"rgb:{p.name}")
                    lst.append({'rgb': img, 'img_key': p.stem, '__source_paths__': {'rgb': str(p), 'x': None}})
                return lst
            else:
                img = InputResolver._load_image(rgb_source)
                InputResolver._validate_array(img, 'rgb_source')
                return {'rgb': img, '__source_paths__': {'rgb': str(rgb_source) if is_file(rgb_source) else None, 'x': None}}

        if x_source is not None and rgb_source is None:
            if is_dir(x_source):
                files = enumerate_images(Path(x_source))
                if not files:
                    raise InputValidationError("X 目录为空或无可用图片")
                lst: List[Dict[str, np.ndarray]] = []
                for p in files:
                    img = InputResolver._load_image(p)
                    InputResolver._validate_array(img, f"x:{p.name}")
                    lst.append({'x': img, 'img_key': p.stem, '__source_paths__': {'rgb': None, 'x': str(p)}})
                return lst
            else:
                img = InputResolver._load_image(x_source)
                InputResolver._validate_array(img, 'x_source')
                return {'x': img, '__source_paths__': {'rgb': None, 'x': str(x_source) if is_file(x_source) else None}}

        # 不应到达此处
        raise InputValidationError("未解析到有效输入。")
