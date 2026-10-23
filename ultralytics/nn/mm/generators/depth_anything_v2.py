# ultralytics/nn/mm/generators/depth_anything_v2.py
"""
DepthGen: 离线深度生成器 - 完全内部化版本。

特点:
- 直接 from ultralytics import DepthGen 即可使用
- 后端 Depth Anything V2，支持 vits/vitb/vitl/vitg
- 无外部依赖，模型代码已整合到 depth_AT2 子模块
- 权重路径由用户任意指定
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_yaml
from .base import ModalGeneratorBase, SaveOptions

# 导入内部化的 Depth-Anything-V2 模块
from .depth_AT2 import DepthAnythingV2, Resize, NormalizeImage, PrepareForNet


# --------------------------
# Encoder 配置
# --------------------------

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def _infer_encoder_from_weights(weights_path: Path) -> Optional[str]:
    """从权重文件名推断 encoder 类型。"""
    name_l = weights_path.name.lower()
    found = [k for k in MODEL_CONFIGS.keys() if k in name_l]
    if len(found) == 1:
        return found[0]
    return None


def _infer_encoder_from_state(state: dict) -> Optional[str]:
    """从权重内容推断 encoder 类型（通过兼容性测试）。"""
    candidates = []
    for enc_key, cfg in MODEL_CONFIGS.items():
        try:
            m = DepthAnythingV2(**cfg)
            m.load_state_dict(state, strict=False)
            candidates.append(enc_key)
        except Exception:
            continue
    if len(candidates) == 1:
        return candidates[0]
    return None


# --------------------------
# 生成器实现
# --------------------------


class DepthGen(ModalGeneratorBase):
    """
    简单接口：DepthGen(weights, encoder=None, input_size=518, device="auto", ...)
    run(source, save_dir=None)
    """

    backend_name = "depth_anything_v2"

    def __init__(
        self,
        weights: str,
        encoder: Optional[str] = None,
        input_size: int = 518,
        device: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        # 保存相关
        save_dir: str | Path | None = None,
        keep_structure: bool = True,
        overwrite: bool = False,
        # 预处理/后处理与可视化
        clamp_percentile: Tuple[float, float] | None = (2.0, 98.0),
        depth_scale: float = 1.0,
        depth_offset: float = 0.0,
        invert: bool = False,
        gaussian_blur: bool = False,
        gaussian_kernel: int = 5,
        gaussian_sigma: float = 1.0,
        median_filter: bool = False,
        median_kernel: int = 5,
        bilateral_filter: bool = False,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75,
        bilateral_sigma_space: float = 75,
        colormap: str = "Spectral_r",
        save_color: bool = True,
        save_raw16: bool = True,
        save_npy: bool = False,
        save_comparison: bool = False,
        grayscale_color: bool = False,
        keep_aspect_ratio: bool = True,
        ensure_multiple_of: int = 14,
        resize_method: str = "lower_bound",
        **kwargs,
    ) -> None:
        # 权重路径必须指定
        if not weights:
            raise ValueError("必须指定 weights 参数，请提供权重文件路径")

        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"权重文件不存在: {self.weights}")

        self.encoder = encoder
        self.input_size = input_size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.ensure_multiple_of = ensure_multiple_of
        self.resize_method = resize_method

        # 后处理配置
        self.clamp_percentile = clamp_percentile
        self.depth_scale = depth_scale
        self.depth_offset = depth_offset
        self.invert = invert
        self.gaussian_blur = gaussian_blur
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma
        self.median_filter = median_filter
        self.median_kernel = median_kernel
        self.bilateral_filter = bilateral_filter
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.colormap = colormap
        self.save_color = save_color
        self.save_raw16 = save_raw16
        self.save_npy = save_npy
        self.save_comparison = save_comparison
        self.grayscale_color = grayscale_color

        # 互斥校验：仅允许三种保存形式之一
        save_flags = [self.save_color, self.save_raw16, self.save_npy]
        if sum(bool(f) for f in save_flags) > 1:
            raise ValueError("save_color / save_raw16 / save_npy 互斥，请只开启一个。")

        save_options = SaveOptions(
            save_dir=save_dir,
            keep_structure=keep_structure,
            overwrite=overwrite,
            enable_save=True,
        )

        super().__init__(
            method=self.backend_name,
            method_cfg={},
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            save_options=save_options,
        )

        self._transform_list = None
        self._img_formats = {
            "bmp", "dng", "jpeg", "jpg", "mpo", "png",
            "tif", "tiff", "webp", "pfm", "heic",
        }
        self._path_meta: Dict[str, Dict[str, Any]] = {}
        self._yaml_root: Optional[Path] = None
        self._debug_logged: set[str] = set()

    # 覆盖源收集逻辑，支持 data.yaml
    def _gather_sources(self, source: str | Path | Iterable[str | Path]) -> List[str]:
        # YAML 情况
        if isinstance(source, (str, Path)) and str(source).lower().endswith((".yaml", ".yml")):
            yaml_path = Path(check_yaml(source, hard=True))
            self._yaml_root = yaml_path.parent
            data = yaml.safe_load(yaml_path.read_text())
            base = Path(data.get("path", yaml_path.parent))
            splits_cfg = []
            for k in ("train", "val", "test"):
                v = data.get(k)
                if v:
                    splits_cfg.append((k, v))
            if not splits_cfg:
                raise ValueError(f"{yaml_path} 中未找到 train/val/test 配置")

            paths: List[str] = []
            self._path_meta.clear()
            for split_name, split_val in splits_cfg:
                if hasattr(self, "split_names") and self.split_names:
                    if split_name not in self.split_names:
                        continue
                split_list = split_val if isinstance(split_val, list) else [split_val]
                for s in split_list:
                    p = Path(s)
                    if not p.is_absolute():
                        p = base / p
                    if not p.exists():
                        raise FileNotFoundError(f"数据集路径不存在: {p}")
                    images_root = p.parent
                    depth_root_default = images_root.parent / "images_depth"
                    for f in p.rglob("*.*"):
                        if f.is_file() and f.suffix[1:].lower() in self._img_formats:
                            fstr = str(f)
                            paths.append(fstr)
                            self._path_meta[fstr] = {
                                "split": p.name,
                                "images_root": images_root,
                                "depth_root_default": depth_root_default,
                            }
            if not paths:
                raise RuntimeError(f"{yaml_path} 未找到任何图像文件")
            return sorted(paths)

        # 普通路径/列表
        paths: List[str] = []
        if isinstance(source, (str, Path)):
            src = Path(source)
            if src.is_dir():
                paths = [
                    str(p)
                    for p in src.rglob("*.*")
                    if p.is_file() and p.suffix[1:].lower() in self._img_formats
                ]
            elif src.is_file():
                if src.suffix[1:].lower() in self._img_formats:
                    paths = [str(src)]
                else:
                    raise ValueError(f"不支持的文件类型: {src}")
            else:
                raise FileNotFoundError(f"找不到输入: {source}")
        else:
            for item in source:
                p = Path(item)
                if p.exists() and p.is_file() and p.suffix[1:].lower() in self._img_formats:
                    paths.append(str(p))
                else:
                    raise FileNotFoundError(f"找不到输入或格式不支持: {item}")

        if not paths:
            raise RuntimeError("未找到任何可处理的输入文件")
        return sorted(paths)

    # ---- 必需接口 ----

    def load_model(self):
        # 1) 决定 encoder
        encoder = (self.encoder.lower() if isinstance(self.encoder, str) else None)
        state = None

        if encoder in (None, "", "auto"):
            # 先尝试从文件名推断
            encoder = _infer_encoder_from_weights(self.weights)
            if encoder:
                LOGGER.info(f"[{self.method}] encoder 自动推断为 {encoder} (from weights filename)")
            else:
                # 从权重内���推断
                state = torch.load(self.weights, map_location="cpu")
                encoder = _infer_encoder_from_state(state)
                if encoder:
                    LOGGER.info(f"[{self.method}] encoder 自动推断为 {encoder} (from weights compatibility)")
                else:
                    raise ValueError(
                        f"无法自动推断 encoder 类型，请显式指定 encoder 参数。"
                        f"支持的类型: {list(MODEL_CONFIGS.keys())}"
                    )

        if encoder not in MODEL_CONFIGS:
            raise ValueError(f"不支持的 encoder: {encoder}，支持: {list(MODEL_CONFIGS.keys())}")
        self.encoder = encoder

        # 2) 构建模型
        self.model = DepthAnythingV2(**MODEL_CONFIGS[self.encoder]).to(self.device).eval()
        if state is None:
            state = torch.load(self.weights, map_location="cpu")
        self.model.load_state_dict(state)

        # 3) 构建变换
        self._transform_list = [
            Resize(
                width=self.input_size,
                height=self.input_size,
                resize_target=False,
                keep_aspect_ratio=self.keep_aspect_ratio,
                ensure_multiple_of=self.ensure_multiple_of,
                resize_method=self.resize_method,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]

        LOGGER.info(f"[{self.method}] 模型加载完成: {self.encoder} @ {self.weights}")

    def preprocess(self, item: str):
        img = cv2.imread(item, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {item}")
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        sample = {"image": img_rgb}
        for t in self._transform_list:
            sample = t(sample)
        tensor = sample["image"]
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device, non_blocking=True)
        meta = {
            "path": item,
            "orig_hw": (h, w),
            "bgr": img,
        }
        if item in self._path_meta:
            meta.update(self._path_meta[item])
        return tensor, meta

    def infer(self, batch_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        batch = torch.cat(batch_inputs, dim=0)
        with torch.no_grad():
            depth = self.model(batch)
        return [d for d in depth]

    def postprocess(self, outputs: List[torch.Tensor], metas: List[Dict[str, Any]]) -> List[np.ndarray]:
        results = []
        for depth_tensor, meta in zip(outputs, metas):
            tgt_h, tgt_w = meta["orig_hw"]
            depth_resized = F.interpolate(
                depth_tensor[None, None, ...], size=(tgt_h, tgt_w), mode="bilinear", align_corners=True
            )[0, 0]
            depth_np = depth_resized.cpu().numpy().astype(np.float32)

            if self.depth_scale != 1.0 or self.depth_offset != 0.0:
                depth_np = depth_np * self.depth_scale + self.depth_offset

            if self.clamp_percentile:
                p_low, p_high = self.clamp_percentile
                lo, hi = np.percentile(depth_np, [p_low, p_high])
                depth_np = np.clip(depth_np, lo, hi)

            if self.gaussian_blur:
                depth_np = cv2.GaussianBlur(
                    depth_np, (self.gaussian_kernel, self.gaussian_kernel), self.gaussian_sigma
                )

            if self.median_filter:
                depth_np = cv2.medianBlur(depth_np, self.median_kernel)

            if self.bilateral_filter:
                d8 = ((depth_np - depth_np.min()) / (depth_np.ptp() + 1e-6) * 255).astype(np.uint8)
                d8 = cv2.bilateralFilter(d8, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space)
                depth_np = d8.astype(np.float32) / 255.0 * (depth_np.ptp() + 1e-6) + depth_np.min()

            if self.invert:
                depth_np = depth_np.max() - depth_np

            results.append(depth_np)
        return results

    def save(self, outputs: List[np.ndarray], metas: List[Dict[str, Any]]) -> List[str]:
        paths = []
        save_opt: SaveOptions = self.save_options

        for depth_np, meta in zip(outputs, metas):
            src_path = Path(meta["path"])
            if save_opt.save_dir is None and "depth_root_default" in meta:
                depth_root = meta["depth_root_default"]
                split = meta.get("split", "unknown")
                out_dir = depth_root / split
            else:
                base_dir = Path(save_opt.save_dir or "runs/depth_gen") / self.method
                rel = (
                    src_path.relative_to(getattr(self, "_source_root", src_path.parent))
                    if save_opt.keep_structure
                    else Path(src_path.name)
                )
                out_dir = (base_dir / rel.parent) if save_opt.keep_structure else base_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = src_path.stem

            debug_key = str(out_dir)
            if debug_key not in self._debug_logged:
                LOGGER.info(f"[{self.method}] 保存目录: {out_dir}")
                self._debug_logged.add(debug_key)

            d_min, d_max = float(depth_np.min()), float(depth_np.max())
            if d_max == d_min:
                d_max = d_min + 1e-6
            depth_8 = ((depth_np - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
            depth_16 = ((depth_np - d_min) / (d_max - d_min) * 65535.0).astype(np.uint16)

            if self.save_color:
                if self.grayscale_color:
                    color = np.repeat(depth_8[..., None], 3, axis=-1)
                else:
                    cmap_fallback = getattr(cv2, "COLORMAP_SPECTRAL", cv2.COLORMAP_JET)
                    try:
                        import matplotlib
                        cm = matplotlib.colormaps.get_cmap(self.colormap)
                        color = (cm(depth_8)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                    except Exception:
                        color = cv2.applyColorMap(depth_8, cmap_fallback)
                color_path = out_dir / f"{stem}_depth.png"
                if save_opt.overwrite or not color_path.exists():
                    cv2.imwrite(str(color_path), color)
                paths.append(str(color_path))

            elif self.save_raw16:
                raw_path = out_dir / f"{stem}_depth_raw.png"
                if save_opt.overwrite or not raw_path.exists():
                    cv2.imwrite(str(raw_path), depth_16)
                paths.append(str(raw_path))

            elif self.save_npy:
                npy_path = out_dir / f"{stem}_depth.npy"
                if save_opt.overwrite or not npy_path.exists():
                    np.save(npy_path, depth_np)
                paths.append(str(npy_path))

            if self.save_comparison and "bgr" in meta:
                bgr = meta["bgr"]
                h1, h2 = bgr.shape[0], depth_8.shape[0]
                # 根据 save_color 决定对比图中深度部分的可视化方式
                if self.save_color and not self.grayscale_color:
                    # 使用与主深度图相同的彩色映射
                    cmap_fallback = getattr(cv2, "COLORMAP_SPECTRAL", cv2.COLORMAP_JET)
                    try:
                        import matplotlib
                        cm = matplotlib.colormaps.get_cmap(self.colormap)
                        depth_vis = (cm(depth_8)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                    except Exception:
                        depth_vis = cv2.applyColorMap(depth_8, cmap_fallback)
                else:
                    # 灰度可视化
                    depth_vis = np.repeat(depth_8[..., None], 3, axis=-1)
                if h1 != h2:
                    depth_vis = cv2.resize(depth_vis, (depth_vis.shape[1], h1))
                split_bar = np.ones((bgr.shape[0], 16, 3), dtype=np.uint8) * 255
                comp = cv2.hconcat([bgr, split_bar, depth_vis])
                if save_opt.save_dir is None and "depth_root_default" in meta:
                    compare_dir = meta["depth_root_default"].parent / "images_depth_compare" / meta.get("split", "unknown")
                else:
                    compare_dir = out_dir / "images_depth_compare"
                compare_dir.mkdir(parents=True, exist_ok=True)
                comp_path = compare_dir / f"{stem}_comparison.png"
                debug_c_key = str(compare_dir)
                if debug_c_key not in self._debug_logged:
                    LOGGER.info(f"[{self.method}] 对比图目录: {compare_dir}")
                    self._debug_logged.add(debug_c_key)
                if save_opt.overwrite or not comp_path.exists():
                    cv2.imwrite(str(comp_path), comp)
                paths.append(str(comp_path))

        return paths

    def run(self, source: str | Path | List[str], save_dir: str | Path | None = None):
        """允许在调用时临时指定 save_dir。"""
        if save_dir is not None:
            self.save_options.save_dir = save_dir
        split = getattr(self, "split", None)
        if split:
            if isinstance(split, str):
                self.split_names = [s.strip() for s in split.split(",") if s.strip()]
            else:
                self.split_names = list(split)
        else:
            self.split_names = []
        return super().run(source)


__all__ = ["DepthGen"]
