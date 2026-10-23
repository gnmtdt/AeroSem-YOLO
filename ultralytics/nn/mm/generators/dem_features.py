"""
DEMGen：离线 DEM 特征生成器。

将 tools/generate_dem_features.py 中的 DEM-like 6 通道特征提取整合到多模态离线生成体系，
对外提供与 DepthGen 一致的 run(source) 接口。

特征通道（与原脚本一致）：
1. Elevation（伪高程/灰度）
2. Slope（梯度幅值）
3. Aspect（梯度方向）
4. Curvature（拉普拉斯二阶导）
5. Roughness（局部标准差）
6. Local Height Difference（高通局部差分）

默认行为：
- 输入为 data.yaml 时，按 train/val/test 的 split 收集 source_modality（默认 rgb）图像；
- 生成结果默认保存到 images_dem/<split>/ 下，与 RGB 文件同 stem 的 .npy 文件。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_yaml

from .base import ModalGeneratorBase, SaveOptions


class DEMKernels:
    """DEM 特征提取所需的卷积核集合。"""

    @staticmethod
    def gaussian_kernel_2d(ksize: int, sigma: float = 0.0) -> torch.Tensor:
        if sigma <= 0:
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        center = ksize // 2
        x = torch.arange(ksize, dtype=torch.float32) - center
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        return gauss_2d / gauss_2d.sum()

    @staticmethod
    def sobel_kernel_x(ksize: int = 3) -> torch.Tensor:
        if ksize == 3:
            return torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
        if ksize == 5:
            return torch.tensor(
                [
                    [-1.0, -2.0, 0.0, 2.0, 1.0],
                    [-4.0, -8.0, 0.0, 8.0, 4.0],
                    [-6.0, -12.0, 0.0, 12.0, 6.0],
                    [-4.0, -8.0, 0.0, 8.0, 4.0],
                    [-1.0, -2.0, 0.0, 2.0, 1.0],
                ],
                dtype=torch.float32,
            )
        raise ValueError(f"Sobel kernel size must be 3 or 5, got {ksize}")

    @staticmethod
    def sobel_kernel_y(ksize: int = 3) -> torch.Tensor:
        return DEMKernels.sobel_kernel_x(ksize).t()

    @staticmethod
    def scharr_kernel_x() -> torch.Tensor:
        return torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
            dtype=torch.float32,
        )

    @staticmethod
    def scharr_kernel_y() -> torch.Tensor:
        return DEMKernels.scharr_kernel_x().t()

    @staticmethod
    def laplacian_kernel(ksize: int = 3) -> torch.Tensor:
        if ksize == 1:
            return torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        if ksize == 3:
            return torch.tensor([[2.0, 0.0, 2.0], [0.0, -8.0, 0.0], [2.0, 0.0, 2.0]], dtype=torch.float32)
        if ksize == 5:
            return torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0, 1.0, 0.0],
                    [1.0, 2.0, -16.0, 2.0, 1.0],
                    [0.0, 1.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        raise ValueError(f"Laplacian kernel size must be 1, 3, or 5, got {ksize}")

    @staticmethod
    def mean_kernel(ksize: int) -> torch.Tensor:
        return torch.ones((ksize, ksize), dtype=torch.float32) / float(ksize * ksize)


class DEMFeatureGenerator(nn.Module):
    """与 tools/generate_dem_features.py 对齐的 6 通道 DEM-like 特征生成网络。"""

    def __init__(
        self,
        gaussian_ksize: int = 3,
        sobel_ksize: int = 3,
        roughness_ksize: int = 5,
        local_diff_ksize: int = 15,
        use_scharr: bool = False,
        device: Optional[str | torch.device] = None,
    ) -> None:
        super().__init__()
        self.gaussian_ksize = gaussian_ksize
        self.sobel_ksize = sobel_ksize
        self.roughness_ksize = roughness_ksize
        self.local_diff_ksize = local_diff_ksize
        self.use_scharr = use_scharr
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self._register_kernels()
        self.to(self.device)

    def _register_kernels(self) -> None:
        gaussian = DEMKernels.gaussian_kernel_2d(self.gaussian_ksize)
        self.register_buffer("gaussian_kernel", gaussian.view(1, 1, self.gaussian_ksize, self.gaussian_ksize))

        if self.use_scharr:
            grad_x = DEMKernels.scharr_kernel_x()
            grad_y = DEMKernels.scharr_kernel_y()
            grad_ksize = 3
        else:
            grad_x = DEMKernels.sobel_kernel_x(self.sobel_ksize)
            grad_y = DEMKernels.sobel_kernel_y(self.sobel_ksize)
            grad_ksize = self.sobel_ksize

        self.register_buffer("sobel_x", grad_x.view(1, 1, grad_ksize, grad_ksize))
        self.register_buffer("sobel_y", grad_y.view(1, 1, grad_ksize, grad_ksize))

        laplacian = DEMKernels.laplacian_kernel(self.sobel_ksize)
        lap_ksize = laplacian.shape[0]
        self.register_buffer("laplacian_kernel", laplacian.view(1, 1, lap_ksize, lap_ksize))

        mean_rough = DEMKernels.mean_kernel(self.roughness_ksize)
        self.register_buffer("mean_kernel_rough", mean_rough.view(1, 1, self.roughness_ksize, self.roughness_ksize))

        mean_local = DEMKernels.mean_kernel(self.local_diff_ksize)
        self.register_buffer("mean_kernel_local", mean_local.view(1, 1, self.local_diff_ksize, self.local_diff_ksize))

    def _conv2d_reflect(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        ksize = kernel.shape[-1]
        pad = ksize // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_padded, kernel)

    def _rgb_to_gray(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x
        if x.shape[1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device)
            return (x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        return x.mean(dim=1, keepdim=True)

    def _normalize_channel(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = x.view(B, -1)
        min_val = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        denom = torch.clamp(max_val - min_val, min=1e-8)
        return (x - min_val) / denom

    @torch.no_grad()
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() <= 1.0:
            x = x * 255.0

        gray = self._rgb_to_gray(x)
        gray = self._conv2d_reflect(gray, self.gaussian_kernel)

        elevation = gray
        gx = self._conv2d_reflect(gray, self.sobel_x)
        gy = self._conv2d_reflect(gray, self.sobel_y)
        slope = torch.sqrt(gx**2 + gy**2)
        aspect = (torch.atan2(gy, gx) + math.pi) / (2 * math.pi)
        curvature = self._conv2d_reflect(gray, self.laplacian_kernel)
        mean_sq = self._conv2d_reflect(gray**2, self.mean_kernel_rough)
        sq_mean = self._conv2d_reflect(gray, self.mean_kernel_rough) ** 2
        roughness = torch.sqrt(torch.clamp(mean_sq - sq_mean, min=0))
        background = self._conv2d_reflect(gray, self.mean_kernel_local)
        local_diff = gray - background

        channels = [elevation, slope, aspect, curvature, roughness, local_diff]
        if normalize:
            channels = [self._normalize_channel(ch) for ch in channels]
        return torch.cat(channels, dim=1)


class DEMGen(ModalGeneratorBase):
    """
    离线 DEM-like 模态生成器。

    通过 run(source) 批量生成 DEM 特征并保存。
    """

    backend_name = "dem_features"

    def __init__(
        self,
        source_modality: str = "rgb",
        device: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        save_dir: str | Path | None = None,
        keep_structure: bool = True,
        overwrite: bool = False,
        split: Optional[str | List[str]] = None,
        save_format: str = "npy",
        normalize: bool = True,
        gaussian_ksize: int = 3,
        sobel_ksize: int = 3,
        roughness_ksize: int = 5,
        local_diff_ksize: int = 15,
        use_scharr: bool = False,
        **kwargs,
    ) -> None:
        self.source_modality = str(source_modality).lower()
        self.save_format = str(save_format).lower()
        if self.save_format not in {"npy", "npz", "png"}:
            raise ValueError("save_format 必须为 'npy'/'npz'/'png'")
        self.normalize = bool(normalize)
        # split 允许传入 "train,val" / ["train","val"] / None
        self.split = split

        self.gaussian_ksize = gaussian_ksize
        self.sobel_ksize = sobel_ksize
        self.roughness_ksize = roughness_ksize
        self.local_diff_ksize = local_diff_ksize
        self.use_scharr = use_scharr

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

        self.model: Optional[DEMFeatureGenerator] = None
        self._img_formats = {
            "bmp",
            "jpeg",
            "jpg",
            "png",
            "tif",
            "tiff",
            "webp",
        }
        self._path_meta: Dict[str, Dict[str, Any]] = {}
        self._yaml_root: Optional[Path] = None
        self._debug_logged: set[str] = set()

    def run(self, source: str | Path | Iterable[str | Path], save_dir: str | Path | None = None):
        """允许在调用时临时指定 save_dir，并通过 split 控制子集。"""
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

    def _gather_sources(self, source: str | Path | Iterable[str | Path]) -> List[str]:
        # YAML 情况：按 data.yaml 收集 source_modality 的图像
        if isinstance(source, (str, Path)) and str(source).lower().endswith((".yaml", ".yml")):
            yaml_path = Path(check_yaml(source, hard=True))
            self._yaml_root = yaml_path.parent
            data = yaml.safe_load(yaml_path.read_text())

            base = Path(data.get("path", yaml_path.parent))
            if not base.is_absolute():
                base = yaml_path.parent / base

            modality_dirs = data.get("modality", {"rgb": "images"})
            if self.source_modality not in modality_dirs:
                raise ValueError(f"data.yaml 未包含 source_modality={self.source_modality} 的目录映射")
            source_dir = modality_dirs[self.source_modality]

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
                if getattr(self, "split_names", None) and self.split_names:
                    if split_name not in self.split_names:
                        continue
                split_list = split_val if isinstance(split_val, list) else [split_val]
                for s in split_list:
                    p = Path(s)
                    if not p.is_absolute():
                        p = base / p
                    if not p.exists():
                        raise FileNotFoundError(f"数据集路径不存在: {p}")

                    split_subdir = p.name
                    images_root = base / source_dir
                    input_dir = images_root / split_subdir
                    if not input_dir.exists():
                        raise FileNotFoundError(f"source_modality={self.source_modality} split 目录不存在: {input_dir}")

                    dem_root_default = images_root.parent / "images_dem"
                    for f in input_dir.rglob("*.*"):
                        if f.is_file() and f.suffix[1:].lower() in self._img_formats:
                            fstr = str(f)
                            paths.append(fstr)
                            self._path_meta[fstr] = {
                                "split": split_subdir,
                                "images_root": images_root,
                                "dem_root_default": dem_root_default,
                            }

            if not paths:
                raise RuntimeError(f"{yaml_path} 未找到任何图像文件")
            return sorted(paths)

        # 普通路径/列表：沿用基类逻辑后按图像格式过滤
        paths = super()._gather_sources(source)
        filtered = [p for p in paths if Path(p).suffix[1:].lower() in self._img_formats]
        if not filtered:
            raise RuntimeError("未找到任何可处理的图像文件")
        return sorted(filtered)

    # ---- 必需接口 ----
    def load_model(self):
        self.model = DEMFeatureGenerator(
            gaussian_ksize=self.gaussian_ksize,
            sobel_ksize=self.sobel_ksize,
            roughness_ksize=self.roughness_ksize,
            local_diff_ksize=self.local_diff_ksize,
            use_scharr=self.use_scharr,
            device=self.device,
        )
        LOGGER.info(f"[{self.method}] DEM 特征生成器初始化完成")

    def preprocess(self, item: str):
        img = cv2.imread(item, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法读取图像: {item}")
        h, w = img.shape[:2]
        if img.ndim == 2:
            tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        else:
            tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, non_blocking=True)
        meta = {"path": item, "orig_hw": (h, w)}
        if item in self._path_meta:
            meta.update(self._path_meta[item])
        return tensor, meta

    def infer(self, batch_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert self.model is not None
        shapes = {tuple(t.shape[-2:]) for t in batch_inputs}
        outputs: List[torch.Tensor] = []
        if len(shapes) == 1:
            batch = torch.cat(batch_inputs, dim=0)
            feats = self.model(batch, normalize=self.normalize)
            outputs = [f for f in feats]
        else:
            for t in batch_inputs:
                feats = self.model(t, normalize=self.normalize)
                outputs.append(feats.squeeze(0))
        return outputs

    def postprocess(self, outputs: List[torch.Tensor], metas: List[Dict[str, Any]]) -> List[np.ndarray]:
        results: List[np.ndarray] = []
        for feat_t, _ in zip(outputs, metas):
            if feat_t.dim() == 4:
                feat_t = feat_t.squeeze(0)
            feat_np = feat_t.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            results.append(feat_np)
        return results

    def save(self, outputs: List[np.ndarray], metas: List[Dict[str, Any]]) -> List[str]:
        paths: List[str] = []
        save_opt: SaveOptions = self.save_options

        for feat_np, meta in zip(outputs, metas):
            src_path = Path(meta["path"])
            split = meta.get("split", "unknown")

            if save_opt.save_dir is None and "dem_root_default" in meta:
                out_dir = Path(meta["dem_root_default"]) / split
            else:
                base_dir = Path(save_opt.save_dir or "runs/dem_gen") / self.method
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

            if self.save_format == "npy":
                out_path = out_dir / f"{stem}.npy"
                if save_opt.overwrite or not out_path.exists():
                    np.save(out_path, feat_np.astype(np.float32))
                paths.append(str(out_path))

            elif self.save_format == "npz":
                out_path = out_dir / f"{stem}.npz"
                if save_opt.overwrite or not out_path.exists():
                    np.savez_compressed(out_path, features=feat_np.astype(np.float32))
                paths.append(str(out_path))

            else:  # png
                feat_u8 = np.clip(feat_np, 0.0, 1.0)
                feat_u8 = (feat_u8 * 255.0).astype(np.uint8)
                p1 = out_dir / f"{stem}.png"
                p2 = out_dir / f"{stem}_ch456.png"
                if save_opt.overwrite or not p1.exists():
                    cv2.imwrite(str(p1), feat_u8[:, :, :3])
                if save_opt.overwrite or not p2.exists():
                    cv2.imwrite(str(p2), feat_u8[:, :, 3:6])
                paths.extend([str(p1), str(p2)])

        return paths


__all__ = ["DEMGen"]
