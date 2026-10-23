"""
EdgeGen：离线边缘（Edge）模态生成器。

目标
----
- 从 RGB 模态图像离线生成 Edge 模态，供多模态训练/推理读取（images_edge/<split>/xxx.png 或 xxx.npy）。
- 与 DepthGen/DEMGen 对齐：提供统一 run(source) 接口，支持 data.yaml 与目录/文件列表。

默认行为
--------
- 输入为 data.yaml 时，按 train/val/test 的 split 收集 source_modality（默认 rgb）图像；
- 生成结果默认保存到 images_edge/<split>/ 下，与 RGB 文件同 stem 的 .png/.npy/.tif 文件；
- 默认输出 3 通道（Xch=3），以适配数据加载侧对标准图像格式的兼容性（也可选 1 通道）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_yaml

from .base import ModalGeneratorBase, SaveOptions


class EdgeKernels:
    """边缘提取相关卷积核。"""

    @staticmethod
    def gaussian_kernel_2d(ksize: int, sigma: float = 0.0) -> torch.Tensor:
        if ksize <= 1:
            return torch.ones((1, 1), dtype=torch.float32)
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
        raise ValueError(f"sobel_ksize 必须为 3 或 5, got {ksize}")

    @staticmethod
    def sobel_kernel_y(ksize: int = 3) -> torch.Tensor:
        return EdgeKernels.sobel_kernel_x(ksize).t()

    @staticmethod
    def scharr_kernel_x() -> torch.Tensor:
        return torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
            dtype=torch.float32,
        )

    @staticmethod
    def scharr_kernel_y() -> torch.Tensor:
        return EdgeKernels.scharr_kernel_x().t()


class EdgeFeatureGenerator(nn.Module):
    """
    轻量边缘强度生成网络（GPU/CPU 统一）。

    输出为单通道 edge strength（建议 normalize=True 后处于 0~1）。
    """

    def __init__(
        self,
        gaussian_ksize: int = 5,
        sobel_ksize: int = 3,
        use_scharr: bool = False,
        normalize: bool = True,
        gamma: float = 1.0,
        binarize: bool = False,
        threshold: Optional[float] = None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        super().__init__()
        self.gaussian_ksize = int(gaussian_ksize)
        self.sobel_ksize = int(sobel_ksize)
        self.use_scharr = bool(use_scharr)
        self.normalize = bool(normalize)
        self.gamma = float(gamma)
        if self.gamma <= 0:
            raise ValueError("gamma 必须 > 0")
        self.binarize = bool(binarize)
        self.threshold = threshold
        if self.threshold is not None and not (0.0 <= float(self.threshold) <= 1.0):
            raise ValueError("threshold 必须在 [0,1] 内（归一化后阈值）")

        if device is None:
            raise ValueError("device 不能为空：请在 EdgeGen 中显式指定 device（或使用其默认 device 参数）。")
        self.device = torch.device(device)
        self._register_kernels()
        self.to(self.device)

    def _register_kernels(self) -> None:
        gauss = EdgeKernels.gaussian_kernel_2d(self.gaussian_ksize)
        self.register_buffer("gaussian_kernel", gauss.view(1, 1, self.gaussian_ksize, self.gaussian_ksize))

        if self.use_scharr:
            gx = EdgeKernels.scharr_kernel_x()
            gy = EdgeKernels.scharr_kernel_y()
            k = 3
        else:
            gx = EdgeKernels.sobel_kernel_x(self.sobel_ksize)
            gy = EdgeKernels.sobel_kernel_y(self.sobel_ksize)
            k = self.sobel_ksize
        self.register_buffer("grad_x", gx.view(1, 1, k, k))
        self.register_buffer("grad_y", gy.view(1, 1, k, k))

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

    def _normalize_01(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x_flat = x.view(b, -1)
        min_val = x_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        max_val = x_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        denom = torch.clamp(max_val - min_val, min=1e-8)
        return (x - min_val) / denom

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()

        # 约定：输入为 0~1 浮点 RGB
        x = x.clamp_(0.0, 1.0)
        gray = self._rgb_to_gray(x)

        # 平滑（降低噪声对梯度的影响）
        if self.gaussian_ksize > 1:
            gray = self._conv2d_reflect(gray, self.gaussian_kernel)

        gx = self._conv2d_reflect(gray, self.grad_x)
        gy = self._conv2d_reflect(gray, self.grad_y)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-12)

        if self.normalize:
            edge = self._normalize_01(edge)

        if self.gamma != 1.0:
            edge = torch.pow(edge, self.gamma)

        if self.binarize:
            thr = float(self.threshold) if self.threshold is not None else 0.5
            edge = (edge >= thr).to(edge.dtype)

        return edge  # (B,1,H,W)


class EdgeGen(ModalGeneratorBase):
    """
    离线 Edge 模态生成器。

    - 输入为 data.yaml 时，默认把输出写到 images_edge/<split>/xxx.(png|npy|npz)
    - 输出通道数由 xch 控制（默认 3 通道）。
    """

    backend_name = "edge"

    def __init__(
        self,
        source_modality: str = "rgb",
        device: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        # 保存相关
        save_dir: str | Path | None = None,
        keep_structure: bool = True,
        overwrite: bool = False,
        split: Optional[str | List[str]] = None,
        save_format: str = "png",
        # 输出通道
        xch: int = 3,
        # 算法参数
        gaussian_ksize: int = 5,
        sobel_ksize: int = 3,
        use_scharr: bool = False,
        normalize: bool = True,
        gamma: float = 1.0,
        binarize: bool = False,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.source_modality = str(source_modality).lower()
        self.split = split

        self.save_format = str(save_format).lower()
        if self.save_format == "tiff":
            self.save_format = "tif"
        if self.save_format not in {"png", "npy", "npz", "tif"}:
            raise ValueError("save_format 必须为 'png'/'npy'/'tif'（可选: 'npz'）")

        self.xch = int(xch)
        if self.xch not in (1, 3):
            raise ValueError("xch 仅支持 1 或 3（单通道/三通道边缘模态）")

        self.gaussian_ksize = int(gaussian_ksize)
        self.sobel_ksize = int(sobel_ksize)
        self.use_scharr = bool(use_scharr)
        self.normalize = bool(normalize)
        self.gamma = float(gamma)
        self.binarize = bool(binarize)
        self.threshold = threshold

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

        self.model: Optional[EdgeFeatureGenerator] = None
        self._img_formats = {
            "bmp",
            "dng",
            "jpeg",
            "jpg",
            "mpo",
            "png",
            "tif",
            "tiff",
            "webp",
            "pfm",
            "heic",
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

                    edge_root_default = images_root.parent / "images_edge"
                    for f in input_dir.rglob("*.*"):
                        if f.is_file() and f.suffix[1:].lower() in self._img_formats:
                            fstr = str(f)
                            paths.append(fstr)
                            self._path_meta[fstr] = {
                                "split": split_subdir,
                                "images_root": images_root,
                                "edge_root_default": edge_root_default,
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
        self.model = EdgeFeatureGenerator(
            gaussian_ksize=self.gaussian_ksize,
            sobel_ksize=self.sobel_ksize,
            use_scharr=self.use_scharr,
            normalize=self.normalize,
            gamma=self.gamma,
            binarize=self.binarize,
            threshold=self.threshold,
            device=self.device,
        )
        LOGGER.info(f"[{self.method}] Edge 生成器初始化完成: xch={self.xch}, format={self.save_format}")

    def preprocess(self, item: str):
        img_bgr = cv2.imread(item, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"无法读取图像: {item}")
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1,3,H,W)
        tensor = tensor.to(self.device, non_blocking=True)
        meta = {"path": item, "orig_hw": (h, w)}
        if item in self._path_meta:
            meta.update(self._path_meta[item])
        return tensor, meta

    def _adapt_channels(self, tensor: torch.Tensor, out_ch: int) -> torch.Tensor:
        c = int(tensor.shape[1])
        if c == out_ch:
            return tensor
        if c == 1 and out_ch == 3:
            return tensor.repeat(1, 3, 1, 1)
        if c == 3 and out_ch == 1:
            return tensor.mean(dim=1, keepdim=True)
        # 通用线性投影：固定权重，确保确定性与可复现
        device, dtype = tensor.device, tensor.dtype
        w = torch.zeros((out_ch, c), device=device, dtype=dtype)
        for i in range(min(out_ch, c)):
            w[i, i] = 1.0
        b, _, hh, ww = tensor.shape
        t = tensor.permute(0, 2, 3, 1).reshape(-1, c)
        out = t @ w.t()
        out = out.reshape(b, hh, ww, out_ch).permute(0, 3, 1, 2).contiguous()
        return out

    def infer(self, batch_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert self.model is not None
        shapes = {tuple(t.shape[-2:]) for t in batch_inputs}
        outputs: List[torch.Tensor] = []

        if len(shapes) == 1:
            batch = torch.cat(batch_inputs, dim=0)
            edge_1ch = self.model(batch)
            edge = self._adapt_channels(edge_1ch, self.xch)
            outputs = [e for e in edge]
        else:
            for t in batch_inputs:
                edge_1ch = self.model(t)
                edge = self._adapt_channels(edge_1ch, self.xch)
                outputs.append(edge.squeeze(0))
        return outputs

    def postprocess(self, outputs: List[torch.Tensor], metas: List[Dict[str, Any]]) -> List[np.ndarray]:
        results: List[np.ndarray] = []
        for edge_t, _ in zip(outputs, metas):
            if edge_t.dim() == 4:
                edge_t = edge_t.squeeze(0)
            edge_np = edge_t.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H,W,C)
            results.append(edge_np)
        return results

    def save(self, outputs: List[np.ndarray], metas: List[Dict[str, Any]]) -> List[str]:
        paths: List[str] = []
        save_opt: SaveOptions = self.save_options

        for edge_np, meta in zip(outputs, metas):
            src_path = Path(meta["path"])
            split = meta.get("split", "unknown")

            if save_opt.save_dir is None and "edge_root_default" in meta:
                out_dir = Path(meta["edge_root_default"]) / split
            else:
                base_dir = Path(save_opt.save_dir or "runs/edge_gen") / self.method
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
                    np.save(out_path, edge_np.astype(np.float32))
                paths.append(str(out_path))

            elif self.save_format == "npz":
                out_path = out_dir / f"{stem}.npz"
                if save_opt.overwrite or not out_path.exists():
                    np.savez_compressed(out_path, features=edge_np.astype(np.float32))
                paths.append(str(out_path))

            elif self.save_format == "tif":
                # TIFF：默认保存为 uint16，提升动态范围与训练侧稳定性
                edge_u16 = np.clip(edge_np, 0.0, 1.0)
                edge_u16 = (edge_u16 * 65535.0).astype(np.uint16)
                out_path = out_dir / f"{stem}.tif"

                if self.xch == 1:
                    img = edge_u16[:, :, 0] if edge_u16.ndim == 3 else edge_u16
                else:
                    img = edge_u16[:, :, :3][:, :, ::-1]  # RGB->BGR

                if save_opt.overwrite or not out_path.exists():
                    cv2.imwrite(str(out_path), img)
                paths.append(str(out_path))

            else:  # png
                edge_u8 = np.clip(edge_np, 0.0, 1.0)
                edge_u8 = (edge_u8 * 255.0).astype(np.uint8)
                out_path = out_dir / f"{stem}.png"

                if self.xch == 1:
                    img = edge_u8[:, :, 0] if edge_u8.ndim == 3 else edge_u8
                else:
                    img = edge_u8[:, :, :3][:, :, ::-1]  # RGB->BGR

                if save_opt.overwrite or not out_path.exists():
                    cv2.imwrite(str(out_path), img)
                paths.append(str(out_path))

        return paths


__all__ = ["EdgeGen"]
