"""Visualization method plugin stubs (to be implemented in later steps)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import cv2
import torch

from .types import CoreVisualizationResult
from .registry import REGISTRY
from .preprocessor import Preprocessor
from .renderer import Renderer


class MethodPlugin:
    """Base interface for visualization method plugins."""

    @staticmethod
    def run(
        *,
        model: Any,
        inputs: Dict[str, Any],
        layers: List[int],
        layer_names: List[str],
        save: bool,
        out_dir: Path,
        modality: str | None,
        family: str,
        **kwargs: Any,
    ) -> List[CoreVisualizationResult]:
        raise NotImplementedError


class HeatmapPlugin(MethodPlugin):
    @staticmethod
    def run(
        *,
        model: Any,
        inputs: Dict[str, Any],
        layers: List[int],
        layer_names: List[str],
        save: bool,
        out_dir: Path,
        modality: str | None,
        family: str,
        **kwargs: Any,
    ) -> List[CoreVisualizationResult]:
        # Import existing heatmap visualizer (reused implementation)
        from ultralytics.models.yolo.multimodal.visualize.heatmap import HeatmapVisualizer
        x_modality_name = str(kwargs.get("x_modality_name", "unknown"))

        # Originals for overlay (保持原始图，不做伪彩)
        originals: Dict[str, np.ndarray] | np.ndarray
        if 'rgb' in inputs and 'x' in inputs:
            originals = {'rgb': inputs['rgb'], 'x': inputs['x']}
            inferred_modality = 'dual'
        elif 'rgb' in inputs:
            originals = inputs['rgb']
            inferred_modality = 'rgb'
        else:
            originals = inputs['x']
            inferred_modality = 'x'

        # 统一前处理（对齐式 letterbox + 显式消融填充），保证输入通道数与模型一致
        align_base = str(kwargs.get('align_base', 'rgb')).lower()
        ablation_fill = str(kwargs.get('ablation_fill', 'zeros')).lower()
        # 支持可视化时显式指定 imgsz（提升热力图空间分辨率/细腻度）
        size_arg = kwargs.get('imgsz', kwargs.get('vis_imgsz', None))
        if size_arg is None:
            # 默认策略：使用模型默认 imgsz；若原图更大，则自动上调到更细腻的可视化尺度（可通过 auto_imgsz 关闭）
            base_size = Preprocessor.model_input_size(model)
            auto_imgsz = bool(kwargs.get('auto_imgsz', True))
            if auto_imgsz:
                try:
                    cap = int(kwargs.get('imgsz_cap', 1280))
                except Exception:
                    raise ValueError(f"imgsz_cap 需要为整数，收到: {kwargs.get('imgsz_cap')!r}")
                if cap <= 0:
                    raise ValueError(f"imgsz_cap 必须为正整数，收到: {cap}")

                def _max_side(x: Any) -> int | None:
                    if isinstance(x, dict):
                        sides = []
                        for v in x.values():
                            s = _max_side(v)
                            if s is not None:
                                sides.append(s)
                        return max(sides) if sides else None
                    if not isinstance(x, np.ndarray) or x.size == 0:
                        return None
                    if x.ndim == 2:
                        return int(max(x.shape[0], x.shape[1]))
                    if x.ndim == 3:
                        return int(max(x.shape[0], x.shape[1]))
                    if x.ndim == 4:
                        # Heuristic: NHWC if last dim looks like channels, else NCHW
                        if x.shape[-1] in (1, 3, 4, 6):
                            return int(max(x.shape[1], x.shape[2]))
                        return int(max(x.shape[2], x.shape[3]))
                    return None

                ms = _max_side(originals)
                if ms is not None and ms > base_size:
                    size = max(base_size, min(ms, cap))
                else:
                    size = base_size
            else:
                size = base_size
        else:
            try:
                size = int(size_arg)
            except Exception:
                raise ValueError(f"imgsz/vis_imgsz 需要为整数（例如 640/1280），收到: {size_arg!r}")
            if size <= 0:
                raise ValueError(f"imgsz/vis_imgsz 必须为正整数，收到: {size}")
        # Unified model-ingress HWC assembly via shared ingress
        hwc, contract = Preprocessor.prepare_model_hwc(
            inputs, model, ablation_fill=ablation_fill, align_base=align_base, size=size,
        )

        input_tensor = torch.from_numpy(np.transpose(hwc, (2, 0, 1))).unsqueeze(0).float()

        # Heatmap visualizer 接收预处理后的张量以避免二次 letterbox，同时使用 originals 做叠加参考
        heat_layers = [str(i) for i in layers]
        alg = kwargs.get('alg', 'gradcam')

        vis = HeatmapVisualizer(model)
        results = vis.visualize(
            images=originals,
            layers=heat_layers,
            alg=alg,
            preprocessed_input=input_tensor,
            original_images=originals,
            **{k: v for k, v in kwargs.items() if k != 'alg'}
        )

        out: List[CoreVisualizationResult] = []
        blend_alpha = float(kwargs.get('blend_alpha', 0.5))
        cmap = kwargs.get('colormap', 'jet')
        cmap_map = {
            'jet': cv2.COLORMAP_JET,
            'turbo': cv2.COLORMAP_TURBO,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'inferno': cv2.COLORMAP_INFERNO,
            'magma': cv2.COLORMAP_MAGMA,
            'plasma': cv2.COLORMAP_PLASMA,
        }
        cmap_name = str(cmap).lower()
        if cmap_name not in cmap_map:
            raise ValueError(f"不支持的 colormap: {cmap}，可选：{', '.join(sorted(cmap_map.keys()))}")
        cmap_cv2 = cmap_map[cmap_name]

        layout = str(kwargs.get('layout', 'panel')).lower().strip()  # overlay|panel|both
        if layout not in {'overlay', 'panel', 'both'}:
            raise ValueError(f"layout 参数非法：{layout}，可选：overlay|panel|both")
        panel_scale = float(kwargs.get('panel_scale', 1.0))
        panel_title = bool(kwargs.get('panel_title', True))

        # 叠加底图控制：overlay ∈ {rgb, x, dual}，默认 rgb；若仅提供 X，则自动转为 x
        overlay_req = kwargs.get('overlay', None)
        has_rgb = 'rgb' in inputs
        has_x = 'x' in inputs
        overlay_base: str
        if overlay_req is None:
            # 双模态默认输出 dual（更符合人类查看习惯：同时看到 RGB 与 X）
            overlay_base = 'dual' if (has_rgb and has_x) else ('rgb' if has_rgb else ('x' if has_x else 'rgb'))
        else:
            overlay_base = str(overlay_req).lower().strip()
            if overlay_base not in {'rgb', 'x', 'dual'}:
                raise ValueError(f"overlay 参数非法：{overlay_req}，可选：rgb|x|dual")
            # 自动调整：若用户仅传 X，但要求 rgb，则调整为 x（按用户要求允许）
            if overlay_base == 'rgb' and (not has_rgb) and has_x:
                overlay_base = 'x'
        # 对于 overlay='x' 但无 X，或 overlay='dual' 但非双模态，Fail-Fast
        if overlay_base == 'x' and not has_x:
            raise ValueError("overlay='x' 需要提供 X 模态输入")
        if overlay_base == 'dual' and not (has_rgb and has_x):
            raise ValueError("overlay='dual' 需要同时提供 RGB 与 X 输入")

        # 导出控制：为论文/报告提供“可组合素材”输出（每层一个文件夹）
        export_components = bool(kwargs.get('export_components', True))
        export_panel = bool(kwargs.get('export_panel', True))
        export_scale_arg = kwargs.get('export_scale', kwargs.get('save_scale', None))
        if export_scale_arg is None:
            export_scale = 1.0
        else:
            try:
                export_scale = float(export_scale_arg)
            except Exception:
                raise ValueError(f"export_scale/save_scale 需要为数字，收到: {export_scale_arg!r}")
            if not np.isfinite(export_scale) or export_scale <= 0:
                raise ValueError(f"export_scale/save_scale 必须为正数，收到: {export_scale}")

        img_key = kwargs.get('img_key', None)

        for li, r in zip(layers, results):
            originals = getattr(r, 'original_image', None)
            heatmaps = getattr(r, 'heatmap', None)
            overlays = getattr(r, 'overlay', None)
            if originals is None or heatmaps is None:
                raise ValueError("HeatmapVisualizer 返回结果缺少 original_image/heatmap 字段，无法生成可视化输出。")

            # 每层一个目录，便于论文中自由组合素材
            layer_subdir = f"layer{int(li):03d}"

            assets: Dict[str, np.ndarray] = {}

            # 输出选择：overlay(兼容) / panel(更适合人看) / both(保存全部组件)
            if overlay_base == 'dual':
                if not isinstance(originals, dict) or not isinstance(heatmaps, dict):
                    raise ValueError("overlay='dual' 时期望 HeatmapResult 返回 dict 形式的 original_image/heatmap。")
                if overlays is None or not isinstance(overlays, dict):
                    # 兼容：若 HeatmapVisualizer 未生成 overlay，则在此生成
                    overlays = Renderer.heat_overlay_multimodal(originals, heatmaps, alpha=blend_alpha, colormap=cmap_cv2)

                need_panel = layout in {'panel', 'both'}
                panel_rgb = None
                panel_x = None
                panel_dual = None
                if need_panel:
                    panel_rgb = Renderer.heat_triptych(
                        originals['rgb'],
                        heatmaps['rgb'],
                        overlays.get('rgb'),
                        alpha=blend_alpha,
                        colormap=cmap_cv2,
                        scale=panel_scale,
                        title=panel_title,
                    )
                    panel_x = Renderer.heat_triptych(
                        originals['x'],
                        heatmaps['x'],
                        overlays.get('x'),
                        alpha=blend_alpha,
                        colormap=cmap_cv2,
                        scale=panel_scale,
                        title=panel_title,
                    )
                    # 双模态：默认纵向拼接（RGB 在上，X 在下）
                    if panel_rgb.shape[1] != panel_x.shape[1]:
                        raise ValueError(
                            f"RGB/X 三联图宽度不一致，无法拼接：rgb={panel_rgb.shape} x={panel_x.shape}。"
                            f"请确保两路输入分辨率一致，或使用 align_base/统一数据预处理后再可视化。"
                        )
                    sep = np.ones((8, panel_rgb.shape[1], 3), dtype=np.uint8) * 255
                    panel_dual = np.vstack([panel_rgb, sep, panel_x])

                if export_components:
                    for mk in ('rgb', 'x'):
                        if mk not in originals or mk not in heatmaps:
                            continue
                        ov_m = overlays.get(mk)
                        if ov_m is None:
                            ov_m = Renderer.heat_overlay(originals[mk], heatmaps[mk], alpha=blend_alpha, colormap=cmap_cv2)
                        asset_prefix = 'rgb' if mk == 'rgb' else x_modality_name
                        assets[f"{asset_prefix}/original"] = originals[mk]
                        assets[f"{asset_prefix}/heatmap"] = heatmaps[mk]
                        assets[f"{asset_prefix}/overlay"] = ov_m
                        if export_panel:
                            # 优先复用主预览 panel（避免重复计算）
                            if mk == 'rgb' and panel_rgb is not None:
                                assets[f"{asset_prefix}/panel"] = panel_rgb
                            elif mk == 'x' and panel_x is not None:
                                assets[f"{asset_prefix}/panel"] = panel_x
                            else:
                                assets[f"{asset_prefix}/panel"] = Renderer.heat_triptych(
                                    originals[mk],
                                    heatmaps[mk],
                                    ov_m,
                                    alpha=blend_alpha,
                                    colormap=cmap_cv2,
                                    scale=panel_scale,
                                    title=panel_title,
                                )
                    # 导出一个双模态总览 panel（无论主布局是否需要）
                    if export_panel:
                        pr = assets.get("rgb/panel", panel_rgb)
                        px = assets.get(f"{x_modality_name}/panel", panel_x)
                        if pr is not None and px is not None:
                            if pr.shape[1] != px.shape[1]:
                                raise ValueError(
                                    f"RGB/X 导出 panel 宽度不一致，无法拼接：rgb={pr.shape} x={px.shape}。"
                                    f"请确保两路输入分辨率一致，或仅导出单路素材。"
                                )
                            sep = np.ones((8, pr.shape[1], 3), dtype=np.uint8) * 255
                            assets["dual/panel"] = np.vstack([pr, sep, px])

                if layout == 'overlay':
                    data = {'rgb': overlays['rgb'], 'x': overlays['x']}
                elif layout == 'panel':
                    data = panel_dual
                else:  # both
                    data = {
                        'dual_panel': panel_dual,
                        'rgb_panel': panel_rgb,
                        'x_panel': panel_x,
                        'rgb_original': originals['rgb'],
                        'x_original': originals['x'],
                        'rgb_heatmap': heatmaps['rgb'],
                        'x_heatmap': heatmaps['x'],
                        'rgb_overlay': overlays['rgb'],
                        'x_overlay': overlays['x'],
                    }
            else:
                # 单底图（rgb 或 x）
                if isinstance(originals, dict):
                    if overlay_base not in originals or overlay_base not in heatmaps:
                        raise ValueError(f"叠加底图 {overlay_base} 在输入中不可用")
                    orig = originals[overlay_base]
                    hm = heatmaps[overlay_base]
                    ov = overlays.get(overlay_base) if isinstance(overlays, dict) else None
                else:
                    orig = originals
                    hm = heatmaps
                    ov = overlays
                if ov is None:
                    ov = Renderer.heat_overlay(orig, hm, alpha=blend_alpha, colormap=cmap_cv2)

                if export_components:
                    # 若同时提供 RGB+X，但 overlay 仅选择其一：仍导出两路素材，便于论文自由组合
                    if isinstance(originals, dict) and isinstance(heatmaps, dict):
                        for mk in ('rgb', 'x'):
                            if mk not in originals or mk not in heatmaps:
                                continue
                            ov_m = overlays.get(mk) if isinstance(overlays, dict) else None
                            if ov_m is None:
                                ov_m = Renderer.heat_overlay(originals[mk], heatmaps[mk], alpha=blend_alpha, colormap=cmap_cv2)
                            asset_prefix = 'rgb' if mk == 'rgb' else x_modality_name
                            assets[f"{asset_prefix}/original"] = originals[mk]
                            assets[f"{asset_prefix}/heatmap"] = heatmaps[mk]
                            assets[f"{asset_prefix}/overlay"] = ov_m
                            if export_panel:
                                assets[f"{asset_prefix}/panel"] = Renderer.heat_triptych(
                                    originals[mk],
                                    heatmaps[mk],
                                    ov_m,
                                    alpha=blend_alpha,
                                    colormap=cmap_cv2,
                                    scale=panel_scale,
                                    title=panel_title,
                                )
                    else:
                        mk = 'rgb' if overlay_base == 'rgb' else x_modality_name
                        assets[f"{mk}/original"] = orig
                        assets[f"{mk}/heatmap"] = hm
                        assets[f"{mk}/overlay"] = ov
                        if export_panel:
                            assets[f"{mk}/panel"] = Renderer.heat_triptych(
                                orig,
                                hm,
                                ov,
                                alpha=blend_alpha,
                                colormap=cmap_cv2,
                                scale=panel_scale,
                                title=panel_title,
                            )

                if layout == 'overlay':
                    data = ov
                elif layout == 'panel':
                    data = Renderer.heat_triptych(
                        orig,
                        hm,
                        ov,
                        alpha=blend_alpha,
                        colormap=cmap_cv2,
                        scale=panel_scale,
                        title=panel_title,
                    )
                else:  # both
                    data = {
                        'panel': Renderer.heat_triptych(
                            orig,
                            hm,
                            ov,
                            alpha=blend_alpha,
                            colormap=cmap_cv2,
                            scale=panel_scale,
                            title=panel_title,
                        ),
                        'original': orig,
                        'heatmap': hm,
                        'overlay': ov,
                    }

            modality_token = x_modality_name if overlay_base == 'x' else overlay_base
            meta = {
                'method': 'heat',
                'layer_idx': li,
                # 文件命名采用叠加底图语义（rgb/x/dual）
                'modality': modality_token,
                'x_modality_name': x_modality_name,
                'family': family,
                'algorithm': alg,
                'alpha': blend_alpha,
                'colormap': cmap,
                'layout': layout,
                'panel_scale': panel_scale,
                'subdir': layer_subdir,
            }
            if isinstance(img_key, str) and len(img_key) > 0:
                meta['img_key'] = img_key
            out.append(CoreVisualizationResult(type='heat', data=data, meta=meta))

            # 额外导出：每层的原图/热图/叠加图（可选含 panel），用于论文拼图
            if export_components and assets:
                asset_meta = dict(meta)
                if export_scale != 1.0:
                    asset_meta['save_scale'] = float(export_scale)
                out.append(CoreVisualizationResult(type='heat_assets', data=assets, meta=asset_meta))
        return out


class FeatureMapPlugin(MethodPlugin):
    @staticmethod
    def run(
        *,
        model: Any,
        inputs: Dict[str, Any],
        layers: List[int],
        layer_names: List[str],
        save: bool,
        out_dir: Path,
        modality: str | None,
        family: str,
        **kwargs: Any,
    ) -> List[CoreVisualizationResult]:
        """
        基础版特征图可视化（与热力图对齐语义）：
        - 自动识别单/双模态：仅一侧时自动做消融（另一侧默认补 0），两侧都有则按双模态。
        - 统一 letterbox（可选对齐基准），拼接为 HWC[3+Xch]，再转 NCHW。
        - 逐层：通道打分（sum/var）→ 选 top_k → 网格渲染（默认灰度）。
        - 返回每层一张网格图，保存命名与热力图风格保持一致（feature_layer{idx}_...）。
        """

        # -----------------
        # 参数与前置校验
        # -----------------
        align_base = str(kwargs.get('align_base', 'rgb')).lower()
        metric = str(kwargs.get('metric', 'sum')).lower()  # 'sum' | 'var'
        top_k = int(kwargs.get('top_k', 8))
        normalize = str(kwargs.get('normalize', 'minmax')).lower()  # 仅实现 minmax
        colormap = str(kwargs.get('colormap', 'gray')).lower()
        split = bool(kwargs.get('split', False))
        x_modality_name = str(kwargs.get("x_modality_name", "unknown"))

        # 自动推断/执行：若只传一侧则自动消融填充，默认 zeros（不新增控制参数）
        has_rgb = 'rgb' in inputs
        has_x = 'x' in inputs
        if modality is None or str(modality).lower() == 'auto':
            m = ('dual' if (has_rgb and has_x) else ('rgb' if has_rgb else 'x'))
        else:
            m = str(modality).lower()
        ablation_fill = str(kwargs.get('ablation_fill', 'zeros')).lower()

        # -----------------
        # 预处理：统一模型入口 HWC 组装（共享 ingress）
        # -----------------
        hwc, contract = Preprocessor.prepare_model_hwc(
            inputs, model, ablation_fill=ablation_fill, align_base=align_base,
        )

        # NCHW + batch
        nchw = np.transpose(hwc, (2, 0, 1))[None, ...].astype(np.float32)
        inp = torch.from_numpy(nchw)

        # -----------------
        # 前向 hook 捕获指定层输出（逐层）
        # -----------------
        feats: Dict[int, torch.Tensor] = {}
        handles = []

        def _first_tensor(x: Any) -> torch.Tensor | None:
            if isinstance(x, torch.Tensor):
                return x
            if isinstance(x, (list, tuple)):
                for t in x:
                    if isinstance(t, torch.Tensor):
                        return t
            return None

        for li in layers:
            if not hasattr(model, 'model') or li < 0 or li >= len(model.model):
                raise ValueError(f"层索引越界：{li}")
            mod = model.model[li]

            def _hook_closure(idx: int):
                def _hook(module, inputs_, output):
                    t = _first_tensor(output)
                    if t is not None:
                        feats[idx] = t.detach()
                return _hook

            handles.append(mod.register_forward_hook(_hook_closure(li)))

        # 前向执行
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        inp = inp.to(device)
        with torch.no_grad():
            _ = model(inp)

        # 清理 hook
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

        # -----------------
        # 打分与选通道 → 网格渲染（逐层输出）
        # -----------------
        def _score_channel_map(t: torch.Tensor) -> torch.Tensor:
            # t: [C,H,W]
            if metric == 'sum':
                return t.abs().sum(dim=(1, 2))
            elif metric == 'var':
                return t.var(dim=(1, 2))
            else:
                raise ValueError(f"不支持的 metric: {metric}（仅 'sum'|'var'）")

        def _norm_uint8(arr: np.ndarray) -> np.ndarray:
            if normalize == 'minmax':
                a_min, a_max = float(arr.min()), float(arr.max())
                if a_max > a_min:
                    out = (arr - a_min) / (a_max - a_min)
                else:
                    out = np.zeros_like(arr)
                return (out * 255.0).astype(np.uint8)
            else:
                raise ValueError(f"不支持的 normalize: {normalize}（基础版仅 'minmax'）")

        def _cv2_colormap(name: str) -> int | None:
            name = str(name).lower()
            if name in {'gray', 'grey', 'grayscale', 'none'}:
                return None
            m = {
                'jet': cv2.COLORMAP_JET,
                'viridis': cv2.COLORMAP_VIRIDIS,
                'inferno': cv2.COLORMAP_INFERNO,
                'magma': cv2.COLORMAP_MAGMA,
                'plasma': cv2.COLORMAP_PLASMA,
            }
            return m.get(name, cv2.COLORMAP_VIRIDIS)

        def _render_grid(feat_list: List[tuple[int, float, np.ndarray]], cell: tuple[int, int] = (128, 128)) -> np.ndarray:
            # feat_list: list of (ch_idx, score, 2D np.array)
            if not feat_list:
                return np.zeros((256, 256, 3), dtype=np.uint8)
            n = len(feat_list)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            h, w = cell
            pad = 5
            label_h = 22
            canvas = np.ones((rows * (h + label_h + pad) + pad, cols * (w + pad) + pad, 3), dtype=np.uint8) * 255

            cmap = _cv2_colormap(colormap)

            for i, (ch, sc, fm) in enumerate(feat_list):
                r, c = i // cols, i % cols
                y = pad + r * (h + label_h + pad)
                x = pad + c * (w + pad)
                u8 = _norm_uint8(fm)
                u8 = cv2.resize(u8, (w, h), interpolation=cv2.INTER_AREA)
                if cmap is None:
                    rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.applyColorMap(u8, cmap)
                    try:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    except Exception:
                        pass
                canvas[y:y + h, x:x + w] = rgb
                label = f"ch:{ch} score:{sc:.1f}"
                cv2.putText(canvas, label, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            return canvas

        modality_meta = {
            ('rgb', True): 'rgb_ablate',
            ('x', True): f'{x_modality_name}_ablate',
        }.get((m, not (has_rgb and has_x)), m)
        if modality_meta == 'x':
            modality_meta = x_modality_name

        results: List[CoreVisualizationResult] = []
        for li in layers:
            t = feats.get(li, None)
            if t is None:
                # 指定层可能无张量输出（例如返回 tuple 无张量）
                continue
            if t.dim() == 4:
                # [B,C,H,W] → 仅支持逐样本；基础版默认 B=1
                b, c, h_, w_ = t.shape
                # 逐样本处理（基础版仅取第 0 个）
                tt = t[0]
            elif t.dim() == 3:
                # [C,H,W]
                tt = t
                c = tt.shape[0]
            else:
                # 非 3/4 维数据不处理
                continue

            scores = _score_channel_map(tt)
            k = int(min(max(1, top_k), int(scores.numel())))
            top_idx = torch.topk(scores, k).indices.cpu().numpy().tolist()

            feat_list: List[tuple[int, float, np.ndarray]] = []
            tiles_imgs: List[np.ndarray] = []
            tiles_channels: List[int] = []
            tiles_scores: List[float] = []

            def _render_tile(fm: np.ndarray, cell: tuple[int, int] = (128, 128)) -> np.ndarray:
                u8 = _norm_uint8(fm)
                u8 = cv2.resize(u8, (cell[1], cell[0]), interpolation=cv2.INTER_AREA)
                cmap_code = _cv2_colormap(colormap)
                if cmap_code is None:
                    return cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.applyColorMap(u8, cmap_code)
                    try:
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    except Exception:
                        pass
                    return rgb

            for ch in top_idx:
                fmap = tt[ch].detach().cpu().float().numpy()
                sc = float(scores[ch].detach().cpu().item())
                feat_list.append((int(ch), sc, fmap))
                if split:
                    tiles_imgs.append(_render_tile(fmap))
                    tiles_channels.append(int(ch))
                    tiles_scores.append(sc)

            grid = _render_grid(feat_list)
            meta = {
                'method': 'feature',
                'layer_idx': li,
                'modality': modality_meta,
                'x_modality_name': x_modality_name,
                'family': family,
                'metric': metric,
                'top_k': top_k,
                'normalize': normalize,
                'align_base': align_base,
            }
            # 附带 img_key（目录/批量模式用于分目录保存）
            img_key = kwargs.get('img_key', inputs.get('img_key') if isinstance(inputs, dict) else None)
            if img_key is not None:
                meta['img_key'] = img_key
            results.append(CoreVisualizationResult(type='feature', data=grid, meta=meta))

            if split and tiles_imgs:
                tiles_meta = {
                    **meta,
                    'channels': tiles_channels,
                    'scores': tiles_scores,
                    'subdir': f'layer{li}',
                }
                if img_key is not None:
                    tiles_meta['img_key'] = img_key
                results.append(CoreVisualizationResult(type='feature_tiles', data=tiles_imgs, meta=tiles_meta))

        return results


# Register plugins
REGISTRY.register('heat', HeatmapPlugin)
REGISTRY.register('feature', FeatureMapPlugin)
