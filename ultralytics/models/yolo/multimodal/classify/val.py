# Ultralytics YOLO, AGPL-3.0 license

"""
多模态分类验证器

本模块提供多模态分类任务的验证器，支持 RGB+X 模态的图像分类验证。
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch

from ultralytics.data import YOLOMultiModalClassifyDataset, build_dataloader
from ultralytics.data.multimodal.precision_contract import (
    apply_dataset_precision_contract,
    extract_batch_precision_contract,
)
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.utils.multimodal.vis import (
    concat_side_by_side,
    extract_precision_contract_from_batch,
    prepare_rgb_for_display,
    split_modalities,
    visualize_x_to_3ch,
)
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images
from ultralytics.nn.mm.utils import normalize_modality_token


class MultiModalClassificationValidator(BaseValidator):
    """
    多模态分类验证器

    继承 BaseValidator，提供多模态分类任务的验证流程，
    包括指标计算、混淆矩阵生成和结果可视化。

    Attributes:
        targets: 真实标签列表
        pred: 预测结果列表
        metrics: 分类指标对象
        names: 类别名称映射
        nc: 类别数量
        confusion_matrix: 混淆矩阵

    Methods:
        get_desc: 返回指标描述字符串
        init_metrics: 初始化指标
        preprocess: 预处理批次数据
        update_metrics: 更新指标
        build_dataset: 构建多模态分类数据集

    Examples:
        >>> from ultralytics.models.yolo.multimodal.classify import MultiModalClassificationValidator
        >>> validator = MultiModalClassificationValidator(args={"data": "mm_cls.yaml"})
        >>> validator()
    """

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None
    ) -> None:
        """
        初始化多模态分类验证器

        Args:
            dataloader: 数据加载器
            save_dir: 保存目录
            args: 配置参数
            _callbacks: 回调函数列表
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.targets = None
        self.pred = None

        # 仅对 rgb/x token 做归一化：rgb/RGB→RGB、x/X→X（其它模态名保持原样）
        self.modality = normalize_modality_token(getattr(self.args, "modality", None))
        if isinstance(self.args, dict):
            self.args["modality"] = self.modality
        else:
            setattr(self.args, "modality", self.modality)

        self.args.task = "classify"
        self.metrics = ClassifyMetrics()

    def get_desc(self) -> str:
        """返回指标描述字符串"""
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """初始化指标"""
        self.names = model.names
        self.nc = len(model.names)
        self.pred = []
        self.targets = []
        self.confusion_matrix = ConfusionMatrix(names=list(model.names.values()))

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """预处理批次数据，contract-aware 精度归一化。"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        raw_dtype = batch["img"].dtype
        batch["img"] = apply_dataset_precision_contract(
            batch["img"].float(),
            extract_batch_precision_contract(batch),
            raw_dtype=raw_dtype,
        )
        if self.args.half:
            batch["img"] = batch["img"].half()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds: torch.Tensor, batch: Dict[str, Any]) -> None:
        """更新指标"""
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        self.targets.append(batch["cls"].type(torch.int32).cpu())

    # ------------------------------------------------------------------
    # AFSS per-sample scoring helpers (classification-specific semantics)
    # ------------------------------------------------------------------

    def _afss_resolve_im_files(self, count: int) -> List[str]:
        """Resolve image file paths for current AFSS scoring batch using sequential dataset cursor."""
        dataloader = getattr(self, "dataloader", None)
        dataset = getattr(dataloader, "dataset", None)
        if dataset is None or not hasattr(dataset, "samples"):
            raise RuntimeError("AFSS classification scoring requires dataloader.dataset.samples")

        dataset_id = id(dataset)
        if getattr(self, "_afss_score_dataset_id", None) != dataset_id:
            self._afss_score_dataset_id = dataset_id
            self._afss_score_cursor = 0

        start = int(getattr(self, "_afss_score_cursor", 0))
        end = start + int(count)
        if end > len(dataset.samples):
            raise RuntimeError(
                "AFSS classification scoring cursor overflow: "
                f"cursor={start}, batch_count={count}, dataset_size={len(dataset.samples)}"
            )

        im_files: List[str] = []
        for si in range(start, end):
            sample = dataset.samples[si]
            if isinstance(sample, (list, tuple)) and sample:
                im_file = sample[0]
            else:
                raise RuntimeError(
                    f"AFSS classification scoring expects dataset.samples[{si}] to contain file path at index 0"
                )
            im_files.append(str(im_file))
        self._afss_score_cursor = end
        return im_files

    def _afss_get_task_override(self, key: str, default: Any = None) -> Any:
        """Read classify-specific AFSS overrides from args in a namespace-safe way."""
        raw_overrides = getattr(self.args, "afss_task_overrides", {})
        if hasattr(raw_overrides, "__dict__"):
            raw_overrides = vars(raw_overrides).copy()
        elif not isinstance(raw_overrides, dict):
            try:
                raw_overrides = dict(raw_overrides)
            except TypeError:
                raw_overrides = {}

        classify_override = raw_overrides.get("classify", {})
        if hasattr(classify_override, "__dict__"):
            classify_override = vars(classify_override).copy()
        elif not isinstance(classify_override, dict):
            try:
                classify_override = dict(classify_override)
            except TypeError:
                classify_override = {}
        return classify_override.get(key, default)

    def afss_score_batch(self, preds: torch.Tensor, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build AFSS score rows for classification using top1_prob_if_correct semantics."""
        if preds.ndim != 2:
            raise RuntimeError(
                f"AFSS classification scoring expects logits/probs with shape [B, C], got {tuple(preds.shape)}"
            )
        sufficiency_mode = str(self._afss_get_task_override("sufficiency_mode", "top1_prob_if_correct"))
        if sufficiency_mode != "top1_prob_if_correct":
            raise RuntimeError(
                "AFSS classify only supports sufficiency_mode='top1_prob_if_correct'. "
                f"Got {sufficiency_mode!r}"
            )

        probs = preds.float().softmax(dim=1)
        batch_size = int(probs.shape[0])
        topk = min(2, int(probs.shape[1]))
        topk_prob, topk_idx = probs.topk(k=topk, dim=1)
        targets = batch["cls"].view(-1).long()
        if targets.numel() != batch_size:
            raise RuntimeError(
                "AFSS classification scoring batch size mismatch: "
                f"logits_batch={batch_size}, targets={targets.numel()}"
            )

        im_files = self._afss_resolve_im_files(batch_size)
        rows: List[Dict[str, Any]] = []
        for si in range(batch_size):
            target = int(targets[si].item())
            pred_class = int(topk_idx[si, 0].item())
            top1_prob = float(topk_prob[si, 0].item())
            margin = (
                float((topk_prob[si, 0] - topk_prob[si, 1]).item())
                if topk > 1
                else top1_prob
            )
            top1_correct = int(pred_class == target)
            sufficiency_raw = top1_prob if top1_correct else 0.0

            rows.append(
                {
                    "im_file": im_files[si],
                    "task_name": "classify",
                    # Compatibility fields only; do not treat as detect-style precision/recall semantics.
                    "precision_op": float(top1_correct),
                    "recall_op": top1_prob,
                    "sufficiency_raw": sufficiency_raw,
                    "valid_for_afss": True,
                    "task_metrics": {
                        "top1_correct": top1_correct,
                        "top1_prob": top1_prob,
                        "margin": margin,
                        "target_class": target,
                        "pred_class": pred_class,
                        "sufficiency_mode": sufficiency_mode,
                    },
                }
            )
        return rows

    def score_batch(self, preds: torch.Tensor, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Backward-compatible AFSS batch helper for external callers."""
        return self.afss_score_batch(preds, batch)

    def finalize_metrics(self) -> None:
        """完成指标计算"""
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir,
                    normalize=normalize,
                    on_plot=self.on_plot
                )
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
        self.metrics.confusion_matrix = self.confusion_matrix

    def postprocess(
        self,
        preds: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
    ) -> torch.Tensor:
        """后处理预测结果"""
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def get_stats(self) -> Dict[str, float]:
        """计算并返回指标"""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def get_dataloader(self, dataset_path, batch_size):
        """构建并返回数据加载器"""
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def build_dataset(self, img_path: str) -> YOLOMultiModalClassifyDataset:
        """
        构建多模态分类数据集

        Args:
            img_path: 图像路径

        Returns:
            YOLOMultiModalClassifyDataset: 多模态分类数据集
        """
        # 获取数据配置
        if hasattr(self, 'data') and self.data:
            data = self.data
        else:
            data = check_det_dataset(self.args.data)

        return YOLOMultiModalClassifyDataset(
            img_path=img_path,
            data=data,
            args=self.args,
            augment=False,
            prefix=self.args.split,
            precision_report_dir=self.save_dir,
        )

    def print_results(self) -> None:
        """打印评估结果"""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """绘制验证样本，多模态 RGB/X/并排三组可视化。"""
        x_modality = self.data.get("x_modality", "unknown")
        xch = self.data.get("Xch", 3)
        batch_idx = torch.arange(len(batch["img"]))

        # 拆分模态
        rgb_images, x_images = split_modalities(batch["img"], xch)
        precision_contract = extract_precision_contract_from_batch(batch)

        # RGB labels 可视化
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])
        plot_images(
            images=rgb_visual,
            batch_idx=batch_idx,
            cls=batch["cls"],
            fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

        # X 模态 labels 可视化
        x_visual = visualize_x_to_3ch(
            x_images, colorize=False, x_modality=x_modality, precision=precision_contract["x"]
        )
        plot_images(
            images=x_visual,
            batch_idx=batch_idx,
            cls=batch["cls"],
            fname=self.save_dir / f"val_batch{ni}_labels_{x_modality}.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

        # 并排 multimodal labels 可视化
        side = concat_side_by_side(rgb_visual, x_visual)
        plot_images(
            images=side,
            batch_idx=batch_idx,
            cls=batch["cls"],
            fname=self.save_dir / f"val_batch{ni}_labels_multimodal.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self,
        batch: Dict[str, Any],
        preds: torch.Tensor,
        ni: int
    ) -> None:
        """绘制预测结果，多模态 RGB/X/并排三组可视化。"""
        x_modality = self.data.get("x_modality", "unknown")
        xch = self.data.get("Xch", 3)
        batch_idx = torch.arange(len(batch["img"]))
        pred_cls = torch.argmax(preds, dim=1)

        # 拆分模态
        rgb_images, x_images = split_modalities(batch["img"], xch)
        precision_contract = extract_precision_contract_from_batch(batch)

        # RGB pred 可视化
        rgb_visual = prepare_rgb_for_display(rgb_images, precision=precision_contract["rgb"])
        plot_images(
            images=rgb_visual,
            batch_idx=batch_idx,
            cls=pred_cls,
            fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

        # X 模态 pred 可视化
        x_visual = visualize_x_to_3ch(
            x_images, colorize=False, x_modality=x_modality, precision=precision_contract["x"]
        )
        plot_images(
            images=x_visual,
            batch_idx=batch_idx,
            cls=pred_cls,
            fname=self.save_dir / f"val_batch{ni}_pred_{x_modality}.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

        # 并排 multimodal pred 可视化
        side = concat_side_by_side(rgb_visual, x_visual)
        plot_images(
            images=side,
            batch_idx=batch_idx,
            cls=pred_cls,
            fname=self.save_dir / f"val_batch{ni}_pred_multimodal.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
