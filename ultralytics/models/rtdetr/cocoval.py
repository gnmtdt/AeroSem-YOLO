import csv
import torch
from ultralytics.models.yolo.multimodal.cocoval import MultiModalCOCOValidator
from ultralytics.utils import ops


class RTDETRMMCOCOValidator(MultiModalCOCOValidator):
    """COCO validator for RT-DETR multi-modal models."""
    
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize RT-DETR multi-modal COCO validator.
        
        Args:
            dataloader: Dataloader for validation dataset
            save_dir: Directory to save validation results
            pbar: Progress bar instance
            args: Validation arguments
            _callbacks: Callback functions
        """
        super().__init__(dataloader=dataloader, save_dir=save_dir, pbar=pbar, args=args, _callbacks=_callbacks)
    
    def init_metrics(self, model):
        """Initialize metrics for RT-DETR multi-modal COCO validation.
        
        Args:
            model: The RT-DETR model to extract metrics configuration from
        """
        # Call parent class init_metrics
        super().init_metrics(model)
        self.model = model
        
        # Initialize RT-DETR specific metrics
        self.end2end = getattr(model, "end2end", False)
        self.names = getattr(model, 'names', {})
        self.seen = 0
        self.jdict = []
        self.num_images_processed = 0
    
    def get_desc(self):
        """Return a formatted string for the progress bar description.
        
        Returns:
            str: Formatted string with column headers for validation output
        """
        # 进度表头只保留 COCO 指标列，避免冗余前缀与列粘连
        return ("%22s" + "%11s" * 3) % ("Class", "Images", "Instances", "COCO-mAP@.5:.95")
    
    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs.
        
        Args:
            preds: Raw predictions from RT-DETR model as a tuple.
        
        Returns:
            List[Dict[str, torch.Tensor]]: List of dictionaries for each image, each containing:
                - 'bboxes': Tensor of shape (N, 4) with bounding box coordinates
                - 'conf': Tensor of shape (N,) with confidence scores
                - 'cls': Tensor of shape (N,) with class indices
        """
        # Handle tuple input format for RT-DETR
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        # Extract batch size and dimensions
        bs, _, nd = preds[0].shape

        # Split predictions into bboxes and scores
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        # Scale bboxes to image size
        # RT-DETR bbox 默认是归一化 xywh（0-1）。若输入为非方形（如 rect=True），应按 (h,w) 分别缩放。
        imgsz_hw = getattr(self, "_imgsz_hw", None)
        if imgsz_hw is None or len(imgsz_hw) != 2:
            raise RuntimeError(
                "RTDETRMMCOCOValidator: 缺少本批次推理输入尺寸信息，无法进行正确 bbox 缩放。"
                "请确认 preprocess() 被调用且 batch['img'] 形状有效。"
            )
        h, w = int(imgsz_hw[0]), int(imgsz_hw[1])
        scale = torch.tensor([w, h, w, h], device=bboxes.device, dtype=bboxes.dtype)
        bboxes = bboxes * scale

        # Initialize outputs
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs

        # Process each image in the batch
        for i, bbox in enumerate(bboxes):
            # Convert from xywh to xyxy format
            bbox = ops.xywh2xyxy(bbox)

            # Get max score and class for each prediction
            score, cls = scores[i].max(-1)

            # Combine bbox, score, and class
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)

            # Filter then sort (避免 score 排序后与 mask 索引错位)
            keep = score > self.args.conf
            pred = pred[keep]
            if pred.numel():
                pred = pred[pred[:, 4].argsort(descending=True)]
            outputs[i] = pred

        # Return formatted results
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]} for x in outputs]

    def preprocess(self, batch):
        """记录本批次模型实际输入尺寸（H,W），供 postprocess 做正确 bbox 缩放。"""
        batch = super().preprocess(batch)
        if "img" not in batch:
            raise RuntimeError("RTDETRMMCOCOValidator: preprocess() 未收到 batch['img']，无法继续 COCO 验证。")
        self._imgsz_hw = tuple(batch["img"].shape[2:4])
        if len(self._imgsz_hw) != 2:
            raise RuntimeError(f"RTDETRMMCOCOValidator: 非法 batch['img'] 形状，无法解析(H,W)：{batch['img'].shape}")
        return batch
    
    def _prepare_pred(self, pred, pbatch):
        """将预测 bbox 从推理输入坐标系还原到原图坐标系（与 DetectionValidator 口径一致）。"""
        cls = pred["cls"]
        if self.args.single_cls:
            cls *= 0
        if "imgsz" not in pbatch or "ori_shape" not in pbatch or "ratio_pad" not in pbatch:
            raise RuntimeError(
                "RTDETRMMCOCOValidator: _prepare_pred() 缺少必要字段：pbatch 必须包含 imgsz/ori_shape/ratio_pad。"
            )
        bboxes = ops.scale_boxes(
            pbatch["imgsz"], pred["bboxes"].clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )
        return {"bboxes": bboxes, "conf": pred["conf"], "cls": cls}
    
    def print_results(self):
        """Print COCO evaluation results and save CSV files.
        
        This method calls the parent class print_results which includes
        automatic CSV file generation through _save_csv_results().
        """
        # Call parent class print_results which handles everything including CSV saving
        super().print_results()
