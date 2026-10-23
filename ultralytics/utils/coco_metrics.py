# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
COCO evaluation metrics implementation for object detection.

This module provides a pure Python implementation of COCO evaluation metrics,
without dependency on pycocotools. It calculates standard COCO metrics including
AP, AP50, AP75, APsmall, APmedium, APlarge, AR1, AR10, AR100, ARsmall, ARmedium, ARlarge.
"""

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import SimpleClass, LOGGER

# COCO size thresholds (in pixels squared)
COCO_AREA_SMALL = 32 ** 2  # area < 1024
COCO_AREA_MEDIUM = 96 ** 2  # 1024 <= area < 9216
# Objects with area >= 9216 are considered large


class COCOMetrics(SimpleClass):
    """
    COCO evaluation metrics for object detection.
    
    This class computes and stores standard COCO evaluation metrics for object detection tasks.
    It provides a clean interface for calculating AP (Average Precision) and AR (Average Recall)
    metrics across different IoU thresholds, object sizes, and detection limits.
    
    Attributes:
        save_dir (Path): Directory for saving evaluation results and plots.
        names (dict): Dictionary mapping class indices to class names.
        plot (bool): Whether to generate evaluation plots.
        on_plot (callable): Callback function for plot events.
        stats (dict): Raw statistics from evaluation computations.
        eval_stats (dict): Processed evaluation statistics.
        
        # Primary COCO metrics (12 standard metrics)
        metrics.coco.AP (float): Average Precision averaged over IoU thresholds [0.5:0.05:0.95].
        metrics.coco.AP50 (float): Average Precision at IoU threshold 0.5.
        metrics.coco.AP75 (float): Average Precision at IoU threshold 0.75.
        metrics.coco.APsmall (float): Average Precision for small objects (area < 32²).
        metrics.coco.APmedium (float): Average Precision for medium objects (32² < area < 96²).
        metrics.coco.APlarge (float): Average Precision for large objects (area > 96²).
        metrics.coco.AR1 (float): Average Recall with at most 1 detection per image.
        metrics.coco.AR10 (float): Average Recall with at most 10 detections per image.
        metrics.coco.AR100 (float): Average Recall with at most 100 detections per image.
        metrics.coco.ARsmall (float): Average Recall for small objects.
        metrics.coco.ARmedium (float): Average Recall for medium objects.
        metrics.coco.ARlarge (float): Average Recall for large objects.
        
    Example:
        ```python
        from ultralytics.utils.coco_metrics import COCOMetrics
        
        # Initialize metrics
        metrics = COCOMetrics(save_dir='runs/val', names={0: 'person', 1: 'car'})
        
        # After evaluation, access metrics
        print(f"mAP: {metrics.AP:.3f}")
        print(f"mAP50: {metrics.AP50:.3f}")
        print(f"mAP75: {metrics.AP75:.3f}")
        ```
    """
    
    def __init__(self, save_dir=Path('.'), names=None, plot=False, on_plot=None):
        """
        Initialize COCOMetrics.
        
        Args:
            save_dir (str | Path): Directory to save results. Defaults to current directory.
            names (dict): Dictionary mapping class indices to names. Defaults to None.
            plot (bool): Whether to generate plots. Defaults to False.
            on_plot (callable): Callback for plot events. Defaults to None.
        """
        self.save_dir = Path(save_dir)
        self.names = names or {}
        self.plot = plot
        self.on_plot = on_plot
        
        # Storage for computation statistics
        self.stats = None  # Raw statistics from evaluation
        self.eval_stats = None  # Processed evaluation statistics
        
        # Initialize all COCO metrics to 0.0
        # Primary metrics (AP - Average Precision)
        self.AP = 0.0  # COCO primary metric - averaged over IoU thresholds [0.5:0.05:0.95]
        self.AP50 = 0.0  # COCO metric at IoU threshold 0.5 (traditional detection metric)
        self.AP75 = 0.0  # COCO metric at IoU threshold 0.75 (strict detection metric)
        
        # Size-specific AP metrics
        self.APsmall = 0.0  # AP for small objects: area < 32²
        self.APmedium = 0.0  # AP for medium objects: 32² < area < 96²
        self.APlarge = 0.0  # AP for large objects: area > 96²
        
        # Size-specific AP50 metrics (新增：按尺寸的AP50指标)
        self.APsmall50 = 0.0  # AP50 for small objects
        self.APmedium50 = 0.0  # AP50 for medium objects  
        self.APlarge50 = 0.0  # AP50 for large objects
        
        # Size-specific AP75 metrics (新增：按尺寸的AP75指标)
        self.APsmall75 = 0.0  # AP75 for small objects
        self.APmedium75 = 0.0  # AP75 for medium objects
        self.APlarge75 = 0.0  # AP75 for large objects
        
        # Recall metrics (AR - Average Recall)
        self.AR1 = 0.0  # AR given 1 detection per image
        self.AR10 = 0.0  # AR given 10 detections per image
        self.AR100 = 0.0  # AR given 100 detections per image (COCO default)
        
        # Size-specific AR metrics
        self.ARsmall = 0.0  # AR for small objects
        self.ARmedium = 0.0  # AR for medium objects
        self.ARlarge = 0.0  # AR for large objects
        
    def __repr__(self):
        """Return string representation showing key metrics."""
        return (
            f"{self.__class__.__name__}("
            f"AP={self.AP:.3f}, "
            f"AP50={self.AP50:.3f}, "
            f"AP75={self.AP75:.3f}, "
            f"AR100={self.AR100:.3f})"
        )
        
    def update(self, stats):
        """
        Update metrics with computed statistics.
        
        Args:
            stats (dict): Dictionary containing computed statistics with keys:
                - 'AP': Average Precision at IoU [0.5:0.05:0.95]
                - 'AP50': Average Precision at IoU 0.5
                - 'AP75': Average Precision at IoU 0.75
                - 'APsmall': AP for small objects
                - 'APmedium': AP for medium objects
                - 'APlarge': AP for large objects
                - 'AR1': Average Recall with 1 detection
                - 'AR10': Average Recall with 10 detections
                - 'AR100': Average Recall with 100 detections
                - 'ARsmall': AR for small objects
                - 'ARmedium': AR for medium objects
                - 'ARlarge': AR for large objects
        """
        self.stats = stats
        
        # Update all metrics from stats dictionary
        self.AP = stats.get('AP', 0.0)
        self.AP50 = stats.get('AP50', 0.0)
        self.AP75 = stats.get('AP75', 0.0)
        self.APsmall = stats.get('APsmall', 0.0)
        self.APmedium = stats.get('APmedium', 0.0)
        self.APlarge = stats.get('APlarge', 0.0)
        # 扩展尺寸+IoU指标（若提供则更新）
        self.APsmall50 = stats.get('APsmall50', getattr(self, 'APsmall50', 0.0))
        self.APsmall75 = stats.get('APsmall75', getattr(self, 'APsmall75', 0.0))
        self.APmedium50 = stats.get('APmedium50', getattr(self, 'APmedium50', 0.0))
        self.APmedium75 = stats.get('APmedium75', getattr(self, 'APmedium75', 0.0))
        self.APlarge50 = stats.get('APlarge50', getattr(self, 'APlarge50', 0.0))
        self.APlarge75 = stats.get('APlarge75', getattr(self, 'APlarge75', 0.0))

        self.AR1 = stats.get('AR1', 0.0)
        self.AR10 = stats.get('AR10', 0.0)
        self.AR100 = stats.get('AR100', 0.0)
        self.ARsmall = stats.get('ARsmall', 0.0)
        self.ARmedium = stats.get('ARmedium', 0.0)
        self.ARlarge = stats.get('ARlarge', 0.0)
        
    def get_summary_dict(self):
        """
        Get a summary dictionary of all metrics.
        
        Returns:
            dict: Dictionary containing all 12 COCO metrics.
        """
        return {
            'metrics/coco/AP': self.AP,
            'metrics/coco/AP50': self.AP50,
            'metrics/coco/AP75': self.AP75,
            'metrics/coco/APsmall': self.APsmall,
            'metrics/coco/APmedium': self.APmedium,
            'metrics/coco/APlarge': self.APlarge,
            'metrics/coco/AR1': self.AR1,
            'metrics/coco/AR10': self.AR10,
            'metrics/coco/AR100': self.AR100,
            'metrics/coco/ARsmall': self.ARsmall,
            'metrics/coco/ARmedium': self.ARmedium,
            'metrics/coco/ARlarge': self.ARlarge,
        }
        
    def print_results(self):
        """Print formatted COCO evaluation results."""
        print("\nCOCO Evaluation Results:")
        print("-" * 50)
        print(f"{'Metric':<20} {'Value':>10}")
        print("-" * 50)
        
        # Print AP metrics
        print(f"{'AP (IoU=0.50:0.95)':<20} {self.AP:>10.3f}")
        print(f"{'AP (IoU=0.50)':<20} {self.AP50:>10.3f}")
        print(f"{'AP (IoU=0.75)':<20} {self.AP75:>10.3f}")
        print(f"{'AP (small)':<20} {self.APsmall:>10.3f}")
        print(f"{'AP (medium)':<20} {self.APmedium:>10.3f}")
        print(f"{'AP (large)':<20} {self.APlarge:>10.3f}")
        
        print("-" * 50)
        
        # Print AR metrics
        print(f"{'AR (max=1)':<20} {self.AR1:>10.3f}")
        print(f"{'AR (max=10)':<20} {self.AR10:>10.3f}")
        print(f"{'AR (max=100)':<20} {self.AR100:>10.3f}")
        print(f"{'AR (small)':<20} {self.ARsmall:>10.3f}")
        print(f"{'AR (medium)':<20} {self.ARmedium:>10.3f}")
        print(f"{'AR (large)':<20} {self.ARlarge:>10.3f}")
        print("-" * 50)
    
    @staticmethod
    def calculate_bbox_area(bbox, ori_shape, from_format: str = 'xyxy', normalized: bool = True):
        """
        Calculate the area of a bounding box.
        
        Args:
            bbox (array-like): Bounding box coordinates.
            ori_shape (tuple): Original image shape as (height, width).
            from_format (str): Input format, 'xyxy' or 'xywh'. Defaults to 'xyxy'.
            normalized (bool): Whether input coordinates are normalized (0-1). Defaults to True.
        
        Returns:
            float: Area of the bounding box in pixels squared.
        
        Example:
            ```python
            # For a normalized bbox [0.1, 0.2, 0.3, 0.4] on a 640x480 image
            area = COCOMetrics.calculate_bbox_area([0.1, 0.2, 0.3, 0.4], (480, 640))
            ```
        """
        bbox = np.array(bbox, dtype=np.float32)
        h, w = ori_shape
        if bbox.shape[-1] != 4:
            raise ValueError(f"Invalid bbox format: expected 4 values, got shape {bbox.shape}")
        x1, y1, x2, y2 = bbox.copy()
        if from_format == 'xywh':
            x2 = x1 + x2
            y2 = y1 + y2
        elif from_format != 'xyxy':
            raise ValueError("from_format must be 'xyxy' or 'xywh'")
        if normalized:
            x1, x2 = x1 * w, x2 * w
            y1, y2 = y1 * h, y2 * h
        area = float(abs((x2 - x1) * (y2 - y1)))
        return area
    
    @staticmethod
    def classify_bbox_size(area):
        """
        Classify a bounding box as small, medium, or large based on its area.
        
        Args:
            area (float): Area of the bounding box in pixels squared.
        
        Returns:
            str: Size category ('small', 'medium', or 'large').
        
        Note:
            - small: area < 32²
            - medium: 32² ≤ area < 96²
            - large: area ≥ 96²
        
        Example:
            ```python
            size_category = COCOMetrics.classify_bbox_size(1500)  # Returns 'medium'
            ```
        """
        if area < COCO_AREA_SMALL:
            return 'small'
        elif area < COCO_AREA_MEDIUM:
            return 'medium'
        else:
            return 'large'
    
    def _calculate_ap_single_class(self, tp, conf, pred_cls, target_cls, plot=False, save_dir=Path('.'), 
                                   names=(), eps=1e-16, iou_thresholds=None):
        """
        Calculate Average Precision (AP) for a single class using COCO's 101-point interpolation method.
        
        Args:
            tp (np.ndarray): True positive indicators of shape (n_detections,) or (n_detections, n_iou_thresholds).
            conf (np.ndarray): Confidence scores of shape (n_detections,).
            pred_cls (np.ndarray): Predicted class indices of shape (n_detections,).
            target_cls (np.ndarray): Target class indices of shape (n_targets,).
            plot (bool): Whether to plot precision-recall curves. Defaults to False.
            save_dir (Path): Directory to save plots. Defaults to current directory.
            names (tuple): Class names for plotting. Defaults to empty tuple.
            eps (float): Small epsilon to prevent division by zero. Defaults to 1e-16.
            iou_thresholds (np.ndarray): IoU thresholds for evaluation. Defaults to np.linspace(0.5, 0.95, 10).
        
        Returns:
            tuple: (ap, p, r, f1, unique_classes) where:
                - ap (np.ndarray): Average precision for each class and IoU threshold.
                - p (np.ndarray): Precision values at confidence thresholds.
                - r (np.ndarray): Recall values at confidence thresholds.
                - f1 (np.ndarray): F1 scores at confidence thresholds.
                - unique_classes (np.ndarray): Array of unique class indices.
        
        Note:
            This implementation follows COCO's evaluation methodology:
            - Uses 101 recall thresholds from 0 to 1 with step 0.01
            - Applies monotonically decreasing interpolation
            - Supports multiple IoU thresholds (default: 0.5:0.05:0.95)
        """
        # Set default IoU thresholds if not provided (COCO standard: 0.5:0.05:0.95)
        if iou_thresholds is None:
            iou_thresholds = np.linspace(0.5, 0.95, 10)
        
        # Ensure inputs are numpy arrays
        tp = np.asarray(tp)
        conf = np.asarray(conf)
        pred_cls = np.asarray(pred_cls)
        target_cls = np.asarray(target_cls)
        
        # Get unique classes that appear in targets
        unique_classes = np.unique(target_cls).astype(int)
        n_classes = unique_classes.shape[0]
        n_iou = len(iou_thresholds) if tp.ndim > 1 else 1
        
        # Initialize outputs
        ap = np.zeros((n_classes, n_iou))
        # Use 1000 points for curve interpolation (same as standard DetMetrics)
        # This is NOT a limit on detections, but interpolation points for the curves
        curve_points = 1000
        p = np.zeros((n_classes, curve_points))  # Precision curve points
        r = np.zeros((n_classes, curve_points))  # Recall curve points
        
        # For curve interpolation
        x = np.linspace(0, 1, curve_points)
        
        # COCO uses 101-point interpolation (recall thresholds at 0.00, 0.01, ..., 1.00)
        recall_thresholds = np.linspace(0, 1, 101)
        
        # Process each class
        for ci, c in enumerate(unique_classes):
            # Get indices for current class
            class_mask = pred_cls == c
            n_positives = (target_cls == c).sum()  # Number of ground truth objects
            
            if class_mask.sum() == 0 or n_positives == 0:
                continue
            
            # Sort by confidence (descending)
            sorted_indices = np.argsort(-conf[class_mask])
            
            # Get sorted true positives and confidence
            if tp.ndim == 1:
                tp_sorted = tp[class_mask][sorted_indices]
                tp_sorted = tp_sorted.reshape(-1, 1)  # Add IoU dimension
            else:
                tp_sorted = tp[class_mask][sorted_indices]
            
            conf_sorted = conf[class_mask][sorted_indices]
            
            # Calculate precision and recall for each IoU threshold
            for iou_idx in range(n_iou):
                tp_cumsum = tp_sorted[:, iou_idx].cumsum()
                fp_cumsum = (1 - tp_sorted[:, iou_idx]).cumsum()
                
                # Precision and recall
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
                recall = tp_cumsum / (n_positives + eps)
                
                # Store interpolated curves for plotting (similar to ap_per_class in metrics.py)
                if iou_idx == 0:  # Store values for IoU=0.5 for plotting
                    # Interpolate precision and recall curves to fixed points
                    # This allows handling any number of detections
                    if len(conf_sorted) > 0:
                        # Interpolate recall curve
                        r[ci] = np.interp(x, recall, recall, left=0, right=recall[-1] if len(recall) > 0 else 0)
                        # Interpolate precision curve (use negative x for correct interpolation)
                        p[ci] = np.interp(-x, -conf_sorted, precision, left=1, right=0)
                
                # COCO-style AP calculation with 101-point interpolation
                # For each recall threshold, find the maximum precision at or above that recall
                ap_at_recall = np.zeros(len(recall_thresholds))
                
                for i, recall_thresh in enumerate(recall_thresholds):
                    # Find all recalls >= threshold
                    mask = recall >= recall_thresh
                    if mask.any():
                        ap_at_recall[i] = precision[mask].max()
                    else:
                        ap_at_recall[i] = 0.0
                
                # Average over all recall thresholds
                ap[ci, iou_idx] = ap_at_recall.mean()
        
        # Calculate F1 scores
        f1 = 2 * p * r / (p + r + eps)
        
        # Plot if requested
        if plot and save_dir.exists():
            # Implementation of plotting can be added here if needed
            pass
        
        return ap, p, r, f1, unique_classes
    
    def _calculate_ap_by_size(self, tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, ori_shapes, size_range):
        """
        Calculate Average Precision (AP) for objects within a specific size range.
        
        This method filters predictions and ground truth objects based on their size (area) and 
        calculates AP only for objects within the specified size range. This is used to compute
        COCO's size-specific metrics: APsmall, APmedium, and APlarge.
        
        Args:
            tp (np.ndarray): True positive indicators of shape (n_detections, n_iou_thresholds).
            conf (np.ndarray): Confidence scores of shape (n_detections,).
            pred_cls (np.ndarray): Predicted class indices of shape (n_detections,).
            target_cls (np.ndarray): Target class indices of shape (n_targets,).
            pred_boxes (np.ndarray): Predicted bounding boxes of shape (n_detections, 4).
            target_boxes (np.ndarray): Target bounding boxes of shape (n_targets, 4).
            ori_shapes (list): List of original image shapes as (height, width) tuples.
            size_range (tuple): Size range as (min_area, max_area) in pixels squared.
                               Use (0, 32²) for small, (32², 96²) for medium, (96², inf) for large.
        
        Returns:
            float: Average Precision for objects within the specified size range.
        
        Note:
            - Boxes should be in normalized coordinates (0-1).
            - The method filters both predictions and targets by size.
            - Uses COCO's standard 101-point interpolation for AP calculation.
        
        Example:
            ```python
            # Calculate AP for small objects
            ap_small = metrics._calculate_ap_by_size(
                tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, 
                ori_shapes, (0, 32**2)
            )
            ```
        """
        min_area, max_area = size_range
        
        # Filter predictions by size
        pred_keep_mask = np.zeros(len(pred_boxes), dtype=bool)
        for i, bbox in enumerate(pred_boxes):
            # Get the image index for this prediction (assuming sequential ordering)
            img_idx = min(i // (len(pred_boxes) // len(ori_shapes)), len(ori_shapes) - 1)
            area = self.calculate_bbox_area(bbox, ori_shapes[img_idx])
            pred_keep_mask[i] = min_area <= area < max_area
        
        # Filter targets by size
        target_keep_mask = np.zeros(len(target_boxes), dtype=bool)
        for i, bbox in enumerate(target_boxes):
            # Get the image index for this target
            img_idx = min(i // (len(target_boxes) // len(ori_shapes)), len(ori_shapes) - 1)
            area = self.calculate_bbox_area(bbox, ori_shapes[img_idx])
            target_keep_mask[i] = min_area <= area < max_area
        
        # If no objects in this size range, return 0
        if not pred_keep_mask.any() and not target_keep_mask.any():
            return 0.0
        
        # Filter inputs
        filtered_tp = tp[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_conf = conf[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_pred_cls = pred_cls[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_target_cls = target_cls[target_keep_mask] if target_keep_mask.any() else np.array([])
        
        # If no predictions or targets after filtering, return 0
        if len(filtered_tp) == 0 or len(filtered_target_cls) == 0:
            return 0.0
        
        # Calculate AP for filtered objects
        ap, _, _, _, _ = self._calculate_ap_single_class(
            filtered_tp, filtered_conf, filtered_pred_cls, filtered_target_cls
        )
        
        # Return mean AP across all classes and IoU thresholds
        return ap.mean()
    
    def _calculate_ar_at_max_dets(self, tp, conf, pred_cls, target_cls, max_dets):
        """
        Calculate Average Recall (AR) at a specific maximum number of detections per image.
        
        This method computes the average recall considering only the top 'max_dets' detections
        per image, sorted by confidence score. This is used to calculate COCO's AR metrics:
        AR1, AR10, and AR100.
        
        Args:
            tp (np.ndarray): True positive indicators of shape (n_detections, n_iou_thresholds).
            conf (np.ndarray): Confidence scores of shape (n_detections,).
            pred_cls (np.ndarray): Predicted class indices of shape (n_detections,).
            target_cls (np.ndarray): Target class indices of shape (n_targets,).
            max_dets (int): Maximum number of detections to consider per image (1, 10, or 100).
        
        Returns:
            float: Average Recall considering at most max_dets detections per image.
        
        Note:
            - Detections are sorted by confidence score before applying the limit.
            - AR is averaged over all IoU thresholds (0.5:0.05:0.95) and classes.
            - Images are assumed to be processed sequentially in the input arrays.
        
        Example:
            ```python
            # Calculate AR with at most 10 detections per image
            ar10 = metrics._calculate_ar_at_max_dets(tp, conf, pred_cls, target_cls, max_dets=10)
            ```
        """
        # Get unique classes
        unique_classes = np.unique(target_cls).astype(int)
        n_iou = tp.shape[1] if tp.ndim > 1 else 1
        
        # Storage for recall values
        recall_values = []
        
        # Process each class
        for c in unique_classes:
            # Get indices for current class
            pred_mask = pred_cls == c
            n_positives = (target_cls == c).sum()
            
            if pred_mask.sum() == 0 or n_positives == 0:
                continue
            
            # Get predictions for this class
            class_tp = tp[pred_mask]
            class_conf = conf[pred_mask]
            
            # Sort by confidence (descending) and keep only top max_dets
            sorted_indices = np.argsort(-class_conf)
            if len(sorted_indices) > max_dets:
                sorted_indices = sorted_indices[:max_dets]
            
            # Calculate recall for each IoU threshold
            for iou_idx in range(n_iou):
                if class_tp.ndim == 1:
                    tp_at_iou = class_tp[sorted_indices]
                else:
                    tp_at_iou = class_tp[sorted_indices, iou_idx]
                
                # Calculate recall with proper bounds
                true_positives = tp_at_iou.sum()
                recall = min(true_positives / max(n_positives, 1e-16), 1.0)
                recall_values.append(recall)
        
        # If no valid detections, return 0
        if not recall_values:
            return 0.0
        
        # Average over all classes and IoU thresholds
        return np.mean(recall_values)
    
    def _calculate_ar_by_size(self, tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, 
                              ori_shapes, size_range, max_dets):
        """
        Calculate Average Recall (AR) for objects within a specific size range and detection limit.
        
        This method combines size-based filtering with maximum detection limits to calculate
        AR metrics like ARsmall, ARmedium, and ARlarge at specific detection thresholds.
        
        Args:
            tp (np.ndarray): True positive indicators of shape (n_detections, n_iou_thresholds).
            conf (np.ndarray): Confidence scores of shape (n_detections,).
            pred_cls (np.ndarray): Predicted class indices of shape (n_detections,).
            target_cls (np.ndarray): Target class indices of shape (n_targets,).
            pred_boxes (np.ndarray): Predicted bounding boxes of shape (n_detections, 4).
            target_boxes (np.ndarray): Target bounding boxes of shape (n_targets, 4).
            ori_shapes (list): List of original image shapes as (height, width) tuples.
            size_range (tuple): Size range as (min_area, max_area) in pixels squared.
            max_dets (int): Maximum number of detections to consider per image.
        
        Returns:
            float: Average Recall for objects within the size range with detection limit.
        
        Example:
            ```python
            # Calculate AR for small objects with at most 100 detections
            ar_small_100 = metrics._calculate_ar_by_size(
                tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, 
                ori_shapes, (0, 32**2), max_dets=100
            )
            ```
        """
        min_area, max_area = size_range
        
        # Filter predictions by size
        pred_keep_mask = np.zeros(len(pred_boxes), dtype=bool)
        for i, bbox in enumerate(pred_boxes):
            img_idx = min(i // (len(pred_boxes) // len(ori_shapes)), len(ori_shapes) - 1)
            area = self.calculate_bbox_area(bbox, ori_shapes[img_idx])
            pred_keep_mask[i] = min_area <= area < max_area
        
        # Filter targets by size
        target_keep_mask = np.zeros(len(target_boxes), dtype=bool)
        for i, bbox in enumerate(target_boxes):
            img_idx = min(i // (len(target_boxes) // len(ori_shapes)), len(ori_shapes) - 1)
            area = self.calculate_bbox_area(bbox, ori_shapes[img_idx])
            target_keep_mask[i] = min_area <= area < max_area
        
        # If no objects in this size range, return 0
        if not pred_keep_mask.any() and not target_keep_mask.any():
            return 0.0
        
        # Filter inputs
        filtered_tp = tp[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_conf = conf[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_pred_cls = pred_cls[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_target_cls = target_cls[target_keep_mask] if target_keep_mask.any() else np.array([])
        
        # If no predictions or targets after filtering, return 0
        if len(filtered_tp) == 0 or len(filtered_target_cls) == 0:
            return 0.0
        
        # Calculate AR with max_dets limit for filtered objects
        return self._calculate_ar_at_max_dets(
            filtered_tp, filtered_conf, filtered_pred_cls, filtered_target_cls, max_dets
        )
    
    @staticmethod
    def convert_bbox_format(bbox, from_format='xywh', to_format='xyxy', normalized=True, img_shape=None):
        """
        Convert bounding box between different formats and coordinate systems.
        
        This method supports conversion between common bounding box formats (xywh, xyxy) and
        handles normalization/denormalization of coordinates. It's essential for ensuring
        compatibility between different detection frameworks and evaluation tools.
        
        Args:
            bbox (np.ndarray | torch.Tensor | list): Bounding box(es) to convert.
                Can be a single box [4] or batch of boxes [N, 4].
            from_format (str): Source format - 'xywh' (x,y,width,height) or 'xyxy' (x1,y1,x2,y2).
                Defaults to 'xywh'.
            to_format (str): Target format - 'xywh' or 'xyxy'. Defaults to 'xyxy'.
            normalized (bool): Whether input coordinates are normalized (0-1). Defaults to True.
            img_shape (tuple): Image shape as (height, width) for denormalization. 
                Required if normalized=True and you want pixel coordinates.
                               
        Returns:
            np.ndarray: Converted bounding box(es) in the same shape as input.
        
        Raises:
            ValueError: If formats are invalid or img_shape is missing when needed.
        
        Note:
            - Supports both numpy arrays and PyTorch tensors.
            - Handles batch processing efficiently.
            - Preserves input data type (numpy/torch).
        
        Example:
            ```python
            # Convert single box from xywh to xyxy format
            bbox_xyxy = COCOMetrics.convert_bbox_format(
                [0.5, 0.5, 0.2, 0.3], 'xywh', 'xyxy'
            )
            
            # Convert batch with denormalization
            bboxes_pixel = COCOMetrics.convert_bbox_format(
                bboxes_norm, 'xywh', 'xyxy', normalized=True, img_shape=(480, 640)
            )
            ```
        """
        # Validate formats
        valid_formats = {'xywh', 'xyxy'}
        if from_format not in valid_formats or to_format not in valid_formats:
            raise ValueError(f"Invalid format. Must be one of {valid_formats}")
        
        # Handle input type
        is_torch = False
        if torch.is_tensor(bbox):
            is_torch = True
            device = bbox.device
            bbox = bbox.cpu().numpy()
        else:
            bbox = np.array(bbox)
        
        # Ensure float type for calculations
        bbox = bbox.astype(np.float32)
        
        # Get original shape
        original_shape = bbox.shape
        if bbox.ndim == 1:
            bbox = bbox.reshape(1, -1)
        
        # Validate bbox shape
        if bbox.shape[-1] != 4:
            raise ValueError(f"Bounding box must have 4 values, got {bbox.shape[-1]}")
        
        # Denormalize if needed
        if normalized and img_shape is not None:
            h, w = img_shape
            bbox = bbox.copy()
            bbox[..., [0, 2]] *= w  # x coordinates
            bbox[..., [1, 3]] *= h  # y coordinates
            normalized = False  # Mark as denormalized
        
        # Convert format if needed
        if from_format != to_format:
            bbox_converted = bbox.copy()
            
            if from_format == 'xywh' and to_format == 'xyxy':
                # xywh to xyxy: (x, y, w, h) -> (x1, y1, x2, y2)
                bbox_converted[..., 2] = bbox[..., 0] + bbox[..., 2]  # x2 = x + w
                bbox_converted[..., 3] = bbox[..., 1] + bbox[..., 3]  # y2 = y + h
            
            elif from_format == 'xyxy' and to_format == 'xywh':
                # xyxy to xywh: (x1, y1, x2, y2) -> (x, y, w, h)
                bbox_converted[..., 2] = bbox[..., 2] - bbox[..., 0]  # w = x2 - x1
                bbox_converted[..., 3] = bbox[..., 3] - bbox[..., 1]  # h = y2 - y1
            
            bbox = bbox_converted
        
        # Restore original shape
        if len(original_shape) == 1:
            bbox = bbox.squeeze(0)
        
        # Convert back to torch if needed
        if is_torch:
            bbox = torch.from_numpy(bbox).to(device)
        
        return bbox
    
    def process(self, tp, conf, pred_cls, target_cls, pred_boxes=None, target_boxes=None, ori_shapes=None,
                pred_img_indices=None, target_img_indices=None):
        """
        Process evaluation results and calculate all COCO metrics.
        
        This method serves as the main entry point for COCO metric calculation. It takes
        prediction results and ground truth data, computes all standard COCO metrics including
        AP at various IoU thresholds and object sizes, as well as AR metrics with different
        detection limits.
        
        Args:
            tp (np.ndarray | torch.Tensor): True positive indicators with shape 
                (n_detections,) for single IoU or (n_detections, n_iou_thresholds) for multiple IoUs.
                Each element indicates whether a detection is a true positive.
            conf (np.ndarray | torch.Tensor): Confidence scores for each detection,
                shape (n_detections,).
            pred_cls (np.ndarray | torch.Tensor): Predicted class indices for each detection,
                shape (n_detections,).
            target_cls (np.ndarray | torch.Tensor): Ground truth class indices,
                shape (n_targets,).
            pred_boxes (np.ndarray | torch.Tensor, optional): Predicted bounding boxes in
                normalized format [x1, y1, x2, y2], shape (n_detections, 4).
                Required for size-specific metrics.
            target_boxes (np.ndarray | torch.Tensor, optional): Ground truth bounding boxes in
                normalized format [x1, y1, x2, y2], shape (n_targets, 4).
                Required for size-specific metrics.
            ori_shapes (list | np.ndarray, optional): Original image shapes as (height, width)
                for each image in the batch. Required for size-specific metrics.
        
        Updates:
            All metric attributes of the class are updated:
            - AP, AP50, AP75: Average Precision at different IoU thresholds
            - APsmall, APmedium, APlarge: Size-specific AP metrics
            - AR1, AR10, AR100: Average Recall at different detection limits
            - ARsmall, ARmedium, ARlarge: Size-specific AR metrics
            - stats: Raw statistics for each class
        
        Note:
            - If pred_boxes, target_boxes, or ori_shapes are not provided, size-specific
              metrics (APsmall, APmedium, APlarge, ARsmall, ARmedium, ARlarge) will be 0.
            - The method handles both single and multiple IoU threshold inputs automatically.
            - All inputs are converted to numpy arrays for processing.
        
        Example:
            ```python
            # Basic usage with detection results
            metrics = COCOMetrics(names={0: 'person', 1: 'car'})
            metrics.process(tp, conf, pred_cls, target_cls)
            
            # Full usage with size-specific metrics
            metrics.process(
                tp, conf, pred_cls, target_cls,
                pred_boxes=pred_boxes,
                target_boxes=target_boxes,
                ori_shapes=[(480, 640), (720, 1280)]
            )
            ```
        """
        # Convert inputs to numpy arrays if they are torch tensors
        if torch.is_tensor(tp):
            tp = tp.cpu().numpy()
        if torch.is_tensor(conf):
            conf = conf.cpu().numpy()
        if torch.is_tensor(pred_cls):
            pred_cls = pred_cls.cpu().numpy()
        if torch.is_tensor(target_cls):
            target_cls = target_cls.cpu().numpy()
        if pred_boxes is not None and torch.is_tensor(pred_boxes):
            pred_boxes = pred_boxes.cpu().numpy()
        if target_boxes is not None and torch.is_tensor(target_boxes):
            target_boxes = target_boxes.cpu().numpy()
        if pred_img_indices is not None and torch.is_tensor(pred_img_indices):
            pred_img_indices = pred_img_indices.cpu().numpy()
        if target_img_indices is not None and torch.is_tensor(target_img_indices):
            target_img_indices = target_img_indices.cpu().numpy()
            
        # Ensure inputs are numpy arrays
        tp = np.asarray(tp)
        conf = np.asarray(conf)
        pred_cls = np.asarray(pred_cls)
        target_cls = np.asarray(target_cls)
        
        # Validate input shapes
        n_detections = len(conf)
        if len(tp) != n_detections or len(pred_cls) != n_detections:
            raise ValueError(f"Input shape mismatch: tp({len(tp)}), conf({len(conf)}), pred_cls({len(pred_cls)})")
        
        # Handle empty predictions or targets
        if n_detections == 0 or len(target_cls) == 0:
            # Set all metrics to 0
            self.AP = self.AP50 = self.AP75 = 0.0
            self.APsmall = self.APmedium = self.APlarge = 0.0
            self.AR1 = self.AR10 = self.AR100 = 0.0
            self.ARsmall = self.ARmedium = self.ARlarge = 0.0
            self.stats = {}
            return
        
        # Set up IoU thresholds (COCO standard: 0.5:0.05:0.95)
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        
        # Ensure tp has correct shape for multiple IoU thresholds
        if tp.ndim == 1:
            # Single IoU threshold provided, replicate for all thresholds
            tp = np.repeat(tp[:, np.newaxis], len(iou_thresholds), axis=1)
        elif tp.shape[1] != len(iou_thresholds):
            # Adjust number of IoU thresholds based on input
            iou_thresholds = np.linspace(0.5, 0.95, tp.shape[1])
        
        # Calculate basic AP metrics
        ap, p, r, f1, unique_classes = self._calculate_ap_single_class(
            tp, conf, pred_cls, target_cls,
            plot=self.plot, save_dir=self.save_dir, names=tuple(self.names.values()),
            iou_thresholds=iou_thresholds
        )
        
        # Store per-class statistics
        self.stats = {
            'ap': ap,  # Shape: (n_classes, n_iou_thresholds)
            'p': p,    # Shape: (n_classes, n_conf_thresholds)
            'r': r,    # Shape: (n_classes, n_conf_thresholds)
            'f1': f1,  # Shape: (n_classes, n_conf_thresholds)
            'unique_classes': unique_classes
        }
        
        # Also store in a separate attribute that won't be overwritten by update()
        self.class_stats = {
            'ap': ap.copy() if hasattr(ap, 'copy') else ap,
            'unique_classes': unique_classes.copy() if hasattr(unique_classes, 'copy') else unique_classes
        }
        
        # Calculate overall metrics (averaged over all classes)
        if ap.size > 0:
            # AP at IoU=0.50:0.95 (primary COCO metric)
            self.AP = ap.mean()
            
            # AP at specific IoU thresholds
            # Find indices for IoU thresholds 0.5 and 0.75
            iou_50_idx = np.argmin(np.abs(iou_thresholds - 0.5))
            iou_75_idx = np.argmin(np.abs(iou_thresholds - 0.75))
            
            self.AP50 = ap[:, iou_50_idx].mean()
            self.AP75 = ap[:, iou_75_idx].mean()
        else:
            self.AP = self.AP50 = self.AP75 = 0.0
        
        # Calculate size-specific metrics if bounding boxes are provided
        if pred_boxes is not None and target_boxes is not None and ori_shapes is not None:
            # Convert ori_shapes to list if needed
            if isinstance(ori_shapes, np.ndarray):
                ori_shapes = ori_shapes.tolist()
            
            # Ensure pred_boxes and target_boxes are numpy arrays
            pred_boxes = np.asarray(pred_boxes)
            target_boxes = np.asarray(target_boxes)
            
            # 严格实现要求提供图像索引，禁止回退
            if pred_img_indices is None or target_img_indices is None:
                raise ValueError("计算按尺寸的COCO指标需要 pred_img_indices 和 target_img_indices。")
            self._calculate_size_specific_metrics_strict(
                conf=conf,
                pred_cls=pred_cls,
                target_cls=target_cls,
                pred_boxes=pred_boxes,
                target_boxes=target_boxes,
                ori_shapes=ori_shapes,
                pred_img_indices=pred_img_indices,
                target_img_indices=target_img_indices,
                iou_thresholds=iou_thresholds,
            )
        else:
            # No size-specific metrics without bounding boxes
            self.APsmall = self.APmedium = self.APlarge = 0.0
            self.APsmall50 = self.APmedium50 = self.APlarge50 = 0.0  # 新增：按尺寸的AP50指标
            self.APsmall75 = self.APmedium75 = self.APlarge75 = 0.0  # 新增：按尺寸的AP75指标
            self.ARsmall = self.ARmedium = self.ARlarge = 0.0
        
        # Calculate AR at different max_dets thresholds（逐图像Top-K，严格实现）
        if pred_boxes is not None and target_boxes is not None and ori_shapes is not None and \
           pred_img_indices is not None and target_img_indices is not None:
            self.AR1 = self._calculate_ar_at_max_dets_strict(
                conf=conf,
                pred_cls=pred_cls,
                target_cls=target_cls,
                pred_boxes=pred_boxes,
                target_boxes=target_boxes,
                pred_img_indices=pred_img_indices,
                target_img_indices=target_img_indices,
                iou_thresholds=iou_thresholds,
                max_dets=1,
            )
            self.AR10 = self._calculate_ar_at_max_dets_strict(
                conf=conf,
                pred_cls=pred_cls,
                target_cls=target_cls,
                pred_boxes=pred_boxes,
                target_boxes=target_boxes,
                pred_img_indices=pred_img_indices,
                target_img_indices=target_img_indices,
                iou_thresholds=iou_thresholds,
                max_dets=10,
            )
            self.AR100 = self._calculate_ar_at_max_dets_strict(
                conf=conf,
                pred_cls=pred_cls,
                target_cls=target_cls,
                pred_boxes=pred_boxes,
                target_boxes=target_boxes,
                pred_img_indices=pred_img_indices,
                target_img_indices=target_img_indices,
                iou_thresholds=iou_thresholds,
                max_dets=100,
            )
        else:
            raise ValueError("计算 AR@K 需要 pred_img_indices 和 target_img_indices。")

    def _pairwise_iou(self, boxes1, boxes2, eps=1e-9):
        """
        计算两组边界框的两两IoU（numpy，xyxy归一化坐标）。
        Args:
            boxes1 (ndarray[N, 4])
            boxes2 (ndarray[M, 4])
        Returns:
            ndarray[N, M]: IoU矩阵
        """
        if boxes1.size == 0 or boxes2.size == 0:
            return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
        b1 = boxes1
        b2 = boxes2
        # 交集
        x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
        y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
        x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
        y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
        inter_w = np.clip(x2 - x1, 0, None)
        inter_h = np.clip(y2 - y1, 0, None)
        inter = inter_w * inter_h
        # 并集
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union = area1[:, None] + area2[None, :] - inter + eps
        return inter / union

    def _calculate_size_specific_metrics_strict(self, conf, pred_cls, target_cls, pred_boxes, target_boxes,
                                               ori_shapes, pred_img_indices, target_img_indices, iou_thresholds):
        """
        严格按COCO：仅基于GT面积划分small/medium/large，预测不过滤；逐图像逐类别匹配计算TP。
        同时输出 AP/AP50/AP75 及 AR@100 尺寸指标。
        """
        # 计算GT面积（像素）：(归一化xyxy) * (w,h)
        def areas_for_targets(tb, t_img_idx):
            areas = np.zeros(len(tb), dtype=np.float64)
            for i in range(len(tb)):
                h, w = ori_shapes[int(t_img_idx[i])]
                x1, y1, x2, y2 = tb[i]
                areas[i] = max(0.0, (x2 - x1) * w * (y2 - y1) * h)
            return areas

        target_areas = areas_for_targets(target_boxes, target_img_indices)
        size_buckets = [
            ((0, COCO_AREA_SMALL), 'small'),
            ((COCO_AREA_SMALL, COCO_AREA_MEDIUM), 'medium'),
            ((COCO_AREA_MEDIUM, float('inf')), 'large'),
        ]

        # 预先准备全局TP矩阵（用于AP计算），初始全False
        n_pred = len(pred_boxes)
        n_iou = len(iou_thresholds)

        def compute_tp_for_size(min_area, max_area):
            tp_size = np.zeros((n_pred, n_iou), dtype=bool)
            # 仅选择该尺寸范围内的GT索引
            gt_keep = np.where((target_areas >= min_area) & (target_areas < max_area))[0]
            if gt_keep.size == 0:
                return tp_size, gt_keep
            # 按(image, class)分组做匹配，预测保留全部
            unique_imgs = np.unique(target_img_indices[gt_keep]).astype(int)
            for img_id in unique_imgs:
                # 当前图像该尺寸内GT索引
                gt_img_idx = gt_keep[target_img_indices[gt_keep] == img_id]
                if gt_img_idx.size == 0:
                    continue
                classes_in_img = np.unique(target_cls[gt_img_idx]).astype(int)
                for c in classes_in_img:
                    gt_idx_group = gt_img_idx[target_cls[gt_img_idx] == c]
                    pred_idx_group = np.where((pred_img_indices == img_id) & (pred_cls == c))[0]
                    if pred_idx_group.size == 0:
                        continue
                    # 按置信度降序
                    order = np.argsort(-conf[pred_idx_group])
                    pred_idx_sorted = pred_idx_group[order]
                    # 计算IoU矩阵（P x G）
                    iou_mat = self._pairwise_iou(pred_boxes[pred_idx_sorted], target_boxes[gt_idx_group])
                    # 逐IoU阈值贪心匹配
                    for k, thr in enumerate(iou_thresholds):
                        matched_gts = np.zeros(len(gt_idx_group), dtype=bool)
                        for p in range(len(pred_idx_sorted)):
                            # 与未匹配GT的最大IoU
                            candidates = np.where((iou_mat[p] >= thr) & (~matched_gts))[0]
                            if candidates.size == 0:
                                continue
                            # 选择IoU最大的GT
                            gbest = candidates[np.argmax(iou_mat[p, candidates])]
                            matched_gts[gbest] = True
                            tp_size[pred_idx_sorted[p], k] = True
            return tp_size, gt_keep

        # 计算每个尺寸桶的AP/AR
        for (size_range, size_name) in size_buckets:
            min_area, max_area = size_range
            tp_size, gt_keep = compute_tp_for_size(min_area, max_area)
            if gt_keep.size == 0:
                if size_name == 'small':
                    self.APsmall = self.APsmall50 = self.APsmall75 = 0.0
                    self.ARsmall = 0.0
                elif size_name == 'medium':
                    self.APmedium = self.APmedium50 = self.APmedium75 = 0.0
                    self.ARmedium = 0.0
                else:
                    self.APlarge = self.APlarge50 = self.APlarge75 = 0.0
                    self.ARlarge = 0.0
                continue

            # AP（基于该尺寸GT子集）
            ap, _, _, _, _ = self._calculate_ap_single_class(
                tp_size, conf, pred_cls, target_cls[gt_keep], plot=False, save_dir=self.save_dir, names=tuple(self.names.values()), iou_thresholds=iou_thresholds
            )
            ap_mean = ap.mean() if ap.size > 0 else 0.0
            iou_50_idx = np.argmin(np.abs(iou_thresholds - 0.5))
            iou_75_idx = np.argmin(np.abs(iou_thresholds - 0.75))
            ap50_mean = ap[:, iou_50_idx].mean() if ap.size > 0 else 0.0
            ap75_mean = ap[:, iou_75_idx].mean() if ap.size > 0 else 0.0

            # AR@100（逐图像Top-100）
            ar100 = self._calculate_ar_at_max_dets_strict(
                conf=conf,
                pred_cls=pred_cls,
                target_cls=target_cls,
                pred_boxes=pred_boxes,
                target_boxes=target_boxes,
                pred_img_indices=pred_img_indices,
                target_img_indices=target_img_indices,
                iou_thresholds=iou_thresholds,
                max_dets=100,
                gt_keep_indices=gt_keep,
            )

            if size_name == 'small':
                self.APsmall, self.APsmall50, self.APsmall75 = float(ap_mean), float(ap50_mean), float(ap75_mean)
                self.ARsmall = float(ar100)
            elif size_name == 'medium':
                self.APmedium, self.APmedium50, self.APmedium75 = float(ap_mean), float(ap50_mean), float(ap75_mean)
                self.ARmedium = float(ar100)
            else:
                self.APlarge, self.APlarge50, self.APlarge75 = float(ap_mean), float(ap50_mean), float(ap75_mean)
                self.ARlarge = float(ar100)

    def _calculate_ar_at_max_dets_strict(self, conf, pred_cls, target_cls, pred_boxes, target_boxes,
                                          pred_img_indices, target_img_indices, iou_thresholds, max_dets=100,
                                          gt_keep_indices=None):
        """
        严格版AR@K：逐图像逐类别取Top-K预测进行贪心匹配，统计全数据集召回（可选限制GT为指定子集）。
        Args:
            gt_keep_indices: 若提供，仅统计该GT子集（用于按尺寸AR）；否则统计全部GT（用于整体AR）。
        Returns:
            float: 平均IoU阈值与类别的召回均值
        """
        unique_classes = np.unique(target_cls).astype(int)
        if unique_classes.size == 0:
            return 0.0
        recalls_per_class_iou = []
        for c in unique_classes:
            # 该类的GT索引（可选尺寸子集）
            if gt_keep_indices is not None:
                gt_c_idx = gt_keep_indices[target_cls[gt_keep_indices] == c]
            else:
                gt_c_idx = np.where(target_cls == c)[0]
            if gt_c_idx.size == 0:
                continue
            total_gt_c = gt_c_idx.size
            # 按图像聚合
            imgs_c = np.unique(target_img_indices[gt_c_idx]).astype(int)
            matched_counts = np.zeros(len(iou_thresholds), dtype=np.float64)
            for img_id in imgs_c:
                gt_ci = gt_c_idx[target_img_indices[gt_c_idx] == img_id]
                pred_ci = np.where((pred_img_indices == img_id) & (pred_cls == c))[0]
                if pred_ci.size == 0:
                    continue
                # Top-K
                order = np.argsort(-conf[pred_ci])
                pred_sorted = pred_ci[order][:max_dets]
                # IoU矩阵 P x G
                iou_mat = self._pairwise_iou(pred_boxes[pred_sorted], target_boxes[gt_ci])
                # 贪心匹配统计每个IoU阈值的TP数量
                for k, thr in enumerate(iou_thresholds):
                    matched_gts = np.zeros(len(gt_ci), dtype=bool)
                    count = 0
                    for p in range(len(pred_sorted)):
                        can = np.where((iou_mat[p] >= thr) & (~matched_gts))[0]
                        if can.size == 0:
                            continue
                        gbest = can[np.argmax(iou_mat[p, can])]
                        matched_gts[gbest] = True
                        count += 1
                    matched_counts[k] += count
            # 类别级别召回：匹配到的GT数 / 总GT数
            recall_c = matched_counts / max(total_gt_c, 1e-16)
            recalls_per_class_iou.append(recall_c)
        if not recalls_per_class_iou:
            return 0.0
        recalls_per_class_iou = np.vstack(recalls_per_class_iou)
        return float(recalls_per_class_iou.mean())
        
        # Update evaluation stats for external access
        self.eval_stats = {
            'AP': self.AP,
            'AP50': self.AP50,
            'AP75': self.AP75,
            'APsmall': self.APsmall,
            'APmedium': self.APmedium,
            'APlarge': self.APlarge,
            'AR1': self.AR1,
            'AR10': self.AR10,
            'AR100': self.AR100,
            'ARsmall': self.ARsmall,
            'ARmedium': self.ARmedium,
            'ARlarge': self.ARlarge,
        }
        
        # Call update method to ensure consistency
        self.update(self.eval_stats)
    
    @staticmethod
    def calculate_iou_batch(boxes1, boxes2, format='xyxy', eps=1e-7):
        """
        批量计算IoU，优化版本。
        
        相比于calculate_iou，这个方法针对批量计算进行了优化。
        
        Args:
            boxes1: 第一组边界框
            boxes2: 第二组边界框
            format: 边界框格式
            eps: 小的epsilon值
            
        Returns:
            批量IoU值
        """
        # 处理输入类型
        is_torch = torch.is_tensor(boxes1) or torch.is_tensor(boxes2)
        if is_torch:
            if not torch.is_tensor(boxes1):
                boxes1 = torch.from_numpy(np.array(boxes1))
            if not torch.is_tensor(boxes2):
                boxes2 = torch.from_numpy(np.array(boxes2))
            device = boxes1.device if torch.is_tensor(boxes1) else boxes2.device
            boxes1, boxes2 = boxes1.to(device), boxes2.to(device)
        else:
            boxes1, boxes2 = np.array(boxes1), np.array(boxes2)
        
        # 批量转换格式
        if format == 'xywh':
            if is_torch:
                boxes1_xyxy = torch.zeros_like(boxes1)
                boxes2_xyxy = torch.zeros_like(boxes2)
                # 向量化转换
                boxes1_xyxy[..., :2] = boxes1[..., :2]  # x1, y1
                boxes1_xyxy[..., 2:] = boxes1[..., :2] + boxes1[..., 2:]  # x2, y2
                boxes2_xyxy[..., :2] = boxes2[..., :2]
                boxes2_xyxy[..., 2:] = boxes2[..., :2] + boxes2[..., 2:]
            else:
                boxes1_xyxy = np.zeros_like(boxes1)
                boxes2_xyxy = np.zeros_like(boxes2)
                boxes1_xyxy[..., :2] = boxes1[..., :2]
                boxes1_xyxy[..., 2:] = boxes1[..., :2] + boxes1[..., 2:]
                boxes2_xyxy[..., :2] = boxes2[..., :2]
                boxes2_xyxy[..., 2:] = boxes2[..., :2] + boxes2[..., 2:]
            boxes1, boxes2 = boxes1_xyxy, boxes2_xyxy
        
        # 向量化计算交集
        if is_torch:
            inter_min = torch.max(boxes1[..., :2], boxes2[..., :2])
            inter_max = torch.min(boxes1[..., 2:], boxes2[..., 2:])
            inter_wh = torch.clamp(inter_max - inter_min, min=0)
        else:
            inter_min = np.maximum(boxes1[..., :2], boxes2[..., :2])
            inter_max = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
            inter_wh = np.clip(inter_max - inter_min, 0, None)
        
        intersection = inter_wh[..., 0] * inter_wh[..., 1]
        
        # 向量化计算面积
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1 + area2 - intersection + eps
        
        return intersection / union
    
    @staticmethod
    def calculate_iou(box1, box2, format='xyxy', eps=1e-7):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes or batches of boxes.
        
        This method computes IoU using efficient vectorized operations and supports both
        single box pairs and batch processing. It's primarily used for matching predictions
        with ground truth during evaluation.
        
        Args:
            box1 (np.ndarray | torch.Tensor): First bounding box(es) with shape (..., 4).
                Each box is in format [x1, y1, x2, y2] if format='xyxy' or [x, y, w, h] if format='xywh'.
            box2 (np.ndarray | torch.Tensor): Second bounding box(es) with shape (..., 4).
                Must have same format as box1.
            format (str): Bounding box format - 'xyxy' (x1,y1,x2,y2) or 'xywh' (x,y,width,height).
                Defaults to 'xyxy'.
            eps (float): Small epsilon to prevent division by zero. Defaults to 1e-7.
        
        Returns:
            np.ndarray | torch.Tensor: IoU values with shape matching the broadcast result
                of box1 and box2. Values range from 0 (no overlap) to 1 (perfect overlap).
        
        Note:
            - Handles both numpy arrays and PyTorch tensors.
            - Supports broadcasting for efficient batch computation.
            - Returns 0 for invalid boxes (zero area).
        
        Example:
            ```python
            # Single box pair
            iou = COCOMetrics.calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])
            
            # Batch processing
            pred_boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
            gt_boxes = np.array([[5, 5, 15, 15], [25, 25, 35, 35]])
            ious = COCOMetrics.calculate_iou(pred_boxes, gt_boxes)
            ```
        """
        # Handle input type
        is_torch = torch.is_tensor(box1) or torch.is_tensor(box2)
        if is_torch:
            if not torch.is_tensor(box1):
                box1 = torch.from_numpy(np.array(box1))
            if not torch.is_tensor(box2):
                box2 = torch.from_numpy(np.array(box2))
            device = box1.device if torch.is_tensor(box1) else box2.device
            box1, box2 = box1.to(device), box2.to(device)
        else:
            box1, box2 = np.array(box1), np.array(box2)
        
        # Convert to xyxy format if needed
        if format == 'xywh':
            if is_torch:
                box1_xyxy = torch.zeros_like(box1)
                box2_xyxy = torch.zeros_like(box2)
            else:
                box1_xyxy = np.zeros_like(box1)
                box2_xyxy = np.zeros_like(box2)
            
            # Convert xywh to xyxy
            box1_xyxy[..., 0] = box1[..., 0]  # x1 = x
            box1_xyxy[..., 1] = box1[..., 1]  # y1 = y
            box1_xyxy[..., 2] = box1[..., 0] + box1[..., 2]  # x2 = x + w
            box1_xyxy[..., 3] = box1[..., 1] + box1[..., 3]  # y2 = y + h
            
            box2_xyxy[..., 0] = box2[..., 0]  # x1 = x
            box2_xyxy[..., 1] = box2[..., 1]  # y1 = y
            box2_xyxy[..., 2] = box2[..., 0] + box2[..., 2]  # x2 = x + w
            box2_xyxy[..., 3] = box2[..., 1] + box2[..., 3]  # y2 = y + h
            
            box1, box2 = box1_xyxy, box2_xyxy
        
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        x1_2, y1_2, x2_2, y2_2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
        
        # Calculate intersection coordinates
        if is_torch:
            x1_inter = torch.max(x1_1, x1_2)
            y1_inter = torch.max(y1_1, y1_2)
            x2_inter = torch.min(x2_1, x2_2)
            y2_inter = torch.min(y2_1, y2_2)
            
            # Calculate intersection area
            inter_width = torch.clamp(x2_inter - x1_inter, min=0)
            inter_height = torch.clamp(y2_inter - y1_inter, min=0)
        else:
            x1_inter = np.maximum(x1_1, x1_2)
            y1_inter = np.maximum(y1_1, y1_2)
            x2_inter = np.minimum(x2_1, x2_2)
            y2_inter = np.minimum(y2_1, y2_2)
            
            # Calculate intersection area
            inter_width = np.clip(x2_inter - x1_inter, 0, None)
            inter_height = np.clip(y2_inter - y1_inter, 0, None)
        
        intersection = inter_width * inter_height
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection + eps
        
        # Calculate IoU
        iou = intersection / union
        
        return iou
    
    def _set_zero_metrics(self):
        """
        设置所有指标为0（用于空数据情况）。
        """
        self.AP = self.AP50 = self.AP75 = 0.0
        self.APsmall = self.APmedium = self.APlarge = 0.0
        self.APsmall50 = self.APmedium50 = self.APlarge50 = 0.0  # 新增：按尺寸的AP50指标
        self.APsmall75 = self.APmedium75 = self.APlarge75 = 0.0  # 新增：按尺寸的AP75指标
        self.AR1 = self.AR10 = self.AR100 = 0.0
        self.ARsmall = self.ARmedium = self.ARlarge = 0.0
        self.stats = {}
    
    def _calculate_ap_single_class_optimized(self, tp, conf, pred_cls, target_cls, iou_thresholds):
        """
        优化的单类别AP计算方法。
        
        使用向量化操作和批处理提高性能。
        
        Args:
            tp: True positive指示器
            conf: 置信度分数
            pred_cls: 预测类别
            target_cls: 目标类别
            iou_thresholds: IoU阈值数组
            
        Returns:
            tuple: (ap, p, r, f1, unique_classes)
        """
        # 获取唯一类别
        unique_classes = np.unique(target_cls).astype(int)
        n_classes = len(unique_classes)
        n_iou = len(iou_thresholds)
        
        # 初始化输出
        ap = np.zeros((n_classes, n_iou))
        # 使用1000个点进行曲线插值（与标准DetMetrics相同）
        # 这不是检测数量的限制，而是曲线的插值点
        curve_points = 1000
        p = np.zeros((n_classes, curve_points))  # Precision曲线点
        r = np.zeros((n_classes, curve_points))  # Recall曲线点
        
        # 用于曲线插值
        x = np.linspace(0, 1, curve_points)
        
        # COCO使用20101点插值
        recall_thresholds = np.linspace(0, 1, 101)
        
        # 使用向量化操作处理每个类别
        for ci, c in enumerate(unique_classes):
            # 获取当前类别的面罩
            class_mask = pred_cls == c
            n_positives = (target_cls == c).sum()
            
            if class_mask.sum() == 0 or n_positives == 0:
                continue
            
            # 使用argsort一次性排序，避免重复排序
            sorted_indices = np.argsort(-conf[class_mask])
            
            # 批量获取排序后的数据
            class_conf = conf[class_mask]  # 获取当前类别的置信度
            if tp.ndim == 1:
                tp_sorted = tp[class_mask][sorted_indices].reshape(-1, 1)
            else:
                tp_sorted = tp[class_mask][sorted_indices]
            
            # 使用向量化操作计算每个IoU阈值的指标
            for iou_idx in range(n_iou):
                # 使用cumsum进行累积计算
                tp_cumsum = np.cumsum(tp_sorted[:, iou_idx])
                fp_cumsum = np.cumsum(1 - tp_sorted[:, iou_idx])
                
                # 向量化计算precision和recall
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                recall = tp_cumsum / (n_positives + 1e-16)
                
                # 保存插值后的曲线用于绘图（类似于metrics.py中的ap_per_class）
                if iou_idx == 0:
                    # 插值precision和recall曲线到固定点
                    # 这允许处理任意数量的检测
                    if len(class_conf) > 0:
                        # 获取排序后的置信度
                        sorted_conf = class_conf[sorted_indices]
                        # 插值recall曲线
                        r[ci] = np.interp(x, recall, recall, left=0, right=recall[-1] if len(recall) > 0 else 0)
                        # 插值precision曲线（使用负x进行正确插值）
                        p[ci] = np.interp(-x, -sorted_conf, precision, left=1, right=0)
                
                # 使用向量化操作计算AP
                ap_values = np.zeros(len(recall_thresholds))
                for i, recall_thresh in enumerate(recall_thresholds):
                    mask = recall >= recall_thresh
                    if mask.any():
                        ap_values[i] = precision[mask].max()
                
                ap[ci, iou_idx] = ap_values.mean()
        
        # 计算F1分数
        f1 = 2 * p * r / (p + r + 1e-16)
        
        return ap, p, r, f1, unique_classes
    
    def _calculate_overall_metrics(self, ap, iou_thresholds):
        """
        计算整体指标（所有类别的平均值）。
        
        Args:
            ap: 每个类别的AP数组
            iou_thresholds: IoU阈值数组
        """
        if ap.size > 0:
            # AP at IoU=0.50:0.95 (主COCO指标)
            self.AP = ap.mean()
            
            # 使用向量化操作查找最接近的IoU阈值
            iou_50_idx = np.argmin(np.abs(iou_thresholds - 0.5))
            iou_75_idx = np.argmin(np.abs(iou_thresholds - 0.75))
            
            self.AP50 = ap[:, iou_50_idx].mean()
            self.AP75 = ap[:, iou_75_idx].mean()
        else:
            self.AP = self.AP50 = self.AP75 = 0.0
    
    def _calculate_ar_at_max_dets_optimized(self, tp, conf, pred_cls, target_cls, max_dets):
        """
        优化的AR计算方法。
        
        使用向量化操作和批处理提高性能。
        
        Args:
            tp: True positive指示器
            conf: 置信度分数
            pred_cls: 预测类别
            target_cls: 目标类别
            max_dets: 最大检测数量
            
        Returns:
            float: 平均召回率
        """
        unique_classes = np.unique(target_cls).astype(int)
        n_iou = tp.shape[1] if tp.ndim > 1 else 1
        
        recall_values = []
        
        # 向量化处理每个类别
        for c in unique_classes:
            pred_mask = pred_cls == c
            n_positives = (target_cls == c).sum()
            
            if pred_mask.sum() == 0 or n_positives == 0:
                continue
            
            # 批量获取类别数据
            class_tp = tp[pred_mask]
            class_conf = conf[pred_mask]
            
            # 使用argpartition进行部分排序（比全排序快）
            if len(class_conf) > max_dets:
                # 只需要最高的max_dets个的索引
                top_indices = np.argpartition(-class_conf, max_dets)[:max_dets]
                # 对选中的部分进行精确排序
                top_indices = top_indices[np.argsort(-class_conf[top_indices])]
            else:
                top_indices = np.argsort(-class_conf)
            
            # 向量化计算每个IoU阈值的recall
            for iou_idx in range(n_iou):
                if class_tp.ndim == 1:
                    tp_at_iou = class_tp[top_indices]
                else:
                    tp_at_iou = class_tp[top_indices, iou_idx]
                
                true_positives = tp_at_iou.sum()
                recall = min(true_positives / max(n_positives, 1e-16), 1.0)
                recall_values.append(recall)
        
        return np.mean(recall_values) if recall_values else 0.0
    
    def _calculate_size_specific_metrics(self, tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, ori_shapes, show_progress=True):
        """
        计算尺寸特定指标。
        
        使用优化的批处理方法计算不同尺寸的AP和AR指标。
        
        Args:
            tp, conf, pred_cls, target_cls: 基本输入
            pred_boxes, target_boxes: 边界框数据
            ori_shapes: 原始图像尺寸
            show_progress: 是否显示进度
        """
        # 转换ori_shapes为列表
        if isinstance(ori_shapes, np.ndarray):
            ori_shapes = ori_shapes.tolist()
        
        pred_boxes = np.asarray(pred_boxes)
        target_boxes = np.asarray(target_boxes)
        
        size_ranges = [
            ((0, COCO_AREA_SMALL), 'small'),
            ((COCO_AREA_SMALL, COCO_AREA_MEDIUM), 'medium'),
            ((COCO_AREA_MEDIUM, float('inf')), 'large')
        ]
        
        if show_progress:
            progress_bar = tqdm(size_ranges, desc="计算尺寸特定指标", unit="size")
        else:
            progress_bar = size_ranges
        
        for (size_range, size_name) in progress_bar:
            # 计算AP
            ap_value = self._calculate_ap_by_size_optimized(
                tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, ori_shapes, size_range
            )
            
            # 计算AR (max_dets=100)
            ar_value = self._calculate_ar_by_size_optimized(
                tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, ori_shapes, size_range, max_dets=100
            )
            
            # 设置对应属性
            if size_name == 'small':
                self.APsmall = ap_value
                self.ARsmall = ar_value
            elif size_name == 'medium':
                self.APmedium = ap_value
                self.ARmedium = ar_value
            else:  # large
                self.APlarge = ap_value
                self.ARlarge = ar_value
    
    def _calculate_ap_by_size_optimized(self, tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, ori_shapes, size_range):
        """
        优化的按尺寸计算AP的方法。
        
        使用向量化操作批量计算边界框面积和过滤。
        """
        min_area, max_area = size_range
        
        # 批量计算预测边界框面积
        pred_areas = self._calculate_areas_batch(pred_boxes, ori_shapes)
        pred_keep_mask = (pred_areas >= min_area) & (pred_areas < max_area)
        
        # 批量计算目标边界框面积
        target_areas = self._calculate_areas_batch(target_boxes, ori_shapes)
        target_keep_mask = (target_areas >= min_area) & (target_areas < max_area)
        
        # 如果没有该尺寸范围的对象，返回0
        if not pred_keep_mask.any() and not target_keep_mask.any():
            return 0.0
        
        # 过滤输入
        filtered_tp = tp[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_conf = conf[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_pred_cls = pred_cls[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_target_cls = target_cls[target_keep_mask] if target_keep_mask.any() else np.array([])
        
        if len(filtered_tp) == 0 or len(filtered_target_cls) == 0:
            return 0.0
        
        # 使用优化的AP计算
        ap, _, _, _, _ = self._calculate_ap_single_class_optimized(
            filtered_tp, filtered_conf, filtered_pred_cls, filtered_target_cls, np.linspace(0.5, 0.95, 10)
        )
        
        return ap.mean()
    
    def _calculate_ar_by_size_optimized(self, tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, ori_shapes, size_range, max_dets):
        """
        优化的按尺寸计算AR的方法。
        """
        min_area, max_area = size_range
        
        # 重用面积计算结果
        pred_areas = self._calculate_areas_batch(pred_boxes, ori_shapes)
        pred_keep_mask = (pred_areas >= min_area) & (pred_areas < max_area)
        
        target_areas = self._calculate_areas_batch(target_boxes, ori_shapes)
        target_keep_mask = (target_areas >= min_area) & (target_areas < max_area)
        
        if not pred_keep_mask.any() and not target_keep_mask.any():
            return 0.0
        
        # 过滤输入
        filtered_tp = tp[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_conf = conf[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_pred_cls = pred_cls[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_target_cls = target_cls[target_keep_mask] if target_keep_mask.any() else np.array([])
        
        if len(filtered_tp) == 0 or len(filtered_target_cls) == 0:
            return 0.0
        
        # 使用优化的AR计算
        return self._calculate_ar_at_max_dets_optimized(
            filtered_tp, filtered_conf, filtered_pred_cls, filtered_target_cls, max_dets
        )
    
    def _calculate_ap_by_size_at_iou(self, tp, conf, pred_cls, target_cls, pred_boxes, target_boxes, ori_shapes, size_range, iou_idx=0):
        """
        计算特定尺寸范围在特定IoU阈值下的AP。
        
        Args:
            tp: True positive指示器
            conf: 置信度分数
            pred_cls: 预测类别
            target_cls: 目标类别
            pred_boxes: 预测边界框
            target_boxes: 目标边界框
            ori_shapes: 原始图像尺寸
            size_range: 尺寸范围 (min_area, max_area)
            iou_idx: IoU阈值索引，0表示IoU=0.5
            
        Returns:
            float: 该尺寸范围在指定IoU阈值下的AP值
        """
        min_area, max_area = size_range
        
        # 批量计算预测边界框面积
        pred_areas = self._calculate_areas_batch(pred_boxes, ori_shapes)
        pred_keep_mask = (pred_areas >= min_area) & (pred_areas < max_area)
        
        # 批量计算目标边界框面积
        target_areas = self._calculate_areas_batch(target_boxes, ori_shapes)
        target_keep_mask = (target_areas >= min_area) & (target_areas < max_area)
        
        # 如果没有该尺寸范围的对象，返回0
        if not pred_keep_mask.any() and not target_keep_mask.any():
            return 0.0
        
        # 过滤输入
        filtered_tp = tp[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_conf = conf[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_pred_cls = pred_cls[pred_keep_mask] if pred_keep_mask.any() else np.array([])
        filtered_target_cls = target_cls[target_keep_mask] if target_keep_mask.any() else np.array([])
        
        if len(filtered_tp) == 0 or len(filtered_target_cls) == 0:
            return 0.0
        
        # 使用优化的AP计算，只计算特定IoU阈值
        ap, _, _, _, _ = self._calculate_ap_single_class_optimized(
            filtered_tp, filtered_conf, filtered_pred_cls, filtered_target_cls, np.linspace(0.5, 0.95, 10)
        )
        
        # 返回指定IoU阈值的AP值（平均所有类别）
        if ap.shape[1] > iou_idx:
            return ap[:, iou_idx].mean()
        else:
            return 0.0
    
    def _calculate_areas_batch(self, boxes, ori_shapes):
        """
        批量计算边界框面积。
        
        使用向量化操作提高计算效率。
        
        Args:
            boxes: 边界框数组
            ori_shapes: 原始图像尺寸列表
            
        Returns:
            np.ndarray: 边界框面积数组
        """
        if len(boxes) == 0:
            return np.array([])
            
        # 优化：假设每个图像的检测数量相等
        boxes_per_image = len(boxes) // len(ori_shapes) if len(ori_shapes) > 0 else len(boxes)
        
        areas = np.zeros(len(boxes))
        
        for i, bbox in enumerate(boxes):
            # 获取对应的图像尺寸
            img_idx = min(i // max(boxes_per_image, 1), len(ori_shapes) - 1)
            ori_shape = ori_shapes[img_idx]
            
            # 使用向量化操作计算面积
            if len(bbox) == 4:
                h, w = ori_shape[:2]
                # 假设是xyxy格式
                x1, y1, x2, y2 = bbox
                # 缩放到原始尺寸
                x1, x2 = x1 * w, x2 * w
                y1, y2 = y1 * h, y2 * h
                # 计算面积
                areas[i] = abs((x2 - x1) * (y2 - y1))
            
        return areas
    
    def _update_eval_stats(self):
        """
        更新评估统计信息。
        """
        self.eval_stats = {
            'AP': self.AP,
            'AP50': self.AP50,
            'AP75': self.AP75,
            'APsmall': self.APsmall,
            'APmedium': self.APmedium,
            'APlarge': self.APlarge,
            'AR1': self.AR1,
            'AR10': self.AR10,
            'AR100': self.AR100,
            'ARsmall': self.ARsmall,
            'ARmedium': self.ARmedium,
            'ARlarge': self.ARlarge,
        }
