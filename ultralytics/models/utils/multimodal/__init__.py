# Ultralytics Multi-modal Utils Package
"""
多模态工具包，提供验证器可视化的复用组件。
"""

from .vis import (
    split_modalities,
    visualize_x_to_3ch,
    concat_side_by_side,
    to_norm_xywh_for_plot,
    duplicate_bboxes_for_side_by_side,
    adjust_bboxes_for_side_by_side,
    ensure_batch_idx_long,
    resolve_x_modality,
    get_x_modality_path,
)

__all__ = [
    'split_modalities',
    'visualize_x_to_3ch', 
    'concat_side_by_side',
    'to_norm_xywh_for_plot',
    'duplicate_bboxes_for_side_by_side',
    'adjust_bboxes_for_side_by_side',
    'ensure_batch_idx_long',
    'resolve_x_modality',
    'get_x_modality_path',
]