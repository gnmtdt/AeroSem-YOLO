# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .base import BaseDataset
from .build import build_dataloader, build_grounding, build_yolo_dataset, load_inference_source
from .dataset import (
    ClassificationDataset,
    GroundingDataset,
    SemSegFormat,
    SemanticDataset,
    YOLOConcatDataset,
    YOLODataset,
    YOLOMultiModalClassifyDataset,
    YOLOMultiModalDataset,
    YOLOMultiModalImageDataset,
    YOLOMultiModalSemSegDataset,
)
from .multimodal_augment import (
    BaseMultiModalTransform,
    MultiModalMixUp,
    MultiModalMosaic,
    MultiModalRandomFlip,
    MultiModalRandomHSV,
    MultiModalAlbumentations,
    MultiModalAsyncPerturb,
    MultiModalModalDropout,
    MultiModalIRNonGeoAug,
    MultiModalDepthNonGeoAug,
    SemSegMultiModalMosaic,
    mm_transforms,
    mm_semseg_transforms,
    create_multimodal_transforms,
)

__all__ = (
    # Base and standard datasets
    "BaseDataset",
    "ClassificationDataset",
    "SemanticDataset",
    "YOLODataset",
    "YOLOConcatDataset",
    "GroundingDataset",

    # Multi-modal datasets
    "YOLOMultiModalClassifyDataset",    # RGB+X classification dataset (YAML + images/labels)
    "YOLOMultiModalDataset",           # Text multi-modal dataset
    "YOLOMultiModalImageDataset",      # RGB+X image multi-modal dataset
    "YOLOMultiModalSemSegDataset",     # RGB+X semantic segmentation dataset
    "SemSegFormat",                    # Semantic segmentation mask format

    # Multi-modal data augmentation
    "BaseMultiModalTransform",         # Multi-modal transform base class
    "MultiModalRandomHSV",             # Multi-modal HSV augmentation
    "MultiModalAlbumentations",       # RGB-only non-geometric augmentation
    "MultiModalRandomFlip",            # Multi-modal flip augmentation
    "MultiModalMosaic",                # Multi-modal mosaic augmentation
    "MultiModalMixUp",                 # Multi-modal mixup augmentation
    "MultiModalAsyncPerturb",          # Micro asynchronous geometric perturbation
    "MultiModalModalDropout",          # Modal dropout augmentation
    "MultiModalIRNonGeoAug",           # IR-only non-geometric augmentation
    "MultiModalDepthNonGeoAug",        # Depth-only non-geometric augmentation
    "mm_transforms",                   # Multi-modal transforms pipeline (pre-Format)
    "mm_semseg_transforms",            # SemSeg transforms pipeline (pre-Format)
    "SemSegMultiModalMosaic",          # SemSeg mosaic with mask canvas
    "create_multimodal_transforms",    # Multi-modal transform factory

    # Dataset building functions
    "build_yolo_dataset",              # Enhanced with multi_modal_image support
    "build_grounding",
    "build_dataloader",
    "load_inference_source",
)
