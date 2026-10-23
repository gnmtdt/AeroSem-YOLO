# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import math
import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import psutil
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from ultralytics.nn.mm.utils import get_modality_path_from_data, resolve_modality_pair, is_rgb_modality

from .augment import (
    Compose,
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .converter import merge_multi_segment
from .multimodal.image_io import MultiModalImageIOMixin
from .utils import (
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for Ultralytics YOLO models
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    This class supports loading data for object detection, segmentation, pose estimation, and oriented bounding box
    (OBB) tasks using the YOLO format.

    Attributes:
        use_segments (bool): Indicates if segmentation masks should be used.
        use_keypoints (bool): Indicates if keypoints should be used for pose estimation.
        use_obb (bool): Indicates if oriented bounding boxes should be used.
        data (dict): Dataset configuration dictionary.

    Methods:
        cache_labels: Cache dataset labels, check images and read shapes.
        get_labels: Return dictionary of labels for YOLO training.
        build_transforms: Build and append transforms to the list.
        close_mosaic: Set mosaic, copy_paste and mixup options to 0.0 and build transformations.
        update_labels_info: Update label format for different tasks.
        collate_fn: Collate data samples into batches.

    Examples:
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> dataset.get_labels()
    """

    def __init__(self, *args, data: Optional[Dict] = None, task: str = "detect", **kwargs):
        """
        Initialize the YOLODataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, channels=self.data["channels"], **kwargs)

    def cache_labels(self, path: Path = Path("./labels.cache")) -> Dict:
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> List[Dict]:
        """
        Return dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        Build and append transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp: Dict) -> None:
        """
        Disable mosaic, copy_paste, mixup and cutmix augmentations by setting their probabilities to 0.0.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label: Dict) -> Dict:
        """
        Update label format for different tasks.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        from ultralytics.utils.timedebug import add_worker_timing, merge_worker_stats

        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order

        has_timedebug = any(b.get("_timedebug_enabled", False) for b in batch)

        if has_timedebug:
            t0 = time.perf_counter()
            sample_stats_list = [b.pop("_timedebug_worker_stats", {}) for b in batch]
            for b in batch:
                b.pop("_timedebug_enabled", None)
        else:
            for b in batch:
                b.pop("_timedebug_worker_stats", None)
                b.pop("_timedebug_enabled", None)

        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)

        if has_timedebug:
            merged_stats = merge_worker_stats(sample_stats_list)
            add_worker_timing(merged_stats, "batch.collate.total", time.perf_counter() - t0)
            new_batch["_timedebug_worker_stats"] = merged_stats
            new_batch["_timedebug_worker_meta"] = {"ready_abs": time.perf_counter()}
            new_batch["_timedebug_enabled"] = True

        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format with multi-modal support.

    This class extends YOLODataset to add text information for multi-modal model training, enabling models to
    process both image and text data.

    Methods:
        update_labels_info: Add text information for multi-modal model training.
        build_transforms: Enhance data transformations with text augmentation.

    Examples:
        >>> dataset = YOLOMultiModalDataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> batch = next(iter(dataset))
        >>> print(batch.keys())  # Should include 'texts'
    """

    def __init__(self, *args, data: Optional[Dict] = None, task: str = "detect", **kwargs):
        """
        Initialize a YOLOMultiModalDataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label: Dict) -> Dict:
        """
        Add text information for multi-modal model training.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances and texts.
        """
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        # NOTE: and `RandomLoadText` would randomly select one of them if there are multiple words.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]

        return labels

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        Enhance data transformations with optional text augmentation for multi-modal training.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """
        Return category names for the dataset.

        Returns:
            (Set[str]): List of class names.
        """
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}  # category names

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):  # to check
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: Dict, threshold: int = 100) -> List[str]:
        """Get negative text samples based on frequency threshold."""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOVODataset(YOLODataset):
    """
    YOLOVO text-augmented multi-modal dataset.

    Extends YOLODataset to provide category text strings for open-vocabulary
    training.  In v1 only 3-channel RGB images are used with standard
    ``v8_transforms``.  RGB+X fusion (``has_x = True``) is NOT supported
    in v1 and will raise NotImplementedError.

    Attributes:
        texts (list[str]): Flat list of category name strings from ``data["names"]``.
        has_x (bool): Whether an auxiliary image modality is present. v1: always False.
    """

    def __init__(
        self,
        *args,
        data: Optional[Dict] = None,
        task: str = "detect",
        yolovo_debug: bool = False,
        yolovo_object_paste_debug_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initialize a YOLOVODataset.

        Args:
            data (dict, optional): Dataset configuration dictionary containing
                ``"names"`` mapping class indices to name strings.
            task (str): Task type, default ``"detect"``.
            yolovo_debug (bool): Enable YOLOVO debug mode for ObjectPaste visualization.
            yolovo_object_paste_debug_dir (str | Path, optional): Directory for ObjectPaste debug images.
            *args (Any): Positional arguments forwarded to YOLODataset.
            **kwargs (Any): Keyword arguments forwarded to YOLODataset.
        """
        self.has_x = False  # v1: only RGB; v2 will add auxiliary modality via bridge
        self.yolovo_debug = bool(yolovo_debug)
        self.yolovo_object_paste_debug_dir = (
            Path(yolovo_object_paste_debug_dir) if yolovo_object_paste_debug_dir else None
        )
        super().__init__(*args, data=data, task=task, **kwargs)
        self.texts = list(self.data["names"].values())

    def get_texts(self) -> List[str]:
        """
        Return full list of category name strings.

        This list is consumed by the trainer's ``sample_batch_texts`` helper to
        build per-batch text prompts for the text tower.

        Returns:
            (list[str]): Category name strings.
        """
        return self.texts

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """Build YOLOVO augmentation pipeline (RGB-only detect v1).

        v1 uses ``yolovo_transforms`` as base. ``has_x=True`` is NOT supported in v1
        and raises NotImplementedError. v2 will add RGB+X bridge.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms ending with ``Format``.
        """
        if self.has_x:
            raise NotImplementedError(
                "YOLOVODataset does not support RGB+X fusion in v1. "
                "Set has_x=False (default). v2 will introduce a bridge "
                "dataset for auxiliary modality loading."
            )

        if not self.augment:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        else:
            from ultralytics.data.yolovo_augment import yolovo_transforms

            transforms = yolovo_transforms(self, self.imgsz, hyp)

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                batch_idx=True,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    def close_mosaic(self, hyp: Dict) -> None:
        """关闭强增强时同步关闭 YOLOVO ObjectPaste。"""
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        if hasattr(hyp, "object_paste"):
            hyp.object_paste = 0.0
        self.transforms = self.build_transforms(hyp)


def auto_detect_modality_channels(dataset_root, modality_dir, split="train", n_samples=10):
    """Sample images to detect modality channel count."""
    dataset_root = Path(dataset_root) if not isinstance(dataset_root, Path) else dataset_root
    img_dir = dataset_root / modality_dir / split
    if not img_dir.exists():
        return 3
    all_files = [f for f in img_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]
    if not all_files:
        return 3
    samples = random.sample(all_files, min(n_samples, len(all_files)))
    channels = set()
    for p in samples:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            ch = img.shape[2] if img.ndim == 3 else 1
            channels.add(ch)
    if len(channels) == 1:
        return channels.pop()
    LOGGER.warning(f"MultiModal: inconsistent channel counts in {modality_dir}: {channels}, using max")
    return max(channels) if channels else 3


class YOLOMultiModalImageDataset(YOLODataset):
    """
    多模态YOLO数据集类 - 支持任意主模态+附模态图像的目标检测数据集

    此类扩展了标准的YOLODataset，添加了对多模态图像（RGB+X）的支持。
    支持的X模态包括：深度图、热红外图、LiDAR点云图等。

    核心特性：
    - RGB+X双模态图像加载和处理
    - 6通道图像构建（RGB 3通道 + X模态 3通道）
    - 智能多模态缓存系统（内存/磁盘）
    - 自体模态生成支持（当X模态缺失时）
    - 与多模态数据增强的无缝集成
    - 高效的索引管理和验证

    数据组织结构：
    ```
    dataset/
    ├── images/
    │   ├── rgb/          # RGB图像目录
    │   └── depth/        # X模态图像目录（如深度图）
    └── labels/           # 标注文件目录
    ```

    Attributes:
        x_modality (str): X模态类型，如'depth', 'thermal', 'lidar'等
        x_modality_dir (str): X模态图像目录名
        x_modality_suffix (str): X模态图像文件后缀
        enable_self_modal_generation (bool): 是否启用自体模态生成
        x_ims (list): X模态图像缓存列表
        x_im_hw0 (list): X模态图像原始尺寸缓存
        x_im_hw (list): X模态图像调整后尺寸缓存

    Methods:
        __getitem__: 获取多模态数据样本（6通道图像+标签）
        get_valid_indices: 获取有完整多模态数据的有效索引
        load_multimodal_image: 加载RGB+X模态图像并构建6通道图像
        cache_images: 缓存RGB和X模态图像到内存或磁盘

    Examples:
        >>> # 基本使用
        >>> dataset = YOLOMultiModalImageDataset(
        ...     img_path="dataset/images/rgb",
        ...     x_modality="depth",
        ...     x_modality_dir="depth"
        ... )
        >>> sample = dataset[0]  # 返回6通道图像和标签
        >>> print(sample["img"].shape)  # (6, H, W) - RGB+Depth

        >>> # 启用自体模态生成
        >>> dataset = YOLOMultiModalImageDataset(
        ...     img_path="dataset/images/rgb",
        ...     x_modality="thermal",
        ...     enable_self_modal_generation=True
        ... )
    """

    def __init__(
        self,
        *args,
        x_modality="depth",
        x_modality_dir=None,
        x_modality_suffix=None,
        enable_self_modal_generation=False,
        precision_report_dir=None,
        **kwargs
    ):
        """
        初始化多模态YOLO数据集

        Args:
            x_modality (str): X模态类型，如'depth', 'thermal', 'lidar'等
            x_modality_dir (str, optional): X模态图像目录名，默认与x_modality相同
            x_modality_suffix (str, optional): X模态图像文件后缀，默认与RGB相同
            enable_self_modal_generation (bool): 是否启用自体模态生成
            *args: 传递给父类的位置参数
            **kwargs: 传递给父类的关键字参数

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(
            ...     img_path="images/rgb",
            ...     x_modality="depth",
            ...     x_modality_dir="depth"
            ... )
        """
        # 设置模态相关属性 (backward compat via property)
        self.primary_modality = None  # will be resolved in _parse_multimodal_config
        self.secondary_modality = x_modality
        self.x_modality = x_modality
        # 延迟解析：等 super().__init__() 后 data 可用时再从配置解析
        self.x_modality_dir = x_modality_dir  # 可能为 None，稍后解析
        self.x_modality_suffix = x_modality_suffix
        self.enable_self_modal_generation = enable_self_modal_generation
        self.precision_report_dir = precision_report_dir
        self.primary_ch = None
        self.secondary_ch = None
        self.precision_contract = {"rgb": "legacy", "x": "legacy"}
        self.precision_report = None
        self.precision_report_path = None

        # 初始化多模态缓存
        self.x_ims = None
        self.x_im_hw0 = None
        self.x_im_hw = None
        self.x_npy_files = None
        self.mm_cache_root_name = ".ultralytics_mm_cache"
        self._x_buffer = []  # X模态LRU缓存追踪，与RGB buffer模式对齐
        self._x_paths: List[Optional[str]] = []  # 预计算的X模态路径列表，与im_files 1:1对齐

        # 提前解析 primary_modality_dir（get_labels 在 super().__init__ 中需要）
        data_dict = kwargs.get('data', None)
        if isinstance(data_dict, dict):
            from ultralytics.nn.mm.utils import get_modality_path_from_data, resolve_modality_pair
            primary, _ = resolve_modality_pair(data_dict, default_secondary=x_modality)
            self.primary_modality_dir = get_modality_path_from_data(data_dict, primary)
        else:
            self.primary_modality_dir = 'images'

        # 调用父类初始化
        super().__init__(*args, **kwargs)

        # data 现在可用，解析 x_modality_dir
        if self._secondary_modality_dir is None:
            from ultralytics.nn.mm.utils import get_modality_path_from_data
            self._secondary_modality_dir = get_modality_path_from_data(
                getattr(self, 'data', None), self.x_modality
            )
            LOGGER.info(f"MultiModal: 从配置解析 secondary 模态目录: {self._secondary_modality_dir}")

        # im_files 稳定后，预计算所有 X 模态路径（必须在 precision_contract 过滤之前）
        self._precompute_x_paths()

        # 启动阶段基于 split 解析精度契约并过滤不匹配样本（仅一次）
        self._initialize_precision_contract()

        # 验证多模态数据完整性
        self._validate_multimodal_data()

        # 记录多模态配置信息
        primary = getattr(self, 'primary_modality', 'rgb') or 'rgb'
        secondary = getattr(self, 'secondary_modality', self.x_modality) or self.x_modality
        LOGGER.info(f"MultiModal: {primary}+{secondary}双模态数据集已初始化")
        primary_ch = getattr(self, 'primary_ch', None)
        secondary_ch = getattr(self, 'secondary_ch', None)
        if primary_ch is not None and secondary_ch is not None:
            LOGGER.info(f"MultiModal: PRIMARY={primary_ch}ch, SECONDARY={secondary_ch}ch")
        LOGGER.info(f"MultiModal: 总样本数={self.ni}, 模态目录='{self.x_modality_dir}'")

    def _initialize_precision_contract(self) -> None:
        """Resolve dataset-run precision contract and filter outlier samples before dataloader usage."""
        from ultralytics.data.multimodal.precision_contract import (
            filter_dataset_by_contract,
            log_dataset_precision_summary,
            resolve_precision_contract,
        )

        if not getattr(self, "im_files", None):
            return

        report = resolve_precision_contract(self, mode="dataset_run")
        keep_indices, filtered = filter_dataset_by_contract(self, report["contracts"])
        split = Path(self.im_files[0]).parent.name if self.im_files else "unknown"
        report["split"] = split
        report["filtered_count"] = len(filtered)
        report["filtered_examples"] = filtered

        if keep_indices and len(keep_indices) < len(self.im_files):
            self._apply_filtered_indices(keep_indices)
            report["filter_applied"] = True
        elif not keep_indices and filtered:
            LOGGER.warning(
                "MultiModal precision contract would filter all %d samples for split=%s, keeping original dataset",
                len(self.im_files),
                split,
            )
            report["filter_applied"] = False
            report["filtered_count"] = 0
            report["filtered_examples"] = []
        else:
            report["filter_applied"] = False

        self.precision_contract = dict(report["contracts"])
        self.precision_report = report
        self.precision_report_path = None
        log_dataset_precision_summary(report, None)

    def _apply_filtered_indices(self, keep_indices: List[int]) -> None:
        """Reindex dataset state after contract-based outlier filtering."""
        self.im_files = [self.im_files[i] for i in keep_indices]
        self.labels = [self.labels[i] for i in keep_indices]
        self.ni = len(self.labels)
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.x_ims, self.x_im_hw0, self.x_im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self._x_buffer = []
        self._x_paths = [self._x_paths[i] for i in keep_indices] if self._x_paths else []
        self._refresh_x_npy_files()
        self.buffer = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        if self.rect and self.ni:
            for i, label in enumerate(self.labels):
                if "shape" not in label:
                    try:
                        _, shape0, _ = self.load_image(i, rect_mode=False)
                    except Exception:
                        shape0 = (self.imgsz, self.imgsz)
                    label["shape"] = shape0
            self.set_rectangle()

    # -- backward compat properties for x_modality / x_modality_dir --

    @property
    def x_modality(self):
        """Backward compat: returns secondary_modality."""
        return getattr(self, '_secondary_modality', getattr(self, '_x_modality', 'unknown'))

    @x_modality.setter
    def x_modality(self, value):
        self._secondary_modality = value

    @property
    def x_modality_dir(self):
        """Backward compat: returns secondary_modality_dir."""
        return getattr(self, '_secondary_modality_dir', getattr(self, '_x_modality_dir', 'images'))

    @x_modality_dir.setter
    def x_modality_dir(self, value):
        self._secondary_modality_dir = value

    def _filter_legacy_x_cache_candidates(self, im_files: List[str]) -> List[str]:
        """Remove legacy X cache files accidentally written into primary image directories."""
        from ultralytics.data.utils import ARRAY_FORMATS, IMG_FORMATS

        x_modality = str(getattr(self, "x_modality", "") or "")
        if not x_modality:
            return im_files

        legacy_suffix = f"_{x_modality}"
        file_index = {
            (str(Path(f).parent), Path(f).stem, Path(f).suffix[1:].lower())
            for f in im_files
        }
        filtered = []
        removed = 0

        for f in im_files:
            p = Path(f)
            suffix = p.suffix[1:].lower()
            if suffix in ARRAY_FORMATS and p.stem.endswith(legacy_suffix):
                base_stem = p.stem[: -len(legacy_suffix)]
                has_primary_sibling = any(
                    (str(p.parent), base_stem, ext) in file_index for ext in IMG_FORMATS
                )
                if has_primary_sibling:
                    removed += 1
                    continue
            filtered.append(f)

        if removed:
            LOGGER.info(
                f"MultiModal: ignored {removed} legacy X cache files in primary image directories "
                f"(*_{x_modality}.npy/.npz). New X cache is stored under {self.mm_cache_root_name}/."
            )
        return filtered

    def get_img_files(self, img_path):
        """Override to support primary NPY/NPZ while ignoring legacy X cache pollution."""
        from ultralytics.data.utils import MM_PRIMARY_FORMATS

        im_files = super().get_img_files(img_path, allowed_formats=MM_PRIMARY_FORMATS)
        return self._filter_legacy_x_cache_candidates(im_files)

    def _img2label_path(self, img_path: Union[str, Path]) -> Path:
        """Convert image path to label path, supporting non-standard primary dirs (e.g. images_ir)."""
        img_path = Path(img_path)
        parts = list(img_path.parts)
        primary_dir = getattr(self, 'primary_modality_dir', None) or 'images'
        for i, part in enumerate(parts):
            if part == primary_dir:
                parts[i] = "labels"
                break
        return Path(*parts).with_suffix(".txt")

    def get_labels(self) -> List[Dict]:
        """Override: use multimodal-aware label path resolution.

        Replaces the hardcoded ``img2label_paths`` (which assumes ``/images/``
        directory) with ``_img2label_path`` that respects ``primary_modality_dir``
        (e.g. ``images_ir`` for IR+Depth datasets).
        """
        self.label_files = [str(self._img2label_path(f)) for f in self.im_files]
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def _validate_multimodal_data(self):
        """验证多模态数据的完整性和一致性"""
        if not hasattr(self, 'im_files') or not self.im_files:
            LOGGER.warning("主模态图像文件列表为空，无法验证多模态数据")
            return

        # 检查X模态目录是否存在（使用与路径构建一致的逻辑）
        primary_path = Path(self.im_files[0])
        dataset_root = primary_path.parent.parent.parent  # /dataset
        split_dir = primary_path.parent.name              # train/val/test
        x_dir = dataset_root / self.x_modality_dir / split_dir

        if not x_dir.exists():
            if self.enable_self_modal_generation:
                LOGGER.info(f"X模态目录 {x_dir} 不存在，将使用自体模态生成")
            else:
                LOGGER.debug(f"X模态目录 {x_dir} 不存在，将尝试从其他位置加载或使用自体生成")

        # 统计有效的多模态样本数量
        valid_count = 0
        sample_size = min(len(self.im_files), 100)  # 采样检查前100个

        for i in range(sample_size):
            x_path = self._x_paths[i] if i < len(self._x_paths) else None
            if x_path is not None and (Path(x_path).exists() or self.enable_self_modal_generation):
                valid_count += 1

        valid_ratio = valid_count / sample_size if sample_size > 0 else 0
        LOGGER.info(f"MultiModal: 多模态数据完整性检查 - {valid_ratio:.1%} ({valid_count}/{sample_size})")

        if valid_ratio < 0.5 and not self.enable_self_modal_generation:
            LOGGER.warning("多模态数据完整性较低，建议启用自体模态生成或检查数据路径")

    def _precompute_x_paths(self) -> None:
        """预计算X模态路径列表，与im_files 1:1对齐。可在_apply_filtered_indices后重建。"""
        self._x_paths = []
        for i in range(self.ni):
            try:
                rgb_path = self.im_files[i]
                self._x_paths.append(self._find_corresponding_x_image(rgb_path))
            except Exception:
                self._x_paths.append(None)

    def __getitem__(self, index):
        """
        获取多模态数据样本

        Args:
            index (int): 样本索引

        Returns:
            dict: 包含6通道多模态图像和标签的字典
                - 'img': 6通道图像张量 [6, H, W] (RGB+X)
                - 其他标签信息（bbox, cls等）

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(...)
            >>> sample = dataset[0]
            >>> img = sample['img']  # 6通道图像
            >>> print(img.shape)  # torch.Size([6, H, W])
        """
        if not getattr(self, "timedebug_enabled", False):
            # 获取基础标签信息（使用父类方法）
            label = self.get_image_and_label(index)

            # 加载多模态图像（替换单模态RGB图像）
            multimodal_img = self.load_multimodal_image(index)
            label["img"] = multimodal_img
            label["precision_contract"] = dict(self.precision_contract)
            label["precision_report_path"] = self.precision_report_path
            label["precision_mode"] = "dataset_run"

            # 应用数据变换
            return self.transforms(label)

        from ultralytics.utils.timedebug import add_sample_timing

        t0 = time.perf_counter()
        # 获取基础标签信息（使用父类方法）
        t_label = time.perf_counter()
        label = self.get_image_and_label(index)
        label["_timedebug_enabled"] = True
        add_sample_timing(label, "sample.get_image_and_label", time.perf_counter() - t_label)

        # 加载多模态图像（替换单模态RGB图像）
        t_load_mm = time.perf_counter()
        timing_stats = label.setdefault("_timedebug_worker_stats", {})
        try:
            multimodal_img = self.load_multimodal_image(index, timing_stats=timing_stats)
        except TypeError as e:
            if "timing_stats" not in str(e):
                raise
            multimodal_img = self.load_multimodal_image(index)
        add_sample_timing(label, "sample.load_multimodal_image", time.perf_counter() - t_load_mm)
        label["img"] = multimodal_img
        label["precision_contract"] = dict(self.precision_contract)
        label["precision_report_path"] = self.precision_report_path
        label["precision_mode"] = "dataset_run"

        # 应用数据变换
        t_transforms = time.perf_counter()
        label = self.transforms(label)
        add_sample_timing(label, "sample.transforms_total", time.perf_counter() - t_transforms)
        add_sample_timing(label, "sample.total", time.perf_counter() - t0)
        return label

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        构建并返回多模态专用增强流水线（独立于通用 v8_transforms）。

        说明：
        - 多模态数据集一律走独立 mm_transforms() 链路，可复用 v8 组件；
        - 单/双模态的路由与消融由路由层负责，数据层不做区分。
        """
        from ultralytics.data.multimodal_augment import mm_transforms, mm_seg_transforms
        from ultralytics.data.augment import Compose, LetterBox, Format

        if self.augment:
            # 强制走多模态增强链；分割任务走 mm_seg_transforms
            if getattr(self, "use_segments", False):
                transforms = mm_seg_transforms(self, self.imgsz, hyp)
            else:
                transforms = mm_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

        # 统一追加 Format；多模态训练不使用 bgr 概率翻转
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=0.0,  # 禁用 BGR 随机翻转，避免触动 X 通道
            )
        )
        return transforms

    def get_valid_indices(self):
        """
        获取有完整多模态数据的有效索引列表

        此方法用于与MultiModalMosaic和MultiModalMixUp数据增强兼容，
        确保增强过程中选择的图像都有完整的RGB+X模态数据。

        Returns:
            list: 有效索引列表

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(...)
            >>> valid_indices = dataset.get_valid_indices()
            >>> print(f"有效样本数: {len(valid_indices)}")
        """
        if not hasattr(self, '_valid_indices_cache'):
            self._valid_indices_cache = []

            # 仅在主进程/主 worker 打印日志，避免多 worker 噪声
            log_enabled = True
            try:
                from torch.utils.data import get_worker_info
                wi = get_worker_info()
                log_enabled = (wi is None) or (wi.id == 0)
            except Exception:
                log_enabled = True

            if log_enabled:
                LOGGER.info("MultiModal: 正在扫描有效的多模态索引...")

            for i in range(self.ni):
                # 检查RGB图像是否存在
                rgb_path = self.im_files[i]
                if not Path(rgb_path).exists():
                    continue

                # 检查X模态图像是否存在或可生成
                x_path = self._x_paths[i] if i < len(self._x_paths) else None
                if x_path is not None and (Path(x_path).exists() or self.enable_self_modal_generation):
                    self._valid_indices_cache.append(i)

            valid_count = len(self._valid_indices_cache)
            valid_ratio = valid_count / self.ni if self.ni > 0 else 0

            if log_enabled:
                LOGGER.info(f"✅ MultiModal: 发现 {valid_count}/{self.ni} ({valid_ratio:.1%}) 个有效多模态样本")

                if valid_count == 0:
                    LOGGER.error("未发现任何有效的多模态样本，请检查数据路径和配置")
                elif valid_ratio < 0.8:
                    LOGGER.warning(f"有效多模态样本比例较低 ({valid_ratio:.1%})，建议检查数据完整性")

        return self._valid_indices_cache

    def load_multimodal_image(self, i, timing_stats=None):
        """
        加载RGB+X模态图像并构建6通道图像

        Args:
            i (int): 图像索引
            timing_stats (dict | None): worker timing stats dict，用于子步骤计时。

        Returns:
            np.ndarray: 6通道图像数组 [H, W, 6] (RGB+X)

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(...)
            >>> img_6ch = dataset.load_multimodal_image(0)
            >>> print(img_6ch.shape)  # (H, W, 6)
        """
        from ultralytics.utils.timedebug import add_worker_timing

        # 加载RGB图像（使用父类方法）
        t_rgb = time.perf_counter()
        rgb_img, (h0, w0), (h, w) = self.load_image(i)
        add_worker_timing(timing_stats, "event.rgb.load", time.perf_counter() - t_rgb)

        # 预先记录X模态候选路径（用于日志）
        x_path_hint = self._x_paths[i] if i < len(self._x_paths) else None

        # 加载X模态图像
        t_x_load = time.perf_counter()
        try:
            x_img, x_hw0, x_hw = self._load_x_image_cached(i, target_shape=(h, w))
        except Exception as e:
            if self.enable_self_modal_generation:
                LOGGER.debug(f"X模态图像加载失败，使用自体生成: {e}")
                x_img = self._generate_self_modal(rgb_img)
            else:
                LOGGER.error(f"无法加载X模态图像 {i}: {e}")
                raise
        add_worker_timing(timing_stats, "event.x.load_cached", time.perf_counter() - t_x_load)

        # 确保X模态图像尺寸与RGB匹配
        t_resize = time.perf_counter()
        if x_img.shape[:2] != rgb_img.shape[:2]:
            x_img = cv2.resize(x_img, (rgb_img.shape[1], rgb_img.shape[0]))
        add_worker_timing(timing_stats, "event.x.resize", time.perf_counter() - t_resize)

        # 获取期望的X模态通道数
        expected_xch = getattr(self, 'data', {}).get('Xch', 3)

        # 处理X模态图像通道数（在启用多模态增强时，严格要求与 data['Xch'] 一致，杜绝静默扩展）
        t_ch_adapt = time.perf_counter()
        # 实际通道（未整形前）
        actual_xch = 1 if len(x_img.shape) == 2 else (x_img.shape[2] if len(x_img.shape) == 3 else 1)
        use_mm_aug = bool(getattr(self, 'hyp', None) and getattr(self.hyp, 'use_multimodal_aug', False))
        if use_mm_aug and actual_xch != expected_xch:
            raise ValueError(
                f"MultiModal: X 通道不一致，data.yaml Xch={expected_xch}，但实际为 {actual_xch}。"
                f" 文件: {x_path_hint or 'unknown'}；请修正数据或配置（禁用静默通道扩展）。"
            )

        # 常规整形（非严格模式下允许），严格模式已在上方阻断
        if len(x_img.shape) == 2:
            # 灰度图像
            if expected_xch == 1:
                x_img = x_img[:, :, np.newaxis]  # 保持1通道
            else:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)  # 转换为3通道
        elif x_img.shape[2] == 1:
            # 单通道图像
            if expected_xch == 1:
                pass  # 保持1通道
            else:
                x_img = np.repeat(x_img, 3, axis=2)  # 扩展为3通道
        elif x_img.shape[2] == 4:
            # RGBA图像
            x_img = x_img[:, :, :3]  # 移除alpha通道
            if expected_xch == 1:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        elif x_img.shape[2] == 3:
            # RGB图像
            if expected_xch == 1:
                x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
            else:
                pass  # 保持3通道

        # 验证X模态通道数
        if x_img.shape[2] != expected_xch:
            LOGGER.warning(
                f"X模态图像通道数({x_img.shape[2]})与配置Xch={expected_xch}不匹配，已自动调整; 文件: {x_path_hint or 'unknown'}"
            )
        add_worker_timing(timing_stats, "event.x.channel_adapt", time.perf_counter() - t_ch_adapt)

        # 构建多通道图像：Primary(前Pch通道) + Secondary(后Xch通道)
        t_concat = time.perf_counter()
        multimodal_img = np.concatenate([rgb_img, x_img], axis=2)
        add_worker_timing(timing_stats, "event.mm.concat", time.perf_counter() - t_concat)

        # MM-DEBUG: 数据加载出口诊断
        import os
        if os.environ.get("MM_DEBUG", "") == "1":
            from ultralytics.data.augment import _mm_dbg
            _mm_dbg("dataset_load", multimodal_img, f"rgb={rgb_img.dtype} x={x_img.dtype}")

        return multimodal_img

    def _load_multimodal_components(self, i, timing_stats=None):
        """返回 (rgb_img, x_img) 元组，不做 concat。供 Mosaic 额外图像直接填入画布。

        与 load_multimodal_image 共享相同的 RGB/X 加载+通道适配逻辑，
        但跳过 np.concatenate，消除 ~75% 的 concat 调用（Mosaic 额外图像路径）。
        4ch/6ch/9ch 统一适用：调用方通过 rgb.shape[2]/x.shape[2] 获取通道数。
        """
        rgb_img, (h0, w0), (h, w) = self.load_image(i)
        try:
            x_img, _, _ = self._load_x_image_cached(i, target_shape=(h, w))
        except FileNotFoundError:
            if self.enable_self_modal_generation:
                x_img = self._generate_self_modal(rgb_img)
            else:
                raise

        # 尺寸对齐
        if x_img.shape[:2] != rgb_img.shape[:2]:
            x_img = cv2.resize(x_img, (rgb_img.shape[1], rgb_img.shape[0]))

        # 通道适配（复用 load_multimodal_image 中的逻辑）
        expected_xch = self.data.get("Xch", 3)
        if x_img.ndim == 2:
            x_img = x_img[:, :, np.newaxis] if expected_xch == 1 else cv2.cvtColor(x_img, cv2.COLOR_GRAY2BGR)
        elif x_img.shape[2] == 1 and expected_xch != 1:
            x_img = np.repeat(x_img, expected_xch, axis=2)
        elif x_img.shape[2] == 4:
            x_img = x_img[:, :, :3] if expected_xch != 1 else cv2.cvtColor(x_img[:, :, :3], cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

        return rgb_img, x_img

    def _find_corresponding_x_image(self, rgb_path):
        """
        根据RGB图像路径找到对应的X模态图像路径

        Args:
            rgb_path (str): RGB图像路径

        Returns:
            str: X模态图像路径

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(x_modality_dir="images_ir")
            >>> rgb_path = "/dataset/images/train/img001.jpg"
            >>> x_path = dataset._find_corresponding_x_image(rgb_path)
            >>> print(x_path)  # "/dataset/images_ir/train/img001.jpg"
        """
        rgb_path = Path(rgb_path)

        # 构建X模态图像路径
        # 保持相同的子目录结构 (train/val/test)，只替换基础模态目录
        # RGB路径: /dataset/images/train/img.jpg
        # X路径:   /dataset/images_ir/train/img.jpg
        dataset_root = rgb_path.parent.parent.parent  # /dataset (跳过images和train)
        split_dir = rgb_path.parent.name              # train/val/test
        x_dir = dataset_root / self.x_modality_dir / split_dir

        # 处理文件后缀
        if self.x_modality_suffix:
            x_filename = rgb_path.stem + self.x_modality_suffix + rgb_path.suffix
        else:
            x_filename = rgb_path.name

        x_path = x_dir / x_filename

        # 如果指定路径不存在，尝试常见的文件扩展名（包含npy/npz特征文件）
        if not x_path.exists():
            common_extensions = ['.npy', '.npz', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
            for ext in common_extensions:
                test_path = x_dir / (rgb_path.stem + ext)
                if test_path.exists():
                    return str(test_path)

        return str(x_path)

    def _load_x_modality(self, x_path):
        """
        加载X模态图像

        Args:
            x_path (str): X模态图像路径

        Returns:
            np.ndarray: X模态图像数组

        Raises:
            FileNotFoundError: 当X模态图像文件不存在时
        """
        x_path = Path(x_path)

        if not x_path.exists():
            raise FileNotFoundError(f"X模态图像不存在: {x_path}")

        # 根据文件扩展名选择加载方式
        if x_path.suffix.lower() in ['.npy']:
            # NumPy数组文件
            x_img = np.load(x_path)
        elif x_path.suffix.lower() in ['.npz']:
            # 多数组压缩文件，按约定顺序择优取出
            with np.load(x_path) as npz_file:
                preferred_keys = ('image', 'arr_0', 'array', 'data')
                selected_key = next((k for k in preferred_keys if k in npz_file.files), None)

                if selected_key is None:
                    if len(npz_file.files) == 1:
                        selected_key = npz_file.files[0]
                    else:
                        raise ValueError(
                            f"npz文件 {x_path} 含多个数组 {npz_file.files}，无法确定默认键，请显式使用标准键(image/arr_0)。"
                        )

                x_img = npz_file[selected_key]
        elif x_path.suffix.lower() in ['.tiff', '.tif']:
            # TIFF文件（常用于深度图）
            x_img = cv2.imread(str(x_path), cv2.IMREAD_UNCHANGED)
        else:
            # 标准图像文件
            x_img = cv2.imread(str(x_path))

        if x_img is None:
            raise ValueError(f"无法读取X模态图像: {x_path}")

        # Legacy 分支: 返回原始数组，归一化在 preprocess 阶段按 contract 处理。
        return x_img

    def _generate_self_modal(self, rgb_img):
        """
        生成自体模态图像（当X模态图像缺失时）

        Args:
            rgb_img (np.ndarray): RGB图像

        Returns:
            np.ndarray: 生成的X模态图像
        """
        # 简单的自体模态生成策略：
        # 1. 转换为灰度图
        # 2. 应用边缘检测
        # 3. 转换回3通道

        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

        # 应用高斯模糊和边缘检测
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 转换为3通道
        x_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return x_img

    def cache_images(self, cache=None):
        """
        缓存RGB和X模态图像到内存或磁盘

        Args:
            cache (str, bool, or None): 缓存模式
                - True or 'ram': 缓存到内存
                - 'disk': 缓存到磁盘
                - False: 不缓存
                - None: 使用self.cache属性

        Examples:
            >>> dataset = YOLOMultiModalImageDataset(...)
            >>> dataset.cache_images('ram')  # 缓存到内存
            >>> dataset.cache_images('disk')  # 缓存到磁盘
            >>> dataset.cache_images()  # 使用self.cache属性
        """
        # Use parameter if provided, otherwise fall back to instance attribute
        cache_mode = cache if cache is not None else self.cache

        # 如果不缓存，初始化空缓存
        if not cache_mode:
            self.x_ims = [None] * self.ni
            self.x_im_hw0 = [None] * self.ni
            self.x_im_hw = [None] * self.ni
            return

        # 传统缓存模式
        # 初始化多模态缓存存储
        self.x_ims = [None] * self.ni
        self.x_im_hw0 = [None] * self.ni
        self.x_im_hw = [None] * self.ni

        # 调用父类方法缓存RGB图像
        super().cache_images(cache_mode)

        # 缓存X模态图像
        if cache_mode:
            self._cache_x_modality_images(cache_mode)

    def _cache_x_modality_images(self, cache):
        """缓存X模态图像"""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self._cache_x_images_to_disk if cache == "disk" else self._load_x_image_for_cache

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == "disk":
                    x_npy_file = Path(x) if x is not None else self._get_x_npy_file(i)
                    if x_npy_file.exists():
                        b += x_npy_file.stat().st_size
                else:  # 'ram'
                    if x is not None:
                        self.x_ims[i], self.x_im_hw0[i], self.x_im_hw[i] = x
                        b += self.x_ims[i].nbytes if self.x_ims[i] is not None else 0
                pbar.desc = f"{self.prefix}Caching {self.x_modality} images ({b / gb:.1f}GB {cache})"
            pbar.close()

    def _load_x_image_for_cache(self, i):
        """为缓存加载X模态图像（使用预计算的X路径）"""
        try:
            if i < len(self._x_paths) and self._x_paths[i] is not None:
                x_path = self._x_paths[i]
            elif self.im_files and i < len(self.im_files):
                x_path = self._find_corresponding_x_image(self.im_files[i])
            else:
                x_path = None
            if x_path is None:
                raise FileNotFoundError(f"X模态路径未预计算: index={i}")

            # 加载X模态图像
            x_img = self._load_x_modality(x_path)

            # 获取RGB图像信息用于尺寸匹配
            if self.ims[i] is not None:
                # 如果RGB已缓存，使用其尺寸
                rgb_img = self.ims[i]
                h0, w0 = self.im_hw0[i]
                target_shape = rgb_img.shape[:2]
            else:
                # 如果RGB未缓存，先加载RGB获取目标尺寸
                rgb_img, (h0, w0), target_shape = self.load_image(i)

            # 调整X模态图像尺寸以匹配RGB
            if x_img.shape[:2] != target_shape:
                x_img = cv2.resize(x_img, (target_shape[1], target_shape[0]))

            return x_img, (h0, w0), x_img.shape[:2]

        except FileNotFoundError as e:
            LOGGER.warning(f"无法加载X模态图像 {i}: {e}")
            return None, None, None

    def _cache_x_images_to_disk(self, i):
        """Cache one X-modality image into the isolated multimodal cache directory."""
        x_npy_file = self._get_x_npy_file(i)
        if x_npy_file.exists():
            return x_npy_file

        try:
            result = self._load_x_image_for_cache(i)
            if result[0] is not None:
                x_img, _, _ = result
                x_npy_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(x_npy_file.as_posix(), x_img, allow_pickle=False)
        except Exception as e:
            LOGGER.warning(f"无法缓存X模态图像到磁盘 {i}: {e}")
        return x_npy_file

    def _dataset_root_from_primary_path(self, primary_path: Union[str, Path]) -> Path:
        """Resolve dataset root from a primary modality sample path."""
        primary_path = Path(primary_path)
        primary_dir = getattr(self, "primary_modality_dir", None) or "images"
        parts = primary_path.parts

        try:
            idx = len(parts) - 1 - parts[::-1].index(primary_dir)
            root_parts = parts[:idx]
            if root_parts:
                return Path(*root_parts)
        except ValueError:
            pass

        try:
            return primary_path.parent.parent.parent
        except IndexError:
            return primary_path.parent

    def _primary_relative_path_for_cache(self, primary_path: Union[str, Path]) -> Path:
        """Return the path under the primary modality directory, preserving split/subdirs."""
        primary_path = Path(primary_path)
        primary_dir = getattr(self, "primary_modality_dir", None) or "images"
        parts = primary_path.parts

        try:
            idx = len(parts) - 1 - parts[::-1].index(primary_dir)
            rel_parts = parts[idx + 1 :]
            if rel_parts:
                return Path(*rel_parts)
        except ValueError:
            pass

        if primary_path.parent.name:
            return Path(primary_path.parent.name) / primary_path.name
        return Path(primary_path.name)

    def _get_x_cache_root(self, primary_path: Union[str, Path]) -> Path:
        """Return isolated X-modality cache root outside primary image scan dirs."""
        dataset_root = self._dataset_root_from_primary_path(primary_path)
        x_modality = str(getattr(self, "x_modality", "x") or "x")
        return dataset_root / self.mm_cache_root_name / x_modality

    def _refresh_x_npy_files(self) -> None:
        """Refresh per-sample X cache file paths after im_files changes."""
        self.x_npy_files = [self._get_x_npy_file(i) for i in range(getattr(self, "ni", 0))]

    def _get_x_npy_file(self, i):
        """Return isolated X-modality .npy cache path for sample index i."""
        primary_path = Path(self.im_files[i])
        relative_path = self._primary_relative_path_for_cache(primary_path).with_suffix(".npy")
        return self._get_x_cache_root(primary_path) / relative_path

    def load_image(self, i, rect_mode=True):
        """重写load_image方法，支持多模态缓存加载"""
        # 加载RGB图像（使用父类方法）
        rgb_img, ori_shape, resized_shape = super().load_image(i, rect_mode)
        return rgb_img, ori_shape, resized_shape

    def _load_x_image_cached(self, i, target_shape=None):
        """从缓存或磁盘加载X模态图像"""
        # 检查内存缓存
        if self.x_ims is not None and self.x_ims[i] is not None:
            return self.x_ims[i], self.x_im_hw0[i], self.x_im_hw[i]

        # 检查磁盘缓存
        x_npy_file = self._get_x_npy_file(i)
        if x_npy_file.exists():
            try:
                x_img = np.load(x_npy_file, allow_pickle=False)
                h0, w0 = x_img.shape[:2]
                if target_shape and x_img.shape[:2] != target_shape:
                    x_img = cv2.resize(x_img, (target_shape[1], target_shape[0]))
                self._cache_x_im(i, x_img, h0, w0)
                return x_img, (h0, w0), x_img.shape[:2]
            except Exception as e:
                LOGGER.warning(f"加载X模态缓存文件失败 {x_npy_file}: {e}")

        # 从原始文件加载（使用预计算的X路径）
        try:
            if i < len(self._x_paths) and self._x_paths[i] is not None:
                x_path = self._x_paths[i]
            elif self.im_files and i < len(self.im_files):
                x_path = self._find_corresponding_x_image(self.im_files[i])
            else:
                x_path = None
            if x_path is None:
                raise FileNotFoundError(f"X模态路径未预计算: index={i}")
            x_img = self._load_x_modality(x_path)

            h0, w0 = x_img.shape[:2]  # 原始尺寸，resize前捕获

            # 如果提供了目标尺寸，调整X模态图像尺寸
            if target_shape and x_img.shape[:2] != target_shape:
                x_img = cv2.resize(x_img, (target_shape[1], target_shape[0]))

            self._cache_x_im(i, x_img, h0, w0)
            return x_img, (h0, w0), x_img.shape[:2]

        except FileNotFoundError as e:
            LOGGER.error(f"无法加载X模态图像 {i}: {e}")
            raise

    def _cache_x_im(self, i, x_img, h0, w0):
        """X模态写回缓存 + LRU驱逐，与RGB的buffer/ims缓存模式对齐。

        Args:
            i: 样本索引
            x_img: 已resize到目标尺寸的X模态图像
            h0, w0: resize前的原始尺寸
        """
        if not self.augment:
            return
        if self.x_ims is None:
            self.x_ims = [None] * self.ni
            self.x_im_hw0 = [None] * self.ni
            self.x_im_hw = [None] * self.ni
        self.x_ims[i] = x_img
        self.x_im_hw0[i] = (h0, w0)
        self.x_im_hw[i] = x_img.shape[:2]
        self._x_buffer.append(i)
        if 1 < len(self._x_buffer) >= self.max_buffer_length:
            j = self._x_buffer.pop(0)
            if self.cache != "ram":
                self.x_ims[j] = None
                self.x_im_hw0[j] = None
                self.x_im_hw[j] = None

    def check_cache_ram(self, safety_margin=0.5):
        """检查多模态图像缓存的内存需求"""
        # 检查RGB图像内存需求（使用父类方法）
        rgb_cache_ok = super().check_cache_ram(safety_margin)

        # 检查X模态图像内存需求
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images

        for _ in range(n):
            try:
                # 随机选择一个样本估算X模态图像大小
                rgb_path = random.choice(self.im_files)
                x_path = self._find_corresponding_x_image(rgb_path)
                x_img = self._load_x_modality(x_path)

                ratio = self.imgsz / max(x_img.shape[0], x_img.shape[1])
                b += x_img.nbytes * ratio**2
            except Exception:
                # 如果加载失败，使用RGB图像大小作为估算
                rgb_img = cv2.imread(random.choice(self.im_files))
                if rgb_img is not None:
                    ratio = self.imgsz / max(rgb_img.shape[0], rgb_img.shape[1])
                    b += rgb_img.nbytes * ratio**2

        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache X modality
        mem = psutil.virtual_memory()
        x_cache_ok = mem_required < mem.available

        if not x_cache_ok:
            LOGGER.info(
                f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache {self.x_modality} images '
                f'with {int(safety_margin * 100)}% safety margin but only '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                f"{'caching images ✅' if x_cache_ok else f'not caching {self.x_modality} images ⚠️'}"
            )

        # 只有当RGB和X模态都能缓存时才返回True
        return rgb_cache_ok and x_cache_ok

    def check_cache_disk(self, safety_margin=0.5):
        """Check if there is enough disk space for both primary and X-modality disk caches."""
        import shutil

        primary_cache_ok = super().check_cache_disk(safety_margin)
        if not primary_cache_ok:
            return False

        if not getattr(self, "im_files", None):
            return primary_cache_ok

        cache_root = self._get_x_cache_root(self.im_files[0])
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.cache = None
            LOGGER.warning(f"{self.prefix}Skipping X modality disk cache, cache root not writeable: {cache_root} ({e})")
            return False

        b, gb = 0, 1 << 30
        n = min(self.ni, 30)
        samples = 0
        for _ in range(n):
            try:
                rgb_path = random.choice(self.im_files)
                x_path = self._find_corresponding_x_image(rgb_path)
                x_img = self._load_x_modality(x_path)
                b += x_img.nbytes
                samples += 1
            except Exception:
                continue

        if samples == 0:
            return primary_cache_ok

        disk_required = b * self.ni / samples * (1 + safety_margin)
        total, used, free = shutil.disk_usage(cache_root)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required to cache {self.x_modality} images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching {self.x_modality} images"
            )
            return False
        return True


class GroundingDataset(YOLODataset):
    """
    Dataset class for object detection tasks using annotations from a JSON file in grounding format.

    This dataset is designed for grounding tasks where annotations are provided in a JSON file rather than
    the standard YOLO format text files.

    Attributes:
        json_file (str): Path to the JSON file containing annotations.

    Methods:
        get_img_files: Return empty list as image files are read in get_labels.
        get_labels: Load annotations from a JSON file and prepare them for training.
        build_transforms: Configure augmentations for training with optional text loading.

    Examples:
        >>> dataset = GroundingDataset(img_path="path/to/images", json_file="annotations.json", task="detect")
        >>> len(dataset)  # Number of valid images with annotations
    """

    def __init__(self, *args, task: str = "detect", json_file: str = "", max_samples: int = 80, **kwargs):
        """
        Initialize a GroundingDataset for object detection.

        Args:
            json_file (str): Path to the JSON file containing annotations.
            task (str): Must be 'detect' or 'segment' for GroundingDataset.
            max_samples (int): Maximum number of samples to load for text augmentation.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        assert task in {"detect", "segment"}, "GroundingDataset currently only supports `detect` and `segment` tasks"
        self.json_file = json_file
        self.max_samples = max_samples
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path: str) -> List:
        """
        The image files would be read in `get_labels` function, return empty list here.

        Args:
            img_path (str): Path to the directory containing images.

        Returns:
            (list): Empty list as image files are read in get_labels.
        """
        return []

    def verify_labels(self, labels: List[Dict[str, Any]]) -> None:
        """
        Verify the number of instances in the dataset matches expected counts.

        This method checks if the total number of bounding box instances in the provided
        labels matches the expected count for known datasets. It performs validation
        against a predefined set of datasets with known instance counts.

        Args:
            labels (List[Dict[str, Any]]): List of label dictionaries, where each dictionary
                contains dataset annotations. Each label dict must have a 'bboxes' key with
                a numpy array or tensor containing bounding box coordinates.

        Raises:
            AssertionError: If the actual instance count doesn't match the expected count
                for a recognized dataset.

        Note:
            For unrecognized datasets (those not in the predefined expected_counts),
            a warning is logged and verification is skipped.
        """
        expected_counts = {
            "final_mixed_train_no_coco_segm": 3662412,
            "final_mixed_train_no_coco": 3681235,
            "final_flickr_separateGT_train_segm": 638214,
            "final_flickr_separateGT_train": 640704,
        }

        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        for data_name, count in expected_counts.items():
            if data_name in self.json_file:
                assert instance_count == count, f"'{self.json_file}' has {instance_count} instances, expected {count}."
                return
        LOGGER.warning(f"Skipping instance count verification for unrecognized dataset '{self.json_file}'")

    def cache_labels(self, path: Path = Path("./labels.cache")) -> Dict[str, Any]:
        """
        Load annotations from a JSON file, filter, and normalize bounding boxes for each image.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (Dict[str, Any]): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls] + s
                        segments.append(s)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)

            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> List[Dict]:
        """
        Load labels from cache or generate them from JSON file.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.json_file)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, _ = self.cache_labels(cache_path), False  # run cache ops
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels = cache["labels"]
        self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels

    def build_transforms(self, hyp: Optional[Dict] = None) -> Compose:
        """
        Configure augmentations for training with optional text loading.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.max_samples, 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """Return unique category names from the dataset."""
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq: Dict, threshold: int = 100) -> List[str]:
        """Get negative text samples based on frequency threshold."""
        threshold = min(max(category_freq.values()), 100)
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets for YOLO training, ensuring they use the same
    collation function.

    Methods:
        collate_fn: Static method that collates data samples into batches using YOLODataset's collation function.

    Examples:
        >>> dataset1 = YOLODataset(...)
        >>> dataset2 = YOLODataset(...)
        >>> combined_dataset = YOLOConcatDataset([dataset1, dataset2])
    """

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp: Dict) -> None:
        """
        Set mosaic, copy_paste and mixup options to 0.0 and build transformations.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """Semantic Segmentation Dataset."""

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Dataset class for image classification tasks extending torchvision ImageFolder functionality.

    This class offers functionalities like image augmentation, caching, and verification. It's designed to efficiently
    handle large datasets for training deep learning models, with optional image transformations and caching mechanisms
    to speed up training.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
        root (str): Root directory of the dataset.
        prefix (str): Prefix for logging and cache filenames.

    Methods:
        __getitem__: Return subset of data and targets corresponding to given indices.
        __len__: Return the total number of samples in the dataset.
        verify_images: Verify all images in dataset.
    """

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """
        Initialize YOLO classification dataset with root directory, arguments, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings.
            augment (bool, optional): Whether to apply augmentations to the dataset.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )

    def __getitem__(self, i: int) -> Dict:
        """
        Return subset of data and targets corresponding to given indices.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            (dict): Dictionary containing the image and its class index.
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self) -> List[Tuple]:
        """
        Verify all images in dataset.

        Returns:
            (list): List of valid samples after verification.
        """
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)  # check image read speeds
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples


class YOLOMultiModalClassifyDataset(MultiModalImageIOMixin):
    """
    多模态分类数据集（RGB+X）

    设计目标：在 `ultralytics/data/dataset.py` 内与现有数据集实现保持一致的组织方式，
    并沿用 YOLO 系列统一的数据管理范式（YAML + images/labels），而不是 torchvision ImageFolder 目录分类范式。

    数据组织结构（示例）:
        dataset/
        ├── images/
        │   ├── train/
        │   │   └── img001.jpg
        │   └── val/
        ├── images_depth/          # X模态
        │   ├── train/
        │   │   └── img001.png     # 同名，可不同扩展
        │   └── val/
        └── labels/
            ├── train/
            │   └── img001.txt     # 第一行第一列: cls_id
            └── val/

    说明：
        - 标签文件仅解析第一行第一列为 cls_id。
        - RGB 的 torchvision 分类增强仅作用在 RGB 上；X 模态使用对齐 + resize 与 RGB 输出尺寸一致后再拼接。
        - 本类对标签缺失/越界、X 模态缺失等情况采取严格报错（Fail-Fast），避免静默跳过样本。
    """

    def __init__(
        self,
        img_path: str,
        data: Dict[str, Any],
        args,
        augment: bool = False,
        prefix: str = "",
        precision_report_dir=None,
        **kwargs,
    ):
        self.data = data
        self.args = args
        self.augment = augment
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.precision_report_dir = precision_report_dir
        self.precision_contract = {"rgb": "legacy", "x": "legacy"}
        self.precision_report = None
        self.precision_report_path = None

        # 解析多模态配置
        self._parse_multimodal_config()

        # Dataset fraction (only for training/augment)
        self.fraction = float(getattr(args, "fraction", 1.0)) if augment else 1.0

        # 获取图像文件列表（复用 BaseDataset.get_img_files 的统一扫描/校验逻辑）
        from ultralytics.data.utils import MM_PRIMARY_FORMATS
        self.im_files = BaseDataset.get_img_files(self, img_path, allowed_formats=MM_PRIMARY_FORMATS)

        # 解析标签 & 预检查 X 模态文件存在性
        self.labels = self._get_labels_strict()
        self.samples = self._build_samples()

        # 缓存相关（对齐 ClassificationDataset 行为）
        self.cache_ram = getattr(args, "cache", False) is True or str(getattr(args, "cache", "")).lower() == "ram"
        if self.cache_ram:
            LOGGER.warning(
                "MultiModal Classification `cache_ram` has known memory leak, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(getattr(args, "cache", "")).lower() == "disk"

        # 使用 LetterBox 预处理（与检测任务统一）
        self.imgsz = getattr(args, "imgsz", 224)
        from ultralytics.data.augment import LetterBox
        self.letterbox = LetterBox(
            new_shape=self.imgsz,
            auto=False,
            scale_fill=False,
            scaleup=augment,  # 训练时允许放大，验证时不放大
            center=True,
            stride=32
        )

        # dataset-run precision contract: 采样探测 -> 过滤 -> 写报告
        self._initialize_classify_precision_contract()

        LOGGER.info(
            f"{self.prefix}MultiModal Classification Dataset initialized: "
            f"samples={len(self.samples)}, modalities=[{getattr(self, 'primary_modality', 'rgb')},{self.x_modality}], "
            f"Pch={getattr(self, 'primary_ch', 3)}, Xch={self.expected_xch}"
        )

    def _parse_multimodal_config(self) -> None:
        """解析多模态配置（严格）"""
        if "modality_used" not in self.data:
            raise ValueError(
                f"{self.prefix}缺少必需字段 `modality_used`（示例: ['rgb','depth'] 或 ['thermal','depth']）。"
            )
        modality_used = self.data["modality_used"]
        if not isinstance(modality_used, list) or len(modality_used) != 2:
            raise ValueError(f"{self.prefix}modality_used 必须是包含2个模态的列表，当前: {modality_used}")

        self.primary_modality, self.secondary_modality = resolve_modality_pair(modality_used, strict=True)
        self.x_modality = self.secondary_modality  # backward compat
        self.primary_modality_dir = get_modality_path_from_data(self.data, self.primary_modality)
        self.secondary_modality_dir = get_modality_path_from_data(self.data, self.secondary_modality)
        self.x_modality_dir = self.secondary_modality_dir  # backward compat
        self.x_modality_suffix = self.data.get("x_modality_suffix", None)

        self.expected_xch = int(self.data.get("Xch", 3))
        self.nc = int(self.data.get("nc", len(self.data.get("names", {}))))
        self.names = self.data.get("names", {})

        # Auto-detect channel counts
        dataset_root = Path(self.data.get('path', '.'))
        self.primary_ch = self.data.get('Pch', None)
        self.secondary_ch = self.data.get('Xch', None)
        if self.primary_ch is None:
            self.primary_ch = auto_detect_modality_channels(dataset_root, self.primary_modality_dir, split="train")
        if self.secondary_ch is None:
            self.secondary_ch = auto_detect_modality_channels(dataset_root, self.secondary_modality_dir, split="train")

    def _img2label_path(self, img_path: Union[str, Path]) -> Path:
        """将图像路径转换为标签路径: .../<images>/<split>/img.jpg -> .../labels/<split>/img.txt"""
        img_path = Path(img_path)
        parts = list(img_path.parts)
        for i, part in enumerate(parts):
            if part == getattr(self, 'primary_modality_dir', None) or (
                getattr(self, 'primary_modality_dir', None) is None and part == "images"
            ):
                parts[i] = "labels"
                break
        return Path(*parts).with_suffix(".txt")

    def _parse_cls_label(self, label_path: Path) -> int:
        """解析分类标签文件：只读取第一行第一列作为 cls_id"""
        if not label_path.exists():
            raise FileNotFoundError(f"{self.prefix}标签文件不存在: {label_path}")

        with label_path.open("r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue
                tok = s.split()
                cls_id = int(tok[0])
                if not (0 <= cls_id < self.nc):
                    raise ValueError(f"{self.prefix}cls_id 越界: {cls_id}, nc={self.nc}, file={label_path}")
                return cls_id

        raise ValueError(f"{self.prefix}标签文件为空或无有效行: {label_path}")

    def _get_labels_strict(self) -> List[int]:
        """扫描并解析全部标签；任何错误将汇总后抛出。"""
        labels: List[int] = []
        errors: List[str] = []

        pbar = TQDM(self.im_files, desc=f"{self.prefix}Scanning labels...", disable=LOCAL_RANK not in {-1, 0})
        for im_file in pbar:
            label_path = self._img2label_path(im_file)
            try:
                labels.append(self._parse_cls_label(label_path))
            except Exception as e:
                errors.append(f"{label_path}: {e}")
                labels.append(-1)

        if errors:
            preview = "\n".join(errors[:20])
            more = f"\n... 另外 {len(errors) - 20} 个错误" if len(errors) > 20 else ""
            raise ValueError(f"{self.prefix}发现无效标签文件，共 {len(errors)} 个：\n{preview}{more}")

        return labels

    def _build_samples(self) -> List[List[Any]]:
        """构建样本列表并预检查 X 模态路径存在性。"""
        samples: List[List[Any]] = []
        errors: List[str] = []

        for im_file, cls_id in zip(self.im_files, self.labels):
            if cls_id < 0:
                # labels 已严格校验，这里仅作防御
                continue

            x_path = self.find_corresponding_x_image(
                rgb_path=im_file,
                x_modality_dir=self.x_modality_dir,
                x_modality_suffix=self.x_modality_suffix,
            )
            if not Path(x_path).exists():
                errors.append(f"X模态缺失: rgb={im_file} -> x={x_path}")

            npy_path = Path(im_file).with_suffix(".npy")
            samples.append([im_file, cls_id, npy_path, None, x_path])

        if errors:
            preview = "\n".join(errors[:20])
            more = f"\n... 另外 {len(errors) - 20} 个错误" if len(errors) > 20 else ""
            raise FileNotFoundError(f"{self.prefix}发现 X 模态缺失，共 {len(errors)} 个：\n{preview}{more}")

        return samples

    def _initialize_classify_precision_contract(self) -> None:
        """Resolve dataset-run precision contract for classify dataset.

        Mirrors YOLOMultiModalImageDataset._initialize_precision_contract but
        adapts to classify's data structure (self.samples instead of self.labels).
        """
        from ultralytics.data.multimodal.precision_contract import (
            filter_dataset_by_contract,
            log_dataset_precision_summary,
            probe_dataset_precision,
        )

        if not getattr(self, "im_files", None):
            return

        report = probe_dataset_precision(self)
        keep_indices, filtered = filter_dataset_by_contract(self, report["contracts"])
        split = Path(self.im_files[0]).parent.name if self.im_files else "unknown"
        report["split"] = split
        report["filtered_count"] = len(filtered)
        report["filtered_examples"] = filtered

        if keep_indices and len(keep_indices) < len(self.im_files):
            # Filter samples / im_files / labels to keep only matching indices
            self.samples = [self.samples[i] for i in keep_indices]
            self.im_files = [self.im_files[i] for i in keep_indices]
            self.labels = [self.labels[i] for i in keep_indices]
            report["filter_applied"] = True
        elif not keep_indices and filtered:
            LOGGER.warning(
                "MultiModal precision contract would filter all %d classify samples for split=%s, keeping original dataset",
                len(self.im_files),
                split,
            )
            report["filter_applied"] = False
            report["filtered_count"] = 0
            report["filtered_examples"] = []
        else:
            report["filter_applied"] = False

        self.precision_contract = dict(report["contracts"])
        self.precision_report = report
        self.precision_report_path = None
        log_dataset_precision_summary(report, None)

    def _find_corresponding_x_image(self, rgb_path):
        """Bridge: precision_contract._resolve_x_path calls dataset._find_corresponding_x_image."""
        return self.find_corresponding_x_image(
            rgb_path=rgb_path,
            x_modality_dir=self.x_modality_dir,
            x_modality_suffix=self.x_modality_suffix,
        )

    def build_transforms(self):
        """LetterBox 已在 __init__ 中初始化，此方法仅为兼容性保留"""
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        im_file, cls_id, npy_path, cached_img, x_path = self.samples[index]

        # 加载 RGB 图像（BGR）
        if self.cache_ram and cached_img is not None:
            rgb_img = cached_img
        elif self.cache_disk:
            if not npy_path.exists():
                rgb_img = cv2.imread(im_file)
                if rgb_img is None:
                    raise ValueError(f"{self.prefix}无法读取RGB图像: {im_file}")
                np.save(npy_path.as_posix(), rgb_img, allow_pickle=False)
            rgb_img = np.load(npy_path)
        else:
            rgb_img = cv2.imread(im_file)

        if rgb_img is None:
            raise ValueError(f"{self.prefix}无法读取RGB图像: {im_file}")

        # 加载并对齐 X 模态图像（仍在原始分辨率空间）
        x_img = self.load_x_modality(x_path)
        x_img = self.align_x_to_rgb(x_img, rgb_img.shape[:2])
        x_img = self.validate_x_channels(x_img, self.expected_xch, strict=False, x_path_hint=str(x_path))

        # Mixin 在 expected_xch>3 时会 warning 而不是 fail-fast，这里强制校验
        final_xch = 1 if len(x_img.shape) == 2 else (x_img.shape[2] if len(x_img.shape) == 3 else -1)
        if final_xch != self.expected_xch:
            raise ValueError(
                f"{self.prefix}X模态通道数不匹配: expected={self.expected_xch}, got={final_xch}, file={x_path}"
            )

        # 确保 X 是 3D
        if len(x_img.shape) == 2:
            x_img = x_img[:, :, np.newaxis]

        # 合并为多通道图像，统一使用 LetterBox
        primary_img = rgb_img  # primary modality image
        secondary_img = x_img  # secondary modality image
        merged = np.concatenate([primary_img, secondary_img], axis=2)  # [H, W, Pch+Xch]
        merged_lb = self.letterbox(image=merged)  # [H', W', Pch+Xch]

        # 分离主模态和附模态
        primary_ch_count = getattr(self, 'primary_ch', 3) or 3
        primary_lb = merged_lb[:, :, :primary_ch_count]
        secondary_lb = merged_lb[:, :, primary_ch_count:]

        # Precision contract: use dataset-run contract resolved at __init__ time
        from ultralytics.data.multimodal.utils import to_tensor_rgb, to_tensor_x

        precision_contract = self.precision_contract

        # Tensorize: 根据 primary_modality 选择正确的 tensorize helper
        if is_rgb_modality(self.primary_modality):
            # RGB 模态: BGR->RGB + HWC->CHW + contract-aware normalize
            primary_rgb = primary_lb[:, :, ::-1].copy()
            primary_tensor = to_tensor_rgb(primary_rgb, precision=precision_contract["rgb"])
        else:
            # 非 RGB 主模态: HWC->CHW + contract-aware normalize
            primary_tensor = to_tensor_x(primary_lb, precision=precision_contract["rgb"])

        secondary_tensor = to_tensor_x(secondary_lb, precision=precision_contract["x"])

        return {
            "img": torch.cat([primary_tensor, secondary_tensor], dim=0),
            "cls": cls_id,
            "precision_contract": precision_contract,
        }


# ===== SemSeg Format + Dataset =====

class SemSegFormat:
    """Convert semantic mask to torch.long after standard Format has processed img/instances."""

    def __call__(self, labels: dict) -> dict:
        sem_mask = labels.get("sem_mask", None)
        if sem_mask is None:
            raise KeyError("SemSeg sample missing 'sem_mask'")
        if isinstance(sem_mask, np.ndarray):
            labels["sem_mask"] = torch.from_numpy(np.ascontiguousarray(sem_mask)).long()
        elif isinstance(sem_mask, torch.Tensor):
            labels["sem_mask"] = sem_mask.long()
        else:
            raise TypeError(f"sem_mask must be ndarray or Tensor, got {type(sem_mask).__name__}")
        return labels


class YOLOMultiModalSemSegDataset(YOLOMultiModalImageDataset):
    """YOLOMM pure semantic segmentation dataset using image mask labels."""

    SEM_LABEL_EXTS = (".png", ".jpg", ".jpeg")
    _warned_jpg_label = False
    _warned_multichannel_label = False
    _warned_invalid_values = False

    @classmethod
    def _warn_once(cls, attr_name: str, message: str):
        if not getattr(cls, attr_name, False):
            setattr(cls, attr_name, True)
            LOGGER.warning(message)

    def _parse_label_mapping(self, mapping):
        """Normalize label_mapping from dataset YAML into int->int ids.

        Args:
            mapping: Raw label_mapping value from dataset YAML (dict or None).

        Returns:
            Normalized dict mapping source pixel value (int) -> target class ID (int).
            Empty dict if mapping is None.
        """
        if mapping is None:
            return {}
        if not isinstance(mapping, dict):
            raise TypeError(f"Expected 'label_mapping' to be a dict, got {type(mapping).__name__}")
        normalized = {}
        for src, dst in mapping.items():
            src = int(src)
            if dst is None:
                dst = 255
            elif isinstance(dst, str):
                dst = dst.strip()
                dst = 255 if dst == "ignore_label" else int(dst)
            else:
                dst = int(dst)
            normalized[src] = dst
        return normalized

    def convert_label(self, label, inverse=False):
        """Convert label values using label_mapping.

        Args:
            label: np.ndarray, segmentation mask.
            inverse: If True, map target->source (for saving predictions).
        """
        if not self.label_mapping:
            return label
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def _resolve_semseg_labels_dir(self) -> Path:
        """Resolve labels directory for SemSeg: explicit `labels` key, then modality.rgb substitution."""
        if "labels" in self.data and self.data["labels"]:
            return Path(self.data["path"]) / self.data["labels"]
        modality_rgb = self.data.get("modality", {}).get("rgb", "images")
        train_or_val = os.path.basename(self.img_path.rstrip("/"))
        if modality_rgb != "images":
            # 替换 modality.rgb 路径中的最后一段为 labels
            parts = os.path.normpath(modality_rgb).split(os.sep)
            parts[-1] = "labels"
            return Path(self.data["path"]) / os.path.join(*parts) / train_or_val
        # 缺省：从 img_path 推断
        return (Path(self.img_path).parent.parent / "labels" / train_or_val).resolve()

    def _find_semseg_label_file(self, im_file, labels_dir: Path) -> Optional[Path]:
        """Find label image matching image file in labels_dir. Priority: .png > .jpg > .jpeg."""
        stem = os.path.splitext(os.path.basename(im_file))[0]
        for ext in self.SEM_LABEL_EXTS:
            candidate = labels_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate.resolve()
        return None

    def _sanitize_invalid_sem_values(self, mask: np.ndarray, nc: int, ignore_index: int) -> np.ndarray:
        """Map invalid class IDs to ignore_index. Returns sanitized array."""
        invalid = (mask != ignore_index) & ((mask < 0) | (mask >= nc))
        if invalid.any():
            self._warn_once("_warned_invalid_values",
                            f"SemSeg mask contains {invalid.sum()} invalid class values, mapped to ignore_index={ignore_index}")
            mask = mask.copy()
            mask[invalid] = ignore_index
        return mask.astype(np.int64)

    def _load_sem_mask(self, sem_file: str, rgb_shape: tuple) -> np.ndarray:
        """Load semantic mask: grayscale, resize to img, sanitize, apply label_mapping.

        Aligns with upstream SemanticDataset.load_mask():
        - IMREAD_GRAYSCALE forces single-channel
        - Auto-resize to img dimensions (tolerant, not error)
        - nc=1 handles 1-bit PNG
        - Applies label_mapping then sanitize
        """
        if not sem_file:
            raise FileNotFoundError("sem_file is None or empty")
        mask = cv2.imread(sem_file, cv2.IMREAD_GRAYSCALE)  # upstream: force grayscale
        if mask is None:
            raise FileNotFoundError(f"Cannot read SemSeg mask: {sem_file}")

        ext = os.path.splitext(sem_file)[1].lower()
        if ext in {".jpg", ".jpeg"}:
            self._warn_once("_warned_jpg_label",
                            f"SemSeg label in JPEG format ({sem_file}). PNG is recommended.")

        # nc=1 binary: handle 1-bit PNG (value range {0, 255} -> {0, 1})
        nc = int(self.data.get("nc", 80))
        if nc == 1:
            from PIL import Image as PILImage
            with PILImage.open(sem_file) as im:
                if im.mode == "1":  # true 1-bit PNG
                    mask[mask == 255] = 1

        # Resize to img dimensions (upstream: tolerant, not error)
        h, w = rgb_shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # label_mapping conversion
        if self.label_mapping:
            mask = self.convert_label(mask, inverse=False)

        # sanitize invalid values
        ignore_index = int(self.data.get("ignore_index", getattr(self, "ignore_index", 255)))
        mask = self._sanitize_invalid_sem_values(mask, nc=nc, ignore_index=ignore_index)
        return mask.astype(np.uint8)  # uint8, not int64
    def _probe_sem_mask(sem_file: str, ignore_index: int, nc: int) -> dict:
        """Probe sem_mask file: hash, shape, invalid count. Does NOT load full array."""
        import hashlib
        mask = cv2.imread(sem_file, cv2.IMREAD_UNCHANGED)
        if mask is None:
            return {"hash": None, "mask_shape": (0, 0), "invalid_count": -1}
        if mask.ndim == 3 and mask.shape[-1] > 1:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.ndim == 3:
            mask = mask[..., 0]
        file_hash = hashlib.sha256(mask.tobytes()).hexdigest()
        mask_shape = mask.shape[:2]
        invalid = (mask != ignore_index) & ((mask < 0) | (mask >= nc))
        invalid_count = int(invalid.sum())
        return {"hash": file_hash, "mask_shape": mask_shape, "invalid_count": invalid_count}

    def _cache_sem_labels(self, labels_dir: Path, sem_files: list) -> dict:
        """Build and persist .cache_semseg file. Returns cache dict."""
        from pathlib import Path as _Path
        from ultralytics.data.utils import save_dataset_cache_file

        nc = int(self.data.get("nc", 80))
        ignore_index = int(self.data.get("ignore_index", getattr(self, "ignore_index", 255)))
        cache_path = _Path(self.im_files[0]).parent / f"{labels_dir.name}.cache_semseg"

        labels = []
        for im_file, sem_file in zip(self.im_files, sem_files):
            probe = self._probe_sem_mask(str(sem_file), ignore_index=ignore_index, nc=nc)
            labels.append({
                "im_file": im_file,
                "sem_file": str(sem_file),
                "shape": probe["mask_shape"],  # (h, w) 来自 _probe_sem_mask
                "cls": [],
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "segments": [],
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
                "mask_hash": probe["hash"],
                "mask_shape": probe["mask_shape"],
                "invalid_count": probe["invalid_count"],
            })

        cache = {
            "hash": get_hash(
                [str(f) for f in sem_files] + [str(self.img_path)] + [f"lm:{json.dumps(self.label_mapping, sort_keys=True)}"]
            ),
            "labels": labels,
            "results": (len(labels), 0, 0, 0, len(labels)),
            "msgs": [],
        }
        save_dataset_cache_file(self.prefix, cache_path, cache, DATASET_CACHE_VERSION)
        LOGGER.info(f"{self.prefix}SemSeg cache created: {cache_path} ({len(labels)} labels)")
        return cache

    def get_labels(self):
        # Initialize label_mapping on first call (self.data is available from parent __init__)
        if not hasattr(self, "label_mapping"):
            self.label_mapping = self._parse_label_mapping(self.data.get("label_mapping"))

        labels_dir = self._resolve_semseg_labels_dir()
        if not labels_dir.exists():
            raise FileNotFoundError(f"{self.prefix}SemSeg labels directory not found: {labels_dir}")

        sem_files = []
        skipped = 0
        for im_file in self.im_files:
            sem_file = self._find_semseg_label_file(im_file, labels_dir)
            if sem_file is None:
                skipped += 1
                continue
            sem_files.append(sem_file)
        if not sem_files:
            raise RuntimeError(f"{self.prefix}No valid SemSeg image/label pairs found under {labels_dir}")
        if skipped:
            LOGGER.warning(f"{self.prefix}SemSeg skipped {skipped} image(s) with missing image-label files")

        from pathlib import Path as _Path
        from ultralytics.data.utils import load_dataset_cache_file

        cache_path = _Path(self.im_files[0]).parent / f"{labels_dir.name}.cache_semseg"
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache.get("version") == DATASET_CACHE_VERSION
            assert cache.get("hash") == get_hash(
                [str(f) for f in sem_files] + [str(self.img_path)] + [f"lm:{json.dumps(self.label_mapping, sort_keys=True)}"]
            )
        except (FileNotFoundError, AssertionError, AttributeError):
            cache = self._cache_sem_labels(labels_dir, sem_files)

        labels = cache["labels"]
        nf, nm, ne, nc_, n = cache.pop("results")
        if LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc_} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache.get("msgs"):
                LOGGER.info("\n".join(cache["msgs"]))
        [cache.pop(k) for k in ("hash", "version", "msgs")]

        # 兼容器旧缓存（v1: shape=None 的 .cache_semseg 文件）
        for lb in labels:
            s = lb.get("shape")
            if s is None or (isinstance(s, tuple) and len(s) == 2 and s == (0, 0)):
                im = cv2.imread(lb["im_file"])
                lb["shape"] = im.shape[:2] if im is not None else (0, 0)

        self.im_files = [lb["im_file"] for lb in labels]
        return labels

    def get_image_and_label(self, index: int) -> dict:
        """Return labels with RGB+X multimodal image and sem_mask.

        Overrides parent so that Mosaic donor loading also carries sem_mask.
        """
        label = super().get_image_and_label(index)
        rgb, x_img = self._load_multimodal_components(index)
        label["_rgb"] = rgb
        label["_x"] = x_img
        label["img"] = np.concatenate([rgb, x_img], axis=2)
        label["sem_mask"] = self._load_sem_mask(self.labels[index]["sem_file"],
                                                  rgb_shape=rgb.shape[:2])
        return label

    def __getitem__(self, index):
        label = self.get_image_and_label(index)
        label["precision_contract"] = dict(self.precision_contract)
        label["precision_report_path"] = self.precision_report_path
        label["precision_mode"] = "dataset_run"
        label["index"] = index
        return self.transforms(label)

    def build_transforms(self, hyp=None):
        from ultralytics.data.augment import Compose, Format, LetterBox
        from ultralytics.data.multimodal_augment import mm_semseg_transforms

        ignore = int(getattr(hyp, "ignore_index", getattr(self, "ignore_index", 255)))
        if self.augment:
            transforms = mm_semseg_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False, ignore_index=ignore)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                bgr=0.0,
            )
        )
        transforms.append(SemSegFormat())
        return transforms

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate SemSeg samples; stack sem_mask into [B,H,W]."""
        new_batch = YOLODataset.collate_fn(batch)
        sem_mask = new_batch.get("sem_mask")
        if isinstance(sem_mask, (tuple, list)):
            new_batch["sem_mask"] = torch.stack([m.long() for m in sem_mask], 0)
        elif isinstance(sem_mask, torch.Tensor) and sem_mask.ndim == 2:
            new_batch["sem_mask"] = sem_mask.unsqueeze(0).long()
        elif isinstance(sem_mask, torch.Tensor):
            new_batch["sem_mask"] = sem_mask.long()
        else:
            raise KeyError("SemSeg batch missing 'sem_mask'")
        return new_batch
