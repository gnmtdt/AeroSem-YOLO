# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""YOLOVO 专用训练增强链。"""

from ultralytics.data.augment import (
    Albumentations,
    Compose,
    CopyPaste,
    CutMix,
    LetterBox,
    MixUp,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.data.multimodal_augment import MultiModalObjectPaste
from ultralytics.utils import LOGGER


def yolovo_transforms(dataset, imgsz: int, hyp) -> Compose:
    """构建 YOLOVO RGB-only detect v1 增强链，不包含 Format。

    ObjectPaste 放在 Mosaic/RandomPerspective 之后，使 target 图像和 bbox 已经处于训练输入尺度。
    该 builder 第一阶段只由 YOLOVODataset 调用，不影响通用 YOLO 或 YOLOMM 链路。
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=getattr(hyp, "mosaic", 0.0))
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
    )
    pre_transform = Compose([mosaic, affine])

    # 保持 v8 copy_paste 语义不变。detect 数据集无 segments 时该增强自然 no-op。
    cp_mode = str(getattr(hyp, "copy_paste_mode", "flip"))
    if cp_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=getattr(hyp, "copy_paste", 0.0), mode=cp_mode))
    else:
        donor_pre = Compose([Mosaic(dataset, imgsz=imgsz, p=getattr(hyp, "mosaic", 0.0)), affine])
        pre_transform.append(
            CopyPaste(dataset, pre_transform=donor_pre, p=getattr(hyp, "copy_paste", 0.0), mode=cp_mode)
        )

    flip_idx = dataset.data.get("flip_idx", [])
    if getattr(dataset, "use_keypoints", False):
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
            hyp.fliplr = hyp.flipud = 0.0
            LOGGER.warning(
                "No 'flip_idx' array defined in data.yaml, disabling 'fliplr' and 'flipud' augmentations."
            )
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    tfl = [pre_transform]

    object_paste_p = float(getattr(hyp, "object_paste", 0.0) or 0.0)
    detect_only = not (
        getattr(dataset, "use_segments", False)
        or getattr(dataset, "use_keypoints", False)
        or getattr(dataset, "use_obb", False)
    )
    if object_paste_p > 0.0 and detect_only:
        tfl.append(
            MultiModalObjectPaste(
                dataset=dataset,
                p=object_paste_p,
                candidate_limit=3,
                scale_range=(0.8, 1.2),
                min_area=0.001,
                max_area=0.35,
                min_side_px=8,
                overlap_thresh=0.20,
                feather_ratio=0.10,
                feather_min_px=4,
                feather_max_px=16,
                debug=bool(getattr(dataset, "yolovo_debug", False)),
                debug_dir=getattr(dataset, "yolovo_object_paste_debug_dir", None),
                debug_limit=8,
            )
        )

    tfl.extend(
        [
            MixUp(dataset, pre_transform=pre_transform, p=getattr(hyp, "mixup", 0.0)),
            CutMix(dataset, pre_transform=pre_transform, p=getattr(hyp, "cutmix", 0.0)),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )
    return Compose(tfl)
