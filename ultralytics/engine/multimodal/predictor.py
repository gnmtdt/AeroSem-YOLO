# Ultralytics Multimodal Inference - Predictor Engine
# Independent multimodal inference engine (no BasePredictor dependency)
# Version: v1.0
# Date: 2026-01-13

import torch
import numpy as np
import time
from pathlib import Path
from typing import Union, List, Dict, Optional, Generator
from ultralytics.utils import LOGGER, ops
from ultralytics.data.multimodal import PairingResolver, MultiModalInferenceDataset
from ultralytics.nn.mm.utils import get_runtime_x_modality_name, get_runtime_primary_modality_name
from .results import MultiModalResults
from .saver import MultiModalSaver


class MultiModalPredictor:
    """
    多模态推理引擎核心（完全独立，不依赖 BasePredictor）

    职责：
    - 从模型读取 Router 配置（Xch）
    - 构建 dataset 并迭代样本
    - 对每个样本执行：forward -> NMS -> scale_boxes -> 组装结果
    - 支持真正的 stream（边迭代边 yield）

    设计原则：
    - 1 sample -> 1 result（不做中途变形）
    - 不解析 YAML 推断融合策略（依赖 Router 的 forward 行为）
    - stream=True 为真正流式：边迭代边 yield
    """

    def __init__(
        self,
        model,
        imgsz: Union[int, tuple] = 640,
        conf: float = 0.25,
        iou: float = 0.5,
        max_det: int = 300,
        device: str = '',
        verbose: bool = True,
        debug: bool = False,
        font_size: Optional[int] = None,
        show_filename: bool = False,
        speed: bool = False
    ):
        """
        初始化多模态推理引擎

        Args:
            model: YOLOMM/RTDETRMM 模型实例
            imgsz: 推理输入尺寸
            conf: 置信度阈值
            iou: NMS IOU 阈值
            max_det: 最大检测框数量
            device: 设备 ('cuda', 'cpu', 或 'cuda:0')
            verbose: 是否输出详细日志（常规日志）
            debug: 是否输出DEBUG调试日志
            font_size: 可视化字体大小（None=自动）
            show_filename: 是否在结果图上显示源文件名
        """
        self.model = model
        self.verbose = verbose
        self.debug = debug  # 新增debug参数
        self.font_size = font_size
        self.show_filename = show_filename
        self.speed = speed  # 速度探针开关（默认关闭，不影响训练管线）

        # 从模型读取 Router 配置
        if hasattr(model, 'model') and hasattr(model.model, 'mm_router'):
            self.router = model.model.mm_router
            self.xch = int(self.router.INPUT_SOURCES.get("SECONDARY", getattr(self.router, "secondary_ch", 3)))
            self.primary_ch = int(self.router.INPUT_SOURCES.get("PRIMARY", getattr(self.router, "primary_ch", 3)))
            self.primary_modality = get_runtime_primary_modality_name(model)
            self.x_modality_type = get_runtime_x_modality_name(model)
        else:
            raise ValueError(
                "模型缺少 mm_router。确保使用 YOLOMM 或 RTDETRMM 模型。"
            )

        # 检测模型类型
        self.is_rtdetr = self._detect_rtdetr_model(model)
        if self.verbose:
            LOGGER.info(f"  模型类型: {'RTDETR' if self.is_rtdetr else 'YOLO'}")

        # 推理参数
        self.imgsz = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

        # 设备配置
        if device == '':
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        if self.verbose:
            LOGGER.info(f"MultiModalPredictor 初始化完成:")
            LOGGER.info(f"  设备: {self.device}")
            LOGGER.info(f"  推理尺寸: {self.imgsz}")
            if self.x_modality_type != 'unknown':
                LOGGER.info(f"  X模态类型: {self.x_modality_type}")
            LOGGER.info(f"  X模态通道数: {self.xch}")
            LOGGER.info(f"  置信度阈值: {self.conf}")
            LOGGER.info(f"  NMS IOU阈值: {self.iou}")

    def __call__(
        self,
        rgb_source: Union[str, Path, List[Union[str, Path]]],
        x_source: Union[str, Path, List[Union[str, Path]]],
        stream: bool = False,
        strict_match: bool = True,
        save: bool = False,
        save_txt: bool = False,
        save_json: bool = False,
        save_dir: Optional[Path] = None,
        crop: bool = False,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        font_size: Optional[int] = None,
        show_filename: Optional[bool] = None,
        speed: Optional[bool] = None,
        **kwargs
    ):
        """
        执行多模态推理（新API - 显式RGB和X模态输入）

        Args:
            rgb_source: RGB图像源
            x_source: X模态图像源
            stream: 是否流式返回结果（True=生成器，False=列表）
            strict_match: 批量推理时的匹配策略（默认严格）
            save: 是否保存可视化结果
            save_txt: 是否保存txt标签
            save_json: 是否保存json结果
            save_dir: 保存目录（可选）
            conf: 置信度阈值（可选，覆盖初始化时的默认值）
            iou: NMS IOU阈值（可选，覆盖初始化时的默认值）
            max_det: 最大检测框数量（可选，覆盖初始化时的默认值）
            font_size: 可视化字体大小（可选，覆盖初始化时的默认值）
            show_filename: 是否在结果图上显示源文件名（可选，覆盖初始化时的默认值）
            **kwargs: 其他参数

        Returns:
            stream=True: Generator[MultiModalResult]
            stream=False: List[MultiModalResult]
        """
        # 解析输入源为配对样本
        resolver = PairingResolver(x_modality=self.x_modality_type, verbose=self.verbose)
        sample_specs = resolver.resolve(rgb_source=rgb_source, x_source=x_source, strict_match=strict_match)

        # 构建推理数据集
        dataset = MultiModalInferenceDataset(
            samples=sample_specs,
            imgsz=self.imgsz,
            dataset_config={
                'Xch': self.xch,
                'x_modality': self.x_modality_type,
                'primary_modality': self.primary_modality,
                'primary_ch': self.primary_ch,
            },
            stride=getattr(self.model, 'stride', 32),
            verbose=self.verbose
        )

        # 使用运行时参数或初始化时的默认值
        _font_size = font_size if font_size is not None else self.font_size
        _show_filename = show_filename if show_filename is not None else self.show_filename
        if speed is not None:
            self.speed = speed  # 运行时覆盖速度探针开关

        # 流式推理
        if stream:
            return self._stream_inference(
                dataset, save=save, save_txt=save_txt, save_json=save_json, save_dir=save_dir,
                crop=crop, conf=conf, iou=iou, max_det=max_det,
                font_size=_font_size, show_filename=_show_filename
            )
        else:
            return list(self._stream_inference(
                dataset, save=save, save_txt=save_txt, save_json=save_json, save_dir=save_dir,
                crop=crop, conf=conf, iou=iou, max_det=max_det,
                font_size=_font_size, show_filename=_show_filename
            ))

    def _stream_inference(
        self,
        dataset: MultiModalInferenceDataset,
        save: bool = False,
        save_txt: bool = False,
        save_json: bool = False,
        save_dir: Optional[Path] = None,
        crop: bool = False,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        font_size: Optional[int] = None,
        show_filename: bool = False
    ) -> Generator:
        """
        流式推理实现（真正的 stream：边迭代边 yield）

        Args:
            dataset: 多模态推理数据集
            save: 是否保存可视化
            save_txt: 是否保存txt标签
            save_json: 是否保存json结果
            save_dir: 保存目录
            conf: 置信度阈值（可选，覆盖self.conf）
            iou: NMS IOU阈值（可选，覆盖self.iou）
            max_det: 最大检测框数量（可选，覆盖self.max_det）
            font_size: 可视化字体大小（可选，覆盖self.font_size）
            show_filename: 是否在结果图上显示源文件名

        Yields:
            MultiModalResult: 单个样本的推理结果
        """
        # 使用动态参数或默认值
        conf_threshold = conf if conf is not None else self.conf
        iou_threshold = iou if iou is not None else self.iou
        max_detections = max_det if max_det is not None else self.max_det

        if self.verbose:
            LOGGER.info(f"推理参数 - conf: {conf_threshold}, iou: {iou_threshold}, max_det: {max_detections}")
        for sample in dataset:
            # 阶段1: 预处理（dataset 迭代 + to device）
            t0 = time.perf_counter() if self.speed else 0
            # 1. Forward（路由在模型内部完成）
            im_tensor = sample['im'].to(self.device)  # [1, 3+Xch, H, W]
            if self.speed:
                t_pre = (time.perf_counter() - t0) * 1000

            # 阶段2: 推理
            t0 = time.perf_counter() if self.speed else 0
            with torch.no_grad():
                preds = self.model(im_tensor)  # 模型 forward
            if self.speed:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_infer = (time.perf_counter() - t0) * 1000

            # 调试日志：查看模型输出（debug控制）
            if self.debug:
                LOGGER.info(f"[DEBUG] 模型输出类型: {type(preds)}")
                if isinstance(preds, torch.Tensor):
                    LOGGER.info(f"[DEBUG] 模型输出形状: {preds.shape}")
                elif isinstance(preds, (list, tuple)):
                    LOGGER.info(f"[DEBUG] 模型输出是list/tuple，长度: {len(preds)}")
                    LOGGER.info(f"[DEBUG] preds[0]类型: {type(preds[0])}, 形状: {preds[0].shape if isinstance(preds[0], torch.Tensor) else 'N/A'}")

            # 2. 根据模型类型选择后处理路径（不再依赖输出格式判断）
            if self.debug:
                LOGGER.info(f"[DEBUG] 模型类型: {'RTDETR' if self.is_rtdetr else 'YOLO'}")

            if self.is_rtdetr:
                # RTDETR 专用后处理：归一化cxcywh -> 像素xyxy
                if self.debug:
                    LOGGER.info("[DEBUG] 使用RTDETR后处理分支")

                # 从AutoBackend输出提取真正的Tensor
                # AutoBackend将RTDETRDecoder的(y, x)转为[y, x]
                preds_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds

                if self.debug:
                    LOGGER.info(f"[DEBUG] 提取后的Tensor形状: {preds_tensor.shape}")

                pred = self._postprocess_rtdetr(preds_tensor, sample, conf_threshold, iou_threshold, max_detections)
            else:
                # YOLO 标准后处理：像素xywh -> NMS -> 像素xyxy
                if self.debug:
                    LOGGER.info("[DEBUG] 使用YOLO后处理分支")
                preds = ops.non_max_suppression(
                    preds,
                    conf_thres=conf_threshold,
                    iou_thres=iou_threshold,
                    max_det=max_detections
                )
                pred = preds[0]  # 取第一个（batch_size=1）

                if self.debug:
                    LOGGER.info(f"[DEBUG] NMS后检测数量: {len(pred)}")
                    if len(pred) > 0:
                        LOGGER.info(f"[DEBUG] NMS后前3个框: {pred[:3, :4]}")

                if len(pred):
                    # 使用 sample.meta['ratio_pad'] 和 sample.meta['ori_shape'] 还原坐标
                    # ratio_pad 格式: (gain, (padw, padh))
                    # scale_boxes 需要: ratio_pad=((gain, gain), (padw, padh))
                    gain = sample['meta']['ratio_pad'][0]
                    pad = sample['meta']['ratio_pad'][1]

                    if self.debug:
                        LOGGER.info(f"[DEBUG] gain={gain}, pad={pad}")
                        LOGGER.info(f"[DEBUG] imgsz={sample['meta']['imgsz']}, ori_shape={sample['meta']['ori_shape']}")

                    pred[:, :4] = ops.scale_boxes(
                        img1_shape=sample['meta']['imgsz'],  # 推理输入尺寸
                        boxes=pred[:, :4],
                        img0_shape=sample['meta']['ori_shape'],  # RGB 原图尺寸
                        ratio_pad=((gain, gain), pad),  # 转换为 scale_boxes 期望的格式
                        padding=True
                    )

                    if self.debug:
                        LOGGER.info(f"[DEBUG] scale_boxes后前3个框: {pred[:3, :4]}")

            # 阶段3: 后处理（NMS + scale_boxes + 组装结果）
            t0 = time.perf_counter() if self.speed else 0

            # 3. 组装 MultiModalResult
            result = self._create_result(sample, pred)

            if self.speed:
                t_post = (time.perf_counter() - t0) * 1000
                result.speed = {'preprocess': t_pre, 'inference': t_infer, 'postprocess': t_post}

            # 4. 保存（如果需要）
            if save or save_txt or save_json or crop:
                self._save_result(result, save=save, save_txt=save_txt, save_json=save_json, save_dir=save_dir, crop=crop, font_size=font_size, show_filename=show_filename)

            # 边迭代边 yield
            yield result

    def _is_rtdetr_output(self, preds: torch.Tensor) -> bool:
        """
        检测模型输出是否为RTDETR格式

        Args:
            preds: 模型输出张量

        Returns:
            bool: True表示RTDETR格式 (bs, num_queries, 4+nc)
                  False表示YOLO格式 (bs, num_boxes, 4+nc) 或 (bs, 4+nc, num_boxes)
        """
        # RTDETR: (bs, 300, 4+nc) 或固定查询数
        # YOLO: (bs, num_boxes, 5+nc) 或 (bs, 4+nc, num_boxes)

        if isinstance(preds, (list, tuple)):
            # YOLO NMS前: list of tensors
            return False

        if not isinstance(preds, torch.Tensor):
            return False

        # RTDETR的典型特征：维度2固定为300（或其他查询数）且较小
        # 输出形状: (bs, 300, 4+nc)
        if len(preds.shape) == 3:
            # 检查是否有固定的查询数（RTDETR典型为300）
            num_queries = preds.shape[1]
            if 100 <= num_queries <= 900:  # RTDETR查询数通常在此范围
                # 进一步验证：最后一维应该是 4 + nc (类别数)
                last_dim = preds.shape[2]
                if last_dim >= 5:  # 至少4个bbox坐标 + 1个类别
                    return True

        return False

    def _detect_rtdetr_model(self, model) -> bool:
        """
        检测模型是否为RTDETR类型

        Args:
            model: 模型实例

        Returns:
            bool: True表示RTDETR，False表示YOLO
        """
        # 检查模型head类型
        try:
            # 获取实际的PyTorch模型
            pt_model = model
            if hasattr(model, 'model'):
                pt_model = model.model

            # 方法1：检查模型类名
            model_class_name = pt_model.__class__.__name__
            if self.verbose:
                LOGGER.info(f"  PyTorch模型类名: {model_class_name}")
            if 'RTDETR' in model_class_name:
                return True

            # 方法2：检查最后一层 (Sequential 或 list)
            if hasattr(pt_model, 'model'):
                model_layers = pt_model.model
                # 处理Sequential或list
                if hasattr(model_layers, '__getitem__'):
                    try:
                        last_layer = model_layers[-1]
                        layer_name = last_layer.__class__.__name__
                        if self.verbose:
                            LOGGER.info(f"  模型head类型: {layer_name}")
                        return 'RTDETR' in layer_name
                    except:
                        pass

        except Exception as e:
            if self.verbose:
                LOGGER.warning(f"无法检测模型类型: {e}")

        return False

    def _postprocess_rtdetr(
        self,
        preds: torch.Tensor,
        sample: Dict,
        conf_threshold: float = None,
        iou_threshold: float = None,
        max_detections: int = None
    ) -> torch.Tensor:
        """
        RTDETR专用后处理：归一化cxcywh -> 像素xyxy -> 原图坐标

        Args:
            preds: RTDETR输出 (bs, num_queries, 4+nc)，bbox为归一化cxcywh [0,1]
            sample: 数据集样本，包含meta信息
            conf_threshold: 置信度阈值（可选，覆盖self.conf）
            iou_threshold: NMS IOU阈值（可选，覆盖self.iou）
            max_detections: 最大检测框数量（可选，覆盖self.max_det）

        Returns:
            torch.Tensor: [N, 6] (x1, y1, x2, y2, conf, cls) 原图坐标
        """
        # 使用动态参数或默认值
        conf_thresh = conf_threshold if conf_threshold is not None else self.conf
        iou_thresh = iou_threshold if iou_threshold is not None else self.iou
        max_det = max_detections if max_detections is not None else self.max_det

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] 输入preds形状: {preds.shape}")

        # 1. 拆分bbox和分类分数
        # RTDETR输出: (bs, 300, 4+nc)
        # bbox: (bs, 300, 4) - 归一化cxcywh
        # scores: (bs, 300, nc) - 类别概率
        bboxes = preds[0, :, :4]  # (300, 4) 归一化cxcywh
        scores = preds[0, :, 4:]  # (300, nc) 类别分数

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] bbox形状: {bboxes.shape}, scores形状: {scores.shape}")
            LOGGER.info(f"[DEBUG RTDETR] 前3个bbox (归一化cxcywh): {bboxes[:3]}")

        # 2. 置信度过滤
        # 对每个查询，取最大类别分数作为置信度
        conf, cls = scores.max(dim=1)  # (300,), (300,)

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] 置信度阈值: {conf_thresh}")
            LOGGER.info(f"[DEBUG RTDETR] 最大置信度: {conf.max()}, 最小置信度: {conf.min()}")

        # 过滤低置信度
        mask = conf >= conf_thresh
        bboxes = bboxes[mask]  # (N, 4)
        conf = conf[mask]      # (N,)
        cls = cls[mask]        # (N,)

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] 过滤后检测数量: {len(bboxes)}")

        # 如果没有检测结果，返回空张量
        if len(bboxes) == 0:
            if self.debug:
                LOGGER.info("[DEBUG RTDETR] 无检测结果，返回空张量")
            return torch.zeros((0, 6), device=preds.device)

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] 过滤后前3个bbox (归一化cxcywh): {bboxes[:3]}")

        # 3. 坐标转换：归一化cxcywh -> 归一化xyxy
        bboxes = ops.xywh2xyxy(bboxes)  # (N, 4) 归一化xyxy

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] 转换为归一化xyxy后前3个bbox: {bboxes[:3]}")

        # 4. 缩放到letterbox尺寸（像素坐标）
        h_letterbox, w_letterbox = sample['meta']['imgsz']
        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] letterbox尺寸: {h_letterbox}x{w_letterbox}")

        bboxes[:, [0, 2]] *= w_letterbox  # x坐标
        bboxes[:, [1, 3]] *= h_letterbox  # y坐标

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] 缩放到letterbox像素坐标后前3个bbox: {bboxes[:3]}")

        # 5. NMS去重（RTDETR虽然用了匈牙利匹配，但可能仍有重叠框）
        # 组装为[N, 6]格式用于NMS
        det = torch.cat([bboxes, conf.unsqueeze(1), cls.unsqueeze(1)], dim=1)  # (N, 6)

        # 应用NMS（使用torchvision的nms）
        from torchvision.ops import nms as torch_nms
        keep_indices = torch_nms(
            boxes=det[:, :4],
            scores=det[:, 4],
            iou_threshold=iou_thresh
        )
        det = det[keep_indices]

        if self.debug:
            LOGGER.info(f"[DEBUG RTDETR] NMS后检测数量: {len(det)}")

        # 6. 限制最大检测数
        if len(det) > max_det:
            det = det[:max_det]
            if self.debug:
                LOGGER.info(f"[DEBUG RTDETR] 限制到max_det={max_det}")

        # 7. 缩放到原图尺寸
        if len(det):
            gain = sample['meta']['ratio_pad'][0]
            pad = sample['meta']['ratio_pad'][1]

            if self.debug:
                LOGGER.info(f"[DEBUG RTDETR] gain={gain}, pad={pad}")
                LOGGER.info(f"[DEBUG RTDETR] ori_shape={sample['meta']['ori_shape']}")
                LOGGER.info(f"[DEBUG RTDETR] scale_boxes前前3个bbox: {det[:3, :4]}")

            det[:, :4] = ops.scale_boxes(
                img1_shape=sample['meta']['imgsz'],
                boxes=det[:, :4],
                img0_shape=sample['meta']['ori_shape'],
                ratio_pad=((gain, gain), pad),
                padding=True
            )

            if self.debug:
                LOGGER.info(f"[DEBUG RTDETR] scale_boxes后前3个bbox: {det[:3, :4]}")

        return det

    def _create_result(self, sample: Dict, pred: torch.Tensor) -> MultiModalResults:
        """
        组装 MultiModalResults

        Args:
            sample: 数据集样本（包含 paths, orig_imgs, meta, im）
            pred: NMS 后的预测结果 [N, 6] (x1,y1,x2,y2,conf,cls)

        Returns:
            MultiModalResults 实例
        """
        boxes = pred.cpu().numpy() if len(pred) else np.zeros((0, 6))

        # 获取类别名称（如果模型有提供）
        names = getattr(self.model, 'names', None)

        # 将 id 添加到 meta 中（从 sample 顶层提取）
        meta = sample['meta'].copy()
        meta['id'] = sample['id']

        result = MultiModalResults(
            boxes=boxes,
            paths=sample['paths'],
            orig_imgs=sample['orig_imgs'],
            meta=meta,
            names=names
        )

        return result

    def _save_result(
        self,
        result: MultiModalResults,
        save: bool = False,
        save_txt: bool = False,
        save_json: bool = False,
        save_dir: Optional[Path] = None,
        crop: bool = False,
        font_size: Optional[int] = None,
        show_filename: bool = False
    ):
        """
        保存推理结果（使用 MultiModalSaver）

        Args:
            result: MultiModalResults 实例
            save: 是否保存可视化图像
            save_txt: 是否保存txt标签
            save_json: 是否保存json结果
            save_dir: 保存目录
            crop: 是否保存实例裁切图
            font_size: 可视化字体大小
            show_filename: 是否在结果图上显示源文件名
        """
        if save_dir is None:
            save_dir = Path('runs/predict')

        save_dir = Path(save_dir)

        # 使用 MultiModalSaver 保存
        saver = MultiModalSaver(
            save_dir=save_dir,
            save_img=save,
            save_txt=save_txt,
            save_json=save_json,
            save_conf=False,
            crop=crop
        )

        saver.save(result, font_size=font_size, show_filename=show_filename)


class MultiModalSegmentPredictor(MultiModalPredictor):
    """
    多模态分割推理引擎

    继承自 MultiModalPredictor，扩展分割任务的后处理逻辑。

    主要改动：
    - 重写 _stream_inference: 处理分割模型的 (preds, protos) 输出
    - 新增 _create_segment_result: 返回 MultiModalSegmentResults
    """

    def _stream_inference(
        self,
        dataset,
        save: bool = False,
        save_txt: bool = False,
        save_json: bool = False,
        save_dir: Optional[Path] = None,
        crop: bool = False,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        font_size: Optional[int] = None,
        show_filename: bool = False
    ) -> Generator:
        """
        分割任务流式推理实现

        Args:
            dataset: 多模态推理数据集
            save: 是否保存可视化
            save_txt: 是否保存txt标签
            save_json: 是否保存json结果
            save_dir: 保存目录
            conf: 置信度阈值
            iou: NMS IOU阈值
            max_det: 最大检测框数量
            font_size: 可视化字体大小
            show_filename: 是否在结果图上显示源文件名

        Yields:
            MultiModalSegmentResults: 单个样本的分割推理结果
        """
        from .results import MultiModalSegmentResults

        conf_threshold = conf if conf is not None else self.conf
        iou_threshold = iou if iou is not None else self.iou
        max_detections = max_det if max_det is not None else self.max_det

        if self.verbose:
            LOGGER.info(f"推理参数 - conf: {conf_threshold}, iou: {iou_threshold}, max_det: {max_detections}")

        for sample in dataset:
            # Stage 1: Preprocess
            t0 = time.perf_counter() if self.speed else 0
            im_tensor = sample['im'].to(self.device)
            if self.speed:
                t_pre = (time.perf_counter() - t0) * 1000

            # Stage 2: Inference
            t0 = time.perf_counter() if self.speed else 0
            with torch.no_grad():
                preds = self.model(im_tensor)
            if self.speed:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_infer = (time.perf_counter() - t0) * 1000

            # DEBUG: 模型输出信息
            if self.debug:
                LOGGER.info(f"[DEBUG] 模型输出类型: {type(preds)}")
                if isinstance(preds, (list, tuple)):
                    LOGGER.info(f"[DEBUG] 模型输出是list/tuple，长度: {len(preds)}")
                    if len(preds) >= 2:
                        LOGGER.info(f"[DEBUG] preds[0]类型: {type(preds[0])}, preds[1]类型: {type(preds[1])}")

            # 分割模型输出: (detection_preds, (proto_masks,)) 或 (detection_preds, proto_masks)
            # 提取 protos
            if isinstance(preds, (list, tuple)) and len(preds) >= 2:
                det_preds = preds[0]
                proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
            else:
                det_preds = preds
                proto = None

            if self.debug:
                LOGGER.info(f"[DEBUG] det_preds形状: {det_preds.shape if isinstance(det_preds, torch.Tensor) else type(det_preds)}")
                if proto is not None:
                    LOGGER.info(f"[DEBUG] proto形状: {proto.shape}")

            # Stage 3: Postprocess (NMS + masks + scale + result assembly)
            t0 = time.perf_counter() if self.speed else 0

            # NMS 处理
            if self.debug:
                LOGGER.info("[DEBUG] 使用YOLO后处理分支（分割）")

            # 分割模型的 NMS 处理
            pred = ops.non_max_suppression(
                det_preds,
                conf_thres=conf_threshold,
                iou_thres=iou_threshold,
                max_det=max_detections,
                nc=len(getattr(self.model, 'names', {})) or 80,
            )
            pred = pred[0]  # batch_size=1

            if self.debug:
                LOGGER.info(f"[DEBUG] NMS后检测数量: {len(pred)}")

            # 处理 masks
            masks = None
            if len(pred) and proto is not None:
                # 提取 mask coefficients
                masks_in = pred[:, 6:]  # [N, 32]
                bboxes = pred[:, :4]     # [N, 4]

                if self.debug:
                    LOGGER.info(f"[DEBUG] masks_in形状: {masks_in.shape}, bboxes形状: {bboxes.shape}")

                # 生成实例 masks
                masks = ops.process_mask(
                    proto[0],  # [32, H, W]
                    masks_in,
                    bboxes,
                    sample['meta']['imgsz'],
                    upsample=True
                )

                if self.debug:
                    LOGGER.info(f"[DEBUG] 生成masks形状: {masks.shape}")

                # 缩放到原图尺寸
                masks = ops.scale_masks(
                    masks.unsqueeze(0),
                    sample['meta']['ori_shape']
                )[0]

                if self.debug:
                    LOGGER.info(f"[DEBUG] 缩放后masks形状: {masks.shape}")

            # 坐标缩放
            if len(pred):
                gain = sample['meta']['ratio_pad'][0]
                pad = sample['meta']['ratio_pad'][1]

                if self.debug:
                    LOGGER.info(f"[DEBUG] gain={gain}, pad={pad}")

                pred[:, :4] = ops.scale_boxes(
                    img1_shape=sample['meta']['imgsz'],
                    boxes=pred[:, :4],
                    img0_shape=sample['meta']['ori_shape'],
                    ratio_pad=((gain, gain), pad),
                    padding=True
                )

                if self.debug:
                    LOGGER.info(f"[DEBUG] scale_boxes后前3个框: {pred[:3, :4]}")

            # 组装结果
            result = self._create_segment_result(sample, pred, masks)

            if self.speed:
                t_post = (time.perf_counter() - t0) * 1000
                result.speed = {'preprocess': t_pre, 'inference': t_infer, 'postprocess': t_post}

            # 保存
            if save or save_txt or save_json or crop:
                self._save_result(
                    result,
                    save=save,
                    save_txt=save_txt,
                    save_json=save_json,
                    save_dir=save_dir,
                    crop=crop,
                    font_size=font_size,
                    show_filename=show_filename,
                )

            yield result

    def _create_segment_result(
        self,
        sample: Dict,
        pred: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ):
        """
        组装 MultiModalSegmentResults

        Args:
            sample: 数据集样本
            pred: NMS 后的预测结果 [N, 6+nm]
            masks: 实例分割 masks [N, H, W]

        Returns:
            MultiModalSegmentResults 实例
        """
        from .results import MultiModalSegmentResults

        # 提取 boxes (前6列: x1,y1,x2,y2,conf,cls)
        boxes = pred[:, :6].cpu().numpy() if len(pred) else np.zeros((0, 6))

        # 转换 masks
        masks_np = None
        if masks is not None and len(masks):
            masks_np = (masks.cpu().numpy() * 255).astype(np.uint8)

        # 获取类别名称
        names = getattr(self.model, 'names', None)

        # 元数据
        meta = sample['meta'].copy()
        meta['id'] = sample['id']

        result = MultiModalSegmentResults(
            boxes=boxes,
            paths=sample['paths'],
            orig_imgs=sample['orig_imgs'],
            meta=meta,
            names=names,
            masks=masks_np
        )

        return result


class MultiModalOBBPredictor(MultiModalPredictor):
    """
    多模态 OBB 推理引擎

    继承自 MultiModalPredictor，扩展旋转框检测的后处理逻辑。
    """

    def _stream_inference(
        self,
        dataset,
        save: bool = False,
        save_txt: bool = False,
        save_json: bool = False,
        save_dir: Optional[Path] = None,
        crop: bool = False,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        font_size: Optional[int] = None,
        show_filename: bool = False
    ) -> Generator:
        """
        OBB 任务流式推理实现

        Args:
            dataset: 多模态推理数据集
            save: 是否保存可视化
            save_txt: 是否保存txt标签
            save_json: 是否保存json结果
            save_dir: 保存目录
            crop: 是否保存实例裁切图
            conf: 置信度阈值
            iou: NMS IOU阈值
            max_det: 最大检测框数量
            font_size: 可视化字体大小
            show_filename: 是否在结果图上显示源文件名
        """
        from .results import MultiModalOBBResults

        conf_threshold = conf if conf is not None else self.conf
        iou_threshold = iou if iou is not None else self.iou
        max_detections = max_det if max_det is not None else self.max_det

        if self.verbose:
            LOGGER.info(f"OBB推理参数 - conf: {conf_threshold}, iou: {iou_threshold}, max_det: {max_detections}")

        for sample in dataset:
            # Stage 1: Preprocess
            t0 = time.perf_counter() if self.speed else 0
            im_tensor = sample['im'].to(self.device)
            if self.speed:
                t_pre = (time.perf_counter() - t0) * 1000

            # Stage 2: Inference
            t0 = time.perf_counter() if self.speed else 0
            with torch.no_grad():
                preds = self.model(im_tensor)
            if self.speed:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_infer = (time.perf_counter() - t0) * 1000

            # DEBUG: 模型输出信息
            if self.debug:
                LOGGER.info(f"[DEBUG][OBB] 模型输出类型: {type(preds)}")
                if isinstance(preds, torch.Tensor):
                    LOGGER.info(f"[DEBUG][OBB] 模型输出形状: {preds.shape}")
                elif isinstance(preds, (list, tuple)):
                    LOGGER.info(f"[DEBUG][OBB] 模型输出长度: {len(preds)}")

            # Stage 3: Postprocess (NMS + scale + result assembly)
            t0 = time.perf_counter() if self.speed else 0

            # OBB 旋转框 NMS
            pred = ops.non_max_suppression(
                preds,
                conf_thres=conf_threshold,
                iou_thres=iou_threshold,
                max_det=max_detections,
                nc=len(getattr(self.model, 'names', {})) or 80,
                rotated=True
            )
            pred = pred[0]  # batch_size=1

            # DEBUG: NMS 结果
            if self.debug:
                ori_shape = sample['meta']['ori_shape']
                imgsz = sample['meta']['imgsz']
                LOGGER.info(f"[DEBUG][OBB] 尺寸信息: 原图{ori_shape} → 推理{imgsz}")
                LOGGER.info(f"[DEBUG][OBB] NMS后检测: {len(pred)}个实例")

            # 坐标处理
            if len(pred):
                # 正则化旋转框角度
                rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))

                # DEBUG: 记录缩放前坐标
                if self.debug:
                    rboxes_before = rboxes.clone()

                # 缩放坐标到原图
                gain = sample['meta']['ratio_pad'][0]
                pad = sample['meta']['ratio_pad'][1]

                if self.debug:
                    LOGGER.info(f"[DEBUG][OBB] 缩放参数: gain={gain:.4f}, pad={pad}")

                rboxes[:, :4] = ops.scale_boxes(
                    img1_shape=sample['meta']['imgsz'],
                    boxes=rboxes[:, :4],
                    img0_shape=sample['meta']['ori_shape'],
                    ratio_pad=((gain, gain), pad),
                    padding=True,
                    xywh=True
                )

                # 重组: [x, y, w, h, angle, conf, cls]
                obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)

                # DEBUG: 逐实例输出缩放前后坐标
                if self.debug:
                    names = getattr(self.model, 'names', {})
                    for i in range(len(obb)):
                        cls_id = int(obb[i, 6])
                        cls_name = names.get(cls_id, str(cls_id))
                        conf_val = float(obb[i, 5])
                        before = rboxes_before[i, :5].tolist()
                        after = rboxes[i, :5].tolist()
                        before_str = f"[{before[0]:.1f},{before[1]:.1f},{before[2]:.1f},{before[3]:.1f},{before[4]:.3f}]"
                        after_str = f"[{after[0]:.1f},{after[1]:.1f},{after[2]:.1f},{after[3]:.1f},{after[4]:.3f}]"
                        LOGGER.info(f"[DEBUG][OBB] #{i} {cls_name}({conf_val:.2f}): 缩放前{before_str} → 缩放后{after_str}")
            else:
                obb = pred

            # 组装结果
            result = self._create_obb_result(sample, obb)

            if self.speed:
                t_post = (time.perf_counter() - t0) * 1000
                result.speed = {'preprocess': t_pre, 'inference': t_infer, 'postprocess': t_post}

            # 保存
            if save or save_txt or save_json or crop:
                self._save_result(
                    result,
                    save=save,
                    save_txt=save_txt,
                    save_json=save_json,
                    save_dir=save_dir,
                    crop=crop,
                    font_size=font_size,
                    show_filename=show_filename,
                )

            yield result

    def _create_obb_result(self, sample: Dict, obb: torch.Tensor):
        """组装 MultiModalOBBResults"""
        from .results import MultiModalOBBResults

        # 提取 OBB 数据
        obb_data = obb.cpu().numpy() if len(obb) else np.zeros((0, 7))

        # 获取类别名称
        names = getattr(self.model, 'names', None)

        # 元数据
        meta = sample['meta'].copy()
        meta['id'] = sample['id']

        result = MultiModalOBBResults(
            obb=obb_data,
            paths=sample['paths'],
            orig_imgs=sample['orig_imgs'],
            meta=meta,
            names=names
        )

        return result


class MultiModalPosePredictor(MultiModalPredictor):
    """
    多模态姿态估计推理引擎

    继承自 MultiModalPredictor，扩展关键点检测的后处理逻辑。
    """

    def _stream_inference(
        self,
        dataset,
        save: bool = False,
        save_txt: bool = False,
        save_json: bool = False,
        save_dir: Optional[Path] = None,
        crop: bool = False,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        font_size: Optional[int] = None,
        show_filename: bool = False
    ) -> Generator:
        """
        Pose 任务流式推理实现

        Args:
            dataset: 多模态推理数据集
            save: 是否保存可视化
            save_txt: 是否保存txt标签
            save_json: 是否保存json结果
            save_dir: 保存目录
            crop: 是否保存实例裁切图
            conf: 置信度阈值
            iou: NMS IOU阈值
            max_det: 最大检测框数量
            font_size: 可视化字体大小
            show_filename: 是否在结果图上显示源文件名
        """
        from .results import MultiModalPoseResults

        conf_threshold = conf if conf is not None else self.conf
        iou_threshold = iou if iou is not None else self.iou
        max_detections = max_det if max_det is not None else self.max_det

        # 获取关键点形状配置
        kpt_shape = getattr(self.model, 'kpt_shape', (17, 3))  # COCO 默认

        if self.verbose:
            LOGGER.info(f"Pose推理参数 - conf: {conf_threshold}, iou: {iou_threshold}, kpt_shape: {kpt_shape}")

        for sample in dataset:
            # Stage 1: Preprocess
            t0 = time.perf_counter() if self.speed else 0
            im_tensor = sample['im'].to(self.device)
            if self.speed:
                t_pre = (time.perf_counter() - t0) * 1000

            # Stage 2: Inference
            t0 = time.perf_counter() if self.speed else 0
            with torch.no_grad():
                preds = self.model(im_tensor)
            if self.speed:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_infer = (time.perf_counter() - t0) * 1000

            # DEBUG: 模型输出信息
            if self.debug:
                LOGGER.info(f"[DEBUG][Pose] 模型输出类型: {type(preds)}")
                if isinstance(preds, torch.Tensor):
                    LOGGER.info(f"[DEBUG][Pose] 模型输出形状: {preds.shape}")
                elif isinstance(preds, (list, tuple)):
                    LOGGER.info(f"[DEBUG][Pose] 模型输出长度: {len(preds)}")

            # Stage 3: Postprocess (NMS + scale + keypoints + result assembly)
            t0 = time.perf_counter() if self.speed else 0

            # 标准 NMS
            pred = ops.non_max_suppression(
                preds,
                conf_thres=conf_threshold,
                iou_thres=iou_threshold,
                max_det=max_detections,
                nc=len(getattr(self.model, 'names', {})) or 1
            )
            pred = pred[0]  # batch_size=1

            # DEBUG: NMS 结果
            if self.debug:
                LOGGER.info(f"[DEBUG][Pose] NMS后检测数量: {len(pred)}")

            # 处理关键点
            keypoints = None
            if len(pred):
                # 缩放边界框
                gain = sample['meta']['ratio_pad'][0]
                pad = sample['meta']['ratio_pad'][1]

                # DEBUG: 坐标变换参数
                if self.debug:
                    LOGGER.info(f"[DEBUG][Pose] gain={gain}, pad={pad}")

                pred[:, :4] = ops.scale_boxes(
                    img1_shape=sample['meta']['imgsz'],
                    boxes=pred[:, :4],
                    img0_shape=sample['meta']['ori_shape'],
                    ratio_pad=((gain, gain), pad),
                    padding=True
                )

                # DEBUG: 边界框变换结果
                if self.debug:
                    LOGGER.info(f"[DEBUG][Pose] scale_boxes后前3个框: {pred[:3, :4] if len(pred) >= 3 else pred[:, :4]}")

                # 提取并缩放关键点
                pred_kpts = pred[:, 6:].view(len(pred), *kpt_shape)
                keypoints = ops.scale_coords(
                    sample['meta']['imgsz'],
                    pred_kpts,
                    sample['meta']['ori_shape']
                )

                # DEBUG: 关键点信息
                if self.debug:
                    LOGGER.info(f"[DEBUG][Pose] 关键点形状: {keypoints.shape}")

            # 组装结果
            result = self._create_pose_result(sample, pred[:, :6] if len(pred) else pred, keypoints)

            if self.speed:
                t_post = (time.perf_counter() - t0) * 1000
                result.speed = {'preprocess': t_pre, 'inference': t_infer, 'postprocess': t_post}

            # 保存
            if save or save_txt or save_json or crop:
                self._save_result(
                    result,
                    save=save,
                    save_txt=save_txt,
                    save_json=save_json,
                    save_dir=save_dir,
                    crop=crop,
                    font_size=font_size,
                    show_filename=show_filename,
                )

            yield result

    def _create_pose_result(self, sample: Dict, pred: torch.Tensor, keypoints: Optional[torch.Tensor]):
        """组装 MultiModalPoseResults"""
        from .results import MultiModalPoseResults

        # 提取 boxes
        boxes = pred.cpu().numpy() if len(pred) else np.zeros((0, 6))

        # 提取 keypoints
        kpts = keypoints.cpu().numpy() if keypoints is not None and len(keypoints) else None

        # 获取类别名称
        names = getattr(self.model, 'names', None)

        # 元数据
        meta = sample['meta'].copy()
        meta['id'] = sample['id']

        result = MultiModalPoseResults(
            boxes=boxes,
            keypoints=kpts,
            paths=sample['paths'],
            orig_imgs=sample['orig_imgs'],
            meta=meta,
            names=names
        )

        return result


class MultiModalClassifyPredictor(MultiModalPredictor):
    """
    多模态分类推理引擎

    继承自 MultiModalPredictor，实现分类任务的后处理逻辑（softmax 而非 NMS）。
    """

    def _stream_inference(
        self,
        dataset,
        save: bool = False,
        save_txt: bool = False,
        save_json: bool = False,
        save_dir: Optional[Path] = None,
        crop: bool = False,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        font_size: Optional[int] = None,
        show_filename: bool = False
    ) -> Generator:
        """
        分类任务流式推理实现

        Args:
            dataset: 多模态推理数据集
            save: 是否保存可视化
            save_txt: 是否保存txt标签
            save_json: 是否保存json结果
            save_dir: 保存目录
            crop: 是否保存实例裁切图
            conf: 分类任务保留兼容参数
            iou: 分类任务保留兼容参数
            max_det: 分类任务保留兼容参数
            font_size: 可视化字体大小
            show_filename: 是否在结果图上显示源文件名
        """
        from .results import MultiModalClassifyResults

        if self.verbose:
            LOGGER.info("分类推理开始")

        for sample in dataset:
            # Stage 1: Preprocess
            t0 = time.perf_counter() if self.speed else 0
            im_tensor = sample['im'].to(self.device)
            if self.speed:
                t_pre = (time.perf_counter() - t0) * 1000

            # Stage 2: Inference
            t0 = time.perf_counter() if self.speed else 0
            with torch.no_grad():
                preds = self.model(im_tensor)
            if self.speed:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_infer = (time.perf_counter() - t0) * 1000

            # DEBUG: 模型输出信息
            if self.debug:
                LOGGER.info(f"[DEBUG][Classify] 模型输出类型: {type(preds)}")
                if isinstance(preds, torch.Tensor):
                    LOGGER.info(f"[DEBUG][Classify] 模型输出形状: {preds.shape}")

            # Stage 3: Postprocess (softmax + result assembly)
            t0 = time.perf_counter() if self.speed else 0

            # 分类后处理：softmax
            probs = preds.softmax(dim=1)

            # DEBUG: softmax 结果
            if self.debug:
                top5_probs, top5_indices = probs.topk(5, dim=1)
                LOGGER.info(f"[DEBUG][Classify] Top5 类别索引: {top5_indices[0].tolist()}")
                LOGGER.info(f"[DEBUG][Classify] Top5 概率: {top5_probs[0].tolist()}")

            # 组装结果
            result = self._create_classify_result(sample, probs)

            if self.speed:
                t_post = (time.perf_counter() - t0) * 1000
                result.speed = {'preprocess': t_pre, 'inference': t_infer, 'postprocess': t_post}

            # 保存
            if save or save_txt or save_json:
                self._save_result(
                    result,
                    save=save,
                    save_txt=save_txt,
                    save_json=save_json,
                    save_dir=save_dir,
                    font_size=font_size,
                    show_filename=show_filename,
                )

            yield result

    def _create_classify_result(self, sample: Dict, probs: torch.Tensor):
        """组装 MultiModalClassifyResults"""
        from .results import MultiModalClassifyResults

        # 提取概率
        probs_np = probs.cpu().numpy()

        # 获取类别名称
        names = getattr(self.model, 'names', None)

        # 元数据
        meta = sample['meta'].copy()
        meta['id'] = sample['id']

        result = MultiModalClassifyResults(
            probs=probs_np,
            paths=sample['paths'],
            orig_imgs=sample['orig_imgs'],
            meta=meta,
            names=names
        )

        return result


class MultiModalSemSegPredictor(MultiModalPredictor):
    """Semantic segmentation predictor for YOLOMM SemSeg models."""

    def __init__(self, model=None, imgsz=640, device="cpu", verbose=True, debug=False, **kwargs):
        super().__init__(model=model, imgsz=imgsz, device=device, verbose=verbose, debug=debug)
        self.save_conf = kwargs.get("save_conf", False)
        self.ignore_index = kwargs.get("ignore_index", 255)
        self.nc = getattr(model, "yaml", {}).get("nc", 80) if model else 80

    def infer_single(self, model, sample, dataset=None):
        import torch.nn.functional as F
        import numpy as np
        from .results import MultiModalSemSegResults

        img = sample["img"].unsqueeze(0).to(self.device)
        if img.ndim == 3:
            img = img.unsqueeze(0)

        model.eval()
        with torch.no_grad():
            logits = model(img)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits[0]  # remove batch dim

        # Upsample to preprocessed input size
        meta = sample["meta"]
        ratio_pad = meta.get("ratio_pad", (1, 1))
        if tuple(logits.shape[-2:]) != tuple(meta.get("input_shape", logits.shape[-2:])):
            logits = F.interpolate(logits.unsqueeze(0), size=meta.get("input_shape", logits.shape[-2:]),
                                    mode="bilinear", align_corners=False)[0]

        # Build probability map for letterbox undo + resize
        if self.nc == 1:
            probs = logits.sigmoid()  # [1, H, W]
        else:
            probs = logits.softmax(dim=0)  # [nc, H, W]

        # Undo letterbox
        if ratio_pad != (1, 1):
            gain = min(img.shape[-1] / meta.get("orig_shape", (img.shape[-1],))[1],
                       img.shape[-2] / meta.get("orig_shape", (img.shape[-2],))[0])
            pad_h = (img.shape[-2] - int(meta.get("orig_shape", (img.shape[-2],))[0] * gain)) // 2
            pad_w = (img.shape[-1] - int(meta.get("orig_shape", (img.shape[-2],))[1] * gain)) // 2
            if pad_h > 0 or pad_w > 0:
                probs = probs[:, pad_h:pad_h + int(meta.get("orig_shape", (img.shape[-2],))[0] * gain),
                               pad_w:pad_w + int(meta.get("orig_shape", (img.shape[-2],))[1] * gain)]

        # Resize to original shape
        orig_shape = meta.get("orig_shape", probs.shape[-2:])
        if tuple(probs.shape[-2:]) != tuple(orig_shape):
            probs = F.interpolate(probs.unsqueeze(0), size=tuple(orig_shape),
                                   mode="bilinear", align_corners=False)[0]

        # nc=1 时使用 sigmoid + 阈值 0（与 gt(0) 等价），nc>1 时使用 softmax + argmax
        if self.nc == 1:
            sem_mask = probs.squeeze(0).gt(0.5).cpu().numpy().astype(np.int64)
            conf_map = probs.squeeze(0).cpu().numpy() if self.save_conf else None
        else:
            sem_mask = probs.argmax(dim=0).cpu().numpy()
            conf_map = probs.max(dim=0).values.cpu().numpy() if self.save_conf else None

        names = getattr(model, "names", {})
        return MultiModalSemSegResults(
            sem_mask=sem_mask,
            paths=sample["paths"],
            orig_imgs=sample["orig_imgs"],
            meta=meta,
            names=names,
            conf_map=conf_map,
        )

    def _save_result(self, result, save=True, save_txt=False, save_json=False, save_dir=None, crop=False,
                     font_size=None, show_filename=False):
        from .results import MultiModalSemSegResults
        from .saver import MultiModalSemSegSaver
        if isinstance(result, MultiModalSemSegResults):
            saver = MultiModalSemSegSaver(save_dir=save_dir, save_img=save, save_json=save_json,
                                          save_conf=self.save_conf)
            saver.save(result)

    def _stream_inference(
        self,
        dataset,
        save: bool = False,
        save_txt: bool = False,
        save_json: bool = False,
        save_dir: Optional[Path] = None,
        crop: bool = False,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
        font_size: Optional[int] = None,
        show_filename: bool = False
    ) -> Generator:
        """SemSeg专用流式推理 — dense logits后处理，不使用NMS/detection路径."""
        if iou is not None or max_det is not None or conf is not None:
            LOGGER.info(
                "SemSeg: conf/iou/max_det parameters are not applicable for semantic "
                "segmentation and will be ignored."
            )

        for sample in dataset:
            if self.speed:
                t_inf_start = time.perf_counter()
            result = self.infer_single(self.model, sample, dataset)
            if self.speed:
                t_inf = (time.perf_counter() - t_inf_start) * 1000
                result.speed = {"preprocess": 0, "inference": t_inf, "postprocess": 0}

            if save or save_json:
                self._save_result(
                    result,
                    save=save,
                    save_txt=False,
                    save_json=save_json,
                    save_dir=save_dir,
                    crop=False,
                    font_size=font_size,
                    show_filename=show_filename,
                )

            yield result
