# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
RT-DETR MultiModal predictor module.

This module provides the RTDETRMMPredictor class for inference with multi-modal RT-DETR models
supporting RGB+X modality inputs. Architecture strictly follows YOLOMM's successful pattern.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import LOGGER, DEFAULT_CFG, ops, colorstr
from ultralytics.engine.results import Results


class RTDETRMMPredictor(BasePredictor):
    """
    A predictor class for RT-DETR MultiModal (RTDETRMM) object detection models.

    This class extends RTDETRPredictor to support multi-modal inputs (RGB + X modality)
    during inference. Architecture strictly follows YOLOMM's MultiModalDetectionPredictor
    successful pattern for consistency and simplicity.

    Key Features:
        - Supports RGB+X dual-modal inference (best performance)
        - Supports single-modal inference with modality parameter
        - Always outputs 6-channel tensor [X,X,X,RGB,RGB,RGB] format
        - Compatible with YOLOMM API for consistent user experience
        - Simple and clean architecture following proven patterns

    Input Format Requirements:
        - Dual-modal: List format [RGB_source, X_source] - ORDER IS IMPORTANT!
        - Single-modal: Single source with modality parameter
        - 6-channel tensor: Pre-processed [X,X,X,RGB,RGB,RGB] format

    Attributes:
        args: Prediction arguments and settings.
        model: The RTDETRMM model used for inference.
        modality: Specific modality for single-modal inference.
        is_dual_modal: Whether current inference is dual-modal.
        is_single_modal: Whether current inference is single-modal.

    Methods:
        preprocess: Preprocess input data for multi-modal inference.
        _parse_inference_input: Parse and validate inference input.
        _process_dual_modality: Process dual-modal input.
        _process_single_modality: Process single-modal input with filling.

    Examples:
        >>> # Dual-modal inference
        >>> predictor = RTDETRMMPredictor(overrides={'model': 'rtdetr-r18-mm.pt'})
        >>> results = predictor(source=['rgb.jpg', 'thermal.jpg'])  # [RGB, X] order!
        
        >>> # Single-modal inference
        >>> predictor = RTDETRMMPredictor(overrides={'modality': 'rgb'})
        >>> results = predictor(source='rgb.jpg')
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize RTDETRMMPredictor for multi-modal inference.

        Args:
            cfg (str | Path | dict, optional): Configuration file path or dict.
            overrides (dict, optional): Dictionary to override default prediction arguments.
            _callbacks (list, optional): List of callback functions.
        """
        # Initialize parent predictor
        super().__init__(cfg, overrides, _callbacks)
        
        # Initialize multimodal-specific attributes (following YOLOMM pattern)
        self.modality = getattr(self.args, 'modality', None)
        # TODO(modality-bridge): runtime/predict should eventually reuse the shared train/val modality resolver
        # once the deferred predict/runtime adoption scope is approved.
        
        # Initialize modal states (following YOLOMM pattern)
        self.is_dual_modal = self.modality is None
        self.is_single_modal = self.modality is not None
        
        # Track input sources for multi-modal visualization
        self.rgb_source = None
        self.x_source = None
        self.input_mode = None  # 'dual', 'single_rgb', 'single_x'
        
        # Log initialization
        LOGGER.info(f"RTDETRMMPredictor initialized: modality={self.modality}, "
                   f"dual={self.is_dual_modal}, single={self.is_single_modal}")

    # -----------------------------
    # Helper methods for MM routing
    # -----------------------------
    def _get_mm_router(self):
        """Return the active MultiModalRouter, handling AutoBackend(Pytorch)包装情形。"""
        m = getattr(self, "model", None)
        if m is None:
            return None
        # 直连PyTorch模型
        if hasattr(m, "mm_router") and m.mm_router is not None:
            return m.mm_router
        # AutoBackend(PyTorch)
        if hasattr(m, "pt") and getattr(m, "pt", False) and hasattr(m, "model"):
            inner = getattr(m, "model", None)
            if inner is not None and hasattr(inner, "mm_router") and inner.mm_router is not None:
                return inner.mm_router
        return None

    def _set_runtime_modality_for_router(self):
        """在单/双模态推理前，向路由器注入运行时模态参数；无路由器时严格 Fail‑Fast。"""
        mm_router = self._get_mm_router()
        if self._dual_input_detected:
            if mm_router:
                mm_router.set_runtime_params(None)
            return
        # 单模态
        if not mm_router:
            raise RuntimeError(
                "检测到单模态输入但未找到多模态路由器(mm_router)。"
                "请确保使用多模态权重并在 PyTorch/AutoBackend(PyTorch) 后端运行。"
            )
        mm_router.set_runtime_params(
            self.modality,
            strategy=getattr(self.args, "ablation_strategy", None),
            seed=getattr(self.args, "seed", None),
        )

    def _get_dual_channels(self) -> int:
        """读取路由器配置的 Dual 通道数(3+Xch)。若不可用则回退为6。"""
        mm_router = self._get_mm_router()
        try:
            if mm_router and hasattr(mm_router, "INPUT_SOURCES"):
                return int(mm_router.INPUT_SOURCES.get("Dual", 6))
        except Exception:
            pass
        return 6

    def _parse_inference_input(self, source):
        """
        Parse and validate inference input from RTDETRMM.predict() method.
        
        Follows YOLOMM's input parsing pattern exactly for consistency.
        
        Args:
            source: Input source from RTDETRMM.predict() method. Can be:
                - Single file path (str/Path): for single-modal inference
                - List of 2 paths: [rgb_path, x_path] for dual-modal inference
                - PIL.Image or np.ndarray: for single image
                - torch.Tensor: preprocessed tensor
                
        Returns:
            tuple: (parsed_source, input_format_info) where:
                - parsed_source: Validated and normalized input source
                - input_format_info: Dict with input analysis information
                
        Raises:
            ValueError: If input format is invalid or incompatible with modality settings
            TypeError: If input type is not supported
        """
        import numpy as np
        from PIL import Image
        from pathlib import Path
        
        # Initialize input format analysis (following YOLOMM pattern)
        input_info = {
            'input_type': type(source).__name__,
            'is_batch': False,
            'source_count': 1,
            'modality_mode': 'dual' if self.is_dual_modal else f'single_{self.modality}',
            'inference_format': None,
            'validation_passed': False
        }
        
        try:
            # Log initial input analysis
            LOGGER.debug(f"解析推理输入: 类型={input_info['input_type']}, 模态模式={input_info['modality_mode']}")
            
            # Case 1: Tensor input (already preprocessed)
            if isinstance(source, torch.Tensor):
                input_info['inference_format'] = 'preprocessed_tensor'
                input_info['tensor_shape'] = list(source.shape)
                
                if source.dim() == 4 and source.shape[1] == 6:
                    LOGGER.debug("检测到6通道预处理tensor，直接使用")
                    input_info['validation_passed'] = True
                    return source, input_info
                else:
                    LOGGER.warning(f"Tensor维度不符合预期: {source.shape}，将重新处理")
            
            # Case 2: List/Tuple input
            elif isinstance(source, (list, tuple)):
                input_info['source_count'] = len(source)
                
                if len(source) == 2 and self.is_dual_modal:
                    # Dual-modal format: [rgb_source, x_source]
                    input_info['inference_format'] = 'dual_modal'
                    rgb_source, x_source = source
                    
                    # Validate individual sources
                    rgb_info = self._analyze_single_source(rgb_source, 'rgb')
                    x_info = self._analyze_single_source(x_source, 'x_modal')
                    
                    input_info['rgb_source'] = rgb_info
                    input_info['x_source'] = x_info
                    input_info['validation_passed'] = True
                    
                    return source, input_info
                    
                elif len(source) == 1 and self.is_single_modal:
                    # Single-modal with list wrapper
                    input_info['inference_format'] = 'single_modal'
                    single_source = source[0]
                    LOGGER.debug(f"单模态输入(列表包装): {type(single_source)}")
                    
                    # Analyze the single source
                    source_info = self._analyze_single_source(single_source, self.modality)
                    input_info.update(source_info)
                    input_info['validation_passed'] = True
                    
                    return single_source, input_info
                    
                else:
                    # Invalid list format for current modality settings
                    if self.is_dual_modal:
                        raise ValueError(f"双模态推理需要2个输入源，但接收到{len(source)}个")
                    else:
                        # Use first element for single-modal
                        single_source = source[0]
                        LOGGER.warning(f"单模态推理接收到{len(source)}个输入，使用第一个: {single_source}")
                        return self._parse_inference_input(single_source)
            
            # Case 3: Single source input
            else:
                if self.is_dual_modal:
                    raise ValueError(
                        f"双模态推理需要列表格式输入 [rgb_source, x_source]，"
                        f"但接收到单个源: {type(source)}"
                    )
                
                # Single-modal input validation
                input_info['inference_format'] = 'single_modal'
                source_info = self._analyze_single_source(source, self.modality)
                input_info.update(source_info)
                input_info['validation_passed'] = True
                
                return source, input_info
                
        except Exception as e:
            input_info['validation_passed'] = False
            input_info['error'] = str(e)
            LOGGER.error(f"输入解析失败: {e}")
            raise
        
        finally:
            # Log final input analysis
            self._log_input_analysis(input_info)

    def _analyze_single_source(self, source, modality_hint=None):
        """
        Analyze a single input source and determine its characteristics.
        Following YOLOMM pattern exactly.
        
        Args:
            source: Single input source (path, PIL.Image, np.ndarray, etc.)
            modality_hint (str): Hint about expected modality type
            
        Returns:
            dict: Analysis information about the source
        """
        import numpy as np
        from PIL import Image
        from pathlib import Path
        
        analysis = {
            'source_type': 'unknown',
            'path': None,
            'exists': False,
            'format': None,
            'modality_hint': modality_hint
        }
        
        if isinstance(source, (str, Path)):
            # File path
            path = Path(source)
            analysis['source_type'] = 'file_path'
            analysis['path'] = str(path)
            analysis['exists'] = path.exists()
            analysis['format'] = path.suffix.lower() if path.suffix else 'no_extension'
            
            if not analysis['exists']:
                raise FileNotFoundError(f"输入文件不存在: {path}")
                
        elif isinstance(source, Image.Image):
            # PIL Image
            analysis['source_type'] = 'pil_image'
            analysis['format'] = source.format or 'unknown'
            analysis['mode'] = source.mode
            analysis['size'] = source.size
            
        elif isinstance(source, np.ndarray):
            # Numpy array
            analysis['source_type'] = 'numpy_array'
            analysis['shape'] = source.shape
            analysis['dtype'] = str(source.dtype)
            
        elif isinstance(source, torch.Tensor):
            # Tensor
            analysis['source_type'] = 'torch_tensor'
            analysis['shape'] = list(source.shape)
            analysis['dtype'] = str(source.dtype)
            analysis['device'] = str(source.device)
            
        else:
            analysis['source_type'] = f'unsupported_{type(source).__name__}'
            
        return analysis

    def _log_input_analysis(self, input_info):
        """
        Log detailed input analysis information.
        Following YOLOMM pattern.
        
        Args:
            input_info (dict): Input analysis information
        """
        LOGGER.debug("=== RTDETRMM输入解析分析报告 ===")
        LOGGER.debug(f"输入类型: {input_info['input_type']}")
        LOGGER.debug(f"推理格式: {input_info['inference_format']}")
        LOGGER.debug(f"模态模式: {input_info['modality_mode']}")
        LOGGER.debug(f"源数量: {input_info['source_count']}")
        LOGGER.debug(f"验证通过: {input_info['validation_passed']}")
        
        if 'rgb_source' in input_info:
            LOGGER.debug(f"RGB源信息: {input_info['rgb_source']}")
            
        if 'x_source' in input_info:
            LOGGER.debug(f"X模态源信息: {input_info['x_source']}")
            
        if 'error' in input_info:
            LOGGER.debug(f"错误信息: {input_info['error']}")
            
        LOGGER.debug("=== 分析报告结束 ===")

    def _update_modality_state(self):
        """
        Update modality state dynamically from args (following YOLOMM pattern).
        This allows for dynamic modality switching during prediction.
        """
        current_modality = getattr(self.args, 'modality', None)
        if current_modality != self.modality:
            LOGGER.debug(f"RTDETRMMPredictor: 动态更新模态状态 {self.modality} → {current_modality}")
            self.modality = current_modality
            self.is_dual_modal = self.modality is None
            self.is_single_modal = self.modality is not None

    def preprocess(self, im):
        """
        Route via MultiModalRouter for single-call prediction.

        - Dual input [rgb, x]: compose into 3+Xch tensor (no filling here).
        - Single input + modality: set router runtime params and keep 3-ch input.
        """
        # Update modality state dynamically
        self._update_modality_state()

        # Detect input mode
        if isinstance(im, (list, tuple)) and len(im) == 2:
            self._dual_input_detected = True
            self.input_mode = 'dual'
        else:
            self._dual_input_detected = False
            self.input_mode = f'single_{self.modality}' if self.modality else 'single'

        # Inject runtime modality into router（兼容 AutoBackend 包装）
        self._set_runtime_modality_for_router()

        # Dual-modal: compose into Dual tensor
        if isinstance(im, (list, tuple)) and len(im) == 2:
            return self._process_dual_modality(im)

        # Single-modal: keep 3 channels, router will synthesize the other side
        return super().preprocess(im)

    def _process_dual_modality(self, im):
        """
        Process dual-modal input: [rgb_path, x_path] or similar formats.
        
        Follows YOLOMM pattern exactly with RT-DETR specific adaptations.
        
        Args:
            im (List): Dual-modal input data [RGB_source, X_source]
            
        Returns:
            torch.Tensor: 6-channel tensor [X, X, X, RGB, RGB, RGB]
        """
        # If already preprocessed tensor with 6 channels, return directly
        if isinstance(im, torch.Tensor) and im.shape[1] == 6:
            LOGGER.debug("输入已为6通道tensor，直接返回")
            return im
        
        # Parse dual-modal input and load images
        rgb_images, x_images = self._parse_dual_modal_input(im)
        
        # Preprocess each modality separately using parent's method (RT-DETR specific)
        rgb_tensor = super().preprocess(rgb_images)  # Shape: (B, 3, H, W)
        x_tensor = super().preprocess(x_images)      # Shape: (B, 3, H, W)
        
        # Ensure same spatial dimensions
        rgb_tensor, x_tensor = self._align_tensor_dimensions(rgb_tensor, x_tensor)
        
        # Combine modalities: [X, X, X, RGB, RGB, RGB] order (matching YOLOMM training)
        combined_tensor = torch.cat([x_tensor, rgb_tensor], dim=1)  # Shape: (B, 6, H, W)
        
        LOGGER.debug(f"双模态预处理完成: {combined_tensor.shape}")
        return combined_tensor

    def _parse_dual_modal_input(self, im):
        """
        Parse dual-modal input and separate RGB and X-modal data.
        Following YOLOMM pattern with RT-DETR adaptations.
        
        Args:
            im (List): Input data - [rgb_source, x_source]
                
        Returns:
            tuple: (rgb_images, x_images) ready for preprocessing
        """
        if isinstance(im, (list, tuple)):
            if len(im) == 2:
                # Standard dual-modal format: [rgb_source, x_source]
                rgb_source, x_source = im
                LOGGER.debug(f"解析标准双模态输入: RGB={type(rgb_source)}, X={type(x_source)}")
                
                # Load images using simple loading
                rgb_images = self._load_image_source(rgb_source)
                x_images = self._load_image_source(x_source)
                
                return rgb_images, x_images
                
            else:
                # Invalid dual-modal input format
                raise ValueError(
                    f"双模态推理需要包含2个元素的列表输入 [rgb_source, x_source]，"
                    f"但接收到: {type(im)} with {len(im)} 元素"
                )
        else:
            # Single source input - invalid for dual-modal
            raise ValueError(
                f"双模态推理需要列表格式输入 [rgb_source, x_source]，"
                f"但接收到单个源: {type(im)}"
            )

    def _load_image_source(self, source):
        """
        Load image(s) from various source types.
        Simplified version of YOLOMM's loading mechanism.
        
        Args:
            source (str | Path | PIL.Image | np.ndarray | torch.Tensor): Image source
            
        Returns:
            List[np.ndarray]: Loaded images ready for preprocessing
        """
        import cv2
        import numpy as np
        from PIL import Image
        from pathlib import Path
        
        LOGGER.debug(f"加载图像源: 类型={type(source)}")
        
        # Case 1: String or Path (file path)
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            
            # Check if file exists
            if not source_path.exists():
                raise FileNotFoundError(f"图像文件不存在: {source_path}")
            
            # Direct OpenCV loading
            img = cv2.imread(str(source_path))
            if img is None:
                raise ValueError(f"无法加载图像: {source_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            LOGGER.debug(f"成功加载图像: {source_path}")
            return [img]
            
        # Case 2: PIL Image
        elif isinstance(source, Image.Image):
            LOGGER.debug("处理PIL图像输入")
            if source.mode != "RGB":
                source = source.convert("RGB")
            img = np.asarray(source)
            return [img]
            
        # Case 3: Numpy array
        elif isinstance(source, np.ndarray):
            LOGGER.debug(f"处理numpy数组输入: shape={source.shape}")
            
            if source.ndim == 3:
                # Single image: (H, W, C)
                return [source]
            elif source.ndim == 4:
                # Batch of images: (B, H, W, C) or (B, C, H, W)
                if source.shape[1] == 3 or source.shape[1] == 1:
                    # Format: (B, C, H, W) -> convert to (B, H, W, C)
                    images = [img.transpose(1, 2, 0) for img in source]
                else:
                    # Format: (B, H, W, C)
                    images = list(source)
                return images
            else:
                raise ValueError(f"不支持的numpy数组维度: {source.shape}")
                
        # Case 4: Torch Tensor
        elif isinstance(source, torch.Tensor):
            LOGGER.debug(f"处理torch.Tensor输入: shape={source.shape}")
            
            # Convert to numpy
            if source.device != torch.device('cpu'):
                source = source.cpu()
            source_np = source.numpy()
            
            # Recursive call with numpy array
            return self._load_image_source(source_np)
            
        # Case 5: List or Tuple (multiple images)
        elif isinstance(source, (list, tuple)):
            LOGGER.debug(f"处理列表输入: 长度={len(source)}")
            
            all_images = []
            for i, item in enumerate(source):
                try:
                    loaded = self._load_image_source(item)
                    all_images.extend(loaded)
                except Exception as e:
                    LOGGER.error(f"加载列表项[{i}]失败: {e}")
                    raise
            
            LOGGER.debug(f"列表加载完成: 总计{len(all_images)}张图像")
            return all_images
            
        else:
            raise TypeError(f"不支持的图像源类型: {type(source)}")

    def _align_tensor_dimensions(self, tensor1, tensor2):
        """
        Ensure two tensors have the same spatial dimensions.
        Following YOLOMM pattern exactly.
        
        Args:
            tensor1 (torch.Tensor): First tensor (B, C, H, W)
            tensor2 (torch.Tensor): Second tensor (B, C, H, W)
            
        Returns:
            tuple: (aligned_tensor1, aligned_tensor2) with same spatial dimensions
        """
        import torch.nn.functional as F
        
        if tensor1.shape[2:] == tensor2.shape[2:]:
            # Already same dimensions
            return tensor1, tensor2
        
        # Get dimensions
        h1, w1 = tensor1.shape[2:]
        h2, w2 = tensor2.shape[2:]
        
        # Use the smaller dimensions as target
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        target_size = (target_h, target_w)
        
        LOGGER.debug(f"对齐tensor维度到: {target_size}")
        
        # Resize if necessary
        if (h1, w1) != target_size:
            tensor1 = F.interpolate(tensor1, size=target_size, mode='bilinear', align_corners=False)
        
        if (h2, w2) != target_size:
            tensor2 = F.interpolate(tensor2, size=target_size, mode='bilinear', align_corners=False)
        
        return tensor1, tensor2

    def _process_single_modality(self, im):
        """
        [Disabled] 单模态输入的缺失模态填充不在此处执行。

        依据项目策略：不在可视化/预处理阶段进行任何自动填充，
        单模态场景请仅传入3通道并通过 MultiModalRouter 在前向中处理。
        """
        raise RuntimeError(
            "Single-modal preprocessing-side filling is disabled. "
            "Pass 3-channel input and rely on MultiModalRouter via modality runtime params."
        )

    def _generate_modality_filling(self, source_tensor: torch.Tensor, 
                                   source_modality: str, target_modality: str) -> torch.Tensor:
        """
        Generate filled modality data for single-modal inference.

        Unified dependency: use ultralytics.nn.mm.filling; no internal auto-degradation mechanisms are used.
        Any import error will surface directly.
        """
        from ultralytics.nn.mm.filling import generate_modality_filling
        return generate_modality_filling(source_tensor, source_modality, target_modality)

    def _finalize_tensor(self, tensor):
        """
        Finalize processed tensor with format validation and device management.
        Following YOLOMM pattern exactly.
        
        Args:
            tensor (torch.Tensor): Processed tensor to finalize
            
        Returns:
            torch.Tensor: Finalized tensor on correct device
            
        Raises:
            ValueError: If tensor format is invalid
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"期望torch.Tensor输出，但得到: {type(tensor)}")
        
        if tensor.dim() != 4:
            raise ValueError(f"期望4维tensor [B, C, H, W]，但得到维度: {tensor.dim()}")
        
        if tensor.shape[1] != 6:
            raise ValueError(f"期望6通道tensor，但得到: {tensor.shape[1]}通道")
        
        # 设备管理 - 确保tensor在正确的设备上
        if hasattr(self, 'device') and self.device != tensor.device:
            LOGGER.debug(f"转移tensor到设备: {self.device}")
            tensor = tensor.to(self.device)
        
        # 数据类型确保
        if tensor.dtype != torch.float32:
            LOGGER.debug(f"转换tensor数据类型: {tensor.dtype} -> torch.float32")
            tensor = tensor.float()
        
        return tensor

    def _log_debug_info(self, im, exception):
        """
        Log detailed debug information when preprocessing fails.
        Following YOLOMM pattern.
        
        Args:
            im: Original input data
            exception: The exception that occurred
        """
        LOGGER.debug("=== 多模态预处理调试信息 ===")
        LOGGER.debug(f"模态设置: modality={self.modality}, is_dual_modal={self.is_dual_modal}")
        LOGGER.debug(f"输入类型: {type(im)}")
        
        if isinstance(im, (list, tuple)):
            LOGGER.debug(f"列表输入长度: {len(im)}")
            for i, item in enumerate(im):
                LOGGER.debug(f"  项目[{i}]: {type(item)} - {item}")
        elif isinstance(im, torch.Tensor):
            LOGGER.debug(f"Tensor形状: {im.shape}")
            LOGGER.debug(f"Tensor设备: {im.device}")
            LOGGER.debug(f"Tensor数据类型: {im.dtype}")
        else:
            LOGGER.debug(f"输入内容: {im}")
        
        LOGGER.debug(f"异常类型: {type(exception).__name__}")
        LOGGER.debug(f"异常信息: {str(exception)}")
        LOGGER.debug("=== 调试信息结束 ===")

    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """
        Streams real-time inference on camera feed and saves results to file.
        
        Overrides parent method to handle 6-channel warmup for multimodal RT-DETR models.
        Always uses 6-channel warmup for RTDETRMM (matching YOLOMM pattern).
        
        Args:
            source (str, optional): The source of the image to make predictions on.
            model (nn.Module, optional): The model to use for predictions.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.
            
        Yields:
            (List[ultralytics.engine.results.Results]): The prediction results.
        """
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # RTDETRMM warmup with dynamic Dual channels
            if not self.done_warmup:
                model_channels = self._get_dual_channels()
                LOGGER.info(f"RTDETRMMPredictor: 使用 {model_channels} 通道进行模型预热")
                self.model.warmup(
                    imgsz=(
                        1 if getattr(self.model, "pt", False) or getattr(self.model, "triton", False) else self.dataset.bs,
                        model_channels,
                        *self.imgsz,
                    )
                )
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                # 记录当批原始图像（严格用于可视化背景）
                try:
                    self._mm_current_im0s = im0s
                except Exception:
                    self._mm_current_im0s = None

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)  # 原始输入图像数量
                results_count = len(self.results)  # 实际生成的结果数量
                
                # 对于多模态推理，原始图像可能是2张，但结果只有1个
                # 需要根据实际results数量来处理
                if results_count != n:
                    LOGGER.debug(f"多模态推理: 输入{n}张图像，生成{results_count}个结果")
                
                for i in range(results_count):  # 使用实际结果数量
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / results_count,  # 使用结果数量计算平均时间
                        "inference": profilers[1].dt * 1e3 / results_count,
                        "postprocess": profilers[2].dt * 1e3 / results_count,
                    }
                    
                    # 为多模态推理调整路径处理
                    if results_count < n:
                        # 多模态情况：多个输入产生1个结果，使用第一个路径作为主路径
                        result_path = Path(paths[0])
                        result_string = s[0] if s else ""
                        
                        # 在结果字符串中添加多模态信息
                        if len(paths) > 1:
                            modality_info = f"({len(paths)}模态输入)"
                            result_string = f"{result_string} {modality_info}" if result_string else modality_info
                    else:
                        # 标准情况：1对1映射
                        result_path = Path(paths[i])
                        result_string = s[i] if i < len(s) else ""
                    
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        # 传入完整的结果字符串列表 s，而非单个字符串，避免父类写盘分支在视频/流等场景下异常短路
                        result_string += self.write_results(i, result_path, im, s)
                    
                    # 保存更新后的字符串
                    if i < len(s):
                        s[i] = result_string
                    elif len(s) == 0:
                        s = [result_string]

                # Print batch results
                if self.args.verbose:
                    # 只打印有效的结果字符串
                    valid_strings = [s_item for s_item in s[:results_count] if s_item]
                    if valid_strings:
                        LOGGER.info("\n".join(valid_strings))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            display_ch = self._get_dual_channels()
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), display_ch, *im.shape[2:])}"
                % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")
    
    def postprocess(self, preds, img, orig_imgs):
        """
        Override parent's postprocess to cache original images for multi-modal visualization.
        
        Following YOLOMM pattern for multi-modal result handling.
        For dual-modal input, creates only one Results object using the RGB path.
        
        Args:
            preds: Model predictions
            img: Preprocessed images
            orig_imgs: Original images
            
        Returns:
            Results: Detection results
        """
        # Store original images for later use in visualization
        self._orig_imgs_cache = orig_imgs
        
        # RT-DETR 预测输出 → Results（独立实现，不依赖 ultralytics.models.rtdetr.*）
        results = self._postprocess_rtdetr(preds, img, orig_imgs)
        
        # For dual-modal input, only keep the first result (RGB)
        # This prevents duplicate outputs since we have 2 inputs but 1 detection result
        if self.is_dual_modal and isinstance(orig_imgs, list) and len(orig_imgs) == 2 and len(results) == 2:
            # Keep only the first result (RGB), as both share the same detections
            results = results[:1]
            LOGGER.debug("RTDETRMMPredictor: 双模态输入合并为单个结果对象")
        
        return results

    def _postprocess_rtdetr(self, preds, img, orig_imgs):
        """
        RT-DETR 后处理：将模型输出转为 Results（拷贝自 RTDETRPredictor 的核心逻辑）。

        Args:
            preds: 模型输出
            img (torch.Tensor): 预处理后的输入
            orig_imgs (list | torch.Tensor): 原始图像

        Returns:
            list[Results]
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)  # (300, 1)
            idx = max_score.squeeze(-1) > self.args.conf  # (300,)
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pre_transform(self, im):
        """
        RT-DETR 预变换：强制 scale_fill 的正方形 letterbox（与 RTDETRPredictor 对齐）。
        """
        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=x) for x in im]
    
    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list) -> str:
        """
        统一的多模态绘图路径（与 YOLOMM 对齐）。

        - Dual 模式：输出 RGB、X（3ch 可视化）与并排对比图
        - 单模态消融：仅输出指定模态；另一侧仅在可视化阶段合成，不影响前向路由

        实现要点：
        - 临时关闭父类图片保存（保留 labels/txt/crops 等副作用），然后仅用复用绘图工具产出图片

        Args:
            i (int): 当前 batch 内结果索引
            p (Path): 原始 RGB 图像路径（用于命名）
            im (torch.Tensor): 预处理后的张量（可能为 3ch 或 6ch）
            s (list): 结果字符串列表（用于详细日志）

        Returns:
            str: 父类返回的结果字符串
        """
        # 同步最新模态状态，避免状态不同步导致的早退判定
        self._update_modality_state()

        # 兜底确保保存目录存在（父类图片保存被关闭时，自定义绘图也能正常写盘）
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # 1) 先调用父类，但关闭图片保存，避免 legacy 可视化输出
        orig_save = getattr(self.args, 'save', False)
        try:
            self.args.save = False
            string = super().write_results(i, p, im, s)
        finally:
            self.args.save = orig_save

        # 如果原本不要求保存图片，直接返回（仅保留父类副作用）
        if not orig_save:
            return string

        # 2) 统一的绘图输出（plot_images + multimodal vis 工具链）
        from ultralytics.utils.plotting import plot_images
        from ultralytics.models.utils.multimodal.vis import (
            concat_side_by_side,
            duplicate_bboxes_for_side_by_side,
            ensure_batch_idx_long,
        )
        from ultralytics.models.utils.multimodal.vis import clip_boxes_norm_xywh as _clip_norm_xywh

        # Helper: 严格且与数据无关的 X 模态命名（与 YOLOMM 保持一致）
        def _resolve_x_modality_strict():
            # 单模态且为 RGB 时保留 'rgb'，其他一律统一为 'x'
            if self.is_single_modal and self.modality and str(self.modality).lower() == 'rgb':
                return 'rgb'
            return 'x'

        # 将原始numpy图像转换为张量(B,3,H,W)，保持原尺寸作为绘图背景
        def _np_to_tensor3ch(img_np: np.ndarray) -> torch.Tensor:
            if img_np is None:
                raise RuntimeError("缺少原始图像用于可视化背景")
            if img_np.ndim == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.ndim == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            else:
                raise RuntimeError(f"不支持的原始图像形状: {img_np.shape}")
            t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            return t.unsqueeze(0)

        # 从当批数据优先提取原始RGB/X（严格按输入顺序）；缺失时退回缓存
        def _get_orig_modal_tensors():
            if hasattr(self, '_mm_current_im0s') and self._mm_current_im0s is not None:
                im0s = self._mm_current_im0s
                if isinstance(im0s, (list, tuple)):
                    if len(im0s) == 2:
                        return _np_to_tensor3ch(im0s[0]), _np_to_tensor3ch(im0s[1])
                    elif len(im0s) == 1:
                        if self.modality and str(self.modality).lower() == 'rgb':
                            return _np_to_tensor3ch(im0s[0]), None
                        else:
                            return None, _np_to_tensor3ch(im0s[0])
                # 单张 ndarray
                if self.modality and str(self.modality).lower() == 'rgb':
                    return _np_to_tensor3ch(im0s), None
                else:
                    return None, _np_to_tensor3ch(im0s)
            # 回退到 postprocess 缓存
            if not hasattr(self, '_orig_imgs_cache') or self._orig_imgs_cache is None:
                raise RuntimeError("未找到原始图像用于可视化背景")
            oi = self._orig_imgs_cache
            if isinstance(oi, (list, tuple)):
                if len(oi) == 2:
                    return _np_to_tensor3ch(oi[0]), _np_to_tensor3ch(oi[1])
                elif len(oi) == 1:
                    if self.modality and str(self.modality).lower() == 'rgb':
                        return _np_to_tensor3ch(oi[0]), None
                    else:
                        return None, _np_to_tensor3ch(oi[0])
            if self.modality and str(self.modality).lower() == 'rgb':
                return _np_to_tensor3ch(oi), None
            else:
                return None, _np_to_tensor3ch(oi)

        # 抽取当前结果对象与基名
        result = self.results[i]
        base = p.stem

        # 使用原始RGB/X图像作为背景（不再从预处理张量拆分）
        rgb_tensor, x_tensor = _get_orig_modal_tensors()

        # 从 Results 构造 plot_images 所需参数（比照 YOLOMM：使用归一化框）
        n_boxes = 0 if result.boxes is None else len(result.boxes)
        if n_boxes:
            cls_ids = result.boxes.cls
            confs = getattr(result.boxes, 'conf', None)
            boxes_norm = result.boxes.xywhn  # 归一化坐标（相对 RGB orig_shape）
            if not isinstance(cls_ids, torch.Tensor):
                cls_ids = torch.as_tensor(cls_ids, dtype=torch.long)
            else:
                cls_ids = cls_ids.long()
            if confs is not None and not isinstance(confs, torch.Tensor):
                confs = torch.as_tensor(confs, dtype=torch.float32)
            batch_idx = ensure_batch_idx_long(torch.zeros(cls_ids.shape[0]))
        else:
            cls_ids = torch.zeros((0,), dtype=torch.long)
            confs = torch.zeros((0,), dtype=torch.float32)
            boxes_norm = torch.zeros((0, 4), dtype=torch.float32)
            batch_idx = ensure_batch_idx_long(torch.zeros((0,), dtype=torch.long))

        names = getattr(self.model, 'names', {})
        x_modality = _resolve_x_modality_strict()  # 'rgb' 或统一 'x'

        # 直接使用归一化框，plot_images 按目标背景尺寸还原像素
        def _norm_for_target(boxes_xywh_norm: torch.Tensor) -> torch.Tensor:
            if boxes_xywh_norm is None or boxes_xywh_norm.numel() == 0:
                return torch.zeros((0, 4), dtype=torch.float32)
            return _clip_norm_xywh(boxes_xywh_norm, 0.0, 1.0, 0.0, 1.0)

        # 单模态：仅保存指定模态（背景使用原始图像）
        if self.is_single_modal:
            if str(self.modality).lower() == 'rgb':
                if rgb_tensor is None:
                    raise RuntimeError("期望RGB原图用于可视化，但缓存缺失")
                boxes_norm_rgb = _norm_for_target(boxes_norm)
                fname_rgb = self.save_dir / f"pred_{base}_labels_rgb.jpg"
                plot_images(rgb_tensor, batch_idx, cls_ids, boxes_norm_rgb, confs=confs,
                            paths=[str(p)], fname=fname_rgb, names=names)
            else:
                if x_tensor is None:
                    raise RuntimeError("期望X原图用于可视化，但缓存缺失")
                boxes_norm_x = _norm_for_target(boxes_norm)
                fname_x = self.save_dir / f"pred_{base}_labels_{x_modality}.jpg"
                plot_images(x_tensor, batch_idx, cls_ids, boxes_norm_x, confs=confs,
                            paths=[str(p.with_name(f"{base}_{x_modality}{p.suffix}"))],
                            fname=fname_x, names=names)
            return string

        # 双模态：保存 RGB、X 与并排
        if rgb_tensor is None or x_tensor is None:
            # 理论上不会进入：仅当上方判断为 dual 模式且却缺失单侧时
            LOGGER.warning("RTDETRMM write_results: missing one modality in dual-modal visualization; skip custom visualization.")
            return string
        # 1) RGB：按 RGB 原图尺寸重新归一化
        Hr, Wr = int(rgb_tensor.shape[-2]), int(rgb_tensor.shape[-1])
        boxes_norm_rgb = _norm_for_target(boxes_norm)
        fname_rgb = self.save_dir / f"pred_{base}_labels_rgb.jpg"
        plot_images(rgb_tensor, batch_idx, cls_ids, boxes_norm_rgb, confs=confs,
                    paths=[str(p)], fname=fname_rgb, names=names)

        # 2) X：直接使用原始X图作为背景（同一套归一化框）
        Hx, Wx = int(x_tensor.shape[-2]), int(x_tensor.shape[-1])
        boxes_norm_x = _norm_for_target(boxes_norm)
        fname_x = self.save_dir / f"pred_{base}_labels_{x_modality}.jpg"
        plot_images(x_tensor, batch_idx, cls_ids, boxes_norm_x, confs=confs,
                    paths=[str(p.with_name(f"{base}_{x_modality}{p.suffix}"))],
                    fname=fname_x, names=names)

        # 3) 并排（复制 bbox 到左右半幅）。若两侧尺寸不一致，将X重采样到RGB尺寸，仅用于并排图
        if (Hr, Wr) != (Hx, Wx):
            x_tensor_resized = torch.nn.functional.interpolate(x_tensor, size=(Hr, Wr), mode='bilinear', align_corners=False)
        else:
            x_tensor_resized = x_tensor
        side = concat_side_by_side(rgb_tensor, x_tensor_resized)
        batch_dup, cls_dup, boxes_dup, confs_dup = duplicate_bboxes_for_side_by_side(
            batch_idx, cls_ids, boxes_norm_rgb, confs
        )
        fname_mm = self.save_dir / f"pred_{base}_labels_multimodal.jpg"
        plot_images(side, batch_dup, cls_dup, boxes_dup, confs=confs_dup,
                    paths=[str(p.with_name(f"{base}_multimodal{p.suffix}"))],
                    fname=fname_mm, names=names)

        return string
    
    # -------------------------------------------------------------------------
    # DEPRECATED: 以下旧的多模态保存与单模态改名逻辑已被统一绘图路径取代，保留仅作历史参考。
    # 不再被调用，请不要在新代码中使用。
    # -------------------------------------------------------------------------
    # def _save_multimodal_results(self, i: int, p: Path, im: torch.Tensor):
    #     """
    #     [DEPRECATED] 旧的多模态保存路径（numpy+result.plot），已由 plot_images+vis 工具链取代。
    #     """
    #     pass
    # 
    # def _update_single_modal_filename(self, p: Path):
    #     """
    #     [DEPRECATED] 旧的“保存后改名”逻辑，已不再使用。
    #     """
    #     pass
    
    def _separate_modalities(self, tensor: torch.Tensor) -> tuple:
        """
        Separate 6-channel tensor into RGB and X modality tensors.
        
        Args:
            tensor (torch.Tensor): 6-channel tensor [X,X,X,RGB,RGB,RGB]
            
        Returns:
            tuple: (rgb_tensor, x_tensor) each with 3 channels
        """
        if tensor.dim() == 3:
            # Single image: (6, H, W)
            x_tensor = tensor[:3]    # First 3 channels (X modality)
            rgb_tensor = tensor[3:]  # Last 3 channels (RGB)
        else:
            # Batch: (B, 6, H, W)
            x_tensor = tensor[:, :3]    # X modality channels
            rgb_tensor = tensor[:, 3:]  # RGB channels
            
        return rgb_tensor, x_tensor
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert preprocessed tensor back to displayable image.
        
        Handles denormalization and format conversion.
        
        Args:
            tensor (torch.Tensor): Normalized tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            np.ndarray: Image in HWC format with uint8 values
        """
        # Remove batch dimension if present
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.dim() == 4:
            # For batch, take first image
            tensor = tensor[0]
            
        # Move to CPU if on GPU
        tensor = tensor.cpu()
        
        # Convert to HWC format
        img = tensor.permute(1, 2, 0).numpy()
        
        # Denormalize (assume standard ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        
        # Clip to [0, 1] and convert to uint8
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        # Handle grayscale
        if img.shape[2] == 1:
            img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
            
        return img
    
    def _plot_on_image(self, result, img: np.ndarray, modality_name: str) -> np.ndarray:
        """
        Plot detection results on a specific modality image.
        
        Args:
            result: Detection result object
            img (np.ndarray): Image to plot on (HWC format)
            modality_name (str): Name of the modality for labeling
            
        Returns:
            np.ndarray: Annotated image
        """
        # Create a copy to avoid modifying original
        img_copy = img.copy()
        
        # Use the result's plot method with the specific image
        annotated = result.plot(
            img=img_copy,
            line_width=self.args.line_width,
            boxes=self.args.show_boxes,
            conf=self.args.show_conf,
            labels=self.args.show_labels
        )
        
        # Add modality label
        h, w = annotated.shape[:2]
        label_bg_color = (0, 0, 0)  # Black background
        label_text_color = (255, 255, 255)  # White text
        
        # Draw modality label in top-left corner
        cv2.rectangle(annotated, (10, 10), (150, 40), label_bg_color, -1)
        cv2.putText(annotated, f"{modality_name} Modality", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_text_color, 2)
        
        return annotated
    
    def _create_multimodal_comparison(self, rgb_img: np.ndarray, x_img: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison of RGB and X modality results.
        
        Args:
            rgb_img (np.ndarray): Annotated RGB image
            x_img (np.ndarray): Annotated X modality image
            
        Returns:
            np.ndarray: Combined side-by-side image
        """
        # Ensure both images have the same height
        h1, w1 = rgb_img.shape[:2]
        h2, w2 = x_img.shape[:2]
        
        if h1 != h2:
            # Resize to match heights
            target_h = max(h1, h2)
            if h1 < target_h:
                scale = target_h / h1
                new_w1 = int(w1 * scale)
                rgb_img = cv2.resize(rgb_img, (new_w1, target_h))
            else:
                scale = target_h / h2
                new_w2 = int(w2 * scale)
                x_img = cv2.resize(x_img, (new_w2, target_h))
        
        # Create side-by-side image
        gap = 10  # Gap between images
        combined_width = rgb_img.shape[1] + x_img.shape[1] + gap
        combined_height = max(rgb_img.shape[0], x_img.shape[0])
        
        # Create black canvas
        combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Place images
        combined[:rgb_img.shape[0], :rgb_img.shape[1]] = rgb_img
        combined[:x_img.shape[0], rgb_img.shape[1] + gap:] = x_img
        
        # Add title
        title_height = 50
        final_img = np.zeros((combined_height + title_height, combined_width, 3), dtype=np.uint8)
        final_img[title_height:] = combined
        
        # Add title text
        cv2.putText(final_img, "RT-DETR Multi-Modal Detection Results", 
                    (combined_width // 2 - 200, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return final_img
