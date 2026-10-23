# Ultralytics Multimodal Router - Universal RGB+X Data Routing
# Supports YOLO and RTDETR with zero-copy tensor routing
# Version: v1.0

import torch
import torch.nn as nn
from ultralytics.utils import LOGGER
from .filling import generate_modality_filling, adapt_xch
from .utils import extract_x_modality_name_from_data, INPUT_SOURCE_ALIASES, is_rgb_modality, resolve_input_source_role, resolve_modality_pair


class MultiModalRouter:
    """
    Universal RGB+X Multimodal Data Router
    
    Supports:
    - RGB: 3-channel visible light images
    - X: 3-channel unified other modality (depth/thermal/lidar/etc.)
    - Dual: 6-channel RGB+X concatenated input
    
    Features:
    - Zero-copy tensor view routing
    - Configuration-driven data flow
    - Support for both YOLO and RTDETR architectures
    """
    
    # Backward-compat defaults (for unpickled legacy objects)
    runtime_modality = None
    runtime_strategy = None
    runtime_seed = None

    def __init__(self, config_dict=None, verbose=True):
        """Initialize multimodal router with configuration."""
        self.x_modality_type = 'unknown'
        self.primary_modality_type = 'rgb'
        self.primary_ch = 3
        self.secondary_ch = 3

        if config_dict and 'dataset_config' in config_dict:
            dataset_config = config_dict['dataset_config']
            self.secondary_ch = dataset_config.get('Xch', 3)
            self.primary_ch = dataset_config.get('Pch', 3)

            x_modality = extract_x_modality_name_from_data(dataset_config if isinstance(dataset_config, dict) else None)
            if x_modality and x_modality != 'unknown':
                self.x_modality_type = x_modality

            primary_modal, _secondary_modal = resolve_modality_pair(
                dataset_config if isinstance(dataset_config, dict) else None, default_secondary='unknown'
            )
            if primary_modal:
                self.primary_modality_type = primary_modal

        # YOLOVO: 从 YAML text_encoder.dim 提取文本模态维度（默认 512）
        text_encoder_cfg = (config_dict or {}).get("text_encoder", {})
        text_dim = int(text_encoder_cfg.get("dim", 512))

        self.INPUT_SOURCES = {
            'PRIMARY': self.primary_ch,
            'SECONDARY': self.secondary_ch,
            'FUSED': self.primary_ch + self.secondary_ch,
            'TEXT': text_dim,  # YOLOVO 文本模态
        }

        # 记录构建时通道契约（不随 update_dataset_config 改变）
        self._build_contract = {
            'primary_ch': self.primary_ch,
            'secondary_ch': self.secondary_ch,
            'fused_ch': self.primary_ch + self.secondary_ch,
        }

        self.has_multimodal_config = self._detect_multimodal_config(config_dict)
        self.verbose = verbose
        self.original_spatial_size = None
        self.original_inputs = {}
        self.runtime_modality = None
        self.runtime_strategy = None
        self.runtime_seed = None

        if self.verbose:
            LOGGER.info("MultiModal: router initialized")
            LOGGER.info(
                f"MultiModal: {self.primary_modality_type}={self.primary_ch}ch, "
                f"{self.x_modality_type}={self.secondary_ch}ch, "
                f"Fused={self.primary_ch + self.secondary_ch}ch"
            )

    def set_runtime_params(self, modality: str | None, strategy: str | None = None, seed: int | None = None):
        """Set runtime ablation/filling params used during setup_multimodal_routing()."""
        self.runtime_modality = (modality.lower() if isinstance(modality, str) else None)
        self.runtime_strategy = strategy
        self.runtime_seed = seed

    # --- Backward compatibility helpers for unpickled legacy routers ---
    def __setstate__(self, state):
        """Restore state with legacy checkpoint migration."""
        self.__dict__.update(state)
        # Runtime defaults
        for attr in ('runtime_modality', 'runtime_strategy', 'runtime_seed'):
            if attr not in self.__dict__:
                setattr(self, attr, None)
        self.original_spatial_size = None

        # --- Legacy checkpoint migration (in-memory only) ---
        if hasattr(self, 'INPUT_SOURCES') and isinstance(self.INPUT_SOURCES, dict):
            old_keys = {'RGB', 'X', 'Dual'}
            if old_keys & set(self.INPUT_SOURCES.keys()):
                # Migrate: RGB->PRIMARY, X->SECONDARY, Dual->FUSED
                self.primary_ch = self.INPUT_SOURCES.pop('RGB', 3)
                self.secondary_ch = self.INPUT_SOURCES.pop('X', 3)
                self.INPUT_SOURCES.pop('Dual', None)
                self.INPUT_SOURCES['PRIMARY'] = self.primary_ch
                self.INPUT_SOURCES['SECONDARY'] = self.secondary_ch
                self.INPUT_SOURCES['FUSED'] = self.primary_ch + self.secondary_ch

        # Migrate original_inputs keys
        old_inputs = getattr(self, 'original_inputs', {})
        new_inputs = {}
        for old_key in ('RGB', 'X', 'Dual'):
            if old_key in old_inputs:
                new_key = INPUT_SOURCE_ALIASES.get(old_key, old_key)
                new_inputs[new_key] = old_inputs[old_key]
        if new_inputs:
            self.original_inputs = new_inputs
        else:
            self.original_inputs = {}

        # Ensure primary_modality_type exists
        if not hasattr(self, 'primary_modality_type'):
            self.primary_modality_type = 'rgb'
        if not hasattr(self, 'primary_ch'):
            self.primary_ch = self.INPUT_SOURCES.get('PRIMARY', 3)
        if not hasattr(self, 'secondary_ch'):
            self.secondary_ch = self.INPUT_SOURCES.get('SECONDARY', 3)

        # 补齐 _build_contract（旧模型无此字段）
        if not hasattr(self, '_build_contract'):
            self._build_contract = {
                'primary_ch': self.primary_ch,
                'secondary_ch': self.secondary_ch,
                'fused_ch': self.primary_ch + self.secondary_ch,
            }

    def __getstate__(self):
        """
        自定义序列化：剔除运行时缓存，避免把数据 batch 张量写入权重文件。

        说明：
        - MultiModalRouter 在每次 forward 中会缓存上一批次输入（original_inputs），用于空间重置等运行时逻辑；
          这些张量与模型权重无关，且体积巨大，序列化会导致 .pt 异常虚高并携带数据内容。
        - 这里仅移除运行时缓存，不影响模型参数与结构的保存/恢复。
        """
        state = dict(self.__dict__)
        state['original_spatial_size'] = None
        state['original_inputs'] = {'PRIMARY': None, 'SECONDARY': None, 'FUSED': None}
        return state

    def _ensure_runtime_defaults(self):
        """Safeguard: define runtime fields if missing on legacy instances."""
        if not hasattr(self, 'runtime_modality'):
            self.runtime_modality = None
        if not hasattr(self, 'runtime_strategy'):
            self.runtime_strategy = None
        if not hasattr(self, 'runtime_seed'):
            self.runtime_seed = None
    
    def parse_layer_config(self, layer_config, layer_index, ch, verbose=True):
        """
        Parse layer configuration with optional 5th field for multimodal routing

        Args:
            layer_config: Layer configuration [from, repeats, module, args, input_source?]
            layer_index: Current layer index
            ch: Channel information
            verbose: Whether to print verbose information

        Returns:
            tuple: (input_channels, mm_input_source, mm_attributes)

        Raises:
            ValueError: When layer_config contains more than 5 fields (6th-field HOOK has been removed).
        """
        # Hard reject: 6th field (HOOK) is no longer supported
        if len(layer_config) > 5:
            raise ValueError(
                f"Layer {layer_index}: layer_config has {len(layer_config)} fields, but only 5 are supported "
                f"(from, repeats, module, args, input_source). "
                f"The 6th-field HOOK system has been removed. "
                f"Please update your model YAML to remove the 6th field."
            )

        # Parse standard 4 fields and optional 5th field (MM input source identifier)
        if len(layer_config) >= 5:
            f, n, m, args, mm_input_source = layer_config[:5]
        else:
            f, n, m, args = layer_config[:4]
            mm_input_source = None
            
        mm_attributes = {}
        
        # Check 5th field: MM input source routing processing
        if mm_input_source:
            resolved_role = resolve_input_source_role(mm_input_source)
            if resolved_role and resolved_role in self.INPUT_SOURCES:
                # Routing identifier detected, redirect input channel count
                c1 = self.INPUT_SOURCES[resolved_role]

                # Set MM attributes for the module
                mm_attributes = {
                    '_mm_input_source': resolved_role,
                    '_mm_layer_index': layer_index,
                    '_mm_version': 'v1.0',
                    '_mm_x_modality': self.x_modality_type,
                    '_mm_primary_modality': self.primary_modality_type,
                }

                # Special handling: if SECONDARY modality and from=-1, mark as new input start
                if resolved_role == 'SECONDARY' and f == -1:
                    mm_attributes['_mm_new_input_start'] = True
                    # Add spatial reset marking for SECONDARY modality new input start
                    mm_attributes['_mm_spatial_reset'] = True
                    # Note: Original size will be dynamically determined from actual input tensor
                    if verbose:
                        LOGGER.info(f"MultiModal Layer {layer_index}: SECONDARY模态新输入起点 (from=-1被重定向)")
                        LOGGER.info(f"MultiModal Layer {layer_index}: 空间重置标记已设置 (尺寸将从输入动态获取)")

                if verbose:
                    if resolved_role == 'PRIMARY':
                        LOGGER.info(f"MultiModal Layer {layer_index}: '{m.__name__ if hasattr(m, '__name__') else m}' <- PRIMARY({self.primary_modality_type})模态输入 ({c1}通道)")
                    elif resolved_role == 'SECONDARY':
                        LOGGER.info(f"MultiModal Layer {layer_index}: '{m.__name__ if hasattr(m, '__name__') else m}' <- SECONDARY({self.x_modality_type})模态输入 ({c1}通道)")
                    else:  # FUSED
                        LOGGER.info(f"MultiModal Layer {layer_index}: '{m.__name__ if hasattr(m, '__name__') else m}' <- FUSED({self.primary_modality_type}+{self.x_modality_type})双模态输入 ({c1}通道)")
        else:
            # Standard format, existing logic remains completely unchanged
            # Handle both single index and list of indices
            if isinstance(f, list):
                if len(f) == 1:
                    f_idx = f[0]
                    c1 = ch[f_idx] if f_idx != -1 else ch[-1]
                else:
                    # Multiple inputs case, calculate total channels
                    c1 = sum(ch[i] if i != -1 else ch[-1] for i in f)
            else:
                c1 = ch[f] if f != -1 else ch[-1]
            
        return c1, mm_input_source, mm_attributes
    
    def setup_multimodal_routing(self, x, profile=False):
        """
        Setup multimodal input sources and routing system initialization

        Args:
            x: Input tensor
            profile: Whether to print profiling information

        Returns:
            tuple: (routing_enabled, input_sources_dict)
        """
        # Backward-compat: make sure runtime_* fields exist even for unpickled legacy objects
        self._ensure_runtime_defaults()

        routing_enabled = False
        input_sources = None

        # Detect multimodal modes
        expected_fused_channels = self.INPUT_SOURCES['FUSED']  # primary_ch + secondary_ch
        primary_ch = self.INPUT_SOURCES['PRIMARY']  # primary channels from config
        secondary_ch = self.INPUT_SOURCES['SECONDARY']  # secondary channels from config
        is_fused_channel_input = x.shape[1] == expected_fused_channels
        is_multimodal_config = self.has_multimodal_config

        if is_fused_channel_input:
            routing_enabled = True
            self.original_spatial_size = (x.shape[2], x.shape[3])
            primary = x[:, :primary_ch, :, :]
            secondary = x[:, primary_ch:primary_ch + secondary_ch, :, :]

            rm = self.runtime_modality
            if rm is None:
                fused = x
            elif rm in ('rgb', self.primary_modality_type.lower()):
                filled_secondary = generate_modality_filling(
                    primary, self.primary_modality_type, self.x_modality_type,
                    target_ch=secondary_ch, strategy=self.runtime_strategy)
                fused = torch.cat([primary, filled_secondary], dim=1)
                secondary = filled_secondary
            else:
                filled_primary = generate_modality_filling(
                    secondary, self.x_modality_type, self.primary_modality_type,
                    target_ch=primary_ch, strategy=self.runtime_strategy)
                fused = torch.cat([filled_primary, secondary], dim=1)
                primary = filled_primary

            input_sources = {
                'PRIMARY': primary,
                'SECONDARY': secondary,
                'FUSED': fused,
            }
            self.cache_original_inputs(input_sources)

        elif is_multimodal_config and x.shape[1] == primary_ch:
            routing_enabled = True
            self.original_spatial_size = (x.shape[2], x.shape[3])
            rm = self.runtime_modality
            if rm is None:
                primary = x
                secondary = x.clone()
                fused = x
            elif rm in ('rgb', self.primary_modality_type.lower()):
                primary = x
                secondary = generate_modality_filling(
                    primary, self.primary_modality_type, self.x_modality_type,
                    target_ch=secondary_ch, strategy=self.runtime_strategy)
                fused = torch.cat([primary, secondary], dim=1)
            else:
                secondary = x
                if secondary.shape[1] != secondary_ch:
                    secondary = adapt_xch(secondary, secondary_ch)
                primary = generate_modality_filling(
                    secondary, self.x_modality_type, self.primary_modality_type,
                    target_ch=primary_ch, strategy=self.runtime_strategy)
                fused = torch.cat([primary, secondary], dim=1)

            input_sources = {
                'PRIMARY': primary,
                'SECONDARY': secondary,
                'FUSED': fused,
            }
            self.cache_original_inputs(input_sources)

        return routing_enabled, input_sources
    
    def route_layer_input(self, x, module, input_sources, profile=False):
        """
        Route input data for a specific layer based on its MM attributes
        
        Args:
            x: Current input tensor
            module: Current module
            input_sources: Multimodal input sources dictionary
            profile: Whether to print profiling information
            
        Returns:
            torch.Tensor or None: Routed input tensor, None if no routing needed
        """
        if not hasattr(module, '_mm_input_source'):
            return None
            
        # Validate input sources availability
        if not input_sources:
            if profile:
                LOGGER.warning(f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - 输入源不可用")
            return None
            
        mm_input_source = module._mm_input_source

        # Resolve legacy aliases (e.g. 'X' -> 'SECONDARY') for backward compat
        if mm_input_source not in ('PRIMARY', 'SECONDARY', 'FUSED'):
            resolved = resolve_input_source_role(mm_input_source)
            if resolved is not None:
                mm_input_source = resolved

        # ===== Routing logic: handle SECONDARY new-input-start first, then normal routing =====
        if hasattr(module, '_mm_new_input_start') and module._mm_new_input_start:
            # SECONDARY modality new input start, directly use SECONDARY modality data
            if 'SECONDARY' not in input_sources:
                if profile:
                    LOGGER.warning(f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                                  f"SECONDARY模态新输入起点需要SECONDARY输入源")
                return None

            routed_x = input_sources['SECONDARY']

            # Validate SECONDARY modality data has correct shape
            expected_secondary_channels = self.INPUT_SOURCES['SECONDARY']
            if routed_x.shape[1] != expected_secondary_channels:
                if profile:
                    LOGGER.error(f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                                f"SECONDARY模态新输入起点期望{expected_secondary_channels}通道，但接收到{routed_x.shape[1]}通道")
                    LOGGER.error("MultiModal: 当前输入源状态:")
                    for k, v in input_sources.items():
                        LOGGER.error(f"   {k}: {v.shape}")
                return None

            if profile:
                x_modality = getattr(module, '_mm_x_modality', 'unknown')
                LOGGER.info(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - SECONDARY({x_modality})模态新输入起点"
                )
                LOGGER.info(f"MultiModal: 输入切换 {x.shape} -> {routed_x.shape}")
        else:
            # Normal modality routing - validate and use requested modality
            if mm_input_source not in input_sources:
                if profile:
                    LOGGER.warning(f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                                  f"请求的模态 '{mm_input_source}' 不存在于输入源中")
                return None

            routed_x = input_sources[mm_input_source]

            if profile:
                LOGGER.info(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 路由到 '{mm_input_source}' - 输入形状: {x.shape} -> {routed_x.shape}"
                )
        
        # Final validation: ensure routed tensor is valid
        if routed_x is None:
            if profile:
                LOGGER.warning(f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} - "
                              f"路由结果为None")
            return None
            
        return routed_x
    
    def set_module_attributes(self, module, mm_attributes):
        """Set multimodal attributes on a module"""
        for attr_name, attr_value in mm_attributes.items():
            setattr(module, attr_name, attr_value)
            
    def get_original_spatial_size(self):
        """Get the original input spatial size for spatial reset"""
        return self.original_spatial_size
    
    def cache_original_inputs(self, input_sources):
        """
        Cache original multimodal inputs for spatial reset operations

        Args:
            input_sources (dict): Multimodal input sources to cache
        """
        # Cache original inputs using references (zero-copy), especially SECONDARY modality for spatial reset
        self.original_inputs = {
            'PRIMARY': input_sources['PRIMARY'] if 'PRIMARY' in input_sources else None,
            'SECONDARY': input_sources['SECONDARY'] if 'SECONDARY' in input_sources else None,
            'FUSED': input_sources['FUSED'] if 'FUSED' in input_sources else None
        }
        
    def get_original_x_input(self, target_size=None):
        """
        Get original SECONDARY modality input with specified target size

        Args:
            target_size (tuple, optional): Target spatial size (H, W). If None, returns original size.

        Returns:
            torch.Tensor or None: Original SECONDARY modality tensor
        """
        if 'SECONDARY' not in self.original_inputs or self.original_inputs['SECONDARY'] is None:
            return None

        x_input = self.original_inputs['SECONDARY']

        # If target_size is specified and different from current size, could add resize logic here
        # For now, we assume the original input already has the correct target size
        if target_size and target_size != x_input.shape[2:4]:
            # Target size validation for future extension
            pass

        return x_input
        
    def reset_spatial_input(self, x, module, mm_input_sources, profile=False):
        """
        Reset SECONDARY modality input to original spatial size for spatial reset layers.

        Args:
            x (torch.Tensor): Current input tensor
            module (nn.Module): Module with spatial reset requirement
            mm_input_sources (dict): Multimodal input sources
            profile (bool): Whether to print profiling information

        Returns:
            torch.Tensor: Reset input tensor with original spatial size
        """
        if not hasattr(module, '_mm_new_input_start') or not module._mm_new_input_start:
            return x

        # Validate that we have the required input sources
        if not mm_input_sources or 'SECONDARY' not in mm_input_sources:
            if profile:
                LOGGER.warning(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置失败 - 缺少SECONDARY模态输入源"
                )
            return x

        # Get original spatial size validation
        if self.original_spatial_size is None:
            if profile:
                LOGGER.warning(
                    f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置失败 - 无法获取原始尺寸"
                )
            return x

        # Use SECONDARY modality data with original spatial size
        reset_x = mm_input_sources['SECONDARY']  # This already has original spatial size

        if profile:
            LOGGER.info(f"MultiModal: Layer {getattr(module, '_mm_layer_index', '?')} 空间重置完成")
            LOGGER.info(f"MultiModal: 尺寸重置 {x.shape} -> {reset_x.shape}")

        return reset_x

    def update_dataset_config(self, dataset_config, update_x_channels: bool = True, update_x_modality: bool = True):
        """
        Update dataset configuration.

        Args:
            dataset_config (dict): Dataset configuration containing Xch/x_modality/Pch/primary_modality.
            update_x_channels (bool): Whether to refresh channel counts.
            update_x_modality (bool): Whether to refresh modality display names.
        """
        if not dataset_config:
            return

        # Update channel counts
        if update_x_channels and 'Xch' in dataset_config:
            self.secondary_ch = int(dataset_config['Xch'])
            self.INPUT_SOURCES['SECONDARY'] = self.secondary_ch
            self.INPUT_SOURCES['FUSED'] = self.primary_ch + self.secondary_ch
            if self.verbose:
                LOGGER.info(f"MultiModal: 更新SECONDARY模态通道数为 {self.secondary_ch}")
                LOGGER.info(f"MultiModal: 更新后路由配置: PRIMARY({self.primary_ch}ch), SECONDARY({self.secondary_ch}ch), FUSED({self.primary_ch + self.secondary_ch}ch)")

        if update_x_channels and 'Pch' in dataset_config:
            self.primary_ch = int(dataset_config['Pch'])
            self.INPUT_SOURCES['PRIMARY'] = self.primary_ch
            self.INPUT_SOURCES['FUSED'] = self.primary_ch + self.secondary_ch
            if self.verbose:
                LOGGER.info(f"MultiModal: 更新PRIMARY模态通道数为 {self.primary_ch}")
                LOGGER.info(f"MultiModal: 更新后路由配置: PRIMARY({self.primary_ch}ch), SECONDARY({self.secondary_ch}ch), FUSED({self.primary_ch + self.secondary_ch}ch)")

        if not update_x_modality:
            return

        # Update X modality type (for logging, e.g. 'ir'/'depth' etc.)
        x_mod = extract_x_modality_name_from_data(dataset_config if isinstance(dataset_config, dict) else None)
        if x_mod and x_mod != 'unknown':
            self.x_modality_type = str(x_mod)

        # Update primary modality type
        if 'primary_modality' in dataset_config:
            primary_mod = dataset_config['primary_modality']
            if primary_mod and isinstance(primary_mod, str) and primary_mod.strip():
                self.primary_modality_type = primary_mod.strip()
    
    def _detect_multimodal_config(self, config_dict):
        """
        Detect if the configuration contains multimodal layers

        Args:
            config_dict (dict, optional): Model configuration dictionary

        Returns:
            bool: True if multimodal configuration detected, False otherwise
        """
        if not config_dict:
            return False

        # Check for top-level 'input' field (single-modal automatic routing)
        if config_dict.get("input") is not None:
            role = resolve_input_source_role(config_dict["input"])
            if role is not None:
                return True

        # Check backbone and head layers for 5th field (MM input source)
        all_layers = config_dict.get('backbone', []) + config_dict.get('head', [])

        for layer_config in all_layers:
            if len(layer_config) >= 5:
                mm_input_source = layer_config[4]
                if resolve_input_source_role(mm_input_source) is not None:
                    return True

        return False
