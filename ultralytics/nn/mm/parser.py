# Ultralytics Multimodal Config Parser
# Universal YAML configuration parsing for arbitrary dual-modality architectures
# Version: v2.0

from ultralytics.utils import LOGGER
from ultralytics.nn.mm.utils import INPUT_SOURCE_ALIASES, resolve_input_source_role


class MultiModalConfigParser:
    """
    Universal Multimodal Configuration Parser

    Handles YAML configuration parsing for both YOLO and RTDETR
    with dynamic modality routing (5-field layer config).
    Supports legacy (RGB/X/Dual) and new (A/B/Fuse) aliases.
    """

    def __init__(self):
        # Support all alias keys (RGB, X, Dual, A, B, Fuse) + internal roles
        self.supported_input_sources = list(INPUT_SOURCE_ALIASES.keys()) + ["PRIMARY", "SECONDARY", "FUSED"]

    def validate_config_format(self, config):
        """Validate multimodal configuration format correctness"""
        role_layers = {"PRIMARY": [], "SECONDARY": [], "FUSED": []}

        for section in ['backbone', 'head']:
            for i, layer_config in enumerate(config.get(section, [])):
                if len(layer_config) == 5:
                    role = resolve_input_source_role(layer_config[4])
                    if role and role in role_layers:
                        role_layers[role].append(i)

        total = sum(len(v) for v in role_layers.values())
        LOGGER.info(f"MultiModal: config validation complete, {total} routing layers")
        return {f'{k.lower()}_layers': v for k, v in role_layers.items()} | {'total_routing_layers': total}

    def extract_multimodal_info(self, config):
        """Extract multimodal information from configuration"""
        x_modality_type = config.get('dataset_config', {}).get('x_modality', 'unknown')
        primary_modality_type = config.get('dataset_config', {}).get('primary_modality', 'rgb')

        mm_layer_count = 0
        for section in ['backbone', 'head']:
            for layer_config in config.get(section, []):
                if len(layer_config) >= 5 and (
                    layer_config[4] in self.supported_input_sources
                    or resolve_input_source_role(layer_config[4]) is not None
                ):
                    mm_layer_count += 1

        return {
            'x_modality_type': x_modality_type,
            'primary_modality_type': primary_modality_type,
            'mm_layer_count': mm_layer_count,
            'supports_multimodal': mm_layer_count > 0,
        }

    def parse_config(self, config: dict) -> dict:
        """
        Build a minimal multimodal model_config dict for MultiModalRouter.

        Detects whether YAML has any 5th-field multimodal routing markers.
        """
        has_mm = False
        input_layers = []
        for section in ['backbone', 'head']:
            for i, layer_config in enumerate(config.get(section, [])):
                if len(layer_config) >= 5 and (
                    layer_config[4] in self.supported_input_sources
                    or resolve_input_source_role(layer_config[4]) is not None
                ):
                    has_mm = True
                    input_layers.append((section, i))
        out = dict(config)
        out['has_multimodal_layers'] = has_mm
        out['input_layers'] = input_layers
        return out
