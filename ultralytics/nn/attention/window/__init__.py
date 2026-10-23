"""
窗口/局部注意力模块集合

包含:
- BiLevelRoutingAttention: 双层路由注意力 (CVPR 2023)
- DilatedGCSA: 膨胀全局上下文自注意力 (arXiv 2024)
- DilatedMWSA: 膨胀多窗口自注意力 (arXiv 2024)
- DPWA: 可变形位置窗口注意力 (IEEE TGRS 2025)
- DWM_MSA: 双窗口多头自注意力 (IEEE TIP 2025)
- DHOGSA: 方向梯度直方图引导空间注意力 (AAAI 2026)
- PatchSA: 补丁自注意力 (ACM MM 2025)
- Token_Selective_Attention: top-k token选择注意力 (Neural Networks 2025)
"""

from .biformer import BiLevelRoutingAttention
from .dilated_gcsa import DilatedGCSA
from .dilated_mwsa import DilatedMWSA
from .dpwa import DPWA
from .dwm_msa import DWM_MSA
from .dhogsa import DHOGSA
from .swsa import PatchSA
from .token_select import Token_Selective_Attention

__all__ = [
    'BiLevelRoutingAttention',
    'DilatedGCSA',
    'DilatedMWSA',
    'DPWA',
    'DWM_MSA',
    'DHOGSA',
    'PatchSA',
    'Token_Selective_Attention',
]
