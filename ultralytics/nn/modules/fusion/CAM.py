import torch
import torch.nn as nn

"""
Cross-Modal Attention Mechanism (CAM)

论文：SalM2: An Extremely Lightweight Saliency Mamba Model for Real-Time Cognitive Awareness of Driver Attention（AAAI 2025）
链接：https://ojs.aaai.org/index.php/AAAI/article/view/32157/34312
导出模块：CAM（跨模态注意力，双输入→单输出）

说明：
- 接口：forward(x1, x2=None) 支持 YAML 双输入 [left, right]
- 通道自适配：两路特征映射到公共维度后做 C×C 注意力，输出对齐左分支通道
- 形状约束：两路输入空间尺寸 (H, W) 必须一致
"""


class CAM(nn.Module):
    """Cross-Modal Attention Mechanism (CAM).

    Args:
        dim (int | None): 公共通道维度。若为 None，则在首次前向时采用左分支通道数。
                          该参数与 tasks.yaml 解析逻辑兼容：会自动注入左分支通道。
    """

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        # 惰性构建的通道投影（依据实际输入通道构建）
        self._built = False
        self._c_left = None
        self._c_right = None
        self._c_common = None
        self.proj_img_in: nn.Module | None = None
        self.proj_txt_in: nn.Module | None = None
        self.proj_out: nn.Module | None = None

    def _build_if_needed(self, c_left: int, c_right: int) -> None:
        if self._built and self._c_left == c_left and self._c_right == c_right:
            return

        c_common = self.dim if self.dim is not None else c_left
        # 输入侧投影，使两路特征具有相同的通道维度
        self.proj_img_in = nn.Identity() if c_left == c_common else nn.Conv2d(c_left, c_common, 1, bias=False)
        self.proj_txt_in = nn.Identity() if c_right == c_common else nn.Conv2d(c_right, c_common, 1, bias=False)
        # 输出侧投影，保证输出通道与左分支一致
        self.proj_out = nn.Identity() if c_common == c_left else nn.Conv2d(c_common, c_left, 1, bias=False)

        self._built = True
        self._c_left = c_left
        self._c_right = c_right
        self._c_common = c_common

    def forward(self, x1, x2=None):
        # 兼容 parse_model 输入为 [x1, x2] 的形式
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1

        if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
            raise TypeError("CAM 需要两路输入张量")

        b1, c1, h1, w1 = x1.shape
        b2, c2, h2, w2 = x2.shape
        if (h1 != h2) or (w1 != w2):
            raise ValueError(f"CAM 要求两路输入空间大小一致，got {(h1, w1)} vs {(h2, w2)}")
        if b1 != b2:
            raise ValueError(f"CAM 要求两路输入 batch 一致，got {b1} vs {b2}")

        # 构建投影层（仅首次或形状变化时）
        self._build_if_needed(c1, c2)

        # 投影至公共维度
        img = self.proj_img_in(x1)
        txt = self.proj_txt_in(x2)

        B, C, H, W = img.shape
        # 通道×通道的跨模态注意力
        q = img.view(B, C, -1)              # [B, C, HW]
        k = txt.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        attn = torch.bmm(q, k)              # [B, C, C]
        attn = self.softmax(attn)

        v = txt.view(B, C, -1)              # [B, C, HW]
        attn_info = torch.bmm(attn, v)      # [B, C, HW]
        attn_info = attn_info.view(B, C, H, W)

        out = self.gamma * attn_info + img    # 残差融合
        out = self.proj_out(out)              # 输出对齐到左分支通道
        return out
