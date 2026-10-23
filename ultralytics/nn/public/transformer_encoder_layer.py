"""Transformer encoder layer (public shared component)."""

from typing import Optional

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    This class implements a standard transformer encoder layer with multi-head attention and feedforward network,
    supporting both pre-normalization and post-normalization configurations.

    Attributes:
        ma (nn.MultiheadAttention): Multi-head attention module.
        fc1 (nn.Linear): First linear layer in the feedforward network.
        fc2 (nn.Linear): Second linear layer in the feedforward network.
        norm1 (nn.LayerNorm): Layer normalization after attention.
        norm2 (nn.LayerNorm): Layer normalization after feedforward network.
        dropout (nn.Dropout): Dropout layer for the feedforward network.
        dropout1 (nn.Dropout): Dropout layer after attention.
        dropout2 (nn.Dropout): Dropout layer after feedforward network.
        act (nn.Module): Activation function.
        normalize_before (bool): Whether to apply normalization before attention and feedforward.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """
        Initialize the TransformerEncoderLayer with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform forward pass with post-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform forward pass with pre-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward propagate the input through the encoder module.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after transformer encoder layer.
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

