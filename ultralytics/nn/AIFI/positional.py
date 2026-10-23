"""AIFI positional embedding utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_2d_sincos_position_embedding(w: int, h: int, embed_dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """
    Build 2D sine-cosine position embedding.

    Returns a tensor with shape [1, h*w, embed_dim] suitable for broadcasting to [B, h*w, embed_dim].
    """
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sin-cos position embedding")
    grid_w = torch.arange(int(w), dtype=torch.float32)
    grid_h = torch.arange(int(h), dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")

    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)

    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]

    pos = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)
    return pos[None]  # [1, h*w, embed_dim]


class LearnedPositionalEncoding(nn.Module):
    """1D learned positional embedding for flattened 2D tokens."""

    def __init__(self, max_position_embeddings: int, embedding_dim: int):
        super().__init__()
        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be > 0")
        self.max_position_embeddings = int(max_position_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.pe = nn.Embedding(self.max_position_embeddings, self.embedding_dim)

    def forward(self, h: int, w: int, device: torch.device | None = None) -> torch.Tensor:
        n = int(h) * int(w)
        if n > self.max_position_embeddings:
            raise ValueError(
                f"LearnedPositionalEncoding overflow: h*w={n} > max_position_embeddings={self.max_position_embeddings}"
            )
        position_ids = torch.arange(n, device=device).unsqueeze(0)  # [1, n]
        return self.pe(position_ids)  # [1, n, embedding_dim]

