"""
YOLOVO MobileCLIP text encoder module.

Provides MobileCLIPTextEncoder and build_yolovo_text_encoder factory for YOLOVO's
open-vocabulary text encoding pipeline. Uses vendored Apple MobileCLIP source code
under models/yolovo/text_tower/mobileclip/_vendor/ to avoid pip install dependency.

Lifecycle:
  - Construction: MobileCLIPTextEncoder loads pretrained weights from asset_dir
  - Training: text_encoder parameters are exposed via iter_optimizer_parameters()
  - Save: save_sidecar() saves text tower state_dict to {path}/text_encoder.pt
  - Load: _load_sidecar() restores from sidecar checkpoint
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default asset directory resolved from this file's location:
# yolovo_text.py is at ultralytics/nn/yolovo_text.py
# assets are at ultralytics/nn/textencoder/mobileclip/
_DEFAULT_ASSET_DIR = str(Path(__file__).resolve().parent / "textencoder" / "mobileclip")


class MobileCLIPTextEncoder(nn.Module):
    """MobileCLIP-S0 text encoder with L2-normalized output for YOLOVO.

    Pretrained weights are loaded from asset_dir during construction.
    The tokenizer returns HF-style dict: {"input_ids": ..., "attention_mask": ...}.

    Offline-first: ``allow_download`` defaults to False. Checkpoint must be
    manually placed in the asset directory. Tokenizer is loaded from vendored
    local files (no HuggingFace remote access).

    Args:
        variant: MobileCLIP variant. Currently only "s0" is supported.
        dim: Output embedding dimension. Default 512.
        asset_dir: Directory containing pretrained checkpoint files.
        sidecar: Optional path to a sidecar checkpoint for restoring fine-tuned weights.
        allow_download: Whether to download checkpoint if not found. Default False.
    """

    def __init__(
        self,
        variant: str = "s0",
        dim: int = 512,
        *,
        asset_dir: str | Path | None = None,
        sidecar: str | Path | None = None,
        allow_download: bool = False,
    ) -> None:
        super().__init__()
        from ultralytics.models.yolovo.text_tower.mobileclip import build_mobileclip_bundle

        self.variant = variant
        self.dim = dim
        self.asset_dir = Path(asset_dir or _DEFAULT_ASSET_DIR)
        self.sidecar = Path(sidecar) if sidecar else None

        bundle = build_mobileclip_bundle(
            variant=variant,
            asset_dir=self.asset_dir,
            allow_download=allow_download,
        )
        self.tokenizer = bundle.tokenizer
        self.text_model = bundle.text_encoder
        # NOTE: bundle.model (full CLIP) is NOT stored here. It contains image_encoder
        # which would pollute state_dict and GPU memory. factory.py detaches
        # text_encoder from the CLIP model before returning the bundle.

        actual_dim = int(bundle.embed_dim)
        self.proj = nn.Linear(actual_dim, dim, bias=False) if actual_dim != dim else nn.Identity()

        if self.sidecar is not None:
            self._load_sidecar(self.sidecar)

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize texts into HF-style dict with input_ids and attention_mask.

        Args:
            texts: List of text strings to tokenize.

        Returns:
            Dict with "input_ids" and "attention_mask" tensors, shape [K, context_length].
        """
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode text strings to L2-normalized embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            L2-normalized text features [K, dim].
        """
        if not texts:
            raise ValueError("MobileCLIPTextEncoder.forward() requires at least one text prompt.")
        return self.forward_tokens(self.tokenize(texts))

    def forward_tokens(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode pre-computed tokens to L2-normalized embeddings.

        Args:
            tokens: Dict with "input_ids" and optionally "attention_mask" tensors.

        Returns:
            L2-normalized text features [K, dim].
        """
        if not tokens or "input_ids" not in tokens:
            raise ValueError(
                "MobileCLIPTextEncoder.forward_tokens() requires 'input_ids' key."
            )
        device = next(self.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # Convert HF attention_mask (1=valid, 0=padding) to MobileCLIP
        # key_padding_mask (True=padding) for correct masking of padding tokens.
        key_padding_mask = None
        if "attention_mask" in tokens:
            key_padding_mask = tokens["attention_mask"].logical_not()

        # MobileCLIP TextTransformer takes text_tokens + key_padding_mask
        text_features = self.text_model(
            text_tokens=tokens["input_ids"],
            key_padding_mask=key_padding_mask,
        )
        text_features = self.proj(text_features)
        return F.normalize(text_features, dim=-1)  # [K, dim]

    def iter_optimizer_parameters(self):
        """Yield all trainable parameters for optimizer grouping.

        Use this instead of hardcoded .encoder attribute access to support
        different text encoder implementations uniformly.
        """
        yield from self.text_model.parameters()
        yield from self.proj.parameters()

    def save_sidecar(self, path: str | Path) -> None:
        """Save trained text tower weights to {path}/text_encoder.pt.

        Args:
            path: Directory where text_encoder.pt will be saved.
        """
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "variant": self.variant,
                "dim": self.dim,
            },
            root / "text_encoder.pt",
        )

    def _load_sidecar(self, path: str | Path) -> None:
        """Load text tower weights from sidecar checkpoint.

        Args:
            path: Path to directory containing text_encoder.pt, or direct .pt file path.
        """
        root = Path(path)
        file = root if root.is_file() else root / "text_encoder.pt"
        state = torch.load(file, map_location="cpu", weights_only=False)
        self.load_state_dict(state["state_dict"], strict=True)


def build_yolovo_text_encoder(cfg: dict) -> nn.Module:
    """Factory function to build YOLOVO text encoder from YAML config.

    Consumes the 'name', 'dim', and optional 'path' fields from the YAML
    text_encoder section. If 'path' is provided, it overrides the default
    asset directory for pretrained weights.

    Args:
        cfg: Dict from YAML text_encoder section. Supported keys: name, dim, path.

    Returns:
        MobileCLIPTextEncoder instance.

    Raises:
        ValueError: If an unsupported text encoder name is specified.
    """
    name = cfg.get("name", "mobileclip_s0")

    if name == "mobileclip_s0":
        asset_dir = cfg.get("path")  # optional: user-specified weight directory
        return MobileCLIPTextEncoder(
            variant="s0",  # variant is encoded by YAML name
            dim=int(cfg.get("dim", 512)),
            asset_dir=asset_dir,  # None falls back to _DEFAULT_ASSET_DIR
        )

    raise ValueError(f"Unknown text encoder: {name}. Supported: mobileclip_s0")
