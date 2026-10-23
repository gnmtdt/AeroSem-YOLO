"""
MobileCLIP factory for YOLOVO.

Builds MobileCLIP text encoder bundles without requiring the upstream pip package.
Uses vendored Apple source code under _vendor/ with sys.path injection.

Key design decision: Instead of using create_model_and_transforms() which builds
the full CLIP model (including image_encoder + timm dependencies), this factory
directly constructs TextTransformer from the config JSON. This eliminates the
need for timm, MCi image models, and other image-related dependencies.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# _vendor sys.path injection (must happen before any mobileclip import)
# ---------------------------------------------------------------------------
_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MobileCLIPVariantSpec:
    """Describes a supported MobileCLIP variant for YOLOVO."""

    variant: str
    model_name: str
    checkpoint_name: str
    text_dim: int
    context_length: int = 77


@dataclass
class YOLOVOMobileCLIPBundle:
    """Bundle of MobileCLIP objects for YOLOVO text encoding."""

    spec: MobileCLIPVariantSpec
    model: nn.Module  # Full CLIP model (text_encoder accessible via .text_encoder)
    text_encoder: nn.Module  # TextTransformer module
    tokenizer: Callable  # HF-style tokenizer returning {"input_ids": ..., "attention_mask": ...}
    embed_dim: int


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------
VARIANT_SPECS = {
    "s0": MobileCLIPVariantSpec(
        variant="s0",
        model_name="mobileclip_s0",
        checkpoint_name="mobileclip_s0.pt",
        text_dim=512,
    ),
}


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------
# Default asset directory resolved from this file's location:
# factory.py is at ultralytics/models/yolovo/text_tower/mobileclip/factory.py
# assets are at ultralytics/nn/textencoder/mobileclip/
DEFAULT_MOBILECLIP_ASSET_DIR = (
    Path(__file__).resolve().parents[4] / "nn" / "textencoder" / "mobileclip"
)


def resolve_asset_dir(asset_dir: str | Path | None = None) -> Path:
    """Resolve the asset directory for MobileCLIP weights.

    Does NOT auto-create the directory. The directory must already exist
    (containing the checkpoint and tokenizer vendor files). Only explicit
    download/prepare flows may create this directory.
    """
    root = Path(asset_dir or DEFAULT_MOBILECLIP_ASSET_DIR)
    return root


def ensure_checkpoint(
    spec: MobileCLIPVariantSpec,
    asset_dir: str | Path | None = None,
    allow_download: bool = False,
) -> Path:
    """Ensure the MobileCLIP checkpoint file exists, optionally downloading it.

    Default is offline-first (``allow_download=False``). Missing checkpoint
    raises a clear error with the exact expected path.

    Args:
        spec: Variant specification (contains checkpoint filename).
        asset_dir: Override directory for the checkpoint.
        allow_download: Whether to download if missing. Default False.

    Returns:
        Path to the checkpoint file.
    """
    root = resolve_asset_dir(asset_dir)
    ckpt = root / spec.checkpoint_name
    if ckpt.exists():
        return ckpt
    if not allow_download:
        raise FileNotFoundError(
            f"MobileCLIP checkpoint not found: {ckpt}. "
            "Place the pretrained file under ultralytics/nn/textencoder/mobileclip/ "
            "or pass allow_download=True to download."
        )
    # Only create directory when explicitly downloading
    root.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import safe_download

    url = f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{spec.checkpoint_name}"
    safe_download(url, dir=root)
    if not ckpt.exists():
        raise FileNotFoundError(f"Downloaded MobileCLIP checkpoint is still missing: {ckpt}")
    return ckpt


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _load_config(model_name: str) -> dict:
    """Load MobileCLIP config JSON from vendored configs directory."""
    configs_dir = os.path.join(_VENDOR_DIR, "mobileclip", "configs")
    cfg_path = os.path.join(configs_dir, f"{model_name}.json")
    if not os.path.exists(cfg_path):
        raise ValueError(f"Config not found for model_name={model_name}: {cfg_path}")
    with open(cfg_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tokenizer builder (delegates to tokenizer.py)
# ---------------------------------------------------------------------------
def build_mobileclip_tokenizer(model_name: str | None = None):
    """Build HF-style tokenizer for MobileCLIP.

    Returns a callable that produces {"input_ids": Tensor, "attention_mask": Tensor}.

    Uses HuggingFace transformers CLIPTokenizer instead of the upstream
    ClipTokenizer (which requires open_clip_torch).
    """
    from ultralytics.models.yolovo.text_tower.mobileclip.tokenizer import (
        build_mobileclip_tokenizer as _build,
    )

    return _build(model_name)


# ---------------------------------------------------------------------------
# Bundle builder
# ---------------------------------------------------------------------------
def build_mobileclip_bundle(
    variant: str = "s0",
    *,
    asset_dir: str | Path | None = None,
    allow_download: bool = False,
) -> YOLOVOMobileCLIPBundle:
    """Build a MobileCLIP text encoding bundle for YOLOVO.

    Constructs the full CLIP model (needed for checkpoint loading), extracts the
    text_encoder, and wraps the tokenizer in HF-style interface.

    Args:
        variant: MobileCLIP variant name. Currently only "s0" is supported.
        asset_dir: Directory containing the pretrained checkpoint.
        allow_download: Whether to download the checkpoint if missing.

    Returns:
        YOLOVOMobileCLIPBundle with model, text_encoder, tokenizer, and embed_dim.
    """
    if variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown MobileCLIP variant: {variant}. Supported: {list(VARIANT_SPECS.keys())}")

    spec = VARIANT_SPECS[variant]
    checkpoint = ensure_checkpoint(spec, asset_dir=asset_dir, allow_download=allow_download)

    # Suppress timm deprecation warnings during mobileclip import
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        from mobileclip import create_model_and_transforms  # noqa: E402 (vendor import)

    # Build full model and load pretrained weights
    model, _, _ = create_model_and_transforms(
        spec.model_name,
        pretrained=str(checkpoint),
        reparameterize=False,  # Keep in training mode for fine-tuning
        device="cpu",
    )

    text_encoder = model.text_encoder
    # Detach text_encoder from CLIP model so image_encoder is not referenced.
    # This prevents image_encoder parameters from appearing in state_dict/GPU memory.
    model.text_encoder = None  # type: ignore[assignment]

    tokenizer = build_mobileclip_tokenizer(spec.model_name)
    embed_dim = int(getattr(text_encoder, "projection_dim", spec.text_dim))

    return YOLOVOMobileCLIPBundle(
        spec=spec,
        model=model,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        embed_dim=embed_dim,
    )
