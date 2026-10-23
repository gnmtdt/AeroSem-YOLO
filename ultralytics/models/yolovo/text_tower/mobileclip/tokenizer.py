"""
YOLOVO MobileCLIP tokenizer -- offline-first, vendor-based.

Tokenizer files are vendored under ``ultralytics/nn/textencoder/mobileclip/tokenizer/``
and loaded purely locally. No HuggingFace remote ID, no cache fallback, no network
access at model construction time.

Missing tokenizer files produce a clear error with the exact expected path.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import CLIPTokenizer as HFCLIPTokenizer

_CONTEXT_LENGTH = 77  # MobileCLIP-S0 context length

# Vendor directory resolved relative to this package:
# tokenizer.py is at ultralytics/models/yolovo/text_tower/mobileclip/tokenizer.py
# vendor files are at ultralytics/nn/textencoder/mobileclip/tokenizer/
_VENDOR_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "nn" / "textencoder" / "mobileclip" / "tokenizer"

_REQUIRED_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json")


def _validate_vendor_dir() -> Path:
    """Check that all required tokenizer files exist in the vendor directory.

    Returns:
        Path: Verified vendor directory.

    Raises:
        FileNotFoundError: If any required file is missing, with the exact path.
    """
    missing = [f for f in _REQUIRED_FILES if not (_VENDOR_DIR / f).exists()]
    if missing:
        paths = "\n  ".join(str(_VENDOR_DIR / f) for f in missing)
        raise FileNotFoundError(
            f"MobileCLIP tokenizer vendor files missing:\n  {paths}\n"
            f"Expected vendor directory: {_VENDOR_DIR}\n"
            f"These files should be tracked in git under ultralytics/nn/textencoder/mobileclip/tokenizer/."
        )
    return _VENDOR_DIR


def _get_tokenizer() -> HFCLIPTokenizer:
    """Load CLIP tokenizer from local vendor directory (no network access)."""
    vendor_dir = _validate_vendor_dir()
    return HFCLIPTokenizer.from_pretrained(str(vendor_dir), local_files_only=True)


class MobileCLIPTokenizer:
    """HF-style tokenizer wrapper for MobileCLIP text encoding.

    Loads from vendored local files only. Returns {"input_ids": [B, 77], "attention_mask": [B, 77]}.
    """

    def __init__(self) -> None:
        self._tokenizer = _get_tokenizer()

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        return self._tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=_CONTEXT_LENGTH,
            return_tensors=return_tensors,
        )


def build_mobileclip_tokenizer(model_name: str | None = None) -> MobileCLIPTokenizer:
    """Build a MobileCLIP-compatible tokenizer from local vendor files.

    Args:
        model_name: Ignored. Kept for backward compatibility with factory interface.

    Returns:
        MobileCLIPTokenizer instance.
    """
    return MobileCLIPTokenizer()