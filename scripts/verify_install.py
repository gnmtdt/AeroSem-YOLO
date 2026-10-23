"""Check the local AeroSem-YOLO installation without loading a model."""

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics import YOLOVO, __version__


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = REPO_ROOT / "configs" / "models" / "aerosem-yolo.yaml"
MOBILECLIP = REPO_ROOT / "ultralytics" / "nn" / "textencoder" / "mobileclip" / "mobileclip_s0.pt"


def main() -> None:
    checks = {
        "YOLOVO API": YOLOVO is not None,
        "model config": MODEL_CONFIG.is_file(),
        "MobileCLIP-S0 checkpoint": MOBILECLIP.is_file(),
    }
    print(f"Ultralytics fork: {__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    for label, passed in checks.items():
        print(f"{'OK' if passed else 'MISSING':>7}  {label}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

