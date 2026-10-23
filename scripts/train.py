"""Portable AeroSem-YOLO training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLOVO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "configs" / "models" / "aerosem-yolo.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model YAML path.")
    parser.add_argument("--weights", type=Path, help="Optional compatible initialization checkpoint.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="CUDA device such as 0, 0,1, or cpu.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "train")
    parser.add_argument("--name", default="aerosem-yolo")
    parser.add_argument("--text-encoder-lr", type=float, default=0.0)
    parser.add_argument("--cache", action="store_true", help="Cache dataset images when memory permits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLOVO(str(args.model.resolve()))
    if args.weights is not None:
        model.load(str(args.weights.resolve()))

    options = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "project": str(args.project),
        "name": args.name,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "warmup_epochs": 3.0,
        "patience": 100,
        "amp": True,
        "close_mosaic": 10,
        "mosaic": 1.0,
        "mixup": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "translate": 0.1,
        "scale": 0.5,
        "cache": args.cache,
        "text_encoder_lr": args.text_encoder_lr,
    }
    if args.device is not None:
        options["device"] = args.device
    model.train(**options)


if __name__ == "__main__":
    main()

