"""Run prompt-based AeroSem-YOLO inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLOVO


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Image, video, directory, glob, or stream URL.")
    parser.add_argument("--text", nargs="+", required=True, help="Ordered class prompts.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "predict")
    parser.add_argument("--name", default="aerosem-yolo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLOVO(str(args.weights.resolve()))
    options = {
        "source": args.source,
        "text": args.text,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "save": True,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": False,
    }
    if args.device is not None:
        options["device"] = args.device
    model.predict(**options)


if __name__ == "__main__":
    main()

