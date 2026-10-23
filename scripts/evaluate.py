"""Evaluate an AeroSem-YOLO checkpoint on a YOLO-format dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLOVO


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "evaluate")
    parser.add_argument("--name", default="aerosem-yolo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLOVO(str(args.weights.resolve()))
    options = {
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": False,
    }
    if args.device is not None:
        options["device"] = args.device
    model.val(**options)


if __name__ == "__main__":
    main()

