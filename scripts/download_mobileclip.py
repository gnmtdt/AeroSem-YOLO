"""Download MobileCLIP-S0 from Apple's official model host."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
URL = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt"
DEFAULT_TARGET = REPO_ROOT / "ultralytics" / "nn" / "textencoder" / "mobileclip" / "mobileclip_s0.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.output.resolve()
    if target.exists() and not args.force:
        print(f"Checkpoint already exists: {target}")
        print(f"SHA-256: {sha256(target)}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".download")
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(target)
    print(f"Saved: {target}")
    print(f"SHA-256: {sha256(target)}")


if __name__ == "__main__":
    main()

