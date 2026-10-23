# MobileCLIP Runtime Assets

This directory contains YOLOVO MobileCLIP runtime assets.

## Resource Layout

```text
ultralytics/nn/textencoder/mobileclip/
  README.md                     # this file
  mobileclip_s0.pt              # checkpoint (runtime asset, do NOT commit)
  tokenizer/                    # vendored tokenizer (tracked in git)
    vocab.json
    merges.txt
    tokenizer_config.json
    special_tokens_map.json
```

## Checkpoint: `mobileclip_s0.pt`

**Offline-first policy:** Model construction does NOT download the checkpoint by default.

- Manually place `mobileclip_s0.pt` in this directory before training.
- To explicitly download: pass `allow_download=True` to `MobileCLIPTextEncoder` or
  `build_mobileclip_bundle`. This is the ONLY way to trigger checkpoint download.
- Missing checkpoint + `allow_download=False` (default) = hard fail with exact path.

Download source: `https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt`

## Tokenizer

**No network access at any time.** Tokenizer files are vendored in `tokenizer/` and
loaded via `local_files_only=True`. There is no fallback to HuggingFace cache,
open_clip cache, or network download.

- If tokenizer vendor files are missing, construction raises `FileNotFoundError`
  listing the exact expected paths.
- These files are tracked in git and should never need manual placement.

## Defaults

| Parameter | Default | Behavior |
|-----------|---------|----------|
| `allow_download` | `False` | Offline-first. Checkpoint must exist locally. |
| `local_files_only` | `True` | Tokenizer loaded from vendor dir only. |

The `allow_download` flag is a runtime-only parameter. It does NOT appear in
model YAML config and cannot be set via YAML defaults.