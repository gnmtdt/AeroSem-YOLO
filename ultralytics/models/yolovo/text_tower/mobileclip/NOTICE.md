# MobileCLIP Source Attribution

## Source

This directory contains source code derived from Apple's MobileCLIP project:
- Upstream repository: https://github.com/apple/ml-mobileclip
- Code license: MIT (see LICENSE file in upstream repo)
- Model license: Apple ML Research Model Terms of Use

## Vendored Code

The `_vendor/mobileclip/` directory contains a copy of Apple's MobileCLIP Python
package source code. This code is used as-is with zero modifications to maintain
compatibility with the upstream package's internal import structure.

## Files included from upstream

- `mobileclip/__init__.py` - Model factory and tokenizer builder
- `mobileclip/clip.py` - CLIP model class
- `mobileclip/text_encoder.py` - TextTransformer text encoder
- `mobileclip/logger.py` - Logging utilities
- `mobileclip/configs/mobileclip_s0.json` - S0 model configuration
- `mobileclip/modules/common/transformer.py` - Transformer encoder, attention, positional embedding
- `mobileclip/modules/common/mobileone.py` - MobileOne re-parameterizable blocks
- `mobileclip/modules/text/tokenizer.py` - ClipTokenizer wrapping open_clip
- `mobileclip/modules/text/repmixer.py` - RepMixer token mixing blocks
- `mobileclip/image_encoder.py` - Image encoder (MCi), retained for CLIP model construction
- `mobileclip/models/` - Model registration (mci, vit), retained for CLIP model construction
- `mobileclip/modules/image/` - Image modules, retained for CLIP model construction

## Runtime Dependencies

The vendored code requires the following Python packages at runtime:
- `torch` (PyTorch)
- `torchvision`
- `timm` (for MobileOne blocks and image encoder models)
- `open_clip_torch` (for the tokenizer)
