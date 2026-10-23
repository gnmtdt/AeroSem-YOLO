# Ultralytics YOLO 🚀, AGPL-3.0 license

"""Disabled YOLOMM package entry for the YOLOVO project."""

_YOLOMM_DISABLED_MESSAGE = (
    "YOLOMM public APIs are disabled in this YOLOVO project. Use the YOLOVO entry point; "
    "this block has no runtime override."
)

__all__ = []


def __getattr__(name):
    raise ImportError(_YOLOMM_DISABLED_MESSAGE)
