# flake8: noqa
"""
模态生成器入口（对外暴露 DepthGen / DEMGen / EdgeGen）。
"""

from .depth_anything_v2 import DepthGen
from .dem_features import DEMGen
from .edge import EdgeGen

__all__ = [
    "DepthGen",
    "DEMGen",
    "EdgeGen",
]
