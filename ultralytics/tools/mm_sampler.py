"""
MultiModal Dataset Sampler - YOLOMM 多模态数据集图像采样工具

提供从多模态数据集 YAML 配置中随机采样匹配图像对的功能，
输出 RGB 和 X 模态图像路径供 YOLOMM.predict() 方法使用。

Usage:
    from mm_sampler import MultiModalSampler

    # 初始化采样器
    sampler = MultiModalSampler('path/to/dataset.yaml')

    # 随机采样一对图像
    rgb_path, x_path = sampler.sample_one()

    # 采样多对图像
    pairs = sampler.sample(n=5)  # [(rgb1, x1), (rgb2, x2), ...]

    # 采样多对图像（以两个列表形式输出，按图片对顺序对齐）
    rgb_list, x_list = sampler.sample_source_list(n=5)  # (['rgb1', 'rgb2', ...], ['x1', 'x2', ...])

    # 直接供 predict 方法使用
    from ultralytics import YOLOMM
    model = YOLOMM('yolo11n-mm.yaml')
    model.predict(source=[rgb_path, x_path])

    # 或使用便捷方法
    sources = sampler.get_predict_sources(n=3)
    for source in sources:
        model.predict(source=source)

Author: YOLOMM Team
"""

import random
import yaml
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Any


class MultiModalSampler:
    """多模态数据集图像采样器"""

    # 常见图像后缀
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.npy', '.npz'}

    def __init__(
        self,
        yaml_path: Union[str, Path],
        split: str = 'val',
        seed: Optional[int] = None
    ):
        """
        初始化多模态采样器

        Args:
            yaml_path: 数据集 YAML 配置文件路径
            split: 数据集分割 ('train', 'val', 'test')
            seed: 随机种子，用于可重复采样
        """
        self.yaml_path = Path(yaml_path)
        self.split = split

        if seed is not None:
            random.seed(seed)

        # 解析 YAML 配置
        self._parse_yaml()

        # 扫描并缓存图像对
        self._scan_image_pairs()

    def _parse_yaml(self) -> None:
        """解析数据集 YAML 配置文件"""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {self.yaml_path}")

        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 解析根目录
        self.root_path = Path(self.config.get('path', ''))
        if not self.root_path.is_absolute():
            # 相对路径基于 YAML 文件所在目录
            self.root_path = self.yaml_path.parent / self.root_path

        # 解析数据分割路径
        split_path = self.config.get(self.split)
        if split_path is None:
            available_splits = [k for k in ['train', 'val', 'test'] if k in self.config]
            raise ValueError(
                f"数据集中未找到 '{self.split}' 分割。"
                f"可用分割: {available_splits}"
            )

        self.split_path = Path(split_path)
        if not self.split_path.is_absolute():
            self.split_path = self.root_path / self.split_path

        # 解析模态配置 (支持 'modality' 和 'modalities' 两种键名)
        self.modality_config = self.config.get('modality') or self.config.get('modalities', {})

        # 解析使用的模态
        self.modality_used = self.config.get('modality_used', ['rgb'])
        if isinstance(self.modality_used, str):
            self.modality_used = [self.modality_used]

        # 确定 RGB 和 X 模态
        self._resolve_modalities()

        # 获取 X 模态通道数
        self.x_channels = self.config.get('Xch', 3)

        # 类别信息
        self.names = self.config.get('names', {})

    def _resolve_modalities(self) -> None:
        """解析并确定 RGB 和 X 模态路径"""
        try:
            from ultralytics.nn.mm.utils import get_modality_path_from_data
            _has_modality_path_util = True
        except ImportError:
            _has_modality_path_util = False

        if len(self.modality_used) < 2:
            raise ValueError(
                f"多模态采样需要至少两个模态，但配置中只有: {self.modality_used}"
            )

        # 第一个通常是 RGB
        rgb_key = self.modality_used[0].lower()
        x_key = self.modality_used[1].lower()

        # 获取模态相对路径
        self.rgb_modality_name = rgb_key
        self.x_modality_name = x_key

        # RGB 模态路径
        if _has_modality_path_util:
            rgb_rel_path = get_modality_path_from_data(self.config, rgb_key)
        else:
            rgb_rel_path = self.modality_config.get(rgb_key, 'images')
        self.rgb_base_path = self.root_path / rgb_rel_path / self.split_path.name

        # 如果上述路径不存在，尝试直接使用 split_path
        if not self.rgb_base_path.exists():
            self.rgb_base_path = self.split_path

        # X 模态路径
        if _has_modality_path_util:
            x_rel_path = get_modality_path_from_data(self.config, x_key)
        else:
            x_rel_path = self.modality_config.get(x_key)
            if x_rel_path is None:
                # 尝试常见命名模式
                possible_x_paths = [
                    f'images_{x_key}',
                    f'{x_key}',
                    f'images-{x_key}'
                ]
                for p in possible_x_paths:
                    test_path = self.root_path / p / self.split_path.name
                    if test_path.exists():
                        x_rel_path = p
                        break
                if x_rel_path is None:
                    x_rel_path = f'images_{x_key}'

        self.x_base_path = self.root_path / x_rel_path / self.split_path.name

        # 如果 X 模态目录不存在，尝试其他结构
        if not self.x_base_path.exists():
            # 尝试直接在根目录下
            alt_x_path = self.root_path / x_rel_path
            if alt_x_path.exists():
                self.x_base_path = alt_x_path

    def _scan_image_pairs(self) -> None:
        """扫描并匹配 RGB 和 X 模态图像对"""
        self.image_pairs: List[Tuple[Path, Path]] = []

        if not self.rgb_base_path.exists():
            raise FileNotFoundError(f"RGB 模态路径不存在: {self.rgb_base_path}")

        if not self.x_base_path.exists():
            raise FileNotFoundError(f"X 模态路径不存在: {self.x_base_path}")

        # 扫描 RGB 图像
        rgb_images = self._scan_images(self.rgb_base_path)

        # 扫描 X 模态图像
        x_images = self._scan_images(self.x_base_path)

        # 建立 X 模态图像的查找字典 (按文件名 stem)
        x_image_dict: Dict[str, Path] = {}
        for x_img in x_images:
            x_image_dict[x_img.stem] = x_img

        # 匹配图像对
        for rgb_img in rgb_images:
            stem = rgb_img.stem
            if stem in x_image_dict:
                self.image_pairs.append((rgb_img, x_image_dict[stem]))

        if not self.image_pairs:
            raise ValueError(
                f"未找到匹配的图像对。\n"
                f"RGB 路径: {self.rgb_base_path} ({len(rgb_images)} 张图像)\n"
                f"X 模态路径: {self.x_base_path} ({len(x_images)} 张图像)"
            )

    def _scan_images(self, directory: Path) -> List[Path]:
        """扫描目录中的图像文件"""
        images = []
        for ext in self.IMAGE_EXTENSIONS:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
        return sorted(images)

    def sample_one(self) -> Tuple[str, str]:
        """
        随机采样一对多模态图像

        Returns:
            Tuple[str, str]: (rgb_path, x_path) 图像路径字符串
        """
        if not self.image_pairs:
            raise RuntimeError("没有可用的图像对")

        rgb_path, x_path = random.choice(self.image_pairs)
        return str(rgb_path), str(x_path)

    def sample(self, n: int = 1, replace: bool = False) -> List[Tuple[str, str]]:
        """
        随机采样多对多模态图像

        Args:
            n: 采样数量
            replace: 是否允许重复采样

        Returns:
            List[Tuple[str, str]]: [(rgb_path, x_path), ...] 图像路径对列表
        """
        if not self.image_pairs:
            raise RuntimeError("没有可用的图像对")

        if not replace and n > len(self.image_pairs):
            raise ValueError(
                f"请求采样 {n} 对，但只有 {len(self.image_pairs)} 对可用。"
                f"设置 replace=True 允许重复采样。"
            )

        if replace:
            selected = [random.choice(self.image_pairs) for _ in range(n)]
        else:
            selected = random.sample(self.image_pairs, n)

        return [(str(rgb), str(x)) for rgb, x in selected]

    def sample_source_list(self, n: int = 1, replace: bool = False) -> tuple[list[str], list[str]]:
        """
        采样一组指定长度的图片对路径，并以两个列表形式输出。

        Returns:
            (RGBlist, Xlist): 两个列表按图片对顺序对齐，分别为 RGB 模态与 X 模态的路径字符串列表。
        """
        pairs = self.sample(n=n, replace=replace)
        if not pairs:
            return [], []
        rgb_list, x_list = zip(*pairs)
        return list(rgb_list), list(x_list)

    def sample_by_index(self, index: int) -> Tuple[str, str]:
        """
        按索引获取特定图像对

        Args:
            index: 图像对索引

        Returns:
            Tuple[str, str]: (rgb_path, x_path) 图像路径字符串
        """
        if index < 0 or index >= len(self.image_pairs):
            raise IndexError(
                f"索引 {index} 超出范围 [0, {len(self.image_pairs)})"
            )
        rgb_path, x_path = self.image_pairs[index]
        return str(rgb_path), str(x_path)

    def get_predict_source(self) -> List[str]:
        """
        获取单个 predict 方法可用的 source 格式

        Returns:
            List[str]: [rgb_path, x_path] 格式，可直接传给 YOLOMM.predict()
        """
        rgb_path, x_path = self.sample_one()
        return [rgb_path, x_path]

    def get_predict_sources(self, n: int = 1) -> List[List[str]]:
        """
        获取多个 predict 方法可用的 source 格式

        Args:
            n: 采样数量

        Returns:
            List[List[str]]: [[rgb1, x1], [rgb2, x2], ...] 格式列表
        """
        pairs = self.sample(n)
        return [list(pair) for pair in pairs]

    def get_source_dirs(self) -> Tuple[str, str]:
        """
        返回 RGB 和 X 模态的目录路径，可直接用于目录批量推理。

        Returns:
            (rgb_dir, x_dir): 两个目录路径字符串

        Example:
            >>> sampler = MultiModalSampler('dataset.yaml', split='val')
            >>> rgb_dir, x_dir = sampler.get_source_dirs()
            >>> model.predict(source=[rgb_dir, x_dir], strict_match=True)
        """
        return str(self.rgb_base_path), str(self.x_base_path)

    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """
        获取所有匹配的图像对

        Returns:
            List[Tuple[str, str]]: 所有 (rgb_path, x_path) 图像对
        """
        return [(str(rgb), str(x)) for rgb, x in self.image_pairs]

    def __len__(self) -> int:
        """返回可用图像对数量"""
        return len(self.image_pairs)

    def __iter__(self):
        """迭代所有图像对"""
        for rgb_path, x_path in self.image_pairs:
            yield str(rgb_path), str(x_path)

    def info(self) -> Dict[str, Any]:
        """
        获取采样器配置信息

        Returns:
            Dict: 配置信息字典
        """
        return {
            'yaml_path': str(self.yaml_path),
            'root_path': str(self.root_path),
            'split': self.split,
            'rgb_modality': self.rgb_modality_name,
            'x_modality': self.x_modality_name,
            'rgb_base_path': str(self.rgb_base_path),
            'x_base_path': str(self.x_base_path),
            'x_channels': self.x_channels,
            'num_pairs': len(self.image_pairs),
            'modality_used': self.modality_used,
            'class_names': self.names
        }

    def __repr__(self) -> str:
        return (
            f"MultiModalSampler(\n"
            f"  yaml='{self.yaml_path.name}',\n"
            f"  split='{self.split}',\n"
            f"  modalities=['{self.rgb_modality_name}', '{self.x_modality_name}'],\n"
            f"  num_pairs={len(self.image_pairs)}\n"
            f")"
        )


def sample_from_yaml(
    yaml_path: Union[str, Path],
    n: int = 1,
    split: str = 'val',
    seed: Optional[int] = None
) -> List[List[str]]:
    """
    便捷函数：从 YAML 配置采样多模态图像对

    Args:
        yaml_path: 数据集 YAML 配置文件路径
        n: 采样数量
        split: 数据集分割
        seed: 随机种子

    Returns:
        List[List[str]]: [[rgb1, x1], [rgb2, x2], ...] 格式列表

    Example:
        sources = sample_from_yaml('dataset.yaml', n=5)
        for source in sources:
            model.predict(source=source)
    """
    sampler = MultiModalSampler(yaml_path, split=split, seed=seed)
    return sampler.get_predict_sources(n)


def quick_sample(yaml_path: Union[str, Path], split: str = 'val') -> List[str]:
    """
    快速采样单对图像，返回可直接用于 predict 的格式

    Args:
        yaml_path: 数据集 YAML 配置文件路径
        split: 数据集分割

    Returns:
        List[str]: [rgb_path, x_path] 格式

    Example:
        from ultralytics import YOLOMM
        model = YOLOMM('yolo11n-mm.pt')
        source = quick_sample('dataset.yaml')
        results = model.predict(source=source)
    """
    sampler = MultiModalSampler(yaml_path, split=split)
    return sampler.get_predict_source()


def sample_source(
    dataset_yaml: str,
    split: str = "val",
    seed: int | None = None,
    index: int | None = None,
) -> tuple[str, str]:
    """
    以“库”的方式从多模态数据集 YAML 中采样一对 (rgb_source, x_source)。

    Args:
        dataset_yaml: 多模态数据集 YAML 配置路径
        split: 数据集分割（'train' | 'val' | 'test'，默认：'val'）
        seed: 随机种子（可选）
        index: 指定图像对索引（可选，优先于随机采样）

    Returns:
        (rgb_source, x_source): 两个字符串路径，可直接传入 YOLOMM.predict(rgb_source=..., x_source=...)
    """
    sampler = MultiModalSampler(dataset_yaml, split=split, seed=seed)
    if index is not None:
        return sampler.sample_by_index(index)
    return sampler.sample_one()


def sample_source_list(
    dataset_yaml: str,
    n: int = 1,
    split: str = "val",
    seed: int | None = None,
    replace: bool = False,
) -> tuple[list[str], list[str]]:
    """
    以“库”的方式从多模态数据集 YAML 中采样一组 (RGBlist, Xlist)。

    Args:
        dataset_yaml: 多模态数据集 YAML 配置路径
        n: 采样数量（图片对数量）
        split: 数据集分割（'train' | 'val' | 'test'，默认：'val'）
        seed: 随机种子（可选）
        replace: 是否允许重复采样（默认：False）

    Returns:
        (RGBlist, Xlist): 两个列表按图片对顺序对齐，分别为 RGB 模态与 X 模态的路径字符串列表。
    """
    sampler = MultiModalSampler(dataset_yaml, split=split, seed=seed)
    return sampler.sample_source_list(n=n, replace=replace)
