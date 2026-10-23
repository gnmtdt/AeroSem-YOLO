# Ultralytics Multimodal Inference - Explicit Pairing Resolver
# Resolves explicit RGB+X inputs into paired sample specifications
# Version: v2.0 (Breaking Change - 废弃隐式source参数)
# Date: 2026-01-13

from pathlib import Path
from typing import List, Dict, Union
from ultralytics.utils import LOGGER
from ultralytics.nn.mm import MultiModalSourceMatcher


class PairingResolver:
    """
    多模态推理显式配对解析器（新API）

    职责：
    - 接收显式的 rgb_source 和 x_source 参数
    - 验证输入合法性和文件存在性
    - 生成统一的配对样本规格

    新API设计：
    - 强制要求同时提供 rgb_source 和 x_source
    - 支持单对、批量推理
    - 不再支持隐式列表 [rgb, x] 格式（breaking change）
    """

    def __init__(self, x_modality: str = "unknown", verbose: bool = True):
        """
        初始化配对解析器

        Args:
            x_modality: X模态类型名称（如 'thermal', 'depth', 'ir'等）
            verbose: 是否输出详细日志
        """
        self.x_modality = x_modality
        self.verbose = verbose

    def resolve(
        self,
        rgb_source: Union[str, Path, List[Union[str, Path]], None] = None,
        x_source: Union[str, Path, List[Union[str, Path]], None] = None,
        strict_match: bool = True
    ) -> List[Dict[str, Union[str, Path, None]]]:
        """
        解析显式RGB和X模态输入为配对样本列表（支持单模态推理）

        Args:
            rgb_source: RGB图像源（可为None表示缺失）
                - 单图: '/path/to/rgb.jpg' 或 Path('/path/to/rgb.jpg')
                - 目录: '/path/to/rgb_dir'（自动扫描目录内所有图像文件）
                - 批量: ['/path/rgb1.jpg', '/path/rgb2.jpg']
                - 缺失: None (将使用零填充)
            x_source: X模态图像源（可为None表示缺失）
                - 单图: '/path/to/thermal.jpg'
                - 目录: '/path/to/thermal_dir'
                - 批量: ['/path/thermal1.jpg', '/path/thermal2.jpg']
                - 缺失: None (将使用零填充)
            strict_match: 批量推理时的匹配策略（默认严格）
                - True: 严格模式，目录必须完全匹配，列表必须等长
                - False: 宽松模式，目录取交集，列表仍要求等长

        Returns:
            配对样本规格列表 [
                {
                    'id': 'sample_001',
                    'rgb_path': Path('/path/rgb.jpg') or None,
                    'x_path': Path('/path/thermal.jpg') or None,
                    'x_modality': 'thermal'
                },
                ...
            ]

        Raises:
            ValueError: 输入格式不合法或数量不匹配
            FileNotFoundError: 文件不存在
        """
        # --- 单模态推理处理 ---
        if rgb_source is None and x_source is not None:
            return self._resolve_single_modality(x_source, modality="x")

        if rgb_source is not None and x_source is None:
            return self._resolve_single_modality(rgb_source, modality="rgb")

        # --- 双模态推理 ---
        # 目录批量：source=[rgb_dir, x_dir]
        if isinstance(rgb_source, (str, Path)) and isinstance(x_source, (str, Path)):
            rgb_p, x_p = Path(rgb_source), Path(x_source)
            if rgb_p.is_dir() and x_p.is_dir():
                matcher = MultiModalSourceMatcher(rgb_source, x_source, strict_match=strict_match)
                pairs = matcher.match()
                samples = [
                    self._create_sample_spec(rgb_path=rp, x_path=xp, sample_idx=i)
                    for i, (rp, xp) in enumerate(pairs)
                ]
                if self.verbose:
                    LOGGER.info(f"目录批量配对完成: {len(samples)} 对有效样本")
                return samples

        # 列表批量：rgb_source=[...], x_source=[...]
        if isinstance(rgb_source, list) and isinstance(x_source, list):
            pairs = MultiModalSourceMatcher.match_lists(rgb_source, x_source, strict_match=strict_match)
            samples = [
                self._create_sample_spec(rgb_path=rp, x_path=xp, sample_idx=i)
                for i, (rp, xp) in enumerate(pairs)
            ]
            if self.verbose:
                LOGGER.info(f"列表批量配对完成: {len(samples)} 对有效样本")
            return samples

        # 单对推理（str/Path对str/Path）: 使用现有逻辑
        rgb_list = self._normalize_to_list(rgb_source, "rgb_source")
        x_list = self._normalize_to_list(x_source, "x_source")

        # 验证数量匹配
        if len(rgb_list) != len(x_list):
            raise ValueError(
                f"RGB和X模态数量不匹配：\n"
                f"  rgb_source: {len(rgb_list)} 张\n"
                f"  x_source: {len(x_list)} 张\n"
                f"请确保两者数量相同。"
            )

        # 配对并验证
        samples = []
        for idx, (rgb_path, x_path) in enumerate(zip(rgb_list, x_list)):
            samples.append(self._create_sample_spec(
                rgb_path=Path(rgb_path),
                x_path=Path(x_path),
                sample_idx=idx
            ))

        if self.verbose:
            LOGGER.info(f"双模态配对完成: {len(samples)} 对有效样本")

        return samples

    def _resolve_single_modality(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        modality: str
    ) -> List[Dict[str, Union[str, Path, None]]]:
        """
        解析单模态输入（支持目录扫描）

        Args:
            source: 图像源（单个、目录或列表）
            modality: 'rgb' 或 'x'
        """
        # 目录输入：扫描目录内所有图像文件
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            src_dir = Path(source)
            files = sorted([
                p for p in src_dir.iterdir()
                if p.is_file() and p.suffix.lower() in MultiModalSourceMatcher.SUPPORTED_FORMATS
            ])
            if not files:
                modality_name = "RGB" if modality == "rgb" else self.x_modality
                raise ValueError(f"{modality_name}目录中未找到受支持的图像文件: {src_dir}")
        else:
            files = self._normalize_to_list(source, f"{modality}_source")

        samples = []
        for idx, path in enumerate(files):
            if modality == "rgb":
                samples.append(self._create_sample_spec(rgb_path=Path(path), x_path=None, sample_idx=idx))
            else:
                samples.append(self._create_sample_spec(rgb_path=None, x_path=Path(path), sample_idx=idx))

        if self.verbose:
            if modality == "rgb":
                LOGGER.info(f"单RGB模态推理: {len(samples)} 个样本（{self.x_modality}将使用零填充）")
            else:
                LOGGER.info(f"单{self.x_modality}模态推理: {len(samples)} 个样本（RGB将使用零填充）")

        return samples

    def _normalize_to_list(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        param_name: str
    ) -> List[Path]:
        """
        统一输入为列表格式

        Args:
            source: 输入源（单个或列表）
            param_name: 参数名称（用于错误提示）

        Returns:
            Path对象列表
        """
        if isinstance(source, (str, Path)):
            return [Path(source)]
        elif isinstance(source, list):
            if not source:
                raise ValueError(f"{param_name} 不能为空列表")
            return [Path(item) for item in source]
        else:
            raise ValueError(
                f"{param_name} 类型不支持: {type(source)}\n"
                f"支持类型: str, Path, List[str], List[Path]"
            )

    def _create_sample_spec(
        self,
        rgb_path: Union[Path, None],
        x_path: Union[Path, None],
        sample_idx: int
    ) -> Dict[str, Union[str, Path, None]]:
        """
        创建配对样本规格并验证文件存在性（支持None占位）

        Args:
            rgb_path: RGB图像路径（可为None表示缺失）
            x_path: X模态图像路径（可为None表示缺失）
            sample_idx: 样本索引（用于生成ID）

        Returns:
            样本规格字典

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 路径不是文件
        """
        # 验证RGB文件（如果提供）
        if rgb_path is not None:
            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB文件不存在: {rgb_path}")
            if not rgb_path.is_file():
                raise ValueError(f"RGB路径不是文件: {rgb_path}")

        # 验证X模态文件（如果提供）
        if x_path is not None:
            if not x_path.exists():
                raise FileNotFoundError(f"X模态文件不存在: {x_path}")
            if not x_path.is_file():
                raise ValueError(f"X模态路径不是文件: {x_path}")

        # 生成样本ID（优先使用RGB文件名，否则使用X模态文件名）
        if rgb_path is not None:
            sample_id = rgb_path.stem
        elif x_path is not None:
            sample_id = x_path.stem
        else:
            sample_id = f"sample_{sample_idx:03d}"

        return {
            "id": sample_id,
            "rgb_path": rgb_path,
            "x_path": x_path,
            "x_modality": self.x_modality
        }
