# Ultralytics Multimodal Source Matcher
# Utility for matching RGB and X-modal image pairs in batch inference
# Version: v1.0

"""
MultiModalSourceMatcher - Batch Inference Image Pair Matching Utility

This module provides utilities for matching multi-modal image pairs from separate
directories for batch inference in YOLOMM/RTDETRMM.

Features:
- Directory scanning with supported image format filtering
- Stem-based matching (ignores file extensions)
- Strict and lenient matching modes
- List-based matching for pre-collected file paths
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

from ultralytics.utils import LOGGER


class MultiModalSourceMatcher:
    """
    Utility class for matching RGB and X-modal image pairs from two directories.

    Matches files by stem (filename without extension), supporting various image formats.
    Provides both directory-based and list-based matching methods.

    Attributes:
        SUPPORTED_FORMATS: Set of supported image file extensions.
        rgb_source: Path to RGB images directory.
        x_source: Path to X-modal images directory.
        strict_match: Whether to require exact matching between directories.

    Example:
        >>> matcher = MultiModalSourceMatcher('path/to/rgb', 'path/to/thermal')
        >>> pairs = matcher.match()
        >>> for rgb_path, x_path in pairs:
        ...     print(f"RGB: {rgb_path}, X: {x_path}")
    """

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(
        self,
        rgb_source: Union[str, Path],
        x_source: Union[str, Path],
        strict_match: bool = True
    ):
        """
        Initialize the MultiModalSourceMatcher.

        Args:
            rgb_source: Path to directory containing RGB images.
            x_source: Path to directory containing X-modal images.
            strict_match: If True, requires all files in both directories to have matches.
                         If False, warns about unmatched files but continues with matched pairs.

        Raises:
            FileNotFoundError: If either directory does not exist.
            ValueError: If either path is not a directory.
        """
        self.rgb_source = Path(rgb_source)
        self.x_source = Path(x_source)
        self.strict_match = strict_match

        self._validate_directories()

    def _validate_directories(self) -> None:
        """
        Validate that both source paths exist and are directories.

        Raises:
            FileNotFoundError: If either directory does not exist.
            ValueError: If either path is not a directory.
        """
        # Check RGB source
        if not self.rgb_source.exists():
            raise FileNotFoundError(f"RGB source directory does not exist: {self.rgb_source}")
        if not self.rgb_source.is_dir():
            raise ValueError(f"RGB source is not a directory: {self.rgb_source}")

        # Check X source
        if not self.x_source.exists():
            raise FileNotFoundError(f"X-modal source directory does not exist: {self.x_source}")
        if not self.x_source.is_dir():
            raise ValueError(f"X-modal source is not a directory: {self.x_source}")

    def _scan_directory(self, folder: Path) -> Dict[str, Path]:
        """
        Scan a directory for supported image files.

        Scans only the top level of the directory (non-recursive).
        Returns a mapping from file stem to full path.

        Args:
            folder: Path to the directory to scan.

        Returns:
            Dictionary mapping file stems to their full paths.
            If duplicate stems exist (different extensions), warns and keeps the first.
        """
        stem_to_path: Dict[str, Path] = {}

        for item in folder.iterdir():
            # Skip directories and non-image files
            if not item.is_file():
                continue

            suffix = item.suffix.lower()
            if suffix not in self.SUPPORTED_FORMATS:
                continue

            stem = item.stem
            if stem in stem_to_path:
                LOGGER.warning(
                    f"Duplicate stem '{stem}' found in {folder}: "
                    f"keeping {stem_to_path[stem].name}, skipping {item.name}"
                )
            else:
                stem_to_path[stem] = item

        return stem_to_path

    def match(self) -> List[Tuple[Path, Path]]:
        """
        Match RGB and X-modal images by file stem.

        Scans both directories and matches files based on their stems
        (filename without extension).

        Returns:
            List of (rgb_path, x_path) tuples, sorted by filename.

        Raises:
            ValueError: If either directory is empty.
            ValueError: If no matching pairs are found.
            ValueError: If strict_match=True and there are unmatched files.
        """
        # Scan both directories
        rgb_files = self._scan_directory(self.rgb_source)
        x_files = self._scan_directory(self.x_source)

        # Check for empty directories
        if not rgb_files:
            raise ValueError(f"No supported image files found in RGB directory: {self.rgb_source}")
        if not x_files:
            raise ValueError(f"No supported image files found in X-modal directory: {self.x_source}")

        # Compute intersection and differences
        rgb_stems = set(rgb_files.keys())
        x_stems = set(x_files.keys())

        matched_stems = rgb_stems & x_stems
        rgb_only = rgb_stems - x_stems
        x_only = x_stems - rgb_stems

        # Check for no matches
        if not matched_stems:
            raise ValueError(
                f"No matching image pairs found between directories. "
                f"RGB has {len(rgb_stems)} files, X-modal has {len(x_stems)} files."
            )

        # Handle unmatched files
        if rgb_only or x_only:
            unmatched_msg_parts = []
            if rgb_only:
                rgb_list = sorted(rgb_only)[:10]  # Show first 10
                suffix = f"... and {len(rgb_only) - 10} more" if len(rgb_only) > 10 else ""
                unmatched_msg_parts.append(f"RGB-only ({len(rgb_only)}): {rgb_list}{suffix}")
            if x_only:
                x_list = sorted(x_only)[:10]
                suffix = f"... and {len(x_only) - 10} more" if len(x_only) > 10 else ""
                unmatched_msg_parts.append(f"X-only ({len(x_only)}): {x_list}{suffix}")

            unmatched_msg = "; ".join(unmatched_msg_parts)

            if self.strict_match:
                raise ValueError(
                    f"Strict matching enabled but found unmatched files. {unmatched_msg}"
                )
            else:
                LOGGER.warning(
                    f"Found unmatched files (continuing with {len(matched_stems)} matched pairs). "
                    f"{unmatched_msg}"
                )

        # Build sorted result
        pairs = [
            (rgb_files[stem], x_files[stem])
            for stem in sorted(matched_stems)
        ]

        LOGGER.info(f"Matched {len(pairs)} image pairs from RGB and X-modal directories")

        return pairs

    @classmethod
    def match_lists(
        cls,
        rgb_list: List[Union[str, Path]],
        x_list: List[Union[str, Path]],
        strict_match: bool = True
    ) -> List[Tuple[Path, Path]]:
        """
        Match two lists of file paths by position.

        Unlike directory matching, this method matches files by their position
        in the lists, not by stem. Lists must have the same length.

        Args:
            rgb_list: List of paths to RGB images.
            x_list: List of paths to X-modal images.
            strict_match: If True, validates that all files exist.

        Returns:
            List of (rgb_path, x_path) tuples.

        Raises:
            ValueError: If lists have different lengths.
            FileNotFoundError: If strict_match=True and any file does not exist.
        """
        # Convert to Path objects
        rgb_paths = [Path(p) for p in rgb_list]
        x_paths = [Path(p) for p in x_list]

        # Validate list lengths
        if len(rgb_paths) != len(x_paths):
            raise ValueError(
                f"List lengths do not match: RGB has {len(rgb_paths)} files, "
                f"X-modal has {len(x_paths)} files."
            )

        if not rgb_paths:
            raise ValueError("Empty file lists provided")

        # Validate file existence if strict
        if strict_match:
            missing_files = []

            for i, rgb_path in enumerate(rgb_paths):
                if not rgb_path.exists():
                    missing_files.append(f"RGB[{i}]: {rgb_path}")

            for i, x_path in enumerate(x_paths):
                if not x_path.exists():
                    missing_files.append(f"X[{i}]: {x_path}")

            if missing_files:
                missing_str = ", ".join(missing_files[:5])
                suffix = f"... and {len(missing_files) - 5} more" if len(missing_files) > 5 else ""
                raise FileNotFoundError(
                    f"Files not found: {missing_str}{suffix}"
                )

        # Build pairs
        pairs = list(zip(rgb_paths, x_paths))

        LOGGER.info(f"Created {len(pairs)} image pairs from provided lists")

        return pairs
