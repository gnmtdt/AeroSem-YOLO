"""
MultiModal Dataset Splitter - 多模态数据集诊断与拆分工具

功能：
1) 扫描诊断：读取多模态 dataset YAML，按项目规范检查目录结构、模态可用性、样本缺失/冲突等问题。
2) 拆分导出：将一个多模态数据集按模态拆分成多个单模态 YOLO 标准数据集（images/labels + data.yaml），采用 copy 策略。

重要约束（发行版级）：
- 拆分仅依赖输入 YAML 的 `modality` 字段决定有哪些模态可拆；不做任何自动推断/降级逻辑。
- `modality_suffix` 不参与拆分匹配，仅在拆分后的单模态 data.yaml 中写入作为标识字段。
- labels 只有一套：以 `labels/<split>` 为样本真值索引；拆分后 labels 文件逐字节 copy。
- 图像匹配：对每个 label stem，在 `dataset_root / modality[mod] / <split>` 下必须找到“同 stem 的唯一图像文件”，
  若 0 个则缺失，>1 则冲突（视为数据构建错误）。
- 当某个模态存在结构错误/缺失/冲突时，该模态整体跳过导出，并在报告中给出原因与例子。

Usage:
    # 扫描诊断（可选输出 JSON）
    python -m ultralytics.tools.mm_dataset_splitter --yaml /path/to/data.yaml scan
    python -m ultralytics.tools.mm_dataset_splitter --yaml /path/to/data.yaml scan --json /tmp/report.json

    # 拆分导出（copy）
    python -m ultralytics.tools.mm_dataset_splitter --yaml /path/to/data.yaml split --out /tmp/mm_split_out
    python -m ultralytics.tools.mm_dataset_splitter --yaml /path/to/data.yaml split --out /tmp/mm_split_out --splits train,val,test
    python -m ultralytics.tools.mm_dataset_splitter --yaml /path/to/data.yaml split --out /tmp/mm_split_out --modalities rgb,ir,depth

    # 交互式（类 TUI，stdin 选择）
    python -m ultralytics.tools.mm_dataset_splitter --yaml /path/to/data.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MAX_EXAMPLES_DEFAULT = 20


class DatasetYamlError(ValueError):
    """Raised when dataset YAML is invalid or missing required fields."""


class DatasetStructureError(RuntimeError):
    """Raised when dataset structure is invalid in a way that blocks scanning/splitting."""


@dataclass
class SplitStats:
    split: str
    labels: int
    found: int
    missing: int
    conflicts: int
    examples_missing: List[str]
    examples_conflicts: List[List[str]]


@dataclass
class ModalityReport:
    modality: str
    status: str  # "ok" | "skipped"
    reason: Optional[str]
    per_split: List[SplitStats]


@dataclass
class ScanReport:
    yaml_path: str
    dataset_root: str
    splits: List[str]
    modalities: List[ModalityReport]
    errors: List[str]
    warnings: List[str]


def load_dataset_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Load dataset yaml as dict (strict)."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset YAML 不存在: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise DatasetYamlError(f"dataset YAML 解析结果必须是 dict，当前为: {type(data)}")
    return data


def resolve_dataset_root(data: Dict[str, Any], yaml_path: Path) -> Path:
    """Resolve dataset root path from YAML `path` (relative -> based on YAML directory)."""
    if "path" not in data:
        raise DatasetYamlError("dataset YAML 缺少必需字段 `path`")
    root = Path(str(data["path"]))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root


def resolve_splits(data: Dict[str, Any]) -> List[str]:
    """Resolve split directory names from YAML keys train/val/test (uses last path segment)."""
    splits: List[str] = []
    for k in ("train", "val", "test"):
        if k not in data:
            continue
        p = Path(str(data[k]))
        if not p.parts:
            continue
        splits.append(p.parts[-1])
    if not splits:
        raise DatasetYamlError("dataset YAML 未找到 train/val/test 任一字段，无法确定 splits")
    return splits


def resolve_modality_map(data: Dict[str, Any]) -> Dict[str, str]:
    """Resolve `modality` mapping (strict)."""
    modality = data.get("modality", None)
    if modality is None or not isinstance(modality, dict) or not modality:
        raise DatasetYamlError("dataset YAML 缺少必需字段 `modality`（且必须为非空 dict）")
    out: Dict[str, str] = {}
    for k, v in modality.items():
        mk = str(k).strip()
        mv = str(v).strip()
        if not mk or not mv:
            continue
        out[mk] = mv
    if not out:
        raise DatasetYamlError("dataset YAML 字段 `modality` 为空或无有效条目")
    return out


def _resolve_split_key_to_name(data: Dict[str, Any]) -> Dict[str, str]:
    """Return mapping like {'train': 'train', 'val': 'val', 'test': 'test'} based on YAML paths' last segment."""
    out: Dict[str, str] = {}
    for k in ("train", "val", "test"):
        if k not in data:
            continue
        p = Path(str(data[k]))
        if p.parts:
            out[k] = p.parts[-1]
    return out


def iter_label_stems(dataset_root: Path, split: str) -> List[str]:
    """Return sorted label stems from dataset_root/labels/<split>."""
    labels_dir = dataset_root / "labels" / split
    if not labels_dir.exists():
        raise DatasetStructureError(f"labels split 目录不存在: {labels_dir}")
    stems = sorted(p.stem for p in labels_dir.glob("*.txt"))
    return stems


def _index_images_by_stem(modality_split_dir: Path) -> Dict[str, List[Path]]:
    """Index images in a directory by stem, only for allowed extensions."""
    if not modality_split_dir.exists():
        raise DatasetStructureError(f"模态 split 目录不存在: {modality_split_dir}")
    index: Dict[str, List[Path]] = {}
    for p in modality_split_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        index.setdefault(p.stem, []).append(p)
    return index


def find_unique_image(index: Dict[str, List[Path]], stem: str) -> Tuple[Optional[Path], Optional[List[Path]]]:
    """
    Find unique image for a given stem.

    Returns:
        (path, None) if found exactly one;
        (None, None) if missing;
        (None, candidates) if conflict (>1).
    """
    candidates = index.get(stem, [])
    if not candidates:
        return None, None
    if len(candidates) > 1:
        return None, sorted(candidates)
    return candidates[0], None


def scan_dataset(
    yaml_path: Path,
    *,
    splits: Optional[List[str]] = None,
    modalities: Optional[List[str]] = None,
    max_examples: int = MAX_EXAMPLES_DEFAULT,
) -> ScanReport:
    data = load_dataset_yaml(yaml_path)
    dataset_root = resolve_dataset_root(data, yaml_path)
    if not dataset_root.exists():
        raise DatasetStructureError(f"dataset root 不存在: {dataset_root}")

    yaml_splits = resolve_splits(data)
    if splits is None:
        splits = yaml_splits
    else:
        unknown = [s for s in splits if s not in yaml_splits]
        if unknown:
            raise DatasetYamlError(f"请求检查的 splits 不在 YAML 中: {unknown}（YAML splits={yaml_splits}）")

    modality_map = resolve_modality_map(data)
    if modalities is None:
        modalities = sorted(modality_map.keys())
    else:
        unknown = [m for m in modalities if m not in modality_map]
        if unknown:
            raise DatasetYamlError(f"请求拆分/检查的 modalities 不在 YAML 的 modality 映射中: {unknown}")

    errors: List[str] = []
    warnings: List[str] = []
    modality_reports: List[ModalityReport] = []

    # labels 作为真值索引：若任一 split 缺 labels 目录，属于阻断性错误
    label_stems_by_split: Dict[str, List[str]] = {}
    for split in splits:
        label_stems_by_split[split] = iter_label_stems(dataset_root, split)

    for mod in modalities:
        per_split: List[SplitStats] = []
        mod_dir = modality_map[mod]
        mod_has_error = False
        mod_reason_parts: List[str] = []

        for split in splits:
            stems = label_stems_by_split[split]
            labels_count = len(stems)

            modality_split_dir = dataset_root / mod_dir / split
            if not modality_split_dir.exists():
                mod_has_error = True
                missing = labels_count
                per_split.append(
                    SplitStats(
                        split=split,
                        labels=labels_count,
                        found=0,
                        missing=missing,
                        conflicts=0,
                        examples_missing=stems[:max_examples],
                        examples_conflicts=[],
                    )
                )
                mod_reason_parts.append(f"{split}:目录缺失")
                continue

            index = _index_images_by_stem(modality_split_dir)
            found = 0
            missing = 0
            conflicts = 0
            examples_missing: List[str] = []
            examples_conflicts: List[List[str]] = []

            for stem in stems:
                p, conflict = find_unique_image(index, stem)
                if p is not None:
                    found += 1
                    continue
                if conflict is None:
                    missing += 1
                    if len(examples_missing) < max_examples:
                        examples_missing.append(stem)
                else:
                    conflicts += 1
                    if len(examples_conflicts) < max_examples:
                        examples_conflicts.append([str(x) for x in conflict])

            if missing or conflicts:
                mod_has_error = True
                parts = []
                if missing:
                    parts.append(f"缺失{missing}")
                if conflicts:
                    parts.append(f"冲突{conflicts}")
                mod_reason_parts.append(f"{split}:{' '.join(parts)}")

            per_split.append(
                SplitStats(
                    split=split,
                    labels=labels_count,
                    found=found,
                    missing=missing,
                    conflicts=conflicts,
                    examples_missing=examples_missing,
                    examples_conflicts=examples_conflicts,
                )
            )

        if mod_has_error:
            status = "skipped"
            reason = "；".join(mod_reason_parts) if mod_reason_parts else "模态存在错误"
        else:
            status = "ok"
            reason = None

        modality_reports.append(ModalityReport(modality=mod, status=status, reason=reason, per_split=per_split))

    return ScanReport(
        yaml_path=str(yaml_path),
        dataset_root=str(dataset_root),
        splits=list(splits),
        modalities=modality_reports,
        errors=errors,
        warnings=warnings,
    )


def _report_to_dict(report: ScanReport) -> Dict[str, Any]:
    return {
        "yaml_path": report.yaml_path,
        "dataset_root": report.dataset_root,
        "splits": report.splits,
        "errors": report.errors,
        "warnings": report.warnings,
        "modalities": [
            {
                "modality": m.modality,
                "status": m.status,
                "reason": m.reason,
                "per_split": [
                    {
                        "split": s.split,
                        "labels": s.labels,
                        "found": s.found,
                        "missing": s.missing,
                        "conflicts": s.conflicts,
                        "examples_missing": s.examples_missing,
                        "examples_conflicts": s.examples_conflicts,
                    }
                    for s in m.per_split
                ],
            }
            for m in report.modalities
        ],
    }


def render_report_text(report: ScanReport) -> str:
    lines: List[str] = []
    lines.append("多模态数据集扫描诊断报告")
    lines.append(f"- YAML: {report.yaml_path}")
    lines.append(f"- Root: {report.dataset_root}")
    lines.append(f"- Splits: {', '.join(report.splits)}")
    lines.append("")

    if report.errors:
        lines.append("错误：")
        for e in report.errors:
            lines.append(f"- {e}")
        lines.append("")

    if report.warnings:
        lines.append("警告：")
        for w in report.warnings:
            lines.append(f"- {w}")
        lines.append("")

    for m in report.modalities:
        title = f"[{m.modality}] {m.status.upper()}"
        if m.reason:
            title += f" - {m.reason}"
        lines.append(title)
        for s in m.per_split:
            lines.append(
                f"  - {s.split}: labels={s.labels} found={s.found} missing={s.missing} conflicts={s.conflicts}"
            )
            if s.examples_missing:
                ex = ", ".join(s.examples_missing[: min(5, len(s.examples_missing))])
                lines.append(f"    - missing例: {ex}")
            if s.examples_conflicts:
                lines.append("    - conflict例:")
                for c in s.examples_conflicts[: min(3, len(s.examples_conflicts))]:
                    lines.append(f"      - {c}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def dump_report_json(report: ScanReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _report_to_dict(report)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_dataset(
    yaml_path: Path,
    out_dir: Path,
    *,
    splits: Optional[List[str]] = None,
    modalities: Optional[List[str]] = None,
) -> ScanReport:
    report = scan_dataset(yaml_path, splits=splits, modalities=modalities)
    data = load_dataset_yaml(yaml_path)
    dataset_root = Path(report.dataset_root)
    modality_map = resolve_modality_map(data)
    split_key_to_name = _resolve_split_key_to_name(data)

    out_dir.mkdir(parents=True, exist_ok=True)

    ok_modalities = [m for m in report.modalities if m.status == "ok"]
    for m in ok_modalities:
        mod = m.modality
        target_root = out_dir / mod
        if target_root.exists() and any(target_root.iterdir()):
            raise DatasetStructureError(f"输出目录已存在且非空，拒绝覆盖: {target_root}")

        for split in report.splits:
            (target_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (target_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        # 逐 split 拷贝
        for split in report.splits:
            stems = iter_label_stems(dataset_root, split)

            src_labels_dir = dataset_root / "labels" / split
            src_images_dir = dataset_root / modality_map[mod] / split
            index = _index_images_by_stem(src_images_dir)

            for stem in stems:
                src_label = src_labels_dir / f"{stem}.txt"
                dst_label = target_root / "labels" / split / f"{stem}.txt"
                shutil.copy2(src_label, dst_label)

                img_path, conflict = find_unique_image(index, stem)
                if img_path is None:
                    if conflict is not None:
                        raise DatasetStructureError(
                            f"模态 {mod} split {split} 出现同 stem 多文件冲突，"
                            f"stem={stem} candidates={[str(x) for x in conflict]}"
                        )
                    raise DatasetStructureError(f"模态 {mod} split {split} 缺失图像，stem={stem}")

                dst_img = target_root / "images" / split / img_path.name
                shutil.copy2(img_path, dst_img)

        # 写入单模态 data.yaml（相对路径 + 标识字段）
        out_yaml: Dict[str, Any] = {}
        out_yaml["path"] = "."
        for k, split_name in split_key_to_name.items():
            if split_name in report.splits:
                out_yaml[k] = f"images/{split_name}"
        if "names" in data:
            out_yaml["names"] = data["names"]
        if "nc" in data:
            out_yaml["nc"] = data["nc"]

        # 仅标识字段：不参与任何拆分/匹配逻辑
        out_yaml["modality_suffix"] = {mod: ""}
        out_yaml["mm_meta"] = {
            "source_yaml": str(yaml_path),
            "source_root": str(dataset_root),
            "source_modality": mod,
        }

        (target_root / "data.yaml").write_text(
            yaml.safe_dump(out_yaml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    return report


def _parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items or None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="mm_dataset_splitter", add_help=True)
    parser.add_argument("--yaml", dest="yaml_path", required=True, help="多模态 dataset YAML 路径")

    sub = parser.add_subparsers(dest="command")
    p_scan = sub.add_parser("scan", help="扫描诊断多模态数据集结构与模态可用性")
    p_scan.add_argument("--json", dest="json_out", default=None, help="可选：输出 JSON 报告路径")

    p_split = sub.add_parser("split", help="按模态拆分为单模态 YOLO 数据集（copy）")
    p_split.add_argument("--out", dest="out_dir", required=True, help="输出目录（会在其下创建 rgb/ir/depth/...）")
    p_split.add_argument("--splits", dest="splits", default=None, help="逗号分隔：train,val,test")
    p_split.add_argument("--modalities", dest="modalities", default=None, help="逗号分隔：rgb,ir,depth")

    args = parser.parse_args(argv)

    if args.command is None:
        # 交互式（类 TUI）：只用 stdin/out，不引入额外依赖
        yaml_path = Path(args.yaml_path)
        while True:
            print("多模态数据集工具（诊断/拆分）")
            print("1) 扫描诊断")
            print("2) 拆分为单模态 YOLO 数据集（copy）")
            print("3) 退出")
            choice = input("请选择 (1/2/3): ").strip()
            if choice == "1":
                report = scan_dataset(yaml_path)
                print(render_report_text(report))
                continue
            if choice == "2":
                out_dir = Path(input("输出目录（会创建 rgb/ir/... 子目录）: ").strip())
                raw_splits = input("splits（逗号分隔，留空=按YAML全部）: ").strip()
                raw_mods = input("modalities（逗号分隔，留空=按YAML全部）: ").strip()
                ss = _parse_csv_list(raw_splits) if raw_splits else None
                ms = _parse_csv_list(raw_mods) if raw_mods else None
                report = split_dataset(yaml_path, out_dir, splits=ss, modalities=ms)
                print(render_report_text(report))
                print(f"拆分完成，输出目录: {out_dir}")
                continue
            if choice == "3":
                return 0
            print("无效选择，请重试。\n")

    yaml_path = Path(args.yaml_path)
    if args.command == "scan":
        report = scan_dataset(yaml_path)
        print(render_report_text(report))
        if getattr(args, "json_out", None):
            dump_report_json(report, Path(args.json_out))
        return 0

    if args.command == "split":
        out_dir = Path(args.out_dir)
        ss = _parse_csv_list(args.splits)
        ms = _parse_csv_list(args.modalities)
        report = split_dataset(yaml_path, out_dir, splits=ss, modalities=ms)
        print(render_report_text(report))
        print(f"拆分完成，输出目录: {out_dir}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
