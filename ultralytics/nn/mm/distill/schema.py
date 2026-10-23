# Ultralytics Multimodal Distillation - Schema & Parser
# Defines read-only configuration objects and YAML parsing with fail-fast validation.

"""
Distillation YAML schema, parser and static validation.

The distillation YAML only describes **teacher definitions** and **distillation
mappings**.  It intentionally does NOT encode loss functions, student model
paths, or training hyper-parameters -- those are owned by the trainer.

Two mapping groups with **distinct semantics**:

- ``feature``: Layer-index mappings for intermediate feature distillation.
  Each entry specifies ``teacher_layer`` and ``student_layer`` integers.
- ``output``: Teacher-level output distillation declarations.
  Each entry only declares which teacher participates; **no layer indices**.

Supported YAML structure::

    version: 1

    teachers:
      - name: teacher_rgb
        role: rgb          # rgb | x | dual
        weights: path/to/rgb_teacher.pt
        yaml: path/to/rgb_teacher.yaml   # optional, auxiliary only

      - name: teacher_x
        role: x
        weights: path/to/x_teacher.pt

    mappings:
      feature:
        - teacher: teacher_rgb
          teacher_layer: 6
          student_layer: 6
          tap: output       # optional, defaults to 'output'

      output:
        - teacher: teacher_rgb
        - teacher: teacher_x
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Data structures (read-only after construction)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherSpec:
    """Specification for a single teacher model."""

    name: str
    role: str  # 'rgb' | 'x' | 'dual'
    weights: str  # must be a .pt path
    yaml: Optional[str] = None  # optional auxiliary YAML


@dataclass(frozen=True)
class FeatureMappingSpec:
    """A single layer-index mapping between a teacher and the student.

    Used exclusively by ``mappings.feature`` -- intermediate feature distillation.

    Fields:
    - teacher_layer: int for atomic mapping, tuple[int,int] for closed range [start, end].
    - student_layer: same as teacher_layer, or None when student_input is used.
    - student_input: 'RGB' | 'X' | 'Dual' | None. Auto-expand entry for student backbone.
    - tap: currently only 'output'.
    """

    teacher: str  # teacher name reference
    teacher_layer: int | tuple[int, int]
    student_layer: int | tuple[int, int] | None = None
    student_input: str | None = None
    tap: str = "output"  # currently only 'output' is allowed


# Backward-compatible alias so that existing code importing ``MappingSpec`` still works.
MappingSpec = FeatureMappingSpec


@dataclass(frozen=True)
class OutputTeacherSpec:
    """Declaration of a teacher that participates in output-level distillation.

    Unlike ``FeatureMappingSpec`` this does **not** carry layer indices -- output
    distillation operates on the full detection-head output package.
    """

    teacher: str  # teacher name reference


@dataclass(frozen=True)
class DistillConfig:
    """Fully validated distillation configuration."""

    version: int
    teachers: List[TeacherSpec]
    feature_mappings: List[FeatureMappingSpec] = field(default_factory=list)
    output_teachers: List[OutputTeacherSpec] = field(default_factory=list)

    # --- convenience helpers ------------------------------------------------

    def teacher_by_name(self, name: str) -> TeacherSpec:
        """Retrieve a teacher by name or raise KeyError."""
        for t in self.teachers:
            if t.name == name:
                return t
        raise KeyError(f"Teacher '{name}' not found in distill config")

    @property
    def teacher_names(self) -> List[str]:
        return [t.name for t in self.teachers]


# ---------------------------------------------------------------------------
# Allowed value sets
# ---------------------------------------------------------------------------

_VALID_ROLES = {"rgb", "x", "dual"}
_VALID_TAPS = {"output"}  # v1 only supports 'output'; extend here later
_VALID_STUDENT_INPUTS = {"RGB", "X", "DUAL"}


def _parse_layer_spec(value, field_name: str):
    """Parse a layer specification: int or [start, end] closed range.

    Args:
        value: Raw YAML value (should be int or 2-element list of ints).
        field_name: Field name for error messages.

    Returns:
        int or tuple[int, int].

    Raises:
        ValueError: If value is not a valid layer spec.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value):
        start, end = value
        if start > end:
            raise ValueError(
                f"{field_name} range must satisfy start <= end, got [{start}, {end}]"
            )
        return (start, end)
    raise ValueError(
        f"{field_name} must be int or [start, end] (2-element int list), got {value!r}"
    )


# ---------------------------------------------------------------------------
# YAML parser
# ---------------------------------------------------------------------------


def load_distill_config(yaml_path: str) -> DistillConfig:
    """Parse and validate a distillation YAML file.

    Args:
        yaml_path: Path to the distillation YAML file.

    Returns:
        A validated ``DistillConfig`` instance.

    Raises:
        FileNotFoundError: If *yaml_path* does not exist.
        ValueError: If the YAML structure or field values are invalid.
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Distillation YAML not found: {yaml_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Distillation YAML root must be a mapping, got {type(raw).__name__}")

    # ---- version ----------------------------------------------------------
    version = raw.get("version", 1)
    if not isinstance(version, int):
        raise ValueError(f"'version' must be an integer, got {type(version).__name__}")

    # ---- teachers ---------------------------------------------------------
    raw_teachers = raw.get("teachers")
    if not raw_teachers or not isinstance(raw_teachers, list):
        raise ValueError("'teachers' must be a non-empty list")

    teachers: List[TeacherSpec] = []
    seen_names: set = set()
    for idx, td in enumerate(raw_teachers):
        if not isinstance(td, dict):
            raise ValueError(f"teachers[{idx}] must be a mapping")

        name = td.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"teachers[{idx}].name must be a non-empty string")
        if name in seen_names:
            raise ValueError(f"Duplicate teacher name: '{name}'")
        seen_names.add(name)

        role = td.get("role", "").lower()
        if role not in _VALID_ROLES:
            raise ValueError(
                f"teachers[{idx}].role must be one of {_VALID_ROLES}, got '{role}'"
            )

        weights = td.get("weights")
        if not weights or not isinstance(weights, str):
            raise ValueError(f"teachers[{idx}].weights must be a non-empty string")
        if not weights.endswith(".pt"):
            raise ValueError(
                f"teachers[{idx}].weights must be a .pt file, got '{weights}'"
            )

        yaml_aux = td.get("yaml")
        if yaml_aux is not None and not isinstance(yaml_aux, str):
            raise ValueError(f"teachers[{idx}].yaml must be a string or omitted")

        teachers.append(TeacherSpec(name=name, role=role, weights=weights, yaml=yaml_aux))

    # ---- mappings ---------------------------------------------------------
    raw_mappings = raw.get("mappings", {})
    if not isinstance(raw_mappings, dict):
        raise ValueError("'mappings' must be a mapping (or omitted)")

    feature_mappings = _parse_feature_mapping_group(raw_mappings.get("feature", []), "feature", seen_names)
    output_teachers = _parse_output_teacher_group(raw_mappings.get("output", []), seen_names)

    return DistillConfig(
        version=version,
        teachers=teachers,
        feature_mappings=feature_mappings,
        output_teachers=output_teachers,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_feature_mapping_group(
    raw_list, group_name: str, valid_teacher_names: set
) -> List[FeatureMappingSpec]:
    """Parse and validate the ``feature`` mapping group (layer-index mappings)."""
    if raw_list is None:
        return []
    if not isinstance(raw_list, list):
        raise ValueError(f"mappings.{group_name} must be a list")

    mappings: List[FeatureMappingSpec] = []
    for idx, md in enumerate(raw_list):
        if not isinstance(md, dict):
            raise ValueError(f"mappings.{group_name}[{idx}] must be a mapping")

        teacher_ref = md.get("teacher")
        if not teacher_ref or not isinstance(teacher_ref, str):
            raise ValueError(
                f"mappings.{group_name}[{idx}].teacher must be a non-empty string"
            )
        if teacher_ref not in valid_teacher_names:
            raise ValueError(
                f"mappings.{group_name}[{idx}].teacher '{teacher_ref}' "
                f"not found in teachers list"
            )

        # teacher_layer: required, supports int or [start, end]
        raw_teacher_layer = md.get("teacher_layer")
        if raw_teacher_layer is None:
            raise ValueError(
                f"mappings.{group_name}[{idx}].teacher_layer is required"
            )
        teacher_layer = _parse_layer_spec(
            raw_teacher_layer,
            f"mappings.{group_name}[{idx}].teacher_layer",
        )

        # student_layer: optional, supports int or [start, end]
        raw_student_layer = md.get("student_layer")
        student_layer = None
        if raw_student_layer is not None:
            student_layer = _parse_layer_spec(
                raw_student_layer,
                f"mappings.{group_name}[{idx}].student_layer",
            )

        # student_input: optional, only RGB/X/DUAL allowed
        raw_student_input = md.get("student_input")
        student_input = None
        if raw_student_input is not None:
            if not isinstance(raw_student_input, str):
                raise ValueError(
                    f"mappings.{group_name}[{idx}].student_input must be a string"
                )
            student_input = raw_student_input.upper()
            if student_input not in _VALID_STUDENT_INPUTS:
                raise ValueError(
                    f"mappings.{group_name}[{idx}].student_input must be one of "
                    f"{_VALID_STUDENT_INPUTS}, got '{raw_student_input}'"
                )

        # at least one of student_layer or student_input must be provided
        if student_layer is None and student_input is None:
            raise ValueError(
                f"mappings.{group_name}[{idx}]: at least one of 'student_layer' "
                f"or 'student_input' must be provided"
            )

        tap = md.get("tap", "output")
        if tap not in _VALID_TAPS:
            raise ValueError(
                f"mappings.{group_name}[{idx}].tap must be one of {_VALID_TAPS}, got '{tap}'"
            )

        mappings.append(
            FeatureMappingSpec(
                teacher=teacher_ref,
                teacher_layer=teacher_layer,
                student_layer=student_layer,
                student_input=student_input,
                tap=tap,
            )
        )

    return mappings


def _parse_output_teacher_group(
    raw_list, valid_teacher_names: set
) -> List[OutputTeacherSpec]:
    """Parse and validate the ``output`` teacher-declaration group.

    Fail-fast: if any entry contains ``teacher_layer`` or ``student_layer`` the
    user is using the old layer-index format -- raise immediately.
    """
    if raw_list is None:
        return []
    if not isinstance(raw_list, list):
        raise ValueError("mappings.output must be a list")

    specs: List[OutputTeacherSpec] = []
    for idx, md in enumerate(raw_list):
        if not isinstance(md, dict):
            raise ValueError(f"mappings.output[{idx}] must be a mapping")

        # Fail-fast: reject old layer-index fields in output group
        for forbidden_key in ("teacher_layer", "student_layer", "tap"):
            if forbidden_key in md:
                raise ValueError(
                    f"mappings.output[{idx}] contains '{forbidden_key}' which is "
                    f"not allowed in the output group. Output distillation is a "
                    f"teacher-level declaration -- only 'teacher' is expected. "
                    f"Layer-index fields belong in mappings.feature only."
                )

        teacher_ref = md.get("teacher")
        if not teacher_ref or not isinstance(teacher_ref, str):
            raise ValueError(
                f"mappings.output[{idx}].teacher must be a non-empty string"
            )
        if teacher_ref not in valid_teacher_names:
            raise ValueError(
                f"mappings.output[{idx}].teacher '{teacher_ref}' "
                f"not found in teachers list"
            )

        specs.append(OutputTeacherSpec(teacher=teacher_ref))

    return specs
