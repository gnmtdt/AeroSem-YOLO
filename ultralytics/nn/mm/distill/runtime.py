# Ultralytics Multimodal Distillation - Teacher Runtime Coordinator
# Manages teacher model building, freezing, and feature collection.

"""
Teacher runtime coordinator for knowledge distillation.

Responsibilities:
1. Build teacher models according to the student's model family.
2. Freeze teacher parameters (eval + requires_grad_(False)).
3. Register temporary collection hooks for feature-level distillation.
4. Run teacher forward passes and collect outputs/features during training.

The coordinator lives exclusively on the *trainer* -- it is never attached to
the student model object.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from pathlib import Path

from ultralytics.utils import LOGGER

from .schema import DistillConfig, FeatureMappingSpec, OutputTeacherSpec, TeacherSpec


# ---------------------------------------------------------------------------
# Performance limits for atomic feature mappings (internal, not exposed to YAML)
# ---------------------------------------------------------------------------
_FEATURE_MAPPING_WARN_LIMIT = 20
_FEATURE_MAPPING_HARD_LIMIT = 50

# Bridge between schema's upper-case student_input and model's _mm_input_source attribute.
# Schema parser stores 'RGB'/'X'/'DUAL'; MultiModalRouter sets 'RGB'/'X'/'Dual'.
_INPUT_SOURCE_MAP = {"RGB": "RGB", "X": "X", "DUAL": "Dual"}


# ---------------------------------------------------------------------------
# Feature collector (hook-based)
# ---------------------------------------------------------------------------


class _FeatureCollector:
    """Lightweight hook-based feature collector for intermediate layers."""

    def __init__(self):
        self.features: Dict[int, torch.Tensor] = {}
        self._handles: list = []

    def register(self, model: nn.Module, layer_indices: List[int]):
        """Register forward hooks for the given layer indices.

        Args:
            model: The model to attach hooks to.
            layer_indices: Layer indices whose outputs should be captured.
        """
        self.clear()
        for idx in layer_indices:
            if idx < 0 or idx >= len(model.model):
                raise ValueError(
                    f"Layer index {idx} out of range for model with "
                    f"{len(model.model)} layers"
                )
            handle = model.model[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(_module, _input, output):
            self.features[layer_idx] = output
        return hook

    def reset(self):
        """Clear captured features (called before each forward pass)."""
        self.features.clear()

    def clear(self):
        """Remove all hooks and clear features."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.features.clear()


# ---------------------------------------------------------------------------
# Per-teacher runtime state
# ---------------------------------------------------------------------------


class _TeacherRuntime:
    """Runtime state for a single teacher model."""

    def __init__(self, spec: TeacherSpec, model: nn.Module, family: str):
        self.spec = spec
        self.model = model
        self.family = family
        self.collector = _FeatureCollector()

    def register_feature_hooks(self, layer_indices: List[int]):
        """Register collection hooks for feature-level distillation."""
        self.collector.register(self.model, layer_indices)

    @staticmethod
    def _build_teacher_batch(batch: dict, role: str) -> dict:
        """Build teacher-specific input batch by slicing channels according to role.

        Args:
            batch: Training batch dict containing 'img' with shape [B, C, H, W].
                   C is typically 6 (RGB=3 + X=3) for multimodal training.
            role: Teacher role from TeacherSpec -- 'rgb', 'x', or 'dual'.

        Returns:
            New batch dict with 'img' sliced to the role-appropriate channels.
            Other batch keys are shared (shallow copy).

        Raises:
            ValueError: If role is not one of 'rgb', 'x', 'dual'.
        """
        img = batch["img"]
        out = dict(batch)  # shallow copy, share non-img entries

        if role == "rgb":
            out["img"] = img[:, :3]
        elif role == "x":
            out["img"] = img[:, 3:]
        elif role == "dual":
            out["img"] = img  # keep full multi-channel
        else:
            raise ValueError(
                f"Unsupported teacher role: '{role}'. "
                f"Must be one of 'rgb', 'x', 'dual'."
            )
        return out

    @staticmethod
    def _set_teacher_router(model: nn.Module, role: str):
        """Configure teacher's mm_router for the given role before forward pass.

        If the teacher model has a ``mm_router`` (i.e., it was trained with a
        multimodal configuration), we must set its runtime params so that
        internal routing matches the role-driven input we are feeding it.

        - role='rgb'  -> set_runtime_params('rgb')  -- router fills X with zeros
        - role='x'    -> set_runtime_params('x')    -- router fills RGB with zeros
        - role='dual' -> set_runtime_params(None)   -- router uses full dual input

        If the teacher has no mm_router, this is a no-op.

        Args:
            model: Teacher model (may or may not have mm_router).
            role: Teacher role ('rgb', 'x', 'dual').
        """
        router = getattr(model, 'mm_router', None)
        if router is None:
            return

        if role == "rgb":
            router.set_runtime_params("rgb")
        elif role == "x":
            router.set_runtime_params("x")
        elif role == "dual":
            router.set_runtime_params(None)

    @staticmethod
    def _check_input_channels(teacher_img: torch.Tensor, spec: TeacherSpec):
        """Validate that teacher input channels match the role expectation.

        Performs basic role-channel consistency checks. Fails fast on mismatch
        -- no silent fallback, no auto-padding.

        Note: We intentionally do NOT inspect the model's first layer here
        because the model may have a mm_router that dynamically adjusts
        channels. Instead, we validate basic role-channel consistency:

        - role='rgb' -> input must be exactly 3 channels
        - role='dual' -> input must be > 3 channels (multi-modal)
        - role='x' -> no strict check (X modality channel count varies by Xch)

        Common mismatch scenarios:
        - 3-ch single-modal teacher declared as 'dual' (model expects 3ch, got 6ch)
        - 6-ch early-fusion teacher declared as 'rgb' (model expects 6ch, got 3ch)

        Args:
            teacher_img: The role-sliced input tensor [B, C, H, W].
            spec: TeacherSpec with name and role for error messages.

        Raises:
            ValueError: If channel count does not match role expectation.
        """
        actual_ch = teacher_img.shape[1]

        if spec.role == "dual" and actual_ch <= 3:
            raise ValueError(
                f"Teacher '{spec.name}' (role=dual) expects a multi-channel input "
                f"but received {actual_ch}-ch tensor. The training batch may not "
                f"contain concatenated RGB+X data."
            )
        if spec.role == "rgb" and actual_ch != 3:
            raise ValueError(
                f"Teacher '{spec.name}' (role=rgb) expects 3-channel RGB input "
                f"but received {actual_ch}-ch tensor. Check that batch['img'] has "
                f"the expected [RGB|X] channel layout."
            )

    def forward(self, batch: dict) -> torch.Tensor:
        """Run teacher forward and return raw predictions in **train-mode semantics**.

        The teacher is always in ``eval()`` mode with ``requires_grad_(False)``.
        Input is sliced from the multimodal batch according to ``self.spec.role``:

        - ``role='rgb'``  -> teacher receives ``batch["img"][:, :3]`` (RGB only)
        - ``role='x'``    -> teacher receives ``batch["img"][:, 3:]`` (X modality only)
        - ``role='dual'`` -> teacher receives ``batch["img"]`` (full multi-channel)

        If the teacher model has a ``mm_router``, its ``runtime_params`` are set
        accordingly before the forward pass.

        - **YOLOMM** (``DetectionModel``): ``predict(img)`` directly returns the
          raw detection-head output -- same as student ``distill_forward``.
        - **RTDETRMM** (``RTDETRDetectionModel``): in eval mode the head returns
          ``(y_decoded, x_raw_tuple)``.  We extract ``x_raw_tuple`` which is the
          5-element train-mode output ``(dec_bboxes, dec_scores, enc_bboxes,
          enc_scores, dn_meta)``, matching the student ``distill_forward``.

        Feature hooks are collected as a side-effect of the forward pass.
        """
        self.collector.reset()

        # Step 1: Build role-specific input batch
        teacher_batch = self._build_teacher_batch(batch, self.spec.role)
        teacher_img = teacher_batch["img"]

        # Step 2: Channel sanity check -- fail fast on mismatch
        self._check_input_channels(teacher_img, self.spec)

        # Step 3: Set mm_router runtime params if teacher has one
        self._set_teacher_router(self.model, self.spec.role)

        # Step 4: Forward pass
        with torch.no_grad():
            preds = self.model.predict(teacher_img)

            # Unpack to train-mode semantics per family
            if self.family == "rtdetrmm":
                # In eval mode RTDETRDecoder returns (y_decoded, x_raw_tuple).
                # x_raw_tuple is the 5-element train-mode output.
                if isinstance(preds, (tuple, list)) and len(preds) == 2:
                    preds = preds[1]  # raw 5-tuple
        return preds

    def get_features(self) -> Dict[int, torch.Tensor]:
        """Return features collected during the last forward pass."""
        return dict(self.collector.features)

    def cleanup(self):
        """Remove hooks."""
        self.collector.clear()


# ---------------------------------------------------------------------------
# Main coordinator
# ---------------------------------------------------------------------------


class DistillRuntime:
    """Coordinates teacher construction, freezing, and per-batch collection.

    Usage (inside a trainer)::

        runtime = DistillRuntime(config, mode, family, student_model, device)
        # ... in training loop ...
        teacher_outputs = runtime.run_teachers(batch)
        # teacher_outputs is a dict: teacher_name -> TeacherOutput
    """

    def __init__(
        self,
        config: DistillConfig,
        mode: str,
        family: str,
        student_model: nn.Module,
        device: torch.device,
    ):
        """
        Args:
            config: Validated DistillConfig.
            mode: One of 'output', 'feature', 'both'.
            family: 'yolomm' or 'rtdetrmm'.
            student_model: The student model (for reference, not modified).
            device: Target device for teacher models.
        """
        self.config = config
        self.mode = mode
        self.family = family
        self.device = device

        # Determine which mapping groups are active
        self.use_feature = mode in ("feature", "both")
        self.use_output = mode in ("output", "both")

        # Active mappings -- compile high-level specs into atomic mappings
        self.feature_mappings: List[FeatureMappingSpec] = (
            self._compile_feature_mappings(
                raw_mappings=list(config.feature_mappings) if self.use_feature else [],
                student_model=student_model,
            )
        )
        self.output_teachers: List[OutputTeacherSpec] = (
            list(config.output_teachers) if self.use_output else []
        )

        # Build teachers
        self._teachers: Dict[str, _TeacherRuntime] = {}
        self._build_teachers(config, student_model)

        # Register feature hooks if needed
        if self.use_feature:
            self._register_feature_hooks()

        LOGGER.info(
            f"DistillRuntime initialized: family={family}, mode={mode}, "
            f"teachers={list(self._teachers.keys())}, "
            f"feature_mappings={len(self.feature_mappings)}, "
            f"output_teachers={len(self.output_teachers)}"
        )

    # ----- feature mapping compilation --------------------------------------

    def _compile_feature_mappings(
        self,
        raw_mappings: List[FeatureMappingSpec],
        student_model: nn.Module,
    ) -> List[FeatureMappingSpec]:
        """Compile high-level feature mapping specs into atomic layer-index pairs.

        Three legal input forms:
        1. Atomic: teacher_layer=int, student_layer=int -> pass through
        2. Range-to-range: teacher_layer=(a,b), student_layer=(c,d) -> zip expand
        3. Teacher range + student_input: teacher_layer=(a,b), student_input='RGB'
           -> auto-find student branch layers, then zip expand

        Args:
            raw_mappings: Raw feature mapping specs from config (may contain ranges).
            student_model: The student model for branch layer lookup.

        Returns:
            List of atomic FeatureMappingSpec (teacher_layer=int, student_layer=int).

        Raises:
            ValueError: On length mismatch, missing branch, or limit exceeded.
        """
        if not raw_mappings:
            return []

        compiled: List[FeatureMappingSpec] = []

        for i, raw in enumerate(raw_mappings):
            t_layer = raw.teacher_layer
            s_layer = raw.student_layer
            s_input = raw.student_input

            # Explicit student_layer takes priority over student_input
            if s_layer is not None:
                if isinstance(t_layer, int) and isinstance(s_layer, int):
                    # Case A: atomic mapping -- pass through directly
                    compiled.append(raw)

                elif isinstance(t_layer, tuple) and isinstance(s_layer, tuple):
                    # Case B: range-to-range zip expansion
                    t_range = list(range(t_layer[0], t_layer[1] + 1))
                    s_range = list(range(s_layer[0], s_layer[1] + 1))
                    if len(t_range) != len(s_range):
                        raise ValueError(
                            f"Feature mapping [{i}]: teacher range [{t_layer[0]}, {t_layer[1]}] "
                            f"has {len(t_range)} layers but student range [{s_layer[0]}, {s_layer[1]}] "
                            f"has {len(s_range)} layers -- lengths must match."
                        )
                    for tl, sl in zip(t_range, s_range):
                        compiled.append(FeatureMappingSpec(
                            teacher=raw.teacher,
                            teacher_layer=tl,
                            student_layer=sl,
                            student_input=None,
                            tap=raw.tap,
                        ))

                else:
                    # Type mismatch: one is int and the other is tuple
                    raise ValueError(
                        f"Feature mapping [{i}]: teacher_layer and student_layer must be "
                        f"the same type (both int or both range), got "
                        f"teacher_layer={t_layer!r}, student_layer={s_layer!r}."
                    )

            elif s_input is not None:
                # student_input auto-expansion
                if isinstance(t_layer, int):
                    # Case D: single teacher layer cannot combine with auto-expansion
                    raise ValueError(
                        f"Feature mapping [{i}]: atomic teacher_layer={t_layer} cannot "
                        f"be used with student_input='{s_input}'. Use a range "
                        f"teacher_layer=[start, end] for auto-expansion."
                    )

                # Case C: teacher range + student_input auto-expansion
                t_range = list(range(t_layer[0], t_layer[1] + 1))
                branch_layers = self._find_student_branch_layers(student_model, s_input)
                if len(t_range) != len(branch_layers):
                    raise ValueError(
                        f"Feature mapping [{i}]: teacher range [{t_layer[0]}, {t_layer[1]}] "
                        f"has {len(t_range)} layers but student '{s_input}' branch has "
                        f"{len(branch_layers)} layers ({branch_layers}) -- lengths must match."
                    )
                for tl, sl in zip(t_range, branch_layers):
                    compiled.append(FeatureMappingSpec(
                        teacher=raw.teacher,
                        teacher_layer=tl,
                        student_layer=sl,
                        student_input=None,
                        tap=raw.tap,
                    ))

            else:
                # Should not reach here if schema parser guarantees at least one
                raise ValueError(
                    f"Feature mapping [{i}]: both student_layer and student_input are None. "
                    f"At least one must be provided."
                )

        # Post-compilation quantity checks
        total = len(compiled)
        if total > _FEATURE_MAPPING_HARD_LIMIT:
            raise ValueError(
                f"Feature mappings expanded to {total} atomic pairs, "
                f"exceeding hard limit {_FEATURE_MAPPING_HARD_LIMIT}. "
                f"Reduce teacher_layer range or use fewer mappings."
            )
        if total > _FEATURE_MAPPING_WARN_LIMIT:
            LOGGER.warning(
                f"Feature mappings expanded to {total} atomic pairs "
                f"(warning threshold={_FEATURE_MAPPING_WARN_LIMIT}). "
                f"This may significantly slow down training."
            )

        return compiled

    def _find_student_branch_layers(
        self, student_model: nn.Module, student_input: str
    ) -> list[int]:
        """Find contiguous backbone layers in the student model for a given input branch.

        Locates the branch start by matching ``_mm_input_source`` attribute on model
        layers, then collects contiguous layers until a fusion point or different
        branch is encountered.

        Args:
            student_model: Student model with ``model`` attribute (nn.Sequential of layers).
            student_input: Input source identifier ('RGB', 'X', 'DUAL').
                           Stored in upper-case from schema parser.

        Returns:
            Sorted list of layer indices belonging to the requested branch.

        Raises:
            ValueError: If no matching branch is found or the branch is empty.
        """
        # Map schema upper-case to model attribute value
        mapped_source = _INPUT_SOURCE_MAP.get(student_input)
        if mapped_source is None:
            raise ValueError(
                f"Unknown student_input '{student_input}'. "
                f"Valid values: {list(_INPUT_SOURCE_MAP.keys())}."
            )

        # Find the branch start index
        start_idx = None
        for idx, m in enumerate(student_model.model):
            source = getattr(m, "_mm_input_source", None)
            if source == mapped_source:
                start_idx = idx
                break

        if start_idx is None:
            raise ValueError(
                f"No layer with _mm_input_source='{mapped_source}' found in student model. "
                f"student_input='{student_input}' branch does not exist in this model configuration."
            )

        # Collect contiguous branch layers starting from start_idx
        branch_layers: list[int] = []
        for idx in range(start_idx, len(student_model.model)):
            m = student_model.model[idx]
            source = getattr(m, "_mm_input_source", None)

            if idx == start_idx:
                # First layer must match the target source
                branch_layers.append(idx)
                continue

            # Stop if this is a multi-input fusion layer (e.g. Concat)
            if isinstance(m.f, list):
                break

            # Stop if this layer belongs to a different branch
            if source is not None and source != mapped_source:
                break

            # Layer is part of the contiguous chain:
            # either it has matching _mm_input_source, or it has m.f == -1 (sequential from previous)
            if source == mapped_source:
                branch_layers.append(idx)
            elif m.f == -1:
                branch_layers.append(idx)
            else:
                # m.f is a non-negative int referencing a specific layer -- not contiguous
                break

        if not branch_layers:
            raise ValueError(
                f"student_input='{student_input}' branch starting at layer {start_idx} "
                f"yielded no contiguous layers."
            )

        return branch_layers

    # ----- teacher construction --------------------------------------------

    def _build_teachers(self, config: DistillConfig, student_model: nn.Module):
        """Build, load and freeze all teacher models."""
        for spec in config.teachers:
            model = self._load_teacher_model(spec, student_model)
            model.eval()
            model.requires_grad_(False)
            model.to(self.device)
            self._teachers[spec.name] = _TeacherRuntime(spec, model, self.family)
            LOGGER.info(
                f"Teacher '{spec.name}' (role={spec.role}) loaded, frozen, "
                f"moved to {self.device}"
            )

    def _load_teacher_model(self, spec: TeacherSpec, student_model: nn.Module) -> nn.Module:
        """Load a teacher model using the current family's loading mechanism.

        The teacher must be loadable by the same family as the student:
        - YOLOMM student -> teacher loaded via DetectionModel + weights
        - RTDETRMM student -> teacher loaded via RTDETRDetectionModel + weights

        Args:
            spec: Teacher specification with weights (and optional yaml).
            student_model: Student model for reference (family detection).

        Returns:
            Loaded teacher model.
        """
        weights_path = Path(spec.weights)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Teacher '{spec.name}' weights not found: {spec.weights}"
            )

        # Load checkpoint
        from ultralytics.nn.tasks import attempt_load_one_weight

        teacher_model, _ = attempt_load_one_weight(str(weights_path))

        # Validate family compatibility
        if self.family == "yolomm":
            from ultralytics.nn.tasks import DetectionModel
            if not isinstance(teacher_model, DetectionModel):
                raise TypeError(
                    f"Teacher '{spec.name}' is not a DetectionModel; "
                    f"YOLOMM only supports same-family distillation"
                )
        elif self.family == "rtdetrmm":
            from ultralytics.nn.tasks import RTDETRDetectionModel
            if not isinstance(teacher_model, RTDETRDetectionModel):
                raise TypeError(
                    f"Teacher '{spec.name}' is not a RTDETRDetectionModel; "
                    f"RTDETRMM only supports same-family distillation"
                )
        else:
            raise ValueError(f"Unknown family: {self.family}")

        return teacher_model

    # ----- feature hook registration ---------------------------------------

    def _register_feature_hooks(self):
        """Register hooks on teacher models for feature collection."""
        # Group feature mappings by teacher
        teacher_layers: Dict[str, List[int]] = {}
        for m in self.feature_mappings:
            teacher_layers.setdefault(m.teacher, []).append(m.teacher_layer)

        for tname, layers in teacher_layers.items():
            unique_layers = sorted(set(layers))
            self._teachers[tname].register_feature_hooks(unique_layers)
            LOGGER.info(
                f"Teacher '{tname}': registered feature hooks at layers {unique_layers}"
            )

    # ----- student feature hook registration -------------------------------

    def register_student_hooks(self, student_model: nn.Module) -> _FeatureCollector:
        """Register feature collection hooks on the student model.

        Returns a ``_FeatureCollector`` whose ``.features`` dict will be
        populated after each student forward pass.
        """
        collector = _FeatureCollector()
        if self.use_feature:
            student_layers = sorted(set(m.student_layer for m in self.feature_mappings))
            collector.register(student_model, student_layers)
            LOGGER.info(
                f"Student: registered feature hooks at layers {student_layers}"
            )
        return collector

    # ----- per-batch teacher forward ---------------------------------------

    def run_teachers(self, batch: dict) -> Dict[str, "TeacherOutput"]:
        """Run all teacher forward passes and collect outputs.

        Args:
            batch: The training batch dict (must contain 'img').

        Returns:
            Dict mapping teacher name to ``TeacherOutput``.
        """
        results: Dict[str, TeacherOutput] = {}
        for tname, trt in self._teachers.items():
            preds = trt.forward(batch)
            features = trt.get_features() if self.use_feature else {}
            results[tname] = TeacherOutput(
                name=tname,
                role=trt.spec.role,
                preds=preds,
                features=features,
            )
        return results

    # ----- lifecycle -------------------------------------------------------

    def cleanup(self):
        """Remove all hooks and release teacher models."""
        for trt in self._teachers.values():
            trt.cleanup()
        self._teachers.clear()
        LOGGER.info("DistillRuntime cleaned up")

    @property
    def teacher_names(self) -> List[str]:
        return list(self._teachers.keys())


# ---------------------------------------------------------------------------
# Teacher output bundle
# ---------------------------------------------------------------------------


class TeacherOutput:
    """Bundle of outputs from a single teacher forward pass."""

    __slots__ = ("name", "role", "preds", "features")

    def __init__(
        self,
        name: str,
        role: str,
        preds: torch.Tensor,
        features: Dict[int, torch.Tensor],
    ):
        self.name = name
        self.role = role
        self.preds = preds
        self.features = features
