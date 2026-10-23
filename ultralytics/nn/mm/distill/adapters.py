# Ultralytics Multimodal Distillation - Family-level Adapters
# Bridges between model-specific outputs and the unified distillation loss API.

"""
Family-level distillation adapters for YOLOMM and RTDETRMM.

Each adapter:
1. Standardises the student ``distill_forward()`` result into a student bundle.
2. Standardises teacher outputs + features into teacher bundles.
3. Delegates to the **family-specific** output distillation module for loss
   computation (NOT a shared flatten-MSE).
4. Delegates to the **family-specific** feature guidance/mask module for
   foreground-guided feature distillation.
5. Aggregates the final distillation loss contribution.

Architecture:
- ``_BaseDistillAdapter``: defines the ``compute_distill_loss`` API, handles
  feature distillation orchestration (guidance cache, mask generation, shared
  4-term loss via ``losses.py``).  Both ``_compute_output_loss`` and the
  family-specific ``_build_feature_guidance`` / ``_build_feature_mask`` are
  abstract -- subclasses MUST override.
- ``YOLOMMDetectDistillAdapter``: delegates output distillation to
  ``output_yolomm.compute_yolomm_output_kd``; delegates feature guidance to
  ``feature_yolomm.build_yolomm_feature_guidance`` /
  ``feature_yolomm.build_yolomm_feature_mask``.
- ``RTDETRMMDetectDistillAdapter``: delegates output distillation to
  ``output_rtdetrmm.compute_rtdetr_output_kd``; delegates feature guidance to
  ``feature_rtdetrmm.build_rtdetr_feature_guidance`` /
  ``feature_rtdetrmm.build_rtdetr_feature_mask``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER

from .losses import compute_feature_distill_loss
from .runtime import TeacherOutput, DistillRuntime, _FeatureCollector
from .schema import DistillConfig, FeatureMappingSpec, OutputTeacherSpec


class ChannelAdapter(nn.Module):
    """Learnable 1x1 Conv for teacher->student channel projection.

    Used when teacher and student feature maps have different channel counts
    in cross-scale distillation scenarios (e.g. teacher s-scale 256ch ->
    student n-scale 128ch).

    Args:
        in_channels: Teacher feature channel count.
        out_channels: Student feature channel count.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project teacher features to student channel dimension."""
        return self.conv(x)


class _BaseDistillAdapter:
    """Base adapter shared between YOLOMM and RTDETRMM.

    Provides shared feature-distillation logic and the ``compute_distill_loss``
    API.  Output distillation is delegated to family-specific subclasses via
    ``_compute_output_loss`` which MUST be overridden.

    Channel adaptation: when teacher and student feature maps have different
    channel counts, a learnable ``ChannelAdapter`` (1x1 Conv) is created
    lazily on first encounter and its parameters are appended to the trainer's
    optimizer.
    """

    def __init__(
        self,
        runtime: DistillRuntime,
        config: DistillConfig,
        student_model: nn.Module = None,
        distill_weight: float = 1.0,
        feature_weight: float = 1.0,
        output_weight: float = 1.0,
        trainer=None,
    ):
        self.runtime = runtime
        self.config = config
        self.student_model = student_model
        self._trainer = trainer

        # Read distill weights from trainer.args, falling back to explicit params
        # then to default.yaml values.
        args = getattr(trainer, 'args', None)
        _a = lambda k, d: getattr(args, k, d) if args is not None else d

        # Overall multipliers
        self.distill_weight = _a('distill_weight', distill_weight)
        self.feature_weight = _a('distill_feature_weight', feature_weight)
        self.output_weight = _a('distill_output_weight', output_weight)

        # Warmup epochs
        self.feature_warmup = _a('distill_feature_warmup', 10)
        self.output_warmup = _a('distill_output_warmup', 5)

        # Feature distillation sub-weights
        self.fg_weight = _a('distill_feature_fg', 1.0)
        self.bg_weight = _a('distill_feature_bg', 0.25)
        self.cwd_weight = _a('distill_feature_cwd', 0.5)
        self.ctx_weight = _a('distill_feature_ctx', 0.25)
        self.cwd_tau = _a('distill_feature_tau', 1.0)

        # Output distillation sub-weights
        self.output_cls_weight = _a('distill_output_cls', 1.0)
        self.output_loc_weight = _a('distill_output_loc', 1.0)
        self.output_overall = _a('distill_output_overall', 0.5)

        # Training state (updated by trainer each epoch)
        self._current_epoch: int = 0
        self._total_epochs: int = 100

        # Channel adapters: mapping_key -> ChannelAdapter (lazy-created)
        self._channel_adapters: Dict[str, ChannelAdapter] = {}

    def set_epoch_state(self, current_epoch: int, total_epochs: int):
        """Update training epoch state (called by trainer at each epoch start)."""
        self._current_epoch = current_epoch
        self._total_epochs = total_epochs

    def _compute_warmup_factor(self, warmup_epochs: int) -> float:
        """Linear warmup ramp from 0 to 1 over ``warmup_epochs``."""
        if warmup_epochs <= 0:
            return 1.0
        return min(1.0, self._current_epoch / warmup_epochs)

    # ----- channel adapter management ----------------------------------------

    def _get_or_create_adapter(
        self,
        mapping_key: str,
        teacher_ch: int,
        student_ch: int,
        device: torch.device,
    ) -> Optional[ChannelAdapter]:
        """Get or lazily create a channel adapter for a feature mapping pair.

        Returns ``None`` when channels already match (no adaptation needed).

        Args:
            mapping_key: Unique identifier for this mapping pair
                (e.g. ``"teacher_rgb_T6_S6"``).
            teacher_ch: Teacher feature channel count.
            student_ch: Student feature channel count.
            device: Device to place the adapter on.

        Returns:
            ``ChannelAdapter`` instance, or ``None`` if channels match.
        """
        if teacher_ch == student_ch:
            return None

        if mapping_key not in self._channel_adapters:
            adapter = ChannelAdapter(teacher_ch, student_ch).to(device)
            self._channel_adapters[mapping_key] = adapter
            LOGGER.info(
                f"Channel adapter created for {mapping_key}: "
                f"{teacher_ch} -> {student_ch} channels"
            )
            self._append_adapter_to_optimizer(adapter)

        return self._channel_adapters[mapping_key]

    def _append_adapter_to_optimizer(self, adapter: ChannelAdapter):
        """Append channel adapter parameters to the trainer's optimizer.

        Uses a dedicated param_group with the current base learning rate and
        zero weight decay (1x1 Conv projection should not be regularised).
        """
        if self._trainer is None or not hasattr(self._trainer, 'optimizer'):
            LOGGER.warning(
                "Channel adapter created but no trainer/optimizer available "
                "to register parameters. Adapter will NOT be optimised."
            )
            return
        optimizer = self._trainer.optimizer
        if optimizer is None:
            LOGGER.warning(
                "Channel adapter created but optimizer is None. "
                "Adapter will NOT be optimised."
            )
            return
        base_lr = optimizer.param_groups[0]['lr']
        optimizer.add_param_group({
            'params': list(adapter.parameters()),
            'lr': base_lr,
            'weight_decay': 0.0,
        })
        LOGGER.info(
            f"Channel adapter params appended to optimizer "
            f"(lr={base_lr:.6f}, weight_decay=0.0)"
        )

    def compute_distill_loss(
        self,
        student_preds,
        student_features: Dict[int, torch.Tensor],
        teacher_outputs: Dict[str, TeacherOutput],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total distillation loss.

        Args:
            student_preds: Raw student detection-head predictions.
            student_features: Student intermediate features (layer_idx -> tensor).
            teacher_outputs: Per-teacher outputs from ``runtime.run_teachers()``.

        Returns:
            (total_distill_loss, loss_items_dict) where *loss_items_dict* maps
            descriptive names to scalar tensors for logging.
        """
        device = _get_device(student_preds)
        total_loss = torch.tensor(0.0, device=device)
        items: Dict[str, torch.Tensor] = {}

        # ---- output distillation (family-specific) ---------------------------
        # Output warmup is handled inside the family-specific KD function
        # (output_yolomm / output_rtdetrmm), so we do NOT apply it here.
        if self.runtime.use_output:
            out_loss, out_items = self._compute_output_loss(
                student_preds, teacher_outputs, device
            )
            total_loss = total_loss + self.output_weight * out_loss
            items["d_out"] = out_loss.detach()
            items.update(out_items)

        # ---- feature distillation (family-aware foreground-guided) -----------
        # Feature warmup is applied HERE (it had none before).
        if self.runtime.use_feature:
            feat_loss, feat_items = self._compute_feature_loss(
                student_features, teacher_outputs, device
            )
            feature_warmup = self._compute_warmup_factor(self.feature_warmup)
            total_loss = total_loss + self.feature_weight * feature_warmup * feat_loss
            items["distill_feature"] = feat_loss.detach()
            items.update(feat_items)

        total_loss = total_loss * self.distill_weight
        return total_loss, items

    # ----- output distillation (MUST be overridden) ----------------------------

    def _compute_output_loss(
        self,
        student_preds,
        teacher_outputs: Dict[str, TeacherOutput],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute output-level distillation loss (family-specific).

        Subclasses MUST override this method to implement family-specific
        output distillation.  The base class does NOT fall back to a generic
        flatten-MSE -- it raises ``NotImplementedError``.

        Returns:
            (output_loss, sub_items_dict)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must override _compute_output_loss(). "
            f"Generic flatten-MSE has been retired from the output distillation "
            f"main path."
        )

    # ----- feature distillation (family-aware foreground-guided) -------------

    def _build_feature_guidance(
        self,
        teacher_name: str,
        teacher_output: TeacherOutput,
    ) -> object:
        """Build family-specific feature guidance from teacher output.

        Subclasses MUST override this method to call their family's
        guidance builder (feature_yolomm or feature_rtdetrmm).

        Args:
            teacher_name: Teacher identifier string.
            teacher_output: Teacher forward pass result (preds + features).

        Returns:
            Family-specific guidance object (YOLOMMFeatureGuidance or
            RTDETRMMFeatureGuidance).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must override _build_feature_guidance()."
        )

    def _build_feature_mask(
        self,
        guidance,
        feature_shape: tuple,
    ) -> torch.Tensor:
        """Build foreground mask from guidance for the given feature shape.

        Subclasses MUST override this method to call their family's
        mask builder.

        Args:
            guidance: Family-specific guidance object.
            feature_shape: (B, C, H, W) of the target feature map.

        Returns:
            Soft foreground mask (B, 1, H, W), values in [0, 1].
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must override _build_feature_mask()."
        )

    def _get_input_size(self) -> Optional[tuple]:
        """Get original input image size from runtime configuration.

        Returns (img_h, img_w) or None if unknown.
        """
        return getattr(self.runtime, '_input_size', None)

    def _build_feature_guidance_cache(
        self,
        teacher_outputs: Dict[str, TeacherOutput],
    ) -> Dict[str, object]:
        """Build guidance cache: one guidance per unique teacher, reused across mappings.

        Iterates over feature mappings, collects unique teacher names, and calls
        ``_build_feature_guidance()`` once per teacher.

        Args:
            teacher_outputs: Per-teacher forward pass results.

        Returns:
            dict mapping teacher name -> guidance object.
        """
        cache: Dict[str, object] = {}
        for mapping in self.runtime.feature_mappings:
            t_name = mapping.teacher
            if t_name in cache:
                continue
            t_out = teacher_outputs.get(t_name)
            if t_out is None:
                continue
            cache[t_name] = self._build_feature_guidance(t_name, t_out)
        return cache

    def _compute_feature_loss(
        self,
        student_features: Dict[int, torch.Tensor],
        teacher_outputs: Dict[str, TeacherOutput],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute feature-level distillation loss with foreground guidance.

        Upgraded from global MSE to family-aware foreground-guided loss:
        1. Build guidance cache (one per unique teacher).
        2. For each mapping, generate fg_mask from guidance.
        3. Call shared 4-term feature loss from losses.py.
        4. Aggregate losses and sub-items across all mappings.

        Returns:
            (aggregated_feature_loss, sub_items_dict)
        """
        # Step 1: Build guidance cache
        guidance_cache = self._build_feature_guidance_cache(teacher_outputs)

        # Determine unique teacher count for prefix logic
        unique_teachers = set(m.teacher for m in self.runtime.feature_mappings)
        multi_teacher = len(unique_teachers) > 1

        losses: List[torch.Tensor] = []
        agg_items: Dict[str, List[torch.Tensor]] = {}

        # Step 2: Per-mapping loss
        for mapping in self.runtime.feature_mappings:
            t_out = teacher_outputs.get(mapping.teacher)
            if t_out is None:
                continue
            t_feat = t_out.features.get(mapping.teacher_layer)
            s_feat = student_features.get(mapping.student_layer)
            if t_feat is None or s_feat is None:
                LOGGER.warning(
                    f"Missing feature for mapping {mapping.teacher}:"
                    f"T[{mapping.teacher_layer}]->S[{mapping.student_layer}], skipped"
                )
                continue

            # Channel adaptation: project teacher features to student channel
            # dimension if they differ (lazy-create 1x1 Conv adapter)
            if t_feat.dim() == 4 and s_feat.dim() == 4:
                mapping_key = (
                    f"{mapping.teacher}_T{mapping.teacher_layer}"
                    f"_S{mapping.student_layer}"
                )
                adapter = self._get_or_create_adapter(
                    mapping_key, t_feat.shape[1], s_feat.shape[1], t_feat.device,
                )
                if adapter is not None:
                    t_feat = adapter(t_feat.detach())

            # Build fg_mask for this mapping
            guidance = guidance_cache.get(mapping.teacher)
            if guidance is None:
                LOGGER.warning(
                    f"No guidance available for teacher {mapping.teacher}, "
                    f"feature mapping skipped"
                )
                continue
            fg_mask = self._build_feature_mask(guidance, t_feat.shape)

            # Call shared feature loss with configured sub-weights
            loss, sub_items = compute_feature_distill_loss(
                s_feat, t_feat, fg_mask,
                bg_weight=self.bg_weight,
                cwd_weight=self.cwd_weight,
                ctx_weight=self.ctx_weight,
            )
            losses.append(loss)

            # Accumulate sub-items for averaging (filter out "total" key to
            # avoid duplicate with items["distill_feature"])
            prefix = f"{mapping.teacher}/" if multi_teacher else ""
            for k, v in sub_items.items():
                if k == "total":
                    continue
                key = f"{prefix}distill_feature_{k}"
                agg_items.setdefault(key, []).append(v)

        # Step 3: Aggregate
        if not losses:
            return torch.tensor(0.0, device=device), {}

        mean_loss = sum(losses) / len(losses)

        # Average sub-items across mappings
        merged: Dict[str, torch.Tensor] = {}
        for k, v_list in agg_items.items():
            merged[k] = sum(v_list) / len(v_list)

        return mean_loss, merged


# ---------------------------------------------------------------------------
# YOLOMM adapter
# ---------------------------------------------------------------------------


class YOLOMMDetectDistillAdapter(_BaseDistillAdapter):
    """Distillation adapter for YOLOMM detection models.

    Output distillation: foreground-guided + cls/loc decoupled.
    Delegates to ``output_yolomm.compute_yolomm_output_kd``.

    Feature guidance: foreground mask from teacher dense detection output.
    Delegates to ``feature_yolomm.build_yolomm_feature_guidance`` /
    ``feature_yolomm.build_yolomm_feature_mask``.
    """

    def _build_feature_guidance(self, teacher_name, teacher_output):
        """Build YOLOMM foreground guidance from teacher detection output."""
        from .feature_yolomm import build_yolomm_feature_guidance

        teacher_model = self.runtime._teachers[teacher_name].model
        return build_yolomm_feature_guidance(
            teacher_preds=teacher_output.preds,
            teacher_model=teacher_model,
        )

    def _build_feature_mask(self, guidance, feature_shape):
        """Generate YOLOMM soft foreground mask."""
        from .feature_yolomm import build_yolomm_feature_mask

        return build_yolomm_feature_mask(
            guidance=guidance,
            feature_shape=feature_shape,
            input_size=self._get_input_size(),
        )

    def _compute_output_loss(
        self,
        student_preds,
        teacher_outputs: Dict[str, TeacherOutput],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute YOLOMM-specific output distillation (per-teacher, then mean)."""
        from .output_yolomm import compute_yolomm_output_kd

        all_losses: List[torch.Tensor] = []
        merged_items: Dict[str, torch.Tensor] = {}

        for ot_spec in self.runtime.output_teachers:
            t_out = teacher_outputs.get(ot_spec.teacher)
            if t_out is None:
                continue

            # Get teacher model from runtime for head access
            teacher_model = self.runtime._teachers[ot_spec.teacher].model

            loss, items = compute_yolomm_output_kd(
                student_preds=student_preds,
                teacher_preds=t_out.preds,
                student_model=self.student_model,
                teacher_model=teacher_model,
                current_epoch=self._current_epoch,
                total_epochs=self._total_epochs,
                cls_weight=self.output_cls_weight,
                loc_weight=self.output_loc_weight,
                overall_weight=self.output_overall,
                warmup_epochs=self.output_warmup,
            )
            all_losses.append(loss)

            # Merge sub-items (prefix with teacher name if multiple)
            n_teachers = len(self.runtime.output_teachers)
            prefix = f"{ot_spec.teacher}/" if n_teachers > 1 else ""
            for k, v in items.items():
                merged_items[f"{prefix}{k}"] = v

        if not all_losses:
            return torch.tensor(0.0, device=device), {}
        return sum(all_losses) / len(all_losses), merged_items


# ---------------------------------------------------------------------------
# RTDETRMM adapter
# ---------------------------------------------------------------------------


class RTDETRMMDetectDistillAdapter(_BaseDistillAdapter):
    """Distillation adapter for RTDETRMM detection models.

    Output distillation: matching-aware / query-aware.
    Delegates to ``output_rtdetrmm.compute_rtdetr_output_kd``.

    Feature guidance: query-aware foreground mask from teacher DETR decoder output.
    Delegates to ``feature_rtdetrmm.build_rtdetr_feature_guidance`` /
    ``feature_rtdetrmm.build_rtdetr_feature_mask``.
    """

    def _build_feature_guidance(self, teacher_name, teacher_output):
        """Build RTDETRMM foreground guidance from teacher decoder output."""
        from .feature_rtdetrmm import build_rtdetr_feature_guidance

        return build_rtdetr_feature_guidance(
            teacher_preds=teacher_output.preds,
            input_size=self._get_input_size(),
        )

    def _build_feature_mask(self, guidance, feature_shape):
        """Generate RTDETRMM soft foreground mask."""
        from .feature_rtdetrmm import build_rtdetr_feature_mask

        return build_rtdetr_feature_mask(
            guidance=guidance,
            feature_shape=feature_shape,
            input_size=self._get_input_size(),
        )

    def _compute_output_loss(
        self,
        student_preds,
        teacher_outputs: Dict[str, TeacherOutput],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute RTDETRMM-specific output distillation (per-teacher, then mean)."""
        from .output_rtdetrmm import compute_rtdetr_output_kd

        all_losses: List[torch.Tensor] = []
        merged_items: Dict[str, torch.Tensor] = {}

        for ot_spec in self.runtime.output_teachers:
            t_out = teacher_outputs.get(ot_spec.teacher)
            if t_out is None:
                continue

            loss, items = compute_rtdetr_output_kd(
                student_preds=student_preds,
                teacher_preds=t_out.preds,
                current_epoch=self._current_epoch,
                total_epochs=self._total_epochs,
                cls_weight=self.output_cls_weight,
                box_weight=self.output_loc_weight,
                overall_weight=self.output_overall,
                warmup_epochs=self.output_warmup,
            )
            all_losses.append(loss)

            # Merge sub-items (prefix with teacher name if multiple)
            n_teachers = len(self.runtime.output_teachers)
            prefix = f"{ot_spec.teacher}/" if n_teachers > 1 else ""
            for k, v in items.items():
                merged_items[f"{prefix}{k}"] = v

        if not all_losses:
            return torch.tensor(0.0, device=device), {}
        return sum(all_losses) / len(all_losses), merged_items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device(preds) -> torch.device:
    """Extract device from predictions (tensor or tuple/list of tensors)."""
    if isinstance(preds, torch.Tensor):
        return preds.device
    if isinstance(preds, (tuple, list)):
        for p in preds:
            if isinstance(p, torch.Tensor):
                return p.device
    return torch.device("cpu")
