"""
Preflight pre-check engine for YOLOMM/RTDETRMM models.

Synthetic-only preflight. This module validates model graph / forward / loss / backward
using synthetic tensors. It does not read dataset samples and must not infer or report
multimodal precision contracts. Dataset precision probing belongs to
ultralytics.data.multimodal.precision_contract.
"""

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from ultralytics.nn.tasks import guess_model_task
from ultralytics.nn.mm.utils import resolve_input_source_role
from ultralytics.utils import LOGGER, colorstr


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PreflightStageError(Exception):
    """Raised when a preflight stage fails, carrying a human-readable suggestion."""

    def __init__(self, message: str, suggestion: str = ""):
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Successful stage result carrying arbitrary data."""

    data: dict = field(default_factory=dict)


@dataclass
class StageReport:
    """Report for a single preflight stage."""

    name: str
    passed: bool = False
    data: dict = field(default_factory=dict)
    error_message: str = ""
    error_suggestion: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "name": self.name,
            "passed": self.passed,
            "data": self.data,
            "error_message": self.error_message,
            "error_suggestion": self.error_suggestion,
        }


@dataclass
class PreflightReport:
    """Aggregated report for the entire preflight run."""

    model_path: str = ""
    stages: List[StageReport] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def ok(self) -> bool:
        """Return True if every stage passed."""
        return all(s.passed for s in self.stages)

    @property
    def failed_stage(self) -> Optional[str]:
        """Return the name of the first failed stage, or None."""
        for s in self.stages:
            if not s.passed:
                return s.name
        return None

    def pass_stage(self, name: str, result: StageResult) -> None:
        """Record a passing stage."""
        self.stages.append(StageReport(name=name, passed=True, data=result.data))

    def fail_stage(self, name: str, error: PreflightStageError) -> None:
        """Record a failing stage."""
        self.stages.append(
            StageReport(
                name=name,
                passed=False,
                error_message=error.message,
                error_suggestion=error.suggestion,
            )
        )

    def warn(self, msg: str) -> None:
        """Append a non-fatal warning."""
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "model_path": self.model_path,
            "ok": self.ok,
            "failed_stage": self.failed_stage,
            "stages": [s.to_dict() for s in self.stages],
            "warnings": self.warnings,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_s": round(self.end_time - self.start_time, 3),
        }

    def summary(self) -> str:
        """Return a human-readable, color-tagged text summary."""
        lines: List[str] = []
        header = colorstr("bold", "Preflight Report")
        lines.append(f"\n{header}")
        lines.append(f"  Model: {self.model_path}")
        elapsed = round(self.end_time - self.start_time, 3)
        lines.append(f"  Elapsed: {elapsed}s")
        lines.append("")

        for s in self.stages:
            tag = colorstr("green", "bold", "[PASS]") if s.passed else colorstr("red", "bold", "[FAIL]")
            lines.append(f"  {tag} {s.name}")
            if not s.passed:
                lines.append(f"         Error: {s.error_message}")
                if s.error_suggestion:
                    lines.append(f"         Hint : {s.error_suggestion}")

        if self.warnings:
            lines.append("")
            warn_tag = colorstr("yellow", "bold", "[WARNINGS]")
            lines.append(f"  {warn_tag}")
            for w in self.warnings:
                lines.append(f"    - {w}")

        status = colorstr("green", "bold", "ALL PASSED") if self.ok else colorstr("red", "bold", "FAILED")
        lines.append(f"\n  Result: {status}\n")
        return "\n".join(lines)


@dataclass
class PreflightConfig:
    """Configuration knobs for the preflight runner."""

    model: str = ""
    iters: int = 1
    device: str = "cpu"
    batch: int = 2
    imgsz: int = 640
    task: str = "detect"
    scale: str = ""
    Xch: int = 3
    x_modality: str = "depth"
    verbose: bool = True
    half: bool = False


# ---------------------------------------------------------------------------
# Synthetic label factory
# ---------------------------------------------------------------------------

def _synthetic_detect_labels(
    batch_size: int,
    nc: int,
    device: torch.device,
    num_objects: int = 3,
) -> Dict[str, torch.Tensor]:
    """
    Create synthetic detection labels for a single batch.

    Returns a dict with keys: batch_idx, cls, bboxes.
    Each image contains *num_objects* randomly placed objects.
    """
    total = batch_size * num_objects
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, num_objects)
    batch_idx = batch_idx.reshape(-1).float()

    cls = torch.randint(0, max(nc, 1), (total,), device=device).float()

    # Normalised centre-x, centre-y, width, height in [0.1, 0.9]
    bboxes = torch.rand(total, 4, device=device) * 0.8 + 0.1

    return {
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
    }


def _synthetic_obb_labels(
    batch_size: int,
    nc: int,
    device: torch.device,
    num_objects: int = 3,
) -> Dict[str, torch.Tensor]:
    """Create synthetic OBB labels with 5-D bboxes [cx, cy, w, h, angle]."""
    total = batch_size * num_objects
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, num_objects)
    batch_idx = batch_idx.reshape(-1).float()

    cls = torch.randint(0, max(nc, 1), (total,), device=device).float()

    # [cx, cy, w, h] in [0.1, 0.9], angle in [-pi/4, pi/4]
    bboxes_xywh = torch.rand(total, 4, device=device) * 0.8 + 0.1
    angles = (torch.rand(total, 1, device=device) - 0.5) * (math.pi / 2)
    bboxes = torch.cat([bboxes_xywh, angles], dim=1)  # [N, 5]

    return {
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
    }


def _synthetic_segment_labels(
    batch_size: int,
    nc: int,
    device: torch.device,
    imgsz: int = 640,
    num_objects: int = 3,
) -> Dict[str, torch.Tensor]:
    """Create synthetic segmentation labels (detect + random binary masks)."""
    labels = _synthetic_detect_labels(batch_size, nc, device, num_objects)
    total = batch_size * num_objects

    # Random binary masks [N, H, W]
    mask_h, mask_w = imgsz // 4, imgsz // 4  # downsampled mask size
    masks = (torch.rand(total, mask_h, mask_w, device=device) > 0.5).float()

    labels["masks"] = masks
    return labels


def _synthetic_pose_labels(
    batch_size: int,
    nc: int,
    device: torch.device,
    num_keypoints: int = 17,
    num_objects: int = 3,
) -> Dict[str, torch.Tensor]:
    """Create synthetic pose labels (detect + random keypoints [N, K, 3])."""
    labels = _synthetic_detect_labels(batch_size, nc, device, num_objects)
    total = batch_size * num_objects

    # Keypoints: [x, y, visibility] — visibility=2 means visible
    kpts = torch.rand(total, num_keypoints, 3, device=device)
    kpts[:, :, 0] = kpts[:, :, 0] * 0.8 + 0.1  # x in [0.1, 0.9]
    kpts[:, :, 1] = kpts[:, :, 1] * 0.8 + 0.1  # y in [0.1, 0.9]
    kpts[:, :, 2] = 2.0  # visible

    labels["keypoints"] = kpts
    return labels


def _synthetic_classify_labels(
    batch_size: int,
    nc: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create synthetic classification labels (class index per image)."""
    cls = torch.randint(0, max(nc, 1), (batch_size,), device=device).long()
    return {"cls": cls}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class PreflightRunner:
    """
    Execute the 6-stage preflight validation pipeline.

    Stages 1-3 are preparation (run once).
    Stages 4-6 are training steps (repeated *iters* times).
    """

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        self.cfg = PreflightConfig(model=model_path, **kwargs)
        if not self.cfg.model:
            raise ValueError("PreflightRunner requires a model path (YAML or .pt file)")
        _dev = self.cfg.device
        if _dev not in ("cpu", "mps") and not _dev.startswith("cuda"):
            _dev = f"cuda:{_dev}"
        self.device = torch.device(_dev)
        # Populated during stages
        self.yaml_cfg: dict = {}
        self.fusion_strategy: str = "none"
        self.input_channels: int = 3
        self.model: Optional[torch.nn.Module] = None
        self.batch: dict = {}
        self._loss_result: Optional[tuple] = None
        self._prev_gpu_mem: Optional[float] = None

    # -- public API ----------------------------------------------------------

    def run(self) -> PreflightReport:
        """Execute all preflight stages and return the report."""
        report = PreflightReport(
            model_path=self.cfg.model,
            start_time=time.time(),
        )

        # Stage 1-3: preparation (once)
        self._exec(report, "YAML parse", self._stage_yaml_parse)
        if report.ok:
            self._exec(report, "Model build", self._stage_model_build)
        if report.ok:
            self._exec(report, "Synthetic data", self._stage_synthetic_data)

        # Stage 4-6: training iterations
        if report.ok:
            for i in range(self.cfg.iters):
                self._loss_result = None
                self._exec(report, "Forward pass", self._stage_forward)
                if not report.ok:
                    break
                self._exec(report, "Loss compute", self._stage_loss)
                if not report.ok:
                    break
                self._exec(report, "Backward pass", self._stage_backward)
                if not report.ok:
                    break
                self._check_gpu_leak(report, i)

        self._finalize(report)
        return report

    # -- stage runners -------------------------------------------------------

    def _stage_yaml_parse(self) -> StageResult:
        """Stage 1: Validate YAML structure and infer fusion strategy."""
        path = Path(self.cfg.model)
        if not path.exists():
            raise PreflightStageError(
                f"Config file not found: {path}",
                "Check the model path and ensure the YAML file exists.",
            )

        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            raise PreflightStageError(
                "YAML root is not a mapping",
                "Ensure the YAML file contains a top-level dict with backbone/head keys.",
            )

        for key in ("backbone", "head"):
            if key not in raw:
                raise PreflightStageError(
                    f"Missing '{key}' section in YAML",
                    f"Add a '{key}' list to the configuration file.",
                )

        # Scan 5th field for multimodal routing tokens
        has_dual = False
        has_rgb = False
        has_x = False
        invalid_modes: List[str] = []

        for section_name in ("backbone", "head"):
            for idx, layer in enumerate(raw.get(section_name, [])):
                if not isinstance(layer, (list, tuple)) or len(layer) < 5:
                    continue
                mode = layer[4]
                if mode is None:
                    continue
                mode_str = str(mode)
                role = resolve_input_source_role(mode_str)
                if role == "FUSED":
                    has_dual = True
                elif role == "PRIMARY":
                    has_rgb = True
                elif role == "SECONDARY":
                    has_x = True
                else:
                    invalid_modes.append(f"{section_name}[{idx}]={mode_str}")

        if invalid_modes:
            raise PreflightStageError(
                f"Invalid 5th-field values: {', '.join(invalid_modes)}",
                "Valid values are: RGB, X, Dual, or None.",
            )

        # Infer fusion strategy
        if has_dual and not has_rgb and not has_x:
            fusion_strategy = "early"
            input_channels = 3 + self.cfg.Xch
        elif has_rgb and has_x:
            fusion_strategy = "mid"
            input_channels = 3 + self.cfg.Xch
        elif has_dual and (has_rgb or has_x):
            fusion_strategy = "mixed"
            input_channels = 3 + self.cfg.Xch
        else:
            fusion_strategy = "none"
            input_channels = 3

        self.yaml_cfg = raw
        self.fusion_strategy = fusion_strategy
        self.input_channels = input_channels

        # Validate and inject scale selection
        scale = self.cfg.scale
        if scale:
            scales_dict = raw.get("scales")
            if not scales_dict:
                raise PreflightStageError(
                    f"scale='{scale}' was requested, but the YAML does not define a 'scales' section.",
                    "Add a 'scales' mapping to the YAML or omit the --scale argument.",
                )
            if scale not in scales_dict:
                raise PreflightStageError(
                    f"Invalid scale='{scale}'. Valid keys: {list(scales_dict)}.",
                    f"Use one of: {', '.join(scales_dict.keys())}.",
                )
            self.yaml_cfg["scale"] = scale

        return StageResult(data={
            "fusion_strategy": fusion_strategy,
            "input_channels": input_channels,
            "nc": raw.get("nc", "unknown"),
            "scale": scale or "default",
        })

    def _stage_model_build(self) -> StageResult:
        """Stage 2: Build the model from parsed YAML using the correct task class."""
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.nn.tasks import (
            ClassificationModel,
            DetectionModel,
            OBBModel,
            PoseModel,
            RTDETRDetectionModel,
            SegmentationModel,
        )
        from ultralytics.utils import IterableSimpleNamespace

        task = self.cfg.task
        if task == "auto":
            try:
                task = guess_model_task(self.yaml_cfg) or "detect"
            except Exception:
                task = "detect"

        # Auto-detect RTDETR architecture by scanning head for RTDETRDecoder
        is_rtdetr = any(
            layer[2] == "RTDETRDecoder"
            for layer in self.yaml_cfg.get("head", [])
            if isinstance(layer, (list, tuple)) and len(layer) >= 3
        )

        if is_rtdetr and task == "detect":
            cls = RTDETRDetectionModel
        else:
            task_to_cls = {
                "detect": DetectionModel,
                "segment": SegmentationModel,
                "classify": ClassificationModel,
                "pose": PoseModel,
                "obb": OBBModel,
            }
            cls = task_to_cls.get(task, DetectionModel)

        model = cls(cfg=deepcopy(self.yaml_cfg), ch=self.input_channels, verbose=False)
        model.to(self.device)

        if self.cfg.half:
            model.half()

        # Attach minimal args so that criterion (v8DetectionLoss etc.) can
        # read hyper-parameters via getattr(model.args, ...).
        overrides = {
            "imgsz": self.cfg.imgsz,
            "batch": self.cfg.batch,
        }
        model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **overrides})

        # NaN weight check
        nan_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
        if nan_params:
            raise PreflightStageError(
                f"NaN weights detected in: {', '.join(nan_params[:5])}",
                "This usually indicates a broken module __init__. "
                "Check custom module weight initialisation.",
            )

        num_params = sum(p.numel() for p in model.parameters())
        self.model = model

        return StageResult(data={
            "parameters": num_params,
            "task": task,
            "model_class": cls.__name__,
        })

    def _stage_synthetic_data(self) -> StageResult:
        """Stage 3: Build a synthetic batch for forward/backward."""
        B = self.cfg.batch
        C = self.input_channels
        H = self.cfg.imgsz
        W = self.cfg.imgsz
        nc = self.yaml_cfg.get("nc", 1)

        dtype = torch.float16 if self.cfg.half else torch.float32
        img = torch.randn(B, C, H, W, device=self.device, dtype=dtype)

        task = self.cfg.task
        if task == "auto":
            try:
                task = guess_model_task(self.yaml_cfg) or "detect"
            except Exception:
                task = "detect"

        _task_label_fn = {
            "detect": lambda: _synthetic_detect_labels(B, nc, self.device),
            "obb": lambda: _synthetic_obb_labels(B, nc, self.device),
            "segment": lambda: _synthetic_segment_labels(B, nc, self.device, H),
            "pose": lambda: _synthetic_pose_labels(B, nc, self.device),
            "classify": lambda: _synthetic_classify_labels(B, nc, self.device),
        }
        labels = _task_label_fn.get(task, _task_label_fn["detect"])()

        self.batch = {"img": img, **labels}
        return StageResult(data={
            "batch_shape": list(img.shape),
            "num_labels": sum(v.shape[0] for v in labels.values() if isinstance(v, torch.Tensor) and v.dim() >= 1),
        })

    def _stage_forward(self) -> StageResult:
        """Stage 4: Forward pass (training mode, dict input triggers loss path)."""
        self.model.train()
        t0 = time.time()

        # Warmup: no-grad forward to let lazy initialisers settle
        with torch.no_grad():
            self.model(self.batch["img"])

        # Forward with dict input -> calls self.loss()
        result = self.model(self.batch)
        elapsed = time.time() - t0

        if result is None:
            raise PreflightStageError(
                "Forward pass returned None",
                "Ensure the model head properly returns (loss_vec, loss_items) "
                "when given a dict batch.",
            )

        # result should be (loss_vec, loss_items)
        loss_vec, loss_items = result
        self._loss_result = (loss_vec, loss_items)

        # NaN / Inf check
        if torch.isnan(loss_vec).any():
            raise PreflightStageError(
                "Forward pass produced NaN loss values",
                "Check for numerical instability in the model or loss function.",
            )
        if torch.isinf(loss_vec).any():
            raise PreflightStageError(
                "Forward pass produced Inf loss values",
                "Check for division-by-zero or overflow in the loss computation.",
            )

        return StageResult(data={
            "loss_vec_shape": list(loss_vec.shape),
            "elapsed_s": round(elapsed, 4),
        })

    def _stage_loss(self) -> StageResult:
        """Stage 5: Inspect loss components."""
        if self._loss_result is None:
            raise PreflightStageError(
                "No loss result available from forward pass",
                "Ensure Stage 4 (Forward pass) completed successfully.",
            )

        loss_vec, loss_items = self._loss_result
        total_loss = loss_vec.sum().item()

        if math.isnan(total_loss):
            raise PreflightStageError(
                "Total loss is NaN",
                "Inspect each loss component for NaN values.",
            )
        if math.isinf(total_loss):
            raise PreflightStageError(
                "Total loss is Inf",
                "Check for overflow in the loss summation.",
            )

        components = {}
        if isinstance(loss_items, (list, tuple)):
            for i, item in enumerate(loss_items):
                val = item.sum().item() if isinstance(item, torch.Tensor) else float(item)
                components[f"loss_{i}"] = val
        elif isinstance(loss_items, torch.Tensor):
            components["loss_items_sum"] = loss_items.sum().item()

        components["total"] = total_loss

        return StageResult(data=components)

    def _stage_backward(self) -> StageResult:
        """Stage 6: Backward pass + optimizer step with gradient checks."""
        if self._loss_result is None:
            raise PreflightStageError(
                "No loss result available for backward pass",
                "Ensure Stage 4 completed successfully.",
            )

        loss_vec, _ = self._loss_result
        total_loss = loss_vec.sum()

        # Snapshot parameters before update
        param_before: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_before[name] = param.data.clone()

        # Build optimizer
        optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=0.01,
            momentum=0.937,
        )
        optimizer.zero_grad()

        # Backward
        total_loss.backward()

        # Gradient checks
        nan_grads: List[str] = []
        zero_grad_count = 0
        total_grad_count = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            total_grad_count += 1
            if torch.isnan(param.grad).any():
                nan_grads.append(name)
            if (param.grad.abs() < 1e-12).all():
                zero_grad_count += 1

        if nan_grads:
            raise PreflightStageError(
                f"NaN gradients in: {', '.join(nan_grads[:5])}",
                "Check for numerical instability or disconnected computation graphs.",
            )

        zero_ratio = zero_grad_count / max(total_grad_count, 1)
        if zero_ratio > 0.5:
            raise PreflightStageError(
                f"{zero_grad_count}/{total_grad_count} parameter layers have zero gradients "
                f"(ratio={zero_ratio:.0%})",
                "More than 50% of layers have zero gradients. "
                "The computation graph may be disconnected or the loss may not "
                "depend on these parameters.",
            )

        # Optimizer step
        optimizer.step()

        # Verify parameters actually changed
        changed = 0
        checked = 0
        for name, param in self.model.named_parameters():
            if name not in param_before:
                continue
            checked += 1
            if not torch.equal(param.data, param_before[name]):
                changed += 1

        if checked > 0 and changed == 0:
            raise PreflightStageError(
                "No parameters changed after optimizer step",
                "Verify that requires_grad is set and the optimizer references "
                "the correct parameters.",
            )

        del param_before  # Free snapshot memory

        return StageResult(data={
            "nan_grads": len(nan_grads),
            "zero_grad_ratio": round(zero_ratio, 4),
            "params_changed": changed,
            "params_checked": checked,
        })

    # -- internal helpers ----------------------------------------------------

    def _exec(self, report: PreflightReport, name: str, fn) -> None:
        """Execute *fn*, catching PreflightStageError and recording the outcome."""
        try:
            result = fn()
            report.pass_stage(name, result)
            if self.cfg.verbose:
                LOGGER.info(f"  {colorstr('green', 'bold', '[PASS]')} {name}")
        except PreflightStageError as exc:
            report.fail_stage(name, exc)
            if self.cfg.verbose:
                LOGGER.info(f"  {colorstr('red', 'bold', '[FAIL]')} {name}: {exc.message}")
        except Exception as exc:
            err = PreflightStageError(str(exc), "Unexpected error. See stack trace above for details.")
            report.fail_stage(name, err)
            if self.cfg.verbose:
                LOGGER.info(f"  {colorstr('red', 'bold', '[FAIL]')} {name}: {exc}")

    def _check_gpu_leak(self, report: PreflightReport, iter_idx: int) -> None:
        """Warn if GPU memory grows >10 MB between iterations."""
        if not self.device.type == "cuda":
            return
        torch.cuda.synchronize(self.device)
        current_mem = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        if self._prev_gpu_mem is not None:
            delta = current_mem - self._prev_gpu_mem
            if delta > 10:
                report.warn(
                    f"GPU memory increased by {delta:.1f} MB between iter {iter_idx} "
                    f"and iter {iter_idx + 1} (possible leak)."
                )
        self._prev_gpu_mem = current_mem

    def _finalize(self, report: PreflightReport) -> None:
        """Record end time and optionally print summary."""
        report.end_time = time.time()
        # Cleanup GPU resources
        if hasattr(self, '_loss_result'):
            self._loss_result = None
        if hasattr(self, 'batch'):
            self.batch = None
        if self.device and self.device.type == "cuda":
            del self.model
            self.model = None
            torch.cuda.empty_cache()
        if self.cfg.verbose:
            LOGGER.info(report.summary())
