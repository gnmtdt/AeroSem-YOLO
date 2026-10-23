# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOVO trainer module.

Extends DetectionTrainer with text sampling, cls remap, text injection,
optimizer grouping (text_encoder_lr control), alpha schedule, and YOLO11 weight
remap for open-vocabulary multi-modal detection training.
"""

import hashlib
import os
import random
from copy import copy
from typing import Optional

import torch

from ultralytics.data.dataset import YOLOVODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import YOLOVOModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.patches import torch_load
from ultralytics.utils.timedebug import YOLOVODebugLogger
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.nn.mm.complexity import compute_default_multimodal_complexity_report, log_default_complexity


class YOLOVOTrainer(DetectionTrainer):
    """
    Trainer for YOLOVO open-vocabulary multi-modal detection.

        Extends DetectionTrainer with per-batch text sampling, global-to-local
        cls remap, text encoding injection, alpha schedule for VCTE modules,
        and YOLO11 pretrained weight remap.

        Attributes:
            max_texts (int): Maximum number of text prompts per batch (default 80).
            loss_names (tuple): Loss component names used during training.

    """

    _YOLOVO_DEBUG_FIRST_N_BATCHES = 3
    _YOLOVO_DEBUG_INTERVAL = 50
    _YOLOVO_DEBUG_TEXT_PREVIEW = 12
    _VCTE_ALPHA_WARMUP_TARGET = 0.15
    _VCTE_ALPHA_WARMUP_END = 1 / 6

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """
        Initialize YOLOVOTrainer.

        Args:
            cfg: Configuration namespace or path.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): Callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.max_texts = getattr(self.args, "max_texts", 80)
        self._batch_i = 0

        # YOLOVO debug is isolated from the global debug flag.
        self.dbg = YOLOVODebugLogger(
            enabled=getattr(self.args, "yolovo_debug", False) and RANK in {-1, 0}
        )
        self.dbg.open(self.save_dir)
        self._debug_tag = ""
        self._debug_trace_batch = False
        self._register_yolovo_callbacks()

    # ------------------------------------------------------------------
    # Dataset & Model
    # ------------------------------------------------------------------

    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        """Build YOLOVODataset for training or validation.

        Args:
            img_path (str): Path to image directory.
            mode (str): 'train' or 'val'.
            batch (int, optional): Batch size for rect mode.

        Returns:
            (YOLOVODataset): Configured dataset.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        debug_enabled = bool(getattr(self.args, "yolovo_debug", False)) and RANK in {-1, 0}
        debug_dir = self.save_dir / "object_paste_debug" if debug_enabled else None
        return YOLOVODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task="detect",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            yolovo_debug=debug_enabled and mode == "train",
            yolovo_object_paste_debug_dir=debug_dir,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLOVOModel instance.

        When loading YOLO11 pretrained weights, the state_dict is remapped
        so backbone/neck/Detect layers align with VCTE's shifted indices.
        YOLOVO/VCTE checkpoints are loaded directly without remapping.

        Args:
            cfg: Model config (YAML path or dict).
            weights (str, optional): Pretrained weights path.
            verbose (bool): Whether to display model info.

        Returns:
            (YOLOVOModel): Initialized model.
        """
        model = YOLOVOModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        model._debug_logger = self.dbg  # propagate once for model/loss debug output
        self._log_model_contract(model, cfg)
        if weights:
            ckpt = torch.load(weights, map_location="cpu") if isinstance(weights, (str, os.PathLike)) else weights
            ckpt_model = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            sd = ckpt_model.state_dict() if hasattr(ckpt_model, "state_dict") else ckpt_model
            if self._is_yolo11_ckpt(sd):
                LOGGER.info("Remapping YOLO11 checkpoint to VCTE layout")
                sd = self._remap_yolo11_to_vcte(sd)
                result = model.load_state_dict(sd, strict=False)
                self.dbg.log(
                    "",
                    "weight_load",
                    source="yolo11_remap",
                    missing=len(result.missing_keys),
                    unexpected=len(result.unexpected_keys),
                    missing_preview=",".join(result.missing_keys[:8]),
                    unexpected_preview=",".join(result.unexpected_keys[:8]),
                )
                if result.missing_keys:
                    LOGGER.info(f"VCTE new modules (random init): {len(result.missing_keys)} keys")
                if result.unexpected_keys:
                    LOGGER.warning(f"Unexpected keys after remap: {result.unexpected_keys}")
            else:
                # YOLOVO/VCTE checkpoint -- use standard load
                model.load(ckpt)
                self.dbg.log("", "weight_load", source="yolovo_checkpoint", path=str(weights))
        if verbose and RANK in {-1, 0}:
            try:
                imgsz = int(getattr(self.args, "imgsz", 640))
                report = compute_default_multimodal_complexity_report(model, imgsz=imgsz)
                log_default_complexity(model, report, LOGGER)
            except Exception as e:
                LOGGER.warning(f"YOLOVO complexity summary skipped: {e}")
        return model

    def get_validator(self):
        """Return a YOLOVOValidator for YOLOVO model validation."""
        from ultralytics.models.yolovo.val import YOLOVOValidator

        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "obj_loss"
        return YOLOVOValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    # ------------------------------------------------------------------
    # YOLOVO debug internals
    # ------------------------------------------------------------------

    def _should_trace_yolovo_batch(self, batch_i: int) -> bool:
        """Internal trace policy: first few batches, then fixed interval."""
        return batch_i < self._YOLOVO_DEBUG_FIRST_N_BATCHES or batch_i % self._YOLOVO_DEBUG_INTERVAL == 0

    @staticmethod
    def _module_names(modules) -> str:
        """Return comma-separated module class names for compact debug logs."""
        return ",".join(type(m).__name__ for m in modules)

    @staticmethod
    def _param_count(module, trainable: bool | None = None) -> int:
        """Count module parameters, optionally filtering by requires_grad."""
        params = module.parameters()
        if trainable is None:
            return sum(p.numel() for p in params)
        return sum(p.numel() for p in params if p.requires_grad is trainable)

    @staticmethod
    def _summarize_prompt_digest(texts: list[str]) -> str:
        """Stable short hash for sampled text prompt lists."""
        payload = "\n".join(texts).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:12]

    def _log_model_contract(self, model, cfg) -> None:
        """Log one construction snapshot proving the YOLOVO module contract."""
        if not self.dbg:
            return
        inner = de_parallel(model)
        head = inner.model[-1]
        alpha_modules = [m for m in inner.modules() if hasattr(m, "alpha")]
        emb_alpha_modules = [m for m in inner.modules() if hasattr(m, "emb_alpha")]
        self.dbg.log(
            "",
            "model_contract",
            cfg=str(cfg),
            model=type(inner).__name__,
            head=type(head).__name__,
            nc=getattr(inner, "nc", inner.yaml.get("nc", self.data.get("nc", ""))),
            names=len(self.data.get("names", {})),
            stride=str(tuple(float(s) for s in getattr(inner, "stride", []))),
            text_encoder=str(inner.text_encoder_cfg.get("name", type(inner.text_encoder).__name__)),
            text_dim=getattr(inner, "text_dim", ""),
            head_embed_dim=getattr(head, "embed_dim", ""),
            params_total=self._param_count(inner),
            params_trainable=self._param_count(inner, trainable=True),
            alpha_mods=self._module_names(alpha_modules),
            emb_alpha_mods=self._module_names(emb_alpha_modules),
        )

    def _param_groups_for_debug(self, model) -> dict[str, list[torch.nn.Parameter]]:
        """Group YOLOVO parameters by architectural role for update/grad diagnostics."""
        inner = de_parallel(model)
        groups = {
            "text_encoder": [],
            "vcte": [],
            "head": [],
            "backbone": [],
            "neck": [],
            "other": [],
        }
        for name, p in inner.named_parameters():
            if name.startswith("text_encoder."):
                groups["text_encoder"].append(p)
            elif name.startswith("model.") and name.split(".", 2)[1].isdigit():
                layer_i = int(name.split(".", 2)[1])
                if layer_i == len(inner.model) - 1:
                    groups["head"].append(p)
                elif layer_i <= 10:
                    groups["backbone"].append(p)
                elif 11 <= layer_i <= 14 or type(inner.model[layer_i]).__name__.startswith("VCTE"):
                    groups["vcte"].append(p)
                else:
                    groups["neck"].append(p)
            else:
                groups["other"].append(p)
        return groups

    @staticmethod
    def _tensor_l2_norm(tensors, attr: str | None = None) -> float:
        """Compute aggregate L2 norm for params or selected tensor attributes."""
        total = 0.0
        for tensor in tensors:
            value = getattr(tensor, attr) if attr else tensor
            if value is None:
                continue
            total += float(value.detach().float().pow(2).sum().item())
        return total ** 0.5

    def _capture_param_snapshot(self) -> dict[str, list[tuple[torch.nn.Parameter, torch.Tensor]]]:
        """Capture trainable parameter snapshots for exact update diagnostics."""
        return {
            name: [(p, p.detach().float().cpu().clone()) for p in params if p.requires_grad]
            for name, params in self._param_groups_for_debug(self.model).items()
            if params
        }

    @staticmethod
    def _compute_update_norms(snapshot: dict[str, list[tuple[torch.nn.Parameter, torch.Tensor]]]) -> dict[str, float]:
        """Compute exact post-step update L2 norm from captured parameter snapshots."""
        update_norms = {}
        for name, items in snapshot.items():
            total = 0.0
            for p, before in items:
                delta = p.detach().float().cpu() - before
                total += float(delta.pow(2).sum().item())
            update_norms[name] = total ** 0.5
        return update_norms

    def _log_optimizer_contract(self, optimizer, model, text_encoder_lr: float) -> None:
        """Log optimizer grouping after YOLOVO text-encoder filtering."""
        if not self.dbg:
            return
        groups = self._param_groups_for_debug(model)
        trainable = {name: sum(p.numel() for p in params if p.requires_grad) for name, params in groups.items()}
        self.dbg.log(
            "",
            "optimizer_contract",
            text_encoder_lr=text_encoder_lr,
            text_encoder_trainable=trainable.get("text_encoder", 0),
            vcte_trainable=trainable.get("vcte", 0),
            head_trainable=trainable.get("head", 0),
            backbone_trainable=trainable.get("backbone", 0),
            neck_trainable=trainable.get("neck", 0),
            param_groups=len(optimizer.param_groups),
            group_lrs=",".join(f"{g.get('lr', 0.0):.8g}" for g in optimizer.param_groups),
            group_params=",".join(str(sum(p.numel() for p in g["params"])) for g in optimizer.param_groups),
        )

    def _log_train_alignment(
        self,
        update_norms: dict[str, float] | None = None,
        include_grad: bool = True,
        phase: str = "step",
    ) -> None:
        """Log grad and update norms around one optimizer step."""
        if not self.dbg or not self._debug_trace_batch:
            return
        groups = self._param_groups_for_debug(self.model)
        meta = {"phase": phase}
        for name, params in groups.items():
            if not params:
                continue
            param_norm = self._tensor_l2_norm(params)
            meta[f"{name}_param_norm"] = param_norm
            if include_grad:
                meta[f"{name}_grad_norm"] = self._tensor_l2_norm(params, attr="grad")
            if update_norms and name in update_norms:
                meta[f"{name}_update_norm"] = update_norms[name]
        self.dbg.log(self._debug_tag, "train_alignment", **meta)

    # ------------------------------------------------------------------
    # Batch preprocessing -- text sampling + cls remap + text injection
    # ------------------------------------------------------------------

    def sample_batch_texts(self, cls):
        """Sample text prompts: all positive classes + random negative classes.

        ``max_texts`` controls the max number of *negative* classes sampled,
        not total K.  All positive (GT) classes are always included so that
        every ground-truth label has a matching text entry.

        Args:
            cls (torch.Tensor): Class IDs in the batch (global IDs).

        Returns:
            (list[str]): Sampled text prompts (positives first, then negatives).
        """
        names = self.data["names"]
        positive_ids = cls.unique().tolist()
        positive_texts = list(dict.fromkeys(names[i] for i in positive_ids if i in names))

        # max_texts caps *negative* sampling only -- positives are never truncated
        remaining = list(set(names.values()) - set(positive_texts))
        n_neg = max(0, min(self.max_texts, len(remaining)))
        negative_texts = random.sample(remaining, n_neg) if n_neg > 0 else []

        return positive_texts + negative_texts

    def remap_cls(self, cls, texts):
        """Remap global class IDs to local indices within sampled texts.

        Every GT class must appear in ``texts``; unmapped IDs raise ValueError
        rather than silently retaining their global ID (which would cause
        ``gt_labels >= K`` out-of-bounds in loss computation).

        Args:
            cls (torch.Tensor): Global class IDs.
            texts (list[str]): Sampled text prompts (from ``sample_batch_texts``).

        Returns:
            (torch.Tensor): Remapped local class IDs (0 .. len(texts)-1).
        """
        names = self.data["names"]
        text2id = {t: i for i, t in enumerate(texts)}
        remapped = torch.full_like(cls, -1)
        for global_id in cls.unique():
            gid = int(global_id.item())
            if gid not in names:
                continue
            text = names[gid]
            if text not in text2id:
                raise ValueError(
                    f"GT class '{text}' (id={gid}) not in sampled texts -- "
                    f"sample_batch_texts must include all positive classes"
                )
            remapped[cls == global_id] = text2id[text]
        return remapped

    def preprocess_batch(self, batch):
        """Preprocess batch with text sampling, cls remap, and text injection.

        Calls parent preprocess (device transfer, float conversion, multi-scale),
        then samples text prompts, remaps class IDs, encodes text in FP32, and
        injects into the model.

        Note: This runs outside the autocast block in _do_train, so text
        encoding is always FP32 for numerical safety.

        Args:
            batch (dict): Batch dict from dataloader.

        Returns:
            (dict): Preprocessed batch with remapped cls and injected text.
        """
        batch = super().preprocess_batch(batch)
        debug_enabled = bool(self.dbg)
        global_cls = batch["cls"].clone() if debug_enabled else batch["cls"]
        texts = self.sample_batch_texts(global_cls)
        batch["cls"] = self.remap_cls(global_cls, texts)

        tag = f"{self.epoch},{self._batch_i}"
        self._debug_tag = tag
        self._debug_trace_batch = debug_enabled and self._should_trace_yolovo_batch(self._batch_i)

        K = len(texts)
        if debug_enabled:
            names = self.data["names"]
            positive_ids = [int(x.item()) for x in global_cls.unique()]
            positive_texts = list(dict.fromkeys(names[i] for i in positive_ids if i in names))
            self.dbg.log(
                tag,
                "preprocess_batch",
                images=batch["img"].shape[0],
                labels=len(batch["cls"]),
                K=K,
                positives=len(positive_texts),
                negatives=max(0, K - len(positive_texts)),
                global_cls_min=global_cls.min().item() if global_cls.numel() else -1,
                global_cls_max=global_cls.max().item() if global_cls.numel() else -1,
                local_cls_min=batch["cls"].min().item() if batch["cls"].numel() else -1,
                local_cls_max=batch["cls"].max().item() if batch["cls"].numel() else -1,
                text_hash=self._summarize_prompt_digest(texts),
            )
            if self._debug_trace_batch:
                mapping = ",".join(
                    f"{gid}->{int(batch['cls'][global_cls == gid].flatten()[0].item())}:{names[gid]}"
                    for gid in positive_ids
                    if gid in names and (global_cls == gid).any()
                )
                self.dbg.log(
                    tag,
                    "text_transfer",
                    mapping=mapping,
                    text_preview="|".join(texts[: self._YOLOVO_DEBUG_TEXT_PREVIEW]),
                    text_cache="runtime",
                )

        # Encode text in FP32 -- preprocess_batch runs outside autocast
        # requires_grad status is controlled by build_optimizer via text_encoder_lr
        inner = de_parallel(self.model)
        inner._debug_tag = tag
        inner._debug_trace_batch = self._debug_trace_batch
        t_clip = inner.encode_text(texts)
        inner.set_input("text", t_clip, cache=False)

        if debug_enabled:
            self.dbg.log(
                tag,
                "encode_text",
                shape=str(tuple(t_clip.shape)),
                norm_min=t_clip.norm(dim=-1).min().item(),
                norm_max=t_clip.norm(dim=-1).max().item(),
                device=str(t_clip.device),
                requires_grad=t_clip.requires_grad,
            )
        self._batch_i += 1
        return batch

    # ------------------------------------------------------------------
    # Optimizer -- text_encoder_lr control
    # ------------------------------------------------------------------

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build optimizer with text_encoder_lr control.

        Inherits parent weight/bn/bias/no_decay grouping rules. When
        ``text_encoder_lr`` is 0 (default), the text encoder is frozen and
        removed from optimizer param groups. When > 0, text encoder params
        are extracted into an independent param group with its own lr,
        inheriting weight_decay from their original group.

        Args:
            model (nn.Module): Model to optimize.
            name (str): Optimizer name.
            lr (float, optional): Base learning rate.
            momentum (float): Momentum factor.
            decay (float): Weight decay.
            iterations (int): Total iterations for auto selection.

        Returns:
            (torch.optim.Optimizer): Configured optimizer.
        """
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        te_lr = float(getattr(self.args, "text_encoder_lr", 0.0))
        inner = de_parallel(model)
        text_ids = {id(p) for p in inner.text_encoder.parameters()}

        if te_lr > 0:
            # Extract text encoder params from parent groups into independent groups,
            # inheriting each param's original group's weight_decay
            te_param_meta = {}  # id(param) -> weight_decay
            for group in optimizer.param_groups:
                wd = group.get("weight_decay", 0.0)
                for p in group["params"]:
                    if id(p) in text_ids:
                        te_param_meta[id(p)] = wd
            # Remove text encoder params from parent groups
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if id(p) not in text_ids]
            # Build independent text encoder groups grouped by weight_decay
            wd_buckets = {}
            for p in inner.text_encoder.parameters():
                wd = te_param_meta.get(id(p), 0.0)
                wd_buckets.setdefault(wd, []).append(p)
            for wd, params in wd_buckets.items():
                optimizer.add_param_group({"params": params, "lr": te_lr, "weight_decay": wd})
        else:
            # Freeze: remove text_encoder params from optimizer entirely
            inner.text_encoder.eval()
            inner.text_encoder.requires_grad_(False)
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if id(p) not in text_ids]

        self._log_optimizer_contract(optimizer, model, te_lr)
        return optimizer

    def optimizer_step(self):
        """Perform optimizer step with YOLOVO gradient/update diagnostics."""
        snapshot = self._capture_param_snapshot() if self.dbg and self._debug_trace_batch else None
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        if self.dbg and self._debug_trace_batch:
            self._log_train_alignment(include_grad=True, phase="pre_step")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.dbg and self._debug_trace_batch:
            self._log_train_alignment(self._compute_update_norms(snapshot or {}), include_grad=False, phase="post_step")
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    # ------------------------------------------------------------------
    # Alpha schedule -- VCTE module warmup
    # ------------------------------------------------------------------

    def set_vcte_alpha(self):
        """Set alpha/emb_alpha for all VCTE modules based on training progress.

        Two-stage progressive schedule:
        - First 1/6 of epochs: alpha ramps from 0 to 0.15
        - Remaining epochs: alpha ramps from 0.15 to 1.0
        """
        p = self.epoch / self.epochs if self.epochs > 0 else 0.0
        warmup_end = self._VCTE_ALPHA_WARMUP_END
        warmup_target = self._VCTE_ALPHA_WARMUP_TARGET
        if p < warmup_end:
            factor = warmup_target * (p / warmup_end)
        else:
            factor = warmup_target + (1.0 - warmup_target) * ((p - warmup_end) / (1.0 - warmup_end))
        factor = max(0.0, min(1.0, factor))

        inner = de_parallel(self.model)
        for m in inner.modules():
            if hasattr(m, "alpha"):
                m.alpha.fill_(factor)
            if hasattr(m, "emb_alpha"):
                m.emb_alpha.fill_(factor)

        alpha_modules = [type(m).__name__ for m in inner.modules() if hasattr(m, "alpha")]
        emb_alpha_modules = [type(m).__name__ for m in inner.modules() if hasattr(m, "emb_alpha")]
        self.dbg.log(
            "",
            "set_vcte_alpha",
            epoch=f"{self.epoch}/{self.epochs}",
            progress=p,
            factor=factor,
            alpha_mods=",".join(alpha_modules),
            emb_alpha_mods=",".join(emb_alpha_modules),
        )

    def _set_vcte_alpha_callback(self, trainer):
        """Callback wrapper for set_vcte_alpha, registered on on_train_epoch_start."""
        self.set_vcte_alpha()

    # ------------------------------------------------------------------
    # Epoch boundary -- batch counter reset
    # ------------------------------------------------------------------

    def _on_epoch_start_callback(self, trainer):
        """Reset batch counter at epoch start."""
        self._batch_i = 0
        self.dbg.log("", "epoch_start", epoch=self.epoch)
        inner = de_parallel(self.model)
        inner._debug_tag = ""
        inner._debug_trace_batch = False

    # ------------------------------------------------------------------
    # Weight remap -- YOLO11 pretrained weights -> VCTE model
    # ------------------------------------------------------------------

    @staticmethod
    def _is_yolo11_ckpt(state_dict):
        """Check if a state_dict belongs to a standard YOLO11 model.

        Returns True if no VCTE-specific marker keys are found.

        Args:
            state_dict (dict): Model state dictionary.

        Returns:
            (bool): True if this is a YOLO11 checkpoint.
        """
        vcte_markers = ("mod_gamma", "mod_beta", "embed_proj", "text_enhancer", "visual_adapter", "text_adapter",
                        "neck_modulator")
        return not any(any(m in k for m in vcte_markers) for k in state_dict)

    @staticmethod
    def _remap_yolo11_to_vcte(state_dict):
        """Remap YOLO11 state_dict keys to VCTE model layout.

        Mapping rules:
        - Backbone (indices 0-10): unchanged
        - Neck (indices 11-22): shifted by +4 (to 15-26)
        - Detect.cv2 (index 23): remapped to YOLOVODetect (index 30)
        - Detect.dfl (index 23): remapped to YOLOVODetect (index 30)
        - Detect.cv3 (index 23): discarded (class branch replaced by text)

        Args:
            state_dict (dict): YOLO11 state dictionary.

        Returns:
            (dict): Remapped state dictionary for VCTE model.
        """
        new_sd = {}
        for k, v in state_dict.items():
            parts = k.split(".")
            if not parts[0].startswith("model"):
                new_sd[k] = v
                continue
            idx = int(parts[1])
            if idx <= 10:
                new_sd[k] = v  # backbone unchanged
            elif 11 <= idx <= 22:
                parts[1] = str(idx + 4)
                new_sd[".".join(parts)] = v  # neck +4
            elif idx == 23 and "cv2" in k:
                new_sd[k.replace("model.23", "model.30")] = v  # box branch
            elif idx == 23 and "dfl" in k:
                new_sd[k.replace("model.23", "model.30")] = v  # DFL
            # cv3 (index 23): discarded -- class branch replaced by text
        return new_sd

    # ------------------------------------------------------------------
    # Checkpoint saving -- text metadata injection
    # ------------------------------------------------------------------

    def save_model(self):
        """Save checkpoint with class_texts and text_encoder_cfg metadata.

        After the parent save_model writes last.pt/best.pt, this method
        re-loads each checkpoint, injects text metadata, and re-saves.
        """
        super().save_model()
        inner = de_parallel(self.model)
        # class_texts and text_encoder_cfg are stable attributes set during model
        # construction; they are identical whether read from self.model or self.ema.ema.
        for p in [self.last, self.best]:
            if p.exists():
                ckpt = torch_load(p, map_location="cpu")
                ckpt["class_texts"] = inner.class_texts
                ckpt["text_encoder_cfg"] = inner.text_encoder_cfg
                torch.save(ckpt, p)

    def on_train_end(self, trainer=None):
        """Save sidecar text encoder weights if it was fine-tuned and close debug log.

        Args:
            trainer: Passed by run_callbacks -- always self. Ignored here
                because the method is bound to the trainer instance.
        """
        if getattr(self.args, "text_encoder_lr", 0.0) > 0:
            inner = de_parallel(self.model)
            name = inner.text_encoder_cfg.get("name", "mobileclip_s0")
            save_path = os.path.join(self.save_dir, f"{name}.pt")
            torch.save(inner.text_encoder.state_dict(), save_path)
            LOGGER.info(f"Saved text encoder sidecar to {save_path}")
        self.dbg.close()

    # ------------------------------------------------------------------
    # Callbacks registration
    # ------------------------------------------------------------------

    def _register_yolovo_callbacks(self):
        """Register YOLOVO-specific training callbacks."""
        self.add_callback("on_train_epoch_start", self._set_vcte_alpha_callback)
        self.add_callback("on_train_epoch_start", self._on_epoch_start_callback)
        self.add_callback("on_train_end", self.on_train_end)
