# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch.distributed as dist

from ultralytics.utils import LOGGER

from .policy import build_epoch_selection
from .sampler import AFSSEpochSampler, AFSSDistributedEpochSampler
from .schema import AFSSConfig, EpochSelectionPlan
from .scorer import AFSSScorer
from .state import AFSSStateStore
from .tasks import resolve_afss_task_adapter


class AFSSRuntime:
    """Runtime for AFSS integration across bootstrap, scoring, and adaptive selection."""

    def __init__(self, config: AFSSConfig, state_store: AFSSStateStore, save_dir: Path):
        self.config = config
        if not config.is_task_enabled(config.task_name):
            raise RuntimeError(
                f"AFSS is not enabled for task={config.task_name}. "
                f"Please set afss_tasks.{config.task_name}=true before enabling AFSS."
            )
        self.state_store = state_store
        self.save_dir = Path(save_dir)
        self.task_adapter = resolve_afss_task_adapter(task_name=config.task_name, config=config)
        self.scorer = AFSSScorer(enabled=config.score_on_train_eval, adapter=self.task_adapter)
        self.selection_plan = None  # lazy init by on_train_epoch_start
        self._sampler = None

    @classmethod
    def from_dataset(cls, dataset: Any, args: Any, save_dir: Path, resume: bool = False) -> "AFSSRuntime":
        """Create runtime from dataset and trainer args."""
        config = AFSSConfig.from_args(args)
        if not config.is_task_enabled(config.task_name):
            raise RuntimeError(
                f"AFSS is enabled globally but disabled for task={config.task_name}. "
                f"Please set afss_tasks.{config.task_name}=true to continue."
            )
        state_store = (
            AFSSStateStore.load_latest(save_dir, config)
            if resume
            else AFSSStateStore.from_dataset(dataset, task_name=config.task_name)
        )
        if resume:
            state_store.validate_against_dataset(dataset)
            state_store.validate_against_config(config)
        rt = cls(config=config, state_store=state_store, save_dir=save_dir)
        # 为 epoch 0 warmup 预初始化 selection_plan
        rt.selection_plan = build_epoch_selection(
            state_store, epoch=0, config=config, warmup=True,
        )
        return rt

    def create_sampler(self, rank: int, shuffle: bool):
        """Create and attach AFSS sampler for current process."""
        active_indices = self.selection_plan.active_indices
        if rank == -1:
            sampler = AFSSEpochSampler(active_indices=active_indices, shuffle=shuffle, seed=self.config.seed)
        else:
            sampler = AFSSDistributedEpochSampler(
                active_indices=active_indices,
                shuffle=shuffle,
                seed=self.config.seed,
                rank=rank,
            )
        self._sampler = sampler
        return sampler

    def on_train_start(self) -> None:
        """Log runtime bootstrap state."""
        LOGGER.info(
            "AFSS [%s] 已启用: runtime ready, "
            "samples=%d, warmup_epochs=%d, score_conf=%.3f, score_iou=%.3f, ema_alpha=%.3f",
            self.config.task_name,
            self.state_store.num_samples,
            self.config.warmup_epochs,
            self.config.score_conf,
            self.config.score_iou,
            self.config.state_ema_alpha,
        )

    def on_train_epoch_start(self, epoch: int) -> None:
        """Refresh epoch selection plan and attached sampler."""
        warmup = epoch < self.config.warmup_epochs
        self.selection_plan = build_epoch_selection(
            self.state_store,
            epoch=epoch,
            config=self.config,
            warmup=warmup,
        )
        if self._sampler is not None:
            self._sampler.set_active_indices(self.selection_plan.active_indices)
        LOGGER.info(
            "AFSS [%s] Epoch %d: reason=%s active=%d/%d easy=%d moderate=%d hard=%d "
            "forced_review=%d forced_coverage=%d",
            self.config.task_name,
            epoch + 1,
            self.selection_plan.reason,
            len(self.selection_plan.active_indices),
            self.selection_plan.total_samples,
            self.selection_plan.counts.get("easy", 0),
            self.selection_plan.counts.get("moderate", 0),
            self.selection_plan.counts.get("hard", 0),
            len(self.selection_plan.forced_review_indices),
            len(self.selection_plan.forced_coverage_indices),
        )

    def on_train_epoch_end(self, epoch: int, trainer=None, validator=None) -> None:
        """Persist AFSS state snapshot and refresh sample scores when scheduled."""
        selection_payload: Dict[str, Any] = {
            "epoch": epoch,
            "task_name": self.config.task_name,
            "reason": self.selection_plan.reason,
            "active_indices": list(self.selection_plan.active_indices),
            "counts": dict(self.selection_plan.counts),
            "forced_review_indices": list(self.selection_plan.forced_review_indices),
            "forced_coverage_indices": list(self.selection_plan.forced_coverage_indices),
        }
        self.state_store.update_train_usage(
            self.selection_plan.active_indices,
            epoch,
            forced_review_indices=self.selection_plan.forced_review_indices,
            forced_coverage_indices=self.selection_plan.forced_coverage_indices,
        )
        should_score = (
            self.config.score_on_train_eval
            and (epoch + 1) > self.config.warmup_epochs
            and ((epoch + 1 - self.config.warmup_epochs) % self.config.state_update_interval == 0)
        )
        if should_score:
            if trainer is None:
                raise RuntimeError("AFSS scoring requires trainer context at epoch end")
            if validator is None:
                raise RuntimeError("AFSS scoring requires a validator at epoch end")
            trainer_task = str(getattr(getattr(trainer, "args", None), "task", self.config.task_name))
            validator_task = str(getattr(getattr(validator, "args", None), "task", trainer_task))
            if trainer_task != self.config.task_name or validator_task != self.config.task_name:
                raise RuntimeError(
                    "AFSS task context mismatch before scoring: "
                    f"config.task_name={self.config.task_name!r}, "
                    f"trainer.args.task={trainer_task!r}, validator.args.task={validator_task!r}"
                )
            is_main = not dist.is_initialized() or dist.get_rank() == 0
            if is_main:
                scoring_loader = self.scorer.build_scoring_dataloader(trainer, validator)
                score_rows = self.scorer.score_epoch(trainer, validator, scoring_loader)
                self.state_store.update_scores(score_rows, epoch, self.config)
                score_summary = self.state_store.score_distribution_summary(self.config)
                task_metric_summary = self.task_adapter.summarize_task_metrics(score_rows)
                # Degenerate check BEFORE persist to avoid saving bad state
                self.state_store.ensure_non_degenerate_scores(self.config, score_summary)
                selection_payload["score_summary"] = score_summary
                if task_metric_summary:
                    selection_payload["task_metric_summary"] = task_metric_summary
                LOGGER.info(
                    "AFSS [%s] Epoch %d: scored %d samples",
                    self.config.task_name,
                    epoch + 1,
                    len(score_rows),
                )
                LOGGER.info(
                    "AFSS [%s] raw score summary: q50=%.4f q90=%.4f q99=%.4f easy=%d moderate=%d hard=%d",
                    self.config.task_name,
                    score_summary["raw"]["q50"],
                    score_summary["raw"]["q90"],
                    score_summary["raw"]["q99"],
                    score_summary["raw"]["counts"]["easy"],
                    score_summary["raw"]["counts"]["moderate"],
                    score_summary["raw"]["counts"]["hard"],
                )
                LOGGER.info(
                    "AFSS [%s] EMA state summary: q50=%.4f q90=%.4f q99=%.4f easy=%d moderate=%d hard=%d",
                    self.config.task_name,
                    score_summary["ema"]["q50"],
                    score_summary["ema"]["q90"],
                    score_summary["ema"]["q99"],
                    score_summary["ema"]["counts"]["easy"],
                    score_summary["ema"]["counts"]["moderate"],
                    score_summary["ema"]["counts"]["hard"],
                )
                if task_metric_summary:
                    LOGGER.info("AFSS [%s] task metric summary: %s", self.config.task_name, task_metric_summary)
                latest_path = self.state_store.save_latest(
                    save_dir=self.save_dir,
                    config=self.config,
                    selection=selection_payload,
                    epoch=epoch,
                )
                self.state_store.save_epoch_artifacts(
                    save_dir=self.save_dir,
                    config=self.config,
                    selection=selection_payload,
                    epoch=epoch,
                )
                LOGGER.debug("AFSS state snapshot updated: %s", latest_path)
            # DDP sync: non-rank-0 loads state from rank-0's persisted snapshot
            if dist.is_initialized():
                dist.barrier()
                if not is_main:
                    self.state_store = AFSSStateStore.load_latest(self.save_dir, self.config)
            return
        latest_path = self.state_store.save_latest(
            save_dir=self.save_dir,
            config=self.config,
            selection=selection_payload,
            epoch=epoch,
        )
        self.state_store.save_epoch_artifacts(
            save_dir=self.save_dir,
            config=self.config,
            selection=selection_payload,
            epoch=epoch,
        )
        LOGGER.debug("AFSS state snapshot updated: %s", latest_path)
