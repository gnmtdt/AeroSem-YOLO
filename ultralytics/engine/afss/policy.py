# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import List, Tuple

from .schema import AFSSConfig, EpochSelectionPlan, SampleState
from .state import AFSSStateStore


def partition_states(sample_states: List[SampleState], config: AFSSConfig) -> Tuple[List[SampleState], List[SampleState], List[SampleState]]:
    """Partition tracked sample states into easy, moderate and hard buckets."""
    easy: List[SampleState] = []
    moderate: List[SampleState] = []
    hard: List[SampleState] = []

    for state in sample_states:
        if not state.valid_for_afss or state.last_score_epoch < 0:
            hard.append(state)
            continue
        score_value = state.sufficiency_ema

        if state.modality_flag != "paired" and not config.allow_generated_x_to_easy:
            if score_value < config.hard_threshold:
                hard.append(state)
            else:
                moderate.append(state)
            continue

        if score_value > config.easy_threshold:
            easy.append(state)
        elif score_value < config.hard_threshold:
            hard.append(state)
        else:
            moderate.append(state)
    return easy, moderate, hard


def select_easy_subset(easy_states: List[SampleState], config: AFSSConfig, epoch: int) -> Tuple[List[int], List[int]]:
    """Select low-frequency easy samples with forced-review priority."""
    if not easy_states:
        return [], []

    quota = max(1, int(len(easy_states) * config.easy_ratio))
    ordered = sorted(easy_states, key=lambda s: (s.last_train_epoch, s.dataset_index))
    forced_candidates = [s.dataset_index for s in ordered if epoch - s.last_train_epoch >= config.easy_review_gap]
    if quota > 0 and forced_candidates and config.easy_forced_review_cap_ratio > 0:
        forced_cap = max(1, int(quota * config.easy_forced_review_cap_ratio))
    else:
        forced_cap = 0
    forced_review = forced_candidates[:forced_cap]
    chosen = list(forced_review)
    chosen_set = set(chosen)
    for state in ordered:
        if state.dataset_index in chosen_set:
            continue
        if len(chosen) >= quota:
            break
        chosen.append(state.dataset_index)
        chosen_set.add(state.dataset_index)
    return chosen, forced_review


def select_moderate_subset(
    moderate_states: List[SampleState], config: AFSSConfig, epoch: int
) -> Tuple[List[int], List[int]]:
    """Select moderate samples with short-term forced coverage."""
    if not moderate_states:
        return [], []

    quota = max(1, int(len(moderate_states) * config.moderate_ratio))
    ordered = sorted(moderate_states, key=lambda s: (s.last_train_epoch, s.dataset_index))
    forced_coverage = [s.dataset_index for s in ordered if epoch - s.last_train_epoch >= config.moderate_cover_gap]
    chosen = list(forced_coverage[:quota])
    chosen_set = set(chosen)
    for state in ordered:
        if state.dataset_index in chosen_set:
            continue
        if len(chosen) >= quota:
            break
        chosen.append(state.dataset_index)
        chosen_set.add(state.dataset_index)
    return chosen, forced_coverage[:quota]


def build_epoch_selection(
    state_store: AFSSStateStore, epoch: int, config: AFSSConfig, warmup: bool
) -> EpochSelectionPlan:
    """Build epoch selection plan using full-dataset fallback until states are scored."""
    total_samples = state_store.num_samples
    if warmup:
        return EpochSelectionPlan(
            epoch=epoch,
            active_indices=state_store.all_indices,
            total_samples=total_samples,
            reason="warmup_full_dataset",
            counts={"active": total_samples, "easy": 0, "moderate": 0, "hard": total_samples},
            forced_review_indices=[],
            forced_coverage_indices=[],
        )

    if not any(state.last_score_epoch >= 0 for state in state_store.sample_states):
        return EpochSelectionPlan(
            epoch=epoch,
            active_indices=state_store.all_indices,
            total_samples=total_samples,
            reason="bootstrap_full_dataset_unscored",
            counts={"active": total_samples, "easy": 0, "moderate": 0, "hard": total_samples},
            forced_review_indices=[],
            forced_coverage_indices=[],
        )

    easy, moderate, hard = partition_states(state_store.sample_states, config)
    easy_active, forced_review = select_easy_subset(easy, config, epoch)
    moderate_active, forced_coverage = select_moderate_subset(moderate, config, epoch)
    hard_active = [s.dataset_index for s in hard]
    active_indices = easy_active + moderate_active + hard_active
    return EpochSelectionPlan(
        epoch=epoch,
        active_indices=active_indices,
        total_samples=total_samples,
        reason="adaptive_selection",
        counts={
            "active": len(active_indices),
            "easy": len(easy),
            "moderate": len(moderate),
            "hard": len(hard),
        },
        forced_review_indices=forced_review,
        forced_coverage_indices=forced_coverage,
    )
