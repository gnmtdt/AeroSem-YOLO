# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

from .tasks import AFSSTaskAdapter


class AFSSScorer:
    """Minimal orchestration layer that delegates scoring to task adapters."""

    def __init__(self, enabled: bool, adapter: AFSSTaskAdapter):
        self.enabled = enabled
        self.adapter = adapter

    def is_ready(self) -> bool:
        """Return whether scoring is available."""
        return self.enabled and self.adapter is not None

    def build_scoring_dataloader(self, trainer: Any, validator: Any):
        """Delegate dataloader construction to the task adapter."""
        return self.adapter.build_scoring_dataloader(trainer, validator)

    def score_epoch(self, trainer: Any, validator: Any, dataloader: Any):
        """Delegate epoch scoring to the task adapter."""
        return self.adapter.score_epoch(trainer, validator, dataloader)
