"""Minimal RouterAdapter for visualize_core.

Provides best-effort access to the model's multimodal router without introducing
hard dependencies. All operations are Fail‑Fast safe: errors are not silenced if
they imply a behavior change; otherwise, they are logged/ignored conservatively.
"""

from __future__ import annotations

from typing import Any, Optional, Dict
from ultralytics.nn.mm.utils import get_runtime_x_modality_name

try:
    from ultralytics.utils import LOGGER
except Exception:  # pragma: no cover
    LOGGER = None


class RouterAdapter:
    def __init__(self, model: Any) -> None:
        self._model = model
        self._router = self._probe_router(model)
        self._last_modality: Optional[str] = None

    @staticmethod
    def _probe_router(model: Any):
        # Common attach points
        for chain in (
            ("mm_router",),
            ("multimodal_router",),
            ("model", "mm_router"),
            ("model", "multimodal_router"),
        ):
            m = model
            ok = True
            for a in chain:
                if hasattr(m, a):
                    m = getattr(m, a)
                else:
                    ok = False
                    break
            if ok and m is not None:
                return m
        return None

    @property
    def available(self) -> bool:
        return self._router is not None

    def set_runtime_params(self, modality: Optional[str] = None) -> None:
        if not self.available:
            return
        try:
            # 规范化：'auto'/'dual'/None -> None（不消融），'rgb'/'x' -> 指定消融
            if modality is None:
                mm = None
                self._last_modality = 'auto'
            else:
                m = str(modality).lower()
                if m in {"auto", "dual"}:
                    mm = None
                    self._last_modality = m
                elif m in {"rgb", "x"}:
                    mm = m
                    self._last_modality = m
                else:
                    raise ValueError(f"不支持的modality: {modality}")
            self._router.set_runtime_params(mm)
        except Exception:
            # To be refined in Step 8; avoid implicit degradation here
            raise

    def get_x_modality_name(self) -> str:
        """Return runtime X modality name from model metadata only."""
        return get_runtime_x_modality_name(self._model)

    def update_dataset_config(self, data: Optional[dict]) -> None:
        if not self.available or not data:
            return
        try:
            self._router.update_dataset_config(data, update_x_modality=False)
        except Exception:
            # Optional path; do not change behavior implicitly
            pass

    def restore(self) -> None:
        if not self.available:
            return
        try:
            if hasattr(self._router, "restore"):
                self._router.restore()
        except Exception:
            # Avoid masking errors; restoration failure should not alter primary outcome
            pass

    def expected_channels(self) -> Dict[str, int]:
        """Return model's PRIMARY/SECONDARY/FUSED channel contract from router."""
        if not self.available:
            return {"primary_ch": 3, "secondary_ch": 0, "fused_ch": 3}
        sources = getattr(self._router, "INPUT_SOURCES", {}) or {}
        primary_ch = int(sources.get("PRIMARY", sources.get("RGB", getattr(self._router, "primary_ch", 3))))
        secondary_ch = int(sources.get("SECONDARY", sources.get("X", getattr(self._router, "secondary_ch", 0))))
        fused_ch = int(sources.get("FUSED", sources.get("Dual", primary_ch + secondary_ch)))
        # Legacy fallback: only "Dual" key present, no PRIMARY/SECONDARY
        if secondary_ch == 0 and fused_ch > primary_ch:
            secondary_ch = fused_ch - primary_ch
        return {"primary_ch": primary_ch, "secondary_ch": secondary_ch, "fused_ch": fused_ch}

    def summary(self) -> Dict[str, Optional[str]]:
        contract = self.expected_channels()
        return {
            "router": "on" if self.available else "off",
            "modality": self._last_modality or "auto",
            "channels_order": f"[PRIMARY({contract['primary_ch']}), SECONDARY({contract['secondary_ch']})]",
        }

    def log_summary(self, prefix: str = "router") -> None:
        if LOGGER is None:
            return
        s = self.summary()
        LOGGER.info(f"{prefix}: status={s['router']}, modality={s['modality']}, order={s['channels_order']}")
