"""Simple in-memory cache for visualization results."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List


class Cache:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    @staticmethod
    def _hash_params(params: List[str]) -> str:
        payload = "|".join(params).encode()
        return hashlib.md5(payload).hexdigest()

    def make_key(
        self,
        *,
        family: str,
        method: str,
        layers: List[int],
        modality: str | None,
        extra: Dict[str, Any] | None = None,
    ) -> str:
        parts = [family, method, ",".join(map(str, layers)), str(modality or "auto")]
        if extra:
            for k in sorted(extra.keys()):
                v = extra[k]
                parts.append(f"{k}={v}")
        return self._hash_params(parts)

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()

