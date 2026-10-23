"""Method registry for visualize_core (plugin discovery)."""

from typing import Any, Dict, List, Optional, Type

from .exceptions import MethodNotRegisteredError


class MethodRegistry:
    """A simple registry to hold visualization method plugins."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def register(self, name: str, plugin: Any) -> None:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("Plugin name cannot be empty")
        self._store[key] = plugin

    def get(self, name: str) -> Any:
        key = str(name).strip().lower()
        if key not in self._store:
            raise MethodNotRegisteredError(key, self.list())
        return self._store[key]

    def list(self) -> List[str]:
        return sorted(self._store.keys())

    def clear(self) -> None:
        self._store.clear()


# Global default registry instance
REGISTRY = MethodRegistry()

