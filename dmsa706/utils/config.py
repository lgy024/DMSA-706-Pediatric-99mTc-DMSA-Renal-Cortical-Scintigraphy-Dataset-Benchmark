"""Small YAML config helper with attribute access."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict

import yaml


class Config(SimpleNamespace):
    """Nested namespace so ``cfg.train.lr`` works."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        obj = cls()
        for key, value in (data or {}).items():
            setattr(obj, key, cls.from_dict(value) if isinstance(value, dict) else value)
        return obj

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in vars(self).items():
            result[key] = value.to_dict() if isinstance(value, Config) else value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def merge(self, other: "Config") -> "Config":
        for key, value in vars(other).items():
            current = getattr(self, key, None)
            if isinstance(current, Config) and isinstance(value, Config):
                current.merge(value)
            else:
                setattr(self, key, value)
        return self


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> Config:
    """Load a YAML config, resolving an optional ``base:`` file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    base = data.pop("base", None)
    if base:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        with open(base_path, "r", encoding="utf-8") as fh:
            base_data = yaml.safe_load(fh) or {}
        data = _deep_merge(base_data, data)

    return Config.from_dict(data)
