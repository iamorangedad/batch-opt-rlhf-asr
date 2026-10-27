"""Minimal YAML config loader with shallow ``extends`` support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required: pip install pyyaml") from exc

    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    extends = data.get("extends")
    if extends:
        parent_path = (path.parent / extends).resolve()
        parent = load_yaml(parent_path)
        data = _deep_merge(parent, data)
    return data
