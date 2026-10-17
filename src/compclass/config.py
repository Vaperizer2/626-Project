"""YAML configs with single inheritance and command-line overrides.

A config file may name a parent with ``base: base.yaml`` (path relative to the file). The parent is loaded
first and the child is deep-merged on top, so a family config only lists what differs from ``base.yaml``.
Overrides such as ``--set train.lr=1e-4 aug.mixup=0.2`` are applied last; values are parsed as YAML, so
``1e-4`` becomes a float, ``null`` becomes None and ``[0.5, 1.0]`` a list.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]


def deep_merge(base: Config, override: Config) -> Config:
    """Return ``base`` updated recursively with ``override`` (neither input is modified)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Apply ``dotted.key=value`` strings to a copy of ``cfg``."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like section.key=value")
        dotted, raw = item.split("=", 1)
        node = cfg
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = _parse_value(raw)
    return cfg


def _parse_value(raw: str) -> Any:
    """YAML-parse a CLI value. PyYAML follows YAML 1.1, where ``1e-3`` is a *string* (it wants ``1.0e-3``),
    so numeric-looking strings are converted to float explicitly. In YAML files, write ``1.0e-3``."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load ``path`` (following ``base:`` links) and apply overrides."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("base", None)
    cfg = deep_merge(load_config(path.parent / parent), raw) if parent else raw
    cfg["name"] = raw.get("name", path.stem)  # run name defaults to the file stem, e.g. "resnet50"
    return apply_overrides(cfg, overrides)


def config_hash(cfg: Config) -> str:
    """Stable 12-hex hash of everything that affects the result (output paths and name excluded)."""
    relevant = {k: v for k, v in cfg.items() if k not in {"output", "name"}}
    blob = json.dumps(relevant, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
