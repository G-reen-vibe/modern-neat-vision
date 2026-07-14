"""YAML config loader with CLI override support.

Configs are nested dicts. We merge a base config, dataset config, model config,
and any CLI overrides into a single dict before training.
"""
from __future__ import annotations
import copy
import yaml
from pathlib import Path
from typing import Any


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(
    base_path: str | Path,
    dataset_path: str | Path | None = None,
    model_path: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """Load and merge base + dataset + model configs, plus optional overrides.

    Merge order (later wins): base < dataset < model < overrides.
    """
    cfg = load_yaml(base_path)
    if dataset_path:
        cfg = deep_update(cfg, load_yaml(dataset_path))
    if model_path:
        cfg = deep_update(cfg, load_yaml(model_path))
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def parse_cli_overrides(pairs: list[str] | None) -> dict:
    """Parse 'a.b.c=value' style CLI overrides into a nested dict."""
    if not pairs:
        return {}
    out: dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Override must be key=value, got: {p}")
        key, val = p.split("=", 1)
        # Type coercion
        if val.lower() in ("true", "false"):
            val_parsed: Any = val.lower() == "true"
        elif val.lower() in ("null", "none"):
            val_parsed = None
        else:
            try:
                val_parsed = int(val)
            except ValueError:
                try:
                    val_parsed = float(val)
                except ValueError:
                    val_parsed = val
        # Set into nested dict
        keys = key.split(".")
        d = out
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val_parsed
    return out
