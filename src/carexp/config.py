"""YAML config loading."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "base.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_stage_config(path: str | Path, smoke: bool = False) -> dict:
    """base.yaml <- stage config <- (stage config's `smoke` section if smoke)."""
    stage = load_config(path)
    smoke_overrides = stage.pop("smoke", {})
    cfg = deep_merge(load_config(DEFAULT_CONFIG), stage)
    if smoke:
        cfg = deep_merge(cfg, smoke_overrides)
    cfg["smoke"] = smoke
    return cfg
