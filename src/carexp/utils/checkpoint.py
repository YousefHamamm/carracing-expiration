"""Atomic checkpoint save/load."""

from __future__ import annotations

import os
from pathlib import Path

import torch

LATEST = "ckpt_latest.pt"


def save_checkpoint(state: dict, out_dir: Path, update: int, keep_milestone: bool = True) -> Path:
    """Writes ckpt_latest.pt (and ckpt_{update:06d}.pt if keep_milestone) via temp file + rename."""
    out_dir = Path(out_dir)
    tmp = out_dir / f".{LATEST}.tmp"
    torch.save(state, tmp)
    if keep_milestone:
        milestone = out_dir / f"ckpt_{update:06d}.pt"
        torch.save(state, milestone.with_suffix(".tmp"))
        os.replace(milestone.with_suffix(".tmp"), milestone)
    latest = out_dir / LATEST
    os.replace(tmp, latest)
    return latest


def load_latest(out_dir: Path, map_location="cpu") -> dict | None:
    path = Path(out_dir) / LATEST
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)
