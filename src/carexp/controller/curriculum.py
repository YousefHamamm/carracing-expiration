"""Delivery-probability curriculum for controller training."""

from __future__ import annotations


def delivery_prob(step: int, total_steps: int, p_start: float = 1.0, p_end: float = 0.3,
                  ramp_fraction: float = 0.4) -> float:
    """Linear from p_start to p_end over the first ramp_fraction of total_steps, then p_end.

    TODO(CLAUDE.md): constant p_end after the ramp vs. a random p_s per env is unconfirmed;
    this implements the constant version.
    """
    ramp = ramp_fraction * total_steps
    if ramp <= 0 or step >= ramp:
        return p_end
    return p_start + (p_end - p_start) * step / ramp
