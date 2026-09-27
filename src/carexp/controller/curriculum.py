"""Delivery curriculum for controller training.

Each env draws a delivery pattern at the start of every episode:
  p_s ~ Uniform[p_min(t), p_max], where p_min anneals linearly from p_min_start to p_min_end over
  the first ramp_fraction of training and then stays at p_min_end;
  with probability periodic_prob the episode is periodic (a frame is sent every m steps, m drawn
  uniformly from `periods`, and each sent frame is lost with probability 1 - p_s); otherwise it
  is Bernoulli (a frame is sent every step and delivered with probability p_s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BERNOULLI = 0  # period value that marks a Bernoulli episode


def p_min_at(step: int, total_steps: int, p_min_start: float = 1.0, p_min_end: float = 0.3,
             ramp_fraction: float = 0.4) -> float:
    """Lower end of the p_s range at a given global step."""
    ramp = ramp_fraction * total_steps
    if ramp <= 0 or step >= ramp:
        return p_min_end
    return p_min_start + (p_min_end - p_min_start) * step / ramp


@dataclass
class DeliveryCurriculum:
    total_steps: int
    p_min_start: float = 1.0
    p_min_end: float = 0.3
    p_max: float = 1.0
    ramp_fraction: float = 0.4
    periodic_prob: float = 0.5
    periods: tuple = (1, 2, 3, 4, 5, 6)

    @classmethod
    def from_config(cls, cfg: dict, total_steps: int) -> "DeliveryCurriculum":
        return cls(total_steps, cfg["p_min_start"], cfg["p_min_end"], cfg["p_max"], cfg["ramp_fraction"],
                   cfg["periodic_prob"], tuple(cfg["periods"]))

    def p_min(self, step: int) -> float:
        return p_min_at(step, self.total_steps, self.p_min_start, self.p_min_end, self.ramp_fraction)

    def sample(self, rng: np.random.Generator, step: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Patterns for n new episodes: (p_s, period), period = BERNOULLI (0) or m."""
        p_s = rng.uniform(self.p_min(step), self.p_max, size=n)
        periodic = rng.random(n) < self.periodic_prob
        m = rng.choice(np.asarray(self.periods), size=n)
        return p_s, np.where(periodic, m, BERNOULLI)


def transmits(period: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Whether step n (>= 1) of an episode sends its frame: always for Bernoulli, n % m == 0 for periodic."""
    period = np.asarray(period)
    return (period == BERNOULLI) | (np.asarray(n) % np.maximum(period, 1) == 0)
