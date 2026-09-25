"""Unreliable, costly forward channel: one frame per use, i.i.d. Bernoulli(p_s) success, cost c per attempt."""

from __future__ import annotations

import numpy as np


class BernoulliChannel:
    def __init__(self, p_s: float, cost: float = 0.0, seed: int | None = None):
        if not 0.0 <= p_s <= 1.0:
            raise ValueError(f"p_s must be in [0, 1], got {p_s}")
        self.p_s = float(p_s)
        self.cost = float(cost)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.attempts = 0
        self.successes = 0

    def transmit(self) -> bool:
        """One channel use. Returns True if the frame is delivered."""
        self.attempts += 1
        ok = bool(self.rng.random() < self.p_s)
        self.successes += ok
        return ok

    @property
    def total_cost(self) -> float:
        return self.cost * self.attempts
