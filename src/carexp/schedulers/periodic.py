"""Periodic baseline: transmit every m control steps."""

from __future__ import annotations

from carexp.schedulers.base import Scheduler


class PeriodicScheduler(Scheduler):
    """Transmits at steps n = m, 2m, 3m, ... (the frame at n = 0 is at the receiver already).

    A failed transmission is not retried; the next attempt is the next slot of the period.
    """

    def __init__(self, m: int):
        if m < 1:
            raise ValueError(f"period must be >= 1, got {m}")
        self.m = int(m)
        self.name = f"periodic_m{m}"

    def decide(self, slot, n, frame, s) -> bool:
        return n % self.m == 0
