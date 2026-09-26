"""Sender-side transmission scheduler interface.

One scheduler object may serve many concurrent episodes; per-episode state is keyed by slot.
The sender sees its own frame y_n and state s_n, plus the delivery acknowledgement of each
transmission it makes (reliable return link).
"""

from __future__ import annotations

import numpy as np


class Scheduler:
    name = "base"

    def reset(self, slot: int, frame: np.ndarray, s: np.ndarray) -> None:
        """A new episode starts in this slot; the initial frame is at the receiver (Delta = 0)."""

    def decide(self, slot: int, n: int, frame: np.ndarray, s: np.ndarray) -> bool:
        """Whether to transmit the frame of control step n (n >= 1)."""
        raise NotImplementedError

    def feedback(self, slot: int, n: int, delivered: bool) -> None:
        """Acknowledgement for a transmission made at step n."""
