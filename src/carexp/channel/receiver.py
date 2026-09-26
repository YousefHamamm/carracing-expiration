"""Receiver buffer: the last K_BUF delivered frames plus Delta_n, the age of the newest one."""

from __future__ import annotations

import numpy as np


class ReceiverBuffer:
    """Frames are ordered oldest -> newest along axis 0.

    reset(frame) fills every slot with the initial frame and sets Delta = 0 (the first frame
    is delivered for free). update(frame) is called once per control step, before the
    controller acts: a delivered frame is pushed and Delta resets to 0; with None the buffer
    is frozen and Delta increments.
    """

    def __init__(self, k_buf: int, frame_shape=(96, 96), dtype=np.uint8):
        if k_buf < 1:
            raise ValueError(f"k_buf must be >= 1, got {k_buf}")
        self.k_buf = k_buf
        self.frames = np.zeros((k_buf, *frame_shape), dtype=dtype)
        self.delta = 0

    def reset(self, frame: np.ndarray) -> None:
        self.frames[:] = frame
        self.delta = 0

    def update(self, frame: np.ndarray | None) -> None:
        if frame is None:
            self.delta += 1
            return
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = frame
        self.delta = 0

    def observation(self) -> tuple[np.ndarray, int]:
        """(copy of the stacked frames, Delta_n)."""
        return self.frames.copy(), self.delta
