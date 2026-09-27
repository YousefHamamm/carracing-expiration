"""Receiver buffer: the last K_BUF delivered frames plus Delta_n, the age of the newest one."""

from __future__ import annotations

import numpy as np


class ReceiverBuffer:
    """Frames are ordered oldest -> newest along axis 0; ages[i] is the age in steps of frames[i].

    reset(frame) fills every slot with the initial frame, all at age 0 (the first frame is
    delivered for free). update(frame) is called once per control step, before the controller
    acts: every age increments, then a delivered frame is pushed at age 0; with None the frames
    are frozen. Delta_n = ages[-1], the age of the newest frame.
    """

    def __init__(self, k_buf: int, frame_shape=(96, 96), dtype=np.uint8):
        if k_buf < 1:
            raise ValueError(f"k_buf must be >= 1, got {k_buf}")
        self.k_buf = k_buf
        self.frames = np.zeros((k_buf, *frame_shape), dtype=dtype)
        self.ages = np.zeros(k_buf, dtype=np.int64)

    @property
    def delta(self) -> int:
        return int(self.ages[-1])

    def reset(self, frame: np.ndarray) -> None:
        self.frames[:] = frame
        self.ages[:] = 0

    def update(self, frame: np.ndarray | None) -> None:
        self.ages += 1
        if frame is None:
            return
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = frame
        self.ages[:-1] = self.ages[1:]
        self.ages[-1] = 0

    def observation(self) -> tuple[np.ndarray, int]:
        """(copy of the stacked frames, Delta_n)."""
        return self.frames.copy(), self.delta
