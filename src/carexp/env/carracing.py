"""CarRacing-v3 wrapper: 2-D actions, action repeat, grayscale frames, s_n, off-track termination."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing

from carexp.env.features import StateExtractor

FRAME_SHAPE = (96, 96)
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class EnvConfig:
    action_repeat: int = 6
    max_steps: int = 500
    offtrack_patience: int = 1
    n_curvature: int = 20

    @classmethod
    def from_dict(cls, d: dict | None) -> "EnvConfig":
        d = d or {}
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown env config keys: {sorted(unknown)}")
        return cls(**d)


def map_action(action) -> np.ndarray:
    """[steering, throttle] in [-1, 1]^2 -> CarRacing [steer, gas, brake]."""
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
    return np.array([a[0], max(a[1], 0.0), max(-a[1], 0.0)], dtype=np.float64)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """(96, 96, 3) uint8 -> (96, 96) uint8 luminance."""
    return np.rint(rgb.astype(np.float32) @ _LUMA).astype(np.uint8)


class CarRacingEnv:
    """One control step = action_repeat physics frames.

    reset(seed) -> (frame, s, info); step(action) -> (frame, s, reward, terminated, truncated, info).
    frame is (96, 96) uint8 grayscale y_n; s is float32 s_n (see features.py).
    Episodes end on lap completion, leaving the playfield (env default), or being off the road
    with all four wheels for offtrack_patience consecutive control steps. truncated at max_steps.
    """

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.env = CarRacing(continuous=True)
        self.features = StateExtractor(self.config.n_curvature)
        self.t = 0
        self._offtrack_steps = 0

    @property
    def unwrapped(self) -> CarRacing:
        return self.env

    def reset(self, seed: int):
        rgb, _ = self.env.reset(seed=int(seed))
        self.features.reset(self.env)
        self.t = 0
        self._offtrack_steps = 0
        s, seg = self.features(self.env)
        return to_gray(rgb), s, self._info(seg)

    def step(self, action):
        a3 = map_action(action)
        reward = 0.0
        terminated = False
        lap_finished = False
        for _ in range(self.config.action_repeat):
            rgb, r, term, _, info = self.env.step(a3)
            reward += float(r)
            if term:
                terminated = True
                lap_finished = bool(info.get("lap_finished", False))
                break
        self.t += 1

        off = self.off_track()
        self._offtrack_steps = self._offtrack_steps + 1 if off else 0
        offtrack_end = not terminated and self._offtrack_steps >= self.config.offtrack_patience
        terminated = terminated or offtrack_end
        truncated = not terminated and self.t >= self.config.max_steps

        s, seg = self.features(self.env)
        info = self._info(seg)
        info.update(lap_finished=lap_finished, off_track=off, offtrack_termination=offtrack_end)
        return to_gray(rgb), s, reward, terminated, truncated, info

    def off_track(self) -> bool:
        return all(len(w.tiles) == 0 for w in self.env.car.wheels)

    def track_fraction(self) -> float:
        return self.env.tile_visited_count / len(self.env.track)

    def _info(self, seg: int) -> dict:
        return {"t": self.t, "segment": seg, "track_fraction": self.track_fraction()}

    def close(self) -> None:
        self.env.close()
