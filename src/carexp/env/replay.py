"""Reconstruct a state by seed + action replay (Box2D worlds cannot be deep-copied)."""

from __future__ import annotations

from carexp.env.carracing import CarRacingEnv


def replay(env: CarRacingEnv, seed: int, actions):
    """Reset env with seed and apply actions in order. Returns the last (frame, s, info).

    Raises if the episode ends before all actions are applied, since the caller asked
    for a state that the recorded trajectory never reached.
    """
    frame, s, info = env.reset(seed)
    for n, a in enumerate(actions):
        frame, s, _, term, trunc, info = env.step(a)
        if (term or trunc) and n < len(actions) - 1:
            raise RuntimeError(f"episode ended at step {n + 1} of {len(actions)} during replay")
    return frame, s, info
