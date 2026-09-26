"""Seed + action replay must reproduce frames, s_n and rewards exactly. The expiration oracle
depends on this: it rebuilds state n by replay, then branches."""

import multiprocessing as mp

import numpy as np

from carexp.env import CarRacingEnv, EnvConfig, replay
from conftest import random_actions

# no off-track termination, so episodes run long enough to exercise the dynamics
CFG = EnvConfig(offtrack_patience=10**9, max_steps=10**9)


def rollout(env, seed, actions):
    frame, s, _ = env.reset(seed)
    frames, states, rewards = [frame], [s], []
    for a in actions:
        frame, s, r, term, trunc, _ = env.step(a)
        frames.append(frame)
        states.append(s)
        rewards.append(r)
        if term or trunc:
            break
    return np.stack(frames), np.stack(states), np.array(rewards)


def assert_same(x, y):
    for u, v in zip(x, y):
        np.testing.assert_array_equal(u, v)


def test_fresh_envs_identical(scale):
    for seed in scale["seeds"]:
        acts = random_actions(scale["steps"], seed)
        assert_same(rollout(CarRacingEnv(CFG), seed, acts), rollout(CarRacingEnv(CFG), seed, acts))


def test_reused_env_identical_after_other_episode(scale):
    """Resetting an env that already ran a different track must not leak state."""
    env = CarRacingEnv(CFG)
    for seed in scale["seeds"]:
        acts = random_actions(scale["steps"], seed)
        ref = rollout(CarRacingEnv(CFG), seed, acts)
        rollout(env, seed + 1000, random_actions(scale["steps"], seed + 1000))
        assert_same(rollout(env, seed, acts), ref)


def test_replay_prefix_then_branch(scale):
    """replay(seed, a[:n]) reaches the same state as the original run, and continuing from it matches."""
    seed = scale["seeds"][0]
    acts = random_actions(scale["steps"], seed)
    frames, states, rewards = rollout(CarRacingEnv(CFG), seed, acts)
    env = CarRacingEnv(CFG)
    for n in (1, len(rewards) // 2, len(rewards)):
        frame, s, _ = replay(env, seed, acts[:n])
        np.testing.assert_array_equal(frame, frames[n])
        np.testing.assert_array_equal(s, states[n])
    n = len(rewards) // 2
    replay(env, seed, acts[:n])
    for k, a in enumerate(acts[n:len(rewards)]):
        frame, s, r, *_ = env.step(a)
        np.testing.assert_array_equal(frame, frames[n + k + 1])
        np.testing.assert_array_equal(s, states[n + k + 1])
        assert r == rewards[n + k]


def test_different_seeds_give_different_tracks():
    a = CarRacingEnv(CFG)
    b = CarRacingEnv(CFG)
    a.reset(0)
    b.reset(1)
    assert not np.array_equal(np.asarray(a.unwrapped.track), np.asarray(b.unwrapped.track))


def _worker(args):
    seed, acts = args
    frames, states, rewards = rollout(CarRacingEnv(CFG), seed, acts)
    return frames.tobytes(), states.tobytes(), rewards.tobytes()


def test_identical_across_processes(scale):
    """Oracle labelling runs in worker processes, so determinism must hold across them."""
    seed = scale["seeds"][0]
    acts = random_actions(scale["steps"], seed)
    local = tuple(x.tobytes() for x in rollout(CarRacingEnv(CFG), seed, acts))
    with mp.get_context("spawn").Pool(2) as pool:
        remote = pool.map(_worker, [(seed, acts), (seed, acts)])
    assert remote[0] == local and remote[1] == local
