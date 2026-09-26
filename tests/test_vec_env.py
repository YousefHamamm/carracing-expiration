import numpy as np
import pytest

from carexp.controller.vec_env import RemoteDrivingVecEnv
from carexp.env import EnvConfig


@pytest.fixture
def vec():
    v = RemoteDrivingVecEnv(num_envs=3, num_workers=2, env_cfg=EnvConfig(max_steps=4), k_buf=4,
                            track_start=100, delivery_seed=0)
    yield v
    v.close()


def test_reset_observation(vec):
    frames, delta, prev_a, start = vec.reset()
    assert frames.shape == (3, 4, 96, 96) and frames.dtype == np.uint8
    assert np.all(delta == 0) and np.all(prev_a == 0) and np.all(start)
    np.testing.assert_array_equal(vec.episode_counters, [1, 1, 1])


def test_no_delivery_freezes_buffers(vec):
    frames0, *_ = vec.reset()
    for k in range(1, 3):
        (frames, delta, prev_a, start), *_ = vec.step(np.full((3, 2), 0.5), p_s=0.0)
        np.testing.assert_array_equal(frames, frames0)
        assert np.all(delta == k) and not start.any()
        np.testing.assert_allclose(prev_a, 0.5)


def test_full_delivery_updates_newest_frame(vec):
    frames0, *_ = vec.reset()
    (frames, delta, *_), *_ = vec.step(np.tile([0.0, 1.0], (3, 1)), p_s=1.0)
    assert np.all(delta == 0)
    np.testing.assert_array_equal(frames[:, :3], frames0[:, 1:])


def test_episode_end_resets_and_advances_seed(vec):
    vec.reset()
    for _ in range(3):
        _, _, term, trunc, finals = vec.step(np.zeros((3, 2)), p_s=0.0)
        assert not finals
    (frames, delta, prev_a, start), _, term, trunc, finals = vec.step(np.zeros((3, 2)), p_s=0.0)
    assert trunc.all() and not term.any()
    assert sorted(f[0] for f in finals) == [0, 1, 2]
    assert all(f[2] == 4 for f in finals)          # Delta of the final (frozen) buffer
    assert start.all() and np.all(delta == 0) and np.all(prev_a == 0)
    np.testing.assert_array_equal(vec.episode_counters, [2, 2, 2])


def test_seed_rule():
    v = RemoteDrivingVecEnv(3, 1, EnvConfig(), 4, track_start=100, delivery_seed=0,
                            episode_counters=[0, 5, 2])
    try:
        np.testing.assert_array_equal(v._next_seeds(np.arange(3)), [100, 116, 108])
    finally:
        v.close()
