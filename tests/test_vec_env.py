import numpy as np
import pytest

from carexp.controller.curriculum import BERNOULLI, DeliveryCurriculum, transmits
from carexp.controller.vec_env import RemoteDrivingVecEnv
from carexp.env import EnvConfig


def fixed(p_s, period=None, total=1000):
    """Curriculum that always yields the same pattern: Bernoulli if period is None, else periodic m."""
    return DeliveryCurriculum(total, p_min_start=p_s, p_min_end=p_s, p_max=p_s,
                              periodic_prob=0.0 if period is None else 1.0, periods=(period or 1,))


def make_vec(curriculum, num_envs=3, max_steps=4, counters=None):
    return RemoteDrivingVecEnv(num_envs=num_envs, num_workers=2, env_cfg=EnvConfig(max_steps=max_steps),
                               k_buf=4, track_start=100, delivery_seed=0, curriculum=curriculum,
                               episode_counters=counters)


@pytest.fixture
def vec_factory():
    made = []

    def make(*args, **kw):
        v = make_vec(*args, **kw)
        made.append(v)
        return v
    yield make
    for v in made:
        v.close()


# --- curriculum ------------------------------------------------------------------------------

def test_p_s_sampled_per_env_within_current_range():
    cur = DeliveryCurriculum(total_steps=1000)  # p_min: 1.0 -> 0.3 over the first 400 steps
    rng = np.random.default_rng(0)
    for step, p_min in [(0, 1.0), (100, 0.825), (200, 0.65), (400, 0.3), (900, 0.3)]:
        p_s, _ = cur.sample(rng, step, 4000)
        assert cur.p_min(step) == pytest.approx(p_min)
        assert p_s.min() >= p_min - 1e-12 and p_s.max() <= 1.0
        if p_min < 1.0:
            # one draw per env, spread over the whole range rather than a single shared value
            assert len(np.unique(p_s)) == len(p_s)
            assert p_s.min() < p_min + 0.05 and p_s.max() > 0.95
            assert p_s.mean() == pytest.approx((p_min + 1.0) / 2, abs=0.02)


def test_pattern_mix():
    cur = DeliveryCurriculum(total_steps=1000)
    _, period = cur.sample(np.random.default_rng(0), 0, 6000)
    assert (period == BERNOULLI).mean() == pytest.approx(0.5, abs=0.03)
    counts = np.bincount(period[period > 0], minlength=7)[1:]
    assert set(np.unique(period[period > 0])) == {1, 2, 3, 4, 5, 6}
    assert counts.min() > 0.8 * counts.mean()


def test_transmits():
    n = np.arange(1, 7)
    assert transmits(np.zeros(6, int), n).all()
    assert list(transmits(np.full(6, 3), n)) == [False, False, True, False, False, True]


# --- vec env ---------------------------------------------------------------------------------

def test_vec_env_draws_p_s_per_env_and_per_episode(vec_factory):
    """At step 0 the range is [1, 1]; episodes that start after the ramp draw from [0.3, 1],
    independently per env."""
    vec = vec_factory(DeliveryCurriculum(total_steps=100), num_envs=6, max_steps=2)
    vec.reset(step=0)
    assert np.all(vec.p_s == 1.0)
    vec.step(np.zeros((6, 2)), step=10)
    assert np.all(vec.p_s == 1.0)                  # no episode ended yet: patterns unchanged
    _, _, _, trunc, finals = vec.step(np.zeros((6, 2)), step=50)
    assert trunc.all()
    assert all(f[3]["p_s"] == 1.0 for f in finals)  # finished episodes report their own p_s
    assert np.all((vec.p_s >= 0.3) & (vec.p_s <= 1.0))
    assert len(np.unique(vec.p_s)) == 6
    assert not np.all(vec.p_s == 1.0)


def test_reset_observation(vec_factory):
    vec = vec_factory(fixed(1.0))
    frames, ages, prev_a, start = vec.reset()
    assert frames.shape == (3, 4, 96, 96) and frames.dtype == np.uint8
    assert ages.shape == (3, 4) and np.all(ages == 0)
    assert np.all(prev_a == 0) and np.all(start)
    np.testing.assert_array_equal(vec.episode_counters, [1, 1, 1])


def test_no_delivery_freezes_buffers(vec_factory):
    vec = vec_factory(fixed(0.0))
    frames0, *_ = vec.reset()
    for k in range(1, 3):
        (frames, ages, prev_a, start), *_ = vec.step(np.full((3, 2), 0.5), step=0)
        np.testing.assert_array_equal(frames, frames0)
        assert np.all(ages == k) and not start.any()
        np.testing.assert_allclose(prev_a, 0.5)
    assert vec.pop_channel_counts() == (6, 0)   # Bernoulli sends every step, nothing arrives


def test_full_delivery_updates_newest_frame(vec_factory):
    vec = vec_factory(fixed(1.0))
    frames0, *_ = vec.reset()
    (frames, ages, *_), *_ = vec.step(np.tile([0.0, 1.0], (3, 1)), step=0)
    assert np.all(ages == [1, 1, 1, 0])
    np.testing.assert_array_equal(frames[:, :3], frames0[:, 1:])


def test_periodic_delivery_pattern(vec_factory):
    vec = vec_factory(fixed(1.0, period=3), max_steps=20)
    vec.reset()
    newest = []
    for _ in range(7):
        (_, ages, *_), *_ = vec.step(np.zeros((3, 2)), step=0)
        newest.append(int(ages[0, -1]))
    assert newest == [1, 2, 0, 1, 2, 0, 1]
    assert list(ages[0]) == [7, 7, 4, 1]
    assert vec.pop_channel_counts() == (6, 6)


def test_periodic_with_loss(vec_factory):
    vec = vec_factory(DeliveryCurriculum(1000, 0.5, 0.5, 0.5, periodic_prob=1.0, periods=(2,)),
                      num_envs=4, max_steps=40)
    vec.reset()
    for _ in range(30):
        vec.step(np.zeros((4, 2)), step=0)
    sent, delivered = vec.pop_channel_counts()
    assert sent == 4 * 15
    assert 0 < delivered < sent


def test_episode_end_resets_and_advances_seed(vec_factory):
    vec = vec_factory(fixed(0.0))
    vec.reset()
    for _ in range(3):
        _, _, term, trunc, finals = vec.step(np.zeros((3, 2)), step=0)
        assert not finals
    (frames, ages, prev_a, start), _, term, trunc, finals = vec.step(np.zeros((3, 2)), step=0)
    assert trunc.all() and not term.any()
    assert sorted(f[0] for f in finals) == [0, 1, 2]
    assert all(list(f[2]) == [4, 4, 4, 4] for f in finals)   # ages of the final (frozen) buffer
    assert start.all() and np.all(ages == 0) and np.all(prev_a == 0)
    np.testing.assert_array_equal(vec.episode_counters, [2, 2, 2])
    assert np.all(vec.n == 0)


def test_seed_rule(vec_factory):
    v = vec_factory(fixed(1.0), counters=[0, 5, 2])
    np.testing.assert_array_equal(v._next_seeds(np.arange(3)), [100, 116, 108])
