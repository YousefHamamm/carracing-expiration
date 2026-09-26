"""The closed loop must show the controller exactly what the channel delivered."""

import numpy as np
import pytest
import torch

from carexp.env import CarRacingEnv, EnvConfig
from carexp.env.pool import EnvPool
from carexp.eval.closed_loop import Interrupted, Job, run_jobs
from carexp.schedulers import PeriodicScheduler

ENV = EnvConfig(max_steps=12, offtrack_patience=10**6)
ACTION = np.array([0.1, 0.6], dtype=np.float32)


class RecordingModel:
    """Stands in for the controller: fixed action, records every observation per slot."""

    k_buf = 4
    hidden_size = 1

    def __init__(self):
        self.seen = []  # (frames, delta, prev_action, start) of slot 0 per call

    def initial_state(self, batch, device=None):
        return torch.zeros(batch, 1)

    def act(self, frames, delta, prev_action, h, start, deterministic=True, generator=None):
        self.seen.append((frames[0].numpy().copy(), int(delta[0]), prev_action[0].numpy().copy(), bool(start[0])))
        B = len(delta)
        a = torch.as_tensor(np.tile(ACTION, (B, 1)))
        return a, a, torch.zeros(B), torch.zeros(B), h


def env_frames(seed, n):
    env = CarRacingEnv(ENV)
    frames = [env.reset(seed)[0]]
    for _ in range(n):
        frames.append(env.step(ACTION)[0])
    return frames


def run_single(p_s, m, seed=5, channel_seed=0):
    model = RecordingModel()
    with EnvPool(1, 1, ENV) as pool:
        res = run_jobs(model, pool, [Job("c", seed, p_s, PeriodicScheduler(m))], channel_seed)
    return model.seen, res["c"][0]


def test_periodic_full_delivery_buffer_contents():
    seen, ep = run_single(p_s=1.0, m=2)
    frames = env_frames(5, ep.length)
    assert ep.length == 12 and ep.end == "timeout"
    assert ep.transmissions == 6 and ep.deliveries == 6
    assert [d for _, d, _, _ in seen] == [0, 1] * 6
    assert seen[0][3] and not any(s for *_, s in seen[1:])
    np.testing.assert_array_equal(seen[0][2], [0, 0])
    np.testing.assert_allclose(seen[1][2], ACTION)
    # what the controller sees at step n: newest delivered frame = y_{2*floor(n/2)}, older ones before it
    for n, (buf, delta, _, _) in enumerate(seen):
        newest = n - delta
        np.testing.assert_array_equal(buf[-1], frames[newest])
        if newest >= 2:
            np.testing.assert_array_equal(buf[-2], frames[newest - 2])


def test_no_delivery_freezes_on_initial_frame():
    seen, ep = run_single(p_s=0.0, m=3)
    frames = env_frames(5, 1)
    assert ep.transmissions == 4 and ep.deliveries == 0
    assert [d for _, d, _, _ in seen] == list(range(12))
    for buf, *_ in seen:
        assert all(np.array_equal(f, frames[0]) for f in buf)


def test_common_random_numbers():
    """Deliveries follow u_n < p_s with u_n from rng([channel_seed, track_seed]), drawn every step."""
    seen, ep = run_single(p_s=0.5, m=1, seed=5, channel_seed=7)
    u = np.random.default_rng([7, 5]).random(12)
    delivered = u < 0.5
    assert ep.deliveries == delivered.sum()
    expected_delta, d = [0], 0
    for ok in delivered[:-1]:
        d = 0 if ok else d + 1
        expected_delta.append(d)
    assert [x for _, x, _, _ in seen] == expected_delta

    # with m=2 the same draws are used: step n is delivered iff n is even and u_n < p_s
    _, ep2 = run_single(p_s=0.5, m=2, seed=5, channel_seed=7)
    assert ep2.deliveries == delivered[1::2].sum()


def test_many_cells_share_slots_and_report_each_cell():
    model = RecordingModel()
    done = []
    jobs = [Job(("a", m), s, 1.0, PeriodicScheduler(m)) for m in (1, 2) for s in (3, 1, 2)]
    with EnvPool(4, 2, ENV) as pool:
        res = run_jobs(model, pool, jobs, 0, on_cell_done=lambda c, r: done.append((c, [e.seed for e in r])))
    assert sorted(done) == [(("a", 1), [1, 2, 3]), (("a", 2), [1, 2, 3])]
    # paired: same track gives the same trajectory here (fixed action), whatever the period
    by_seed = {m: {e.seed: e.ret for e in res[("a", m)]} for m in (1, 2)}
    assert by_seed[1] == pytest.approx(by_seed[2])
    assert [e.transmissions for e in res[("a", 2)]] == [6, 6, 6]


def test_stop_check_interrupts():
    with EnvPool(1, 1, ENV) as pool, pytest.raises(Interrupted):
        run_jobs(RecordingModel(), pool, [Job("c", 0, 1.0, PeriodicScheduler(1))], 0, stop_check=lambda: True)
