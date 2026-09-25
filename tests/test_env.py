import numpy as np
import pytest

from carexp.config import DEFAULT_CONFIG, load_config
from carexp.env import CarRacingEnv, EnvConfig, map_action, state_dim, to_gray


def test_frame_and_state_types():
    env = CarRacingEnv()
    frame, s, info = env.reset(0)
    assert frame.shape == (96, 96) and frame.dtype == np.uint8
    assert s.shape == (state_dim(20),)
    frame, s, r, term, trunc, info = env.step([0.0, 0.5])
    assert frame.shape == (96, 96) and frame.dtype == np.uint8
    assert isinstance(r, float) and not term and not trunc
    assert info["t"] == 1


def test_to_gray():
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    rgb[..., 1] = 200
    assert np.all(to_gray(rgb) == round(0.587 * 200))


@pytest.mark.parametrize("repeat", [1, 6])
def test_action_repeat_advances_physics(repeat):
    env = CarRacingEnv(EnvConfig(action_repeat=repeat))
    env.reset(0)
    t0 = env.unwrapped.t
    for _ in range(3):
        env.step([0.0, 0.5])
    assert env.unwrapped.t - t0 == pytest.approx(3 * repeat / 50.0)


def test_reward_is_sum_over_repeated_frames():
    from gymnasium.envs.box2d.car_racing import CarRacing
    raw = CarRacing(continuous=True)
    raw.reset(seed=0)
    env = CarRacingEnv()
    env.reset(0)
    for a in ([0.0, 0.5], [0.3, 1.0], [-0.2, -0.5]):
        expected = sum(raw.step(map_action(a))[1] for _ in range(6))
        _, _, r, *_ = env.step(a)
        assert r == pytest.approx(expected)


def test_truncation_at_max_steps():
    env = CarRacingEnv(EnvConfig(max_steps=5))
    env.reset(0)
    for n in range(5):
        _, _, _, term, trunc, _ = env.step([0.0, 0.0])
    assert trunc and not term


def test_offtrack_termination():
    env = CarRacingEnv(EnvConfig(offtrack_patience=1))
    env.reset(0)
    for _ in range(100):
        _, _, _, term, trunc, info = env.step([1.0, 1.0])  # hard right at full throttle
        if term:
            break
    assert term and info["offtrack_termination"] and info["off_track"]


def test_offtrack_patience_delays_termination():
    env = CarRacingEnv(EnvConfig(offtrack_patience=3))
    env.reset(0)
    off_run = 0
    for _ in range(100):
        _, _, _, term, _, info = env.step([1.0, 1.0])
        off_run = off_run + 1 if info["off_track"] else 0
        if term:
            break
    assert term and off_run == 3


def test_track_fraction_increases_when_driving():
    env = CarRacingEnv()
    _, _, info0 = env.reset(0)
    for _ in range(10):
        _, _, _, _, _, info = env.step([0.0, 0.5])
    assert info["track_fraction"] > info0["track_fraction"]


def test_config_file_builds_env():
    cfg = load_config(DEFAULT_CONFIG)
    env_cfg = EnvConfig.from_dict(cfg["env"])
    assert env_cfg.action_repeat == 6
    with pytest.raises(ValueError):
        EnvConfig.from_dict({"bogus": 1})
