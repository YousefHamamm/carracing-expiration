import numpy as np
import pytest

from carexp.env import CarRacingEnv, EnvConfig, state_dim
from carexp.env.features import segment_curvatures, track_relative
from conftest import random_actions


def circle(radius=50.0, n=200, ccw=True):
    th = 2 * np.pi * np.arange(n) / n
    if not ccw:
        th = -th
    return np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1)


@pytest.mark.parametrize("ccw, sign", [(True, 1.0), (False, -1.0)])
def test_curvature_sign_and_magnitude(ccw, sign):
    r = 50.0
    k = segment_curvatures(circle(r, ccw=ccw))
    np.testing.assert_allclose(k, sign / r, rtol=1e-3)


def test_straight_line_has_zero_curvature():
    pts = np.stack([np.arange(100.0), np.zeros(100)], axis=1)
    # the closing segment of the loop turns back; check only the interior
    assert np.allclose(segment_curvatures(pts)[:90], 0.0)


def test_lateral_offset_and_heading_error_on_circle():
    r = 50.0
    pts = circle(r)  # CCW: driving direction at angle 0 is +y, left is toward the center
    tangent_heading = np.pi / 2
    _, lat_in, herr, curv = track_relative(pts, (r - 2.0, 0.1), tangent_heading, 20)
    _, lat_out, _, _ = track_relative(pts, (r + 2.0, 0.1), tangent_heading + 0.3, 20)
    assert lat_in == pytest.approx(2.0, abs=0.05)
    assert lat_out == pytest.approx(-2.0, abs=0.05)
    assert herr == pytest.approx(0.0, abs=0.05)
    assert curv.shape == (20,)
    _, _, herr2, _ = track_relative(pts, (r, 0.1), tangent_heading + 0.3, 20)
    assert herr2 == pytest.approx(0.3, abs=0.05)


def test_heading_error_wraps():
    pts = circle()
    _, _, herr, _ = track_relative(pts, (50.0, 0.1), np.pi / 2 + 2 * np.pi - 0.1, 20)
    assert herr == pytest.approx(-0.1, abs=0.05)


def test_state_vector_on_real_track(scale):
    env = CarRacingEnv(EnvConfig(offtrack_patience=10**6))
    _, s, info = env.reset(scale["seeds"][0])
    assert s.shape == (state_dim(20),) and s.dtype == np.float32
    # the car starts on the centerline, aligned with the track, at rest
    assert abs(s[10]) < 0.5
    assert abs(s[4]) < 0.05
    assert s[2] < 1.0

    for a in random_actions(scale["steps"], seed=0):
        _, s, _, term, trunc, info = env.step(a)
        assert np.all(np.isfinite(s))
        assert s[2] == pytest.approx(np.hypot(s[0], s[1]), rel=1e-5)
        if term or trunc:
            break


def test_forward_velocity_and_curvature_agree_with_env():
    env = CarRacingEnv(EnvConfig(offtrack_patience=10**6))
    env.reset(0)
    for _ in range(10):
        _, s, *_ = env.step([0.0, 1.0])
    assert s[1] > 5.0              # driving forward along body +y
    assert abs(s[0]) < 0.1 * s[1]  # little sideslip in a straight line
    assert np.all(s[5:9] > 0)      # wheels spin forward

    # track curvature agrees with the env's own tile headings (beta), independent of our geometry
    track = np.asarray(env.unwrapped.track)
    k = segment_curvatures(track[:, 2:4])
    dbeta = np.angle(np.exp(1j * (np.roll(track[:, 1], -1) - track[:, 1])))
    best = max(np.corrcoef(k, np.roll(dbeta, sh))[0, 1] for sh in range(-2, 3))
    assert best > 0.95


def test_steering_sign_matches_action():
    env = CarRacingEnv(EnvConfig(offtrack_patience=10**6))
    env.reset(0)
    for _ in range(3):
        _, s, *_ = env.step([1.0, 0.2])  # steer right
    assert s[9] > 0
