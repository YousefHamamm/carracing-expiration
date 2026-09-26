"""Sender-side state vector s_n, extracted from CarRacing internals.

Layout (31 dims with n_curvature=20), all in Box2D world units / radians:
    0:2   body-frame velocity (lateral, forward); forward is the car's +y axis
    2     speed
    3     yaw rate (positive = counter-clockwise)
    4     heading error w.r.t. the track tangent at the nearest segment, in [-pi, pi)
    5:9   wheel angular velocities (front-left, front-right, rear-left, rear-right)
    9     steering angle, sign matched to action[0] (positive = steer right)
    10    lateral offset from the lane center (positive = left of the driving direction)
    11:31 signed curvature of the next n_curvature segments (positive = left turn), 1/unit length

Heading is relative to the track rather than global, because the global angle carries no
information on a random track. Values are unnormalized; models normalize.
"""

from __future__ import annotations

import numpy as np

N_BASE = 11


def state_dim(n_curvature: int = 20) -> int:
    return N_BASE + n_curvature


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def track_points(track) -> np.ndarray:
    """(N, 2) centerline points from env.unwrapped.track entries (alpha, beta, x, y)."""
    return np.asarray(track, dtype=np.float64)[:, 2:4]


def segment_headings(points: np.ndarray) -> np.ndarray:
    """Heading of segment i (points[i] -> points[i+1]) on the closed loop."""
    d = np.roll(points, -1, axis=0) - points
    return np.arctan2(d[:, 1], d[:, 0])


def segment_curvatures(points: np.ndarray) -> np.ndarray:
    """Signed curvature at segment i: heading change to segment i+1 over the mean segment length."""
    d = np.roll(points, -1, axis=0) - points
    ds = np.linalg.norm(d, axis=1)
    phi = np.arctan2(d[:, 1], d[:, 0])
    dphi = wrap_angle(np.roll(phi, -1) - phi)
    return dphi / (0.5 * (ds + np.roll(ds, -1)))


def track_relative(points: np.ndarray, pos, heading: float, n_curvature: int,
                   curvatures: np.ndarray | None = None):
    """Nearest segment, lateral offset, heading error and upcoming curvature for a pose.

    Returns (segment_index, lateral_offset, heading_error, curvature[n_curvature]).
    """
    pos = np.asarray(pos, dtype=np.float64)
    a = points
    d = np.roll(points, -1, axis=0) - points
    seg_len2 = np.einsum("ij,ij->i", d, d)
    t = np.clip(np.einsum("ij,ij->i", pos - a, d) / seg_len2, 0.0, 1.0)
    proj = a + t[:, None] * d
    dist2 = np.einsum("ij,ij->i", pos - proj, pos - proj)
    i = int(np.argmin(dist2))

    tangent = d[i] / np.sqrt(seg_len2[i])
    rel = pos - a[i]
    lateral = float(tangent[0] * rel[1] - tangent[1] * rel[0])
    heading_err = float(wrap_angle(heading - np.arctan2(tangent[1], tangent[0])))

    if curvatures is None:
        curvatures = segment_curvatures(points)
    idx = (i + np.arange(n_curvature)) % len(points)
    return i, lateral, heading_err, curvatures[idx]


class StateExtractor:
    """Computes s_n from a CarRacing env. Caches per-track geometry; call reset() after env.reset()."""

    def __init__(self, n_curvature: int = 20):
        self.n_curvature = n_curvature
        self._points = None
        self._curv = None

    @property
    def dim(self) -> int:
        return state_dim(self.n_curvature)

    def reset(self, env) -> None:
        self._points = track_points(env.unwrapped.track)
        self._curv = segment_curvatures(self._points)

    def __call__(self, env) -> tuple[np.ndarray, int]:
        """Returns (s_n as float32, index of the nearest track segment)."""
        car = env.unwrapped.car
        hull = car.hull
        v_body = hull.GetLocalVector(hull.linearVelocity)
        fwd = hull.GetWorldVector((0.0, 1.0))
        heading = float(np.arctan2(fwd[1], fwd[0]))
        seg, lateral, heading_err, curv = track_relative(
            self._points, hull.position, heading, self.n_curvature, self._curv)

        s = np.empty(self.dim, dtype=np.float64)
        s[0] = v_body[0]
        s[1] = v_body[1]
        s[2] = np.hypot(v_body[0], v_body[1])
        s[3] = hull.angularVelocity
        s[4] = heading_err
        s[5:9] = [w.omega for w in car.wheels]
        s[9] = -car.wheels[0].joint.angle  # env applies steer(-action[0])
        s[10] = lateral
        s[11:] = curv
        return s.astype(np.float32), seg
