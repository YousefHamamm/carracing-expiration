import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--smoke", action="store_true", default=False,
                     help="tiny version of each test (few seeds, few steps)")


@pytest.fixture(scope="session")
def smoke(request) -> bool:
    return request.config.getoption("--smoke")


@pytest.fixture(scope="session")
def scale(smoke) -> dict:
    """Number of track seeds and control steps per episode used by the env-driving tests."""
    return {"seeds": [0, 1], "steps": 40} if smoke else {"seeds": [0, 1, 2, 3, 4], "steps": 300}


def random_actions(n: int, seed: int) -> np.ndarray:
    """Smooth random [steer, throttle] sequence with a forward bias, so the car moves and turns."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n, 2))
    steer, thr = 0.0, 0.3
    for i in range(n):
        steer = 0.8 * steer + 0.2 * rng.uniform(-1, 1)
        thr = 0.8 * thr + 0.2 * rng.uniform(-0.3, 1.0)
        a[i] = steer, thr
    return a
