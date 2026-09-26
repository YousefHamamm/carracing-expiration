import numpy as np
import pytest

from carexp.env import map_action


@pytest.mark.parametrize("a, expected", [
    ([0.0, 0.0], [0.0, 0.0, 0.0]),
    ([0.5, 0.7], [0.5, 0.7, 0.0]),
    ([-0.3, -0.4], [-0.3, 0.0, 0.4]),
    ([1.0, 1.0], [1.0, 1.0, 0.0]),
    ([-1.0, -1.0], [-1.0, 0.0, 1.0]),
    ([2.0, -5.0], [1.0, 0.0, 1.0]),   # clipped
])
def test_map_action(a, expected):
    np.testing.assert_allclose(map_action(a), expected)


def test_gas_and_brake_never_both_positive():
    for thr in np.linspace(-1, 1, 21):
        _, gas, brake = map_action([0.0, thr])
        assert gas * brake == 0.0


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        map_action([0.0, 0.0, 0.0])
