import numpy as np
import pytest

from carexp.channel import ReceiverBuffer


def frame(v):
    return np.full((96, 96), v, dtype=np.uint8)


def newest_values(buf):
    frames, _ = buf.observation()
    return [int(f[0, 0]) for f in frames]


def test_reset_fills_all_slots():
    buf = ReceiverBuffer(4)
    buf.reset(frame(9))
    assert newest_values(buf) == [9, 9, 9, 9]
    assert buf.delta == 0


def test_delivery_shifts_and_resets_delta():
    buf = ReceiverBuffer(4)
    buf.reset(frame(0))
    for v in (1, 2, 3):
        buf.update(frame(v))
    assert newest_values(buf) == [0, 1, 2, 3]
    buf.update(frame(4))
    assert newest_values(buf) == [1, 2, 3, 4]
    assert buf.delta == 0


def test_no_delivery_freezes_buffer_and_increments_delta():
    buf = ReceiverBuffer(4)
    buf.reset(frame(0))
    buf.update(frame(1))
    before = buf.observation()[0]
    for k in range(1, 6):
        buf.update(None)
        assert buf.delta == k
    np.testing.assert_array_equal(buf.observation()[0], before)
    buf.update(frame(2))
    assert buf.delta == 0
    assert newest_values(buf) == [0, 0, 1, 2]


def test_observation_is_a_copy():
    buf = ReceiverBuffer(2)
    buf.reset(frame(1))
    obs, _ = buf.observation()
    obs[:] = 0
    assert newest_values(buf) == [1, 1]


@pytest.mark.parametrize("k", [1, 6])
def test_other_buffer_sizes(k):
    buf = ReceiverBuffer(k)
    buf.reset(frame(0))
    buf.update(frame(5))
    assert buf.observation()[0].shape == (k, 96, 96)
    assert newest_values(buf)[-1] == 5


def test_invalid_k():
    with pytest.raises(ValueError):
        ReceiverBuffer(0)
