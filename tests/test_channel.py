import numpy as np
import pytest

from carexp.channel import BernoulliChannel


def test_success_rate_matches_p_s():
    ch = BernoulliChannel(p_s=0.3, cost=2.0, seed=0)
    n = 20000
    hits = sum(ch.transmit() for _ in range(n))
    assert abs(hits / n - 0.3) < 4 * np.sqrt(0.3 * 0.7 / n)
    assert ch.attempts == n and ch.successes == hits
    assert ch.total_cost == pytest.approx(2.0 * n)


def test_seeded_reproducible_and_reset():
    a = BernoulliChannel(0.5, seed=7)
    b = BernoulliChannel(0.5, seed=7)
    seq_a = [a.transmit() for _ in range(100)]
    assert seq_a == [b.transmit() for _ in range(100)]
    a.reset(seed=7)
    assert a.attempts == 0
    assert seq_a == [a.transmit() for _ in range(100)]


@pytest.mark.parametrize("p, expected", [(0.0, False), (1.0, True)])
def test_extremes(p, expected):
    ch = BernoulliChannel(p, seed=0)
    assert all(ch.transmit() == expected for _ in range(100))


def test_invalid_p():
    with pytest.raises(ValueError):
        BernoulliChannel(1.5)
