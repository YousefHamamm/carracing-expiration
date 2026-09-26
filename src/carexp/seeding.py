"""Track seed ranges: stages must not share tracks."""

from __future__ import annotations


def check_range(name: str, lo: int, hi: int, allowed: tuple[int, int]) -> None:
    """[lo, hi) must lie inside allowed = [a, b)."""
    a, b = allowed
    if not (a <= lo and hi <= b):
        raise ValueError(f"{name} seeds [{lo}, {hi}) fall outside their range [{a}, {b})")


def check_disjoint(ranges: dict[str, tuple[float, float]]) -> None:
    """Raises if any two half-open ranges overlap."""
    items = sorted(ranges.items(), key=lambda kv: kv[1][0])
    for (n1, (lo1, hi1)), (n2, (lo2, hi2)) in zip(items, items[1:]):
        if lo2 < hi1:
            raise ValueError(f"seed ranges overlap: {n1} [{lo1}, {hi1}) and {n2} [{lo2}, {hi2})")
