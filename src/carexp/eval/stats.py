"""Summary statistics over per-episode rows (paired by track seed)."""

from __future__ import annotations

import numpy as np


def mean_se(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(se)


def summarize(rows: list[dict]) -> dict:
    ret = [r["ret"] for r in rows]
    frac = [r["track_fraction"] for r in rows]
    steps = sum(r["length"] for r in rows)
    tx = sum(r["transmissions"] for r in rows)
    dl = sum(r["deliveries"] for r in rows)
    ends = [r["end"] for r in rows]
    ret_m, ret_se = mean_se(ret)
    frac_m, frac_se = mean_se(frac)
    return {
        "n": len(rows),
        "return_mean": ret_m, "return_se": ret_se,
        "track_mean": frac_m, "track_se": frac_se,
        "lap_rate": float(np.mean([bool(r["lap_finished"]) for r in rows])) if rows else float("nan"),
        "length_mean": steps / len(rows) if rows else float("nan"),
        "tx_per_step": tx / steps if steps else float("nan"),
        "deliveries_per_step": dl / steps if steps else float("nan"),
        **{f"end_{e}": ends.count(e) for e in ("lap", "offtrack", "playfield", "timeout")},
    }


def paired_difference(a: list[dict], b: list[dict], key: str = "ret") -> tuple[float, float]:
    """Mean and s.e. of a - b over tracks present in both (matched by seed)."""
    bs = {r["seed"]: r[key] for r in b}
    d = [r[key] - bs[r["seed"]] for r in a if r["seed"] in bs]
    return mean_se(d)
