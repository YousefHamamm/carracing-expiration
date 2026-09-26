"""Table III: periodic transmission, p_s x m grid, paired tracks, return +/- s.e. and track completion."""

from __future__ import annotations

import csv
import math
import signal
import time
from pathlib import Path

import torch
from tqdm import tqdm

from carexp.config import deep_merge
from carexp.controller.model import load_controller
from carexp.env import EnvConfig
from carexp.env.pool import EnvPool
from carexp.eval.closed_loop import Interrupted, Job, run_jobs
from carexp.eval.stats import summarize
from carexp.eval.store import CellStore, file_sha256
from carexp.schedulers import PeriodicScheduler
from carexp.seeding import check_disjoint, check_range


def cell_key(p_s: float, m: int) -> str:
    return f"ps{p_s:.2f}_m{m}"


def format_cell(p_s: float, m: int, s: dict, secs: float | None = None) -> str:
    t = f" {secs:6.0f}s" if secs is not None else " (cached)"
    return (f"[cell p_s={p_s:.2f} m={m}] return {s['return_mean']:8.1f} ± {s['return_se']:5.1f} | "
            f"track {100 * s['track_mean']:5.1f}% ± {100 * s['track_se']:4.1f} | laps {100 * s['lap_rate']:5.1f}% | "
            f"tx/step {s['tx_per_step']:.3f} | len {s['length_mean']:6.1f} | n={s['n']}{t}")


def write_table(store: CellStore, out_dir: Path, p_list, m_list, n_tracks: int) -> str:
    """Writes table_iii.csv (long) and table_iii.md (grids) from finished cells; returns the markdown."""
    rows, grid = [], {}
    for p in p_list:
        for m in m_list:
            key = cell_key(p, m)
            if store.is_done(key, n_tracks):
                s = store.load_summary(key)
                grid[p, m] = s
                rows.append({"p_s": p, "m": m, **{k: s[k] for k in (
                    "n", "return_mean", "return_se", "track_mean", "track_se", "lap_rate",
                    "tx_per_step", "length_mean")}})
    if rows:
        with open(out_dir / "table_iii.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def md_grid(title, fmt):
        head = "| p_s \\ m | " + " | ".join(str(m) for m in m_list) + " |"
        sep = "|---|" + "---|" * len(m_list)
        body = []
        for p in p_list:
            cells = [fmt(grid[p, m]) if (p, m) in grid else "–" for m in m_list]
            body.append(f"| {p:.1f} | " + " | ".join(cells) + " |")
        return "\n".join([f"**{title}**", "", head, sep, *body])

    md = "\n\n".join([
        f"Periodic transmission, {n_tracks} paired tracks per cell (mean ± s.e.)",
        md_grid("Return", lambda s: f"{s['return_mean']:.1f} ± {s['return_se']:.1f}"),
        md_grid("Track completed (%)", lambda s: f"{100 * s['track_mean']:.1f} ± {100 * s['track_se']:.1f}"),
    ]) + "\n"
    (out_dir / "table_iii.md").write_text(md)
    return md


class _StopFlag:
    def __init__(self):
        self.stop = False
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, self._set)

    def _set(self, *_):
        self.stop = True


def run_periodic_table(cfg: dict, checkpoint: str | Path, out_dir: str | Path) -> dict:
    """Evaluates every missing cell; finished cells are loaded from disk.
    Returns {'reason': 'done' | 'interrupted', 'cells_run': int}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rt = cfg["runtime"]
    device = rt["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    stop = _StopFlag()

    model, train_cfg = load_controller(checkpoint, device)
    env_dict = deep_merge(train_cfg["env"], cfg.get("env_overrides") or {})
    lo, n = cfg["tracks"]["start"], cfg["tracks"]["n"]
    check_range("evaluation", lo, lo + n, tuple(cfg["seeds"]["eval"]))
    check_disjoint({"evaluation": (lo, lo + n), "controller training": (train_cfg["seeds"]["track_start"], math.inf)})
    seeds = list(range(lo, lo + n))
    p_list = [float(p) for p in cfg["grid"]["p_s"]]
    m_list = [int(m) for m in cfg["grid"]["m"]]

    meta = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "tracks": [lo, n],
        "env": env_dict,
        "k_buf": model.k_buf,
        "channel_seed": cfg["channel_seed"],
        "deterministic": cfg["policy"]["deterministic"],
        "policy_seed": cfg["policy"]["seed"],
    }
    store = CellStore(out_dir, meta)

    tqdm.write(f"[table3] checkpoint={checkpoint} sha256={meta['checkpoint_sha256'][:12]} device={device}")
    tqdm.write(f"[table3] tracks=[{lo}, {lo + n}) channel_seed={cfg['channel_seed']} "
               f"deterministic={meta['deterministic']} env={env_dict}")
    todo = []
    for p in p_list:
        for m in m_list:
            if store.is_done(cell_key(p, m), n):
                tqdm.write(format_cell(p, m, store.load_summary(cell_key(p, m))))
            else:
                todo.append((p, m))
    tqdm.write(f"[table3] {len(p_list) * len(m_list) - len(todo)} cells on disk, {len(todo)} to run")

    reason = "done"
    if todo:
        jobs = []
        for p, m in todo:
            sched = PeriodicScheduler(m)
            jobs += [Job((p, m), seed, p, sched) for seed in seeds]
        t0 = {"t": time.time()}
        bar = tqdm(total=len(jobs), unit="ep", desc="table3", dynamic_ncols=True)

        def on_cell_done(cell, results):
            p, m = cell
            rows = [r.as_row() for r in results]
            summary = {"p_s": p, "m": m, **summarize(rows)}
            store.save(cell_key(p, m), rows, summary)
            write_table(store, out_dir, p_list, m_list, n)
            tqdm.write(format_cell(p, m, summary, time.time() - t0["t"]))
            t0["t"] = time.time()

        num_slots = min(rt["num_slots"], len(jobs))
        try:
            with EnvPool(num_slots, rt["num_workers"], EnvConfig.from_dict(env_dict)) as pool:
                run_jobs(model, pool, jobs, cfg["channel_seed"], device,
                         deterministic=cfg["policy"]["deterministic"], policy_seed=cfg["policy"]["seed"],
                         on_cell_done=on_cell_done, on_episode_done=lambda *_: bar.update(1),
                         stop_check=lambda: stop.stop)
        except Interrupted:
            reason = "interrupted"
            tqdm.write("[table3] stop signal received; finished cells are saved, exiting")
        finally:
            bar.close()

    md = write_table(store, out_dir, p_list, m_list, n)
    if reason == "done":
        tqdm.write("\n" + md)
    return {"reason": reason, "cells_run": len(todo)}
