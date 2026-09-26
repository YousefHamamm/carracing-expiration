"""Pool of CarRacing envs in worker processes, addressed by slot, with explicit per-slot resets.

Used by evaluation, where each slot runs one episode on a chosen track seed at a time.
This module does not import torch, so spawned workers stay small.
"""

from __future__ import annotations

import multiprocessing as mp
import signal

import numpy as np

from carexp.env.carracing import CarRacingEnv, EnvConfig


def _pool_worker(conn, env_cfg: EnvConfig, n_envs: int):
    # SLURM signals every process in the step; the main process decides what to do.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    envs = [CarRacingEnv(env_cfg) for _ in range(n_envs)]
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "run":
                resets, actions = data
                reset_out = {i: envs[i].reset(seed) for i, seed in resets.items()}
                step_out = {i: envs[i].step(a) for i, a in actions.items()}
                conn.send((reset_out, step_out))
            elif cmd == "close":
                break
    finally:
        for e in envs:
            e.close()
        conn.close()


class EnvPool:
    """run(resets={slot: seed}, actions={slot: action}) resets and steps the named slots.

    Returns ({slot: (frame, s, info)}, {slot: (frame, s, reward, terminated, truncated, info)}).
    A slot is either reset or stepped in one call, not both.
    """

    def __init__(self, n_slots: int, num_workers: int, env_cfg: EnvConfig):
        self.n_slots = n_slots
        num_workers = max(1, min(num_workers, n_slots))
        groups = np.array_split(np.arange(n_slots), num_workers)
        self._where = {}  # slot -> (worker, local index)
        for w, g in enumerate(groups):
            for j, slot in enumerate(g):
                self._where[int(slot)] = (w, j)
        self._base = [int(g[0]) for g in groups]
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for g in groups:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_pool_worker, args=(child, env_cfg, len(g)), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)

    def run(self, resets: dict, actions: dict):
        both = set(resets) & set(actions)
        if both:
            raise ValueError(f"slots both reset and stepped: {sorted(both)}")
        per_worker = [({}, {}) for _ in self.conns]
        for slot, seed in resets.items():
            w, j = self._where[slot]
            per_worker[w][0][j] = int(seed)
        for slot, a in actions.items():
            w, j = self._where[slot]
            per_worker[w][1][j] = np.asarray(a, dtype=np.float32)
        busy = [w for w, (r, a) in enumerate(per_worker) if r or a]
        for w in busy:
            self.conns[w].send(("run", per_worker[w]))
        reset_out, step_out = {}, {}
        for w in busy:
            r, s = self.conns[w].recv()
            reset_out.update({self._base[w] + j: v for j, v in r.items()})
            step_out.update({self._base[w] + j: v for j, v in s.items()})
        return reset_out, step_out

    def close(self):
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
