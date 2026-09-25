"""Parallel CarRacing envs with Bernoulli frame delivery and a receiver buffer per env.

Physics runs in worker processes (each owns a slice of the envs). The channel draws and receiver
buffers live in the main process, so workers only ship the newest frame per env.

Track seeds: env i's j-th episode uses track_start + i + num_envs * j, so seeds are deterministic,
never repeat, and resume from the saved per-env episode counters.
"""

from __future__ import annotations

import multiprocessing as mp
import signal

import numpy as np

from carexp.channel import ReceiverBuffer
from carexp.env import FRAME_SHAPE, CarRacingEnv, EnvConfig


def _worker(conn, env_cfg: EnvConfig, n_envs: int):
    # SLURM signals every process in the step; the main process checkpoints and closes workers.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    envs = [CarRacingEnv(env_cfg) for _ in range(n_envs)]
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send(np.stack([e.reset(s)[0] for e, s in zip(envs, data)]))
            elif cmd == "step":
                actions, next_seeds = data
                out = []
                for e, a, seed in zip(envs, actions, next_seeds):
                    frame, _, r, term, trunc, info = e.step(a)
                    final = None
                    if term or trunc:
                        final = (frame, info["track_fraction"], info["lap_finished"])
                        frame = e.reset(seed)[0]
                    out.append((frame, r, term, trunc, final))
                conn.send(out)
            elif cmd == "close":
                break
    finally:
        for e in envs:
            e.close()
        conn.close()


class RemoteDrivingVecEnv:
    """Batched view for the controller: receiver frames (N, K, 96, 96), Delta_n (N,),
    previous action (N, 2), episode-start flags (N,)."""

    def __init__(self, num_envs: int, num_workers: int, env_cfg: EnvConfig, k_buf: int,
                 track_start: int, delivery_seed: int, episode_counters=None):
        self.num_envs = num_envs
        self.track_start = track_start
        self.episode_counters = (np.zeros(num_envs, dtype=np.int64) if episode_counters is None
                                 else np.asarray(episode_counters, dtype=np.int64).copy())
        self.rng = np.random.default_rng(delivery_seed)
        self.receivers = [ReceiverBuffer(k_buf, FRAME_SHAPE) for _ in range(num_envs)]
        self.prev_action = np.zeros((num_envs, 2), dtype=np.float32)
        self.episode_start = np.ones(num_envs, dtype=bool)

        num_workers = max(1, min(num_workers, num_envs))
        self.slices = np.array_split(np.arange(num_envs), num_workers)
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for sl in self.slices:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, env_cfg, len(sl)), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)

    def _next_seeds(self, idx) -> np.ndarray:
        """Seeds for the next episode of each env in idx (consumes them)."""
        seeds = self.track_start + idx + self.num_envs * self.episode_counters[idx]
        self.episode_counters[idx] += 1
        return seeds

    def reset(self):
        for conn, sl in zip(self.conns, self.slices):
            conn.send(("reset", self._next_seeds(sl).tolist()))
        for conn, sl in zip(self.conns, self.slices):
            for i, frame in zip(sl, conn.recv()):
                self.receivers[i].reset(frame)
        self.prev_action[:] = 0.0
        self.episode_start[:] = True
        return self.observation()

    def observation(self):
        frames = np.stack([r.frames for r in self.receivers])
        delta = np.array([r.delta for r in self.receivers], dtype=np.int64)
        return frames, delta, self.prev_action.copy(), self.episode_start.copy()

    def step(self, actions: np.ndarray, p_s: float):
        """Apply actions, then deliver each new frame with probability p_s.

        Returns (obs, rewards, terminated, truncated, finals). finals is a list of
        (env index, final receiver frames, final Delta, episode info) for envs whose episode
        ended; the returned obs for those envs is already the first obs of the next episode.
        Seeds for the next episode are drawn before stepping, so an env that does not finish
        this step keeps its seed (the counter is rolled back).
        """
        actions = np.asarray(actions, dtype=np.float32)
        pending = [self._next_seeds(sl) for sl in self.slices]
        for conn, sl, seeds in zip(self.conns, self.slices, pending):
            conn.send(("step", (actions[sl], seeds.tolist())))

        rewards = np.zeros(self.num_envs, dtype=np.float64)
        term = np.zeros(self.num_envs, dtype=bool)
        trunc = np.zeros(self.num_envs, dtype=bool)
        deliver = self.rng.random(self.num_envs) < p_s
        finals = []
        for conn, sl in zip(self.conns, self.slices):
            for i, (frame, r, te, tr, final) in zip(sl, conn.recv()):
                rewards[i], term[i], trunc[i] = r, te, tr
                rec = self.receivers[i]
                if final is None:
                    self.episode_counters[i] -= 1  # seed not used
                    rec.update(frame if deliver[i] else None)
                    self.prev_action[i] = actions[i]
                    self.episode_start[i] = False
                else:
                    last_frame, frac, lap = final
                    rec.update(last_frame if deliver[i] else None)
                    finals.append((i, rec.frames.copy(), rec.delta,
                                   {"track_fraction": frac, "lap_finished": lap}))
                    rec.reset(frame)
                    self.prev_action[i] = 0.0
                    self.episode_start[i] = True
        return self.observation(), rewards, term, trunc, finals

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
