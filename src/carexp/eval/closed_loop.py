"""Closed-loop remote driving: env -> sender scheduler -> channel -> receiver -> frozen controller.

Episodes (jobs) from many cells share one pool of env slots, so slow tracks in one cell do not
leave workers idle. Results are grouped by cell, and on_cell_done fires as soon as a cell's
last episode finishes.

Per control step n >= 1 of an episode:
  1. the controller acts on the receiver buffer (the initial frame is there at n = 0, Delta = 0);
  2. the env advances, producing y_n and s_n;
  3. the scheduler decides whether to send y_n; a sent frame is delivered iff u_n < p_s,
     where u_n is drawn every step from rng([channel_seed, track_seed]) whether or not the frame
     is sent (common random numbers: the same track sees the same channel draws in every cell);
  4. the receiver pushes the frame if delivered, otherwise freezes and Delta increments.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Callable, Hashable

import numpy as np
import torch

from carexp.channel import ReceiverBuffer
from carexp.controller.model import Controller
from carexp.env import FRAME_SHAPE
from carexp.env.pool import EnvPool
from carexp.schedulers.base import Scheduler


class Interrupted(Exception):
    pass


@dataclass
class Job:
    cell: Hashable
    seed: int
    p_s: float
    scheduler: Scheduler


@dataclass
class EpisodeResult:
    seed: int
    ret: float
    track_fraction: float
    lap_finished: bool
    length: int
    transmissions: int
    deliveries: int
    end: str  # lap | offtrack | playfield | timeout

    def as_row(self) -> dict:
        return asdict(self)


def _end_reason(term: bool, info: dict) -> str:
    if not term:
        return "timeout"
    if info.get("lap_finished"):
        return "lap"
    if info.get("offtrack_termination"):
        return "offtrack"
    return "playfield"


class _Slot:
    def __init__(self, k_buf: int):
        self.receiver = ReceiverBuffer(k_buf, FRAME_SHAPE)
        self.job: Job | None = None
        self.running = False

    def start(self, job: Job, frame, channel_seed: int):
        self.job = job
        self.receiver.reset(frame)
        self.rng = np.random.default_rng([channel_seed, job.seed])
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.episode_start = True
        self.n = 0
        self.ret = 0.0
        self.transmissions = 0
        self.deliveries = 0
        self.running = True


def run_jobs(model: Controller, pool: EnvPool, jobs: list[Job], channel_seed: int,
             device="cpu", deterministic: bool = True, policy_seed: int = 0,
             on_cell_done: Callable[[Hashable, list[EpisodeResult]], None] | None = None,
             on_episode_done: Callable[[Job, EpisodeResult], None] | None = None,
             stop_check: Callable[[], bool] | None = None) -> dict:
    """Runs every job to the end of its episode. Returns {cell: [EpisodeResult sorted by seed]}."""
    device = torch.device(device)
    S = pool.n_slots
    slots = [_Slot(model.k_buf) for _ in range(S)]
    h = model.initial_state(S, device)
    gen = torch.Generator(device=device).manual_seed(policy_seed)

    remaining = defaultdict(int)
    for j in jobs:
        remaining[j.cell] += 1
    results = defaultdict(list)
    queue = deque(jobs)
    to_reset = {}
    for i in range(min(S, len(queue))):
        to_reset[i] = queue.popleft()

    while to_reset or any(s.running for s in slots):
        if stop_check is not None and stop_check():
            raise Interrupted()

        active = [i for i, s in enumerate(slots) if s.running]
        actions = {}
        if active:
            idx = torch.as_tensor(active, device=device)
            frames = torch.as_tensor(np.stack([slots[i].receiver.frames for i in active]), device=device)
            delta = torch.as_tensor([slots[i].receiver.delta for i in active], device=device)
            prev_a = torch.as_tensor(np.stack([slots[i].prev_action for i in active]), device=device)
            start = torch.as_tensor([slots[i].episode_start for i in active], device=device)
            a, _, _, _, h_new = model.act(frames, delta, prev_a, h[idx], start,
                                          deterministic=deterministic, generator=gen)
            h[idx] = h_new
            a = a.cpu().numpy()
            actions = {i: a[k] for k, i in enumerate(active)}

        reset_out, step_out = pool.run({i: j.seed for i, j in to_reset.items()}, actions)
        starting, to_reset = to_reset, {}

        for i, (frame, s, r, term, trunc, info) in step_out.items():
            sl = slots[i]
            job = sl.job
            sl.n += 1
            sl.ret += r
            sl.prev_action = actions[i].astype(np.float32)
            sl.episode_start = False
            u = sl.rng.random()
            if job.scheduler.decide(i, sl.n, frame, s):
                sl.transmissions += 1
                delivered = bool(u < job.p_s)
                sl.deliveries += delivered
                sl.receiver.update(frame if delivered else None)
                job.scheduler.feedback(i, sl.n, delivered)
            else:
                sl.receiver.update(None)

            if term or trunc:
                res = EpisodeResult(job.seed, sl.ret, info["track_fraction"], bool(info.get("lap_finished")),
                                    sl.n, sl.transmissions, sl.deliveries, _end_reason(term, info))
                sl.running = False
                results[job.cell].append(res)
                remaining[job.cell] -= 1
                if on_episode_done is not None:
                    on_episode_done(job, res)
                if remaining[job.cell] == 0:
                    results[job.cell].sort(key=lambda e: e.seed)
                    if on_cell_done is not None:
                        on_cell_done(job.cell, results[job.cell])
                if queue:
                    to_reset[i] = queue.popleft()

        for i, (frame, s, _) in reset_out.items():
            job = starting[i]
            slots[i].start(job, frame, channel_seed)
            job.scheduler.reset(i, frame, s)

    return dict(results)
