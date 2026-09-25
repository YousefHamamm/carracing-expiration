"""Recurrent PPO: rollout storage, GAE, KL-adaptive learning rate, and the update with KL early stop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from carexp.controller.model import Controller, gaussian_entropy, squashed_log_prob


class RolloutBuffer:
    """One segment of T steps for N envs, stored on the CPU. h0 is the GRU state entering step 0,
    before the episode-start mask, so the update can replay the segment exactly."""

    def __init__(self, T: int, N: int, k_buf: int, frame_shape, hidden: int):
        self.T, self.N = T, N
        self.frames = np.zeros((T, N, k_buf, *frame_shape), dtype=np.uint8)
        self.delta = np.zeros((T, N), dtype=np.int64)
        self.prev_action = np.zeros((T, N, 2), dtype=np.float32)
        self.episode_start = np.zeros((T, N), dtype=bool)
        self.u = np.zeros((T, N, 2), dtype=np.float32)
        self.log_prob = np.zeros((T, N), dtype=np.float32)
        self.value = np.zeros((T, N), dtype=np.float32)
        self.reward = np.zeros((T, N), dtype=np.float32)
        self.done = np.zeros((T, N), dtype=bool)
        self.h0 = torch.zeros(N, hidden)
        self.advantage = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)


def compute_gae(reward, value, done, last_value, gamma: float, lam: float):
    """GAE over (T, N) arrays. done[t] means the episode ended after step t (no bootstrap
    through it; truncation bootstrap is folded into reward by the caller)."""
    T = reward.shape[0]
    adv = np.zeros_like(reward, dtype=np.float32)
    last = np.zeros(reward.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else value[t + 1]
        nonterminal = 1.0 - done[t].astype(np.float32)
        delta = reward[t] + gamma * next_value * nonterminal - value[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + value


def adapt_lr(lr: float, kl: float, kl_target: float, lr_min: float, lr_max: float,
             factor: float = 1.5) -> float:
    """Shrink lr when KL > 2*target, grow it when KL < target/2, clip to [lr_min, lr_max]."""
    if kl > 2.0 * kl_target:
        lr /= factor
    elif kl < 0.5 * kl_target:
        lr *= factor
    return float(min(max(lr, lr_min), lr_max))


def sampled_kl(log_ratio: torch.Tensor) -> torch.Tensor:
    """Sample estimate of KL(old || new) from log(pi_new/pi_old) at old-policy samples:
    E[(r - 1) - log r], unbiased and non-negative."""
    return ((log_ratio.exp() - 1) - log_ratio).mean()


@dataclass
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
    clip_frac: float
    epochs_done: float  # fractional: minibatches completed / minibatches per epoch
    early_stopped: bool
    lr: float


def ppo_update(model: Controller, optimizer: torch.optim.Optimizer, buf: RolloutBuffer, cfg: dict,
               lr: float, device, generator: torch.Generator | None = None):
    """Runs up to cfg['epochs'] epochs over env-wise minibatches of full sequences.

    Before each minibatch's gradient step, the sampled KL between the rollout policy and the
    current policy is measured on that minibatch; if it exceeds early_stop_kl_mult * kl_target,
    the update stops without taking that step. The KL used to adapt lr is the mean over the
    minibatches measured in the last epoch that ran. Returns (UpdateStats, new lr).
    """
    kl_stop = cfg["early_stop_kl_mult"] * cfg["kl_target"]
    n_mb = cfg["minibatches"]
    to = lambda x, dt=None: torch.as_tensor(x, device=device, dtype=dt)  # noqa: E731

    for g in optimizer.param_groups:
        g["lr"] = lr

    stats = {"pl": [], "vl": [], "ent": [], "cf": []}
    epoch_kls: list[float] = []
    done_mbs, stopped = 0, False
    for _ in range(cfg["epochs"]):
        epoch_kls = []
        perm = torch.randperm(buf.N, generator=generator).numpy()
        for mb in np.array_split(perm, n_mb):
            mean, log_std, value = model.unroll(
                to(buf.frames[:, mb]), to(buf.delta[:, mb]), to(buf.prev_action[:, mb]),
                buf.h0[mb].to(device), to(buf.episode_start[:, mb]))
            u = to(buf.u[:, mb])
            logp = squashed_log_prob(u, mean, log_std)
            log_ratio = logp - to(buf.log_prob[:, mb])
            kl = sampled_kl(log_ratio.detach()).item()
            epoch_kls.append(kl)
            if kl > kl_stop:
                stopped = True
                break

            adv = to(buf.advantage[:, mb])
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = log_ratio.exp()
            clip = cfg["clip"]
            pl = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            vl = 0.5 * ((value - to(buf.returns[:, mb])) ** 2).mean()
            ent = gaussian_entropy(log_std).mean()
            loss = pl + cfg["vf_coef"] * vl - cfg["ent_coef"] * ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()

            stats["pl"].append(pl.item())
            stats["vl"].append(vl.item())
            stats["ent"].append(ent.item())
            stats["cf"].append(((ratio - 1).abs() > clip).float().mean().item())
            done_mbs += 1
        if stopped:
            break

    kl = float(np.mean(epoch_kls))
    new_lr = adapt_lr(lr, kl, cfg["kl_target"], cfg["lr_min"], cfg["lr_max"], cfg["lr_factor"])
    avg = lambda k: float(np.mean(stats[k])) if stats[k] else float("nan")  # noqa: E731
    return UpdateStats(avg("pl"), avg("vl"), avg("ent"), kl, avg("cf"), done_mbs / n_mb,
                       stopped, lr), new_lr
