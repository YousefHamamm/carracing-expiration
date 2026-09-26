"""Controller training loop: rollouts under random Bernoulli delivery, PPO updates, checkpoint/resume."""

from __future__ import annotations

import csv
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from carexp.controller.curriculum import delivery_prob
from carexp.controller.model import Controller, count_parameters
from carexp.controller.ppo import RolloutBuffer, compute_gae, ppo_update
from carexp.controller.vec_env import RemoteDrivingVecEnv
from carexp.env import FRAME_SHAPE, EnvConfig
from carexp.utils.checkpoint import load_latest, save_checkpoint

METRIC_FIELDS = ["update", "global_step", "p_s", "lr", "episodes", "ep_return", "ep_return_se",
                 "ep_len", "track_fraction", "lap_rate", "delivery_rate", "policy_loss",
                 "value_loss", "entropy", "kl", "clip_frac", "epochs_done", "early_stopped",
                 "sps", "wall_time"]


class _StopFlag:
    """Set on SIGTERM/SIGUSR1 (SLURM preemption/time limit): finish the update, checkpoint, exit."""

    def __init__(self):
        self.stop = False
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, self._handler)

    def _handler(self, signum, frame):
        self.stop = True


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _prepare_metrics(path: Path, resumed_update: int) -> None:
    """Keep only rows up to the resumed checkpoint, so a restart does not duplicate updates."""
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, METRIC_FIELDS).writeheader()
        return
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["update"]) <= resumed_update]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, METRIC_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _diff_keys(a, b, prefix=""):
    """Dotted keys whose values differ between two nested dicts."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return [prefix.rstrip(".")] if a != b else []
    out = []
    for k in sorted(set(a) | set(b), key=str):
        out += _diff_keys(a.get(k), b.get(k), f"{prefix}{k}.")
    return out


def train(cfg: dict, out_dir: str | Path) -> dict:
    """Trains (or resumes) the controller. Returns a summary dict. Exit reason is
    'done' when total_steps is reached, 'interrupted' after a stop signal."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ppo_cfg, rt = cfg["ppo"], cfg["runtime"]
    N, T, total = ppo_cfg["num_envs"], ppo_cfg["rollout_len"], ppo_cfg["total_steps"]
    k_buf = cfg["receiver"]["k_buf"]
    device = _resolve_device(rt["device"])
    stop_flag = _StopFlag()

    torch.manual_seed(cfg["seed"])
    model = Controller.from_config(k_buf, cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg["lr_init"], eps=1e-5)
    sample_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 1)
    mb_gen = torch.Generator().manual_seed(cfg["seed"] + 2)
    delivery_seed = cfg["seed"] + 3

    lr, global_step, update, episode_counters = ppo_cfg["lr_init"], 0, 0, None
    ckpt = load_latest(out_dir, map_location=device)
    if ckpt is not None:
        if ckpt["num_envs"] != N:
            raise ValueError(f"checkpoint has num_envs={ckpt['num_envs']}, config has {N}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr, global_step, update = ckpt["lr"], ckpt["global_step"], ckpt["update"]
        episode_counters = ckpt["episode_counters"]
        torch.set_rng_state(ckpt["torch_rng"])
        sample_gen.set_state(ckpt["sample_gen"].to(sample_gen.get_state().device))
        mb_gen.set_state(ckpt["mb_gen"])
        changed = _diff_keys(ckpt["config"], cfg)
        if changed:
            tqdm.write(f"[controller] WARNING: config differs from checkpoint at: {', '.join(changed)}")
    else:
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    seeds_log = {"seed": cfg["seed"], "sample_gen": cfg["seed"] + 1, "minibatch_gen": cfg["seed"] + 2,
                 "delivery": delivery_seed, "track_start": cfg["seeds"]["track_start"],
                 "track_rule": "track_start + env_index + num_envs * episode_index"}
    with open(out_dir / "seeds.json", "w") as f:
        json.dump(seeds_log, f, indent=2)

    metrics_path = out_dir / "metrics.csv"
    _prepare_metrics(metrics_path, update)

    tqdm.write(f"[controller] device={device} params={count_parameters(model):,} k_buf={k_buf} "
               f"envs={N} T={T} total_steps={total:,} out={out_dir}")
    tqdm.write(f"[controller] seeds: {seeds_log}")
    if ckpt is not None:
        tqdm.write(f"[controller] resumed from update {update}, step {global_step:,}, lr {lr:.2e}")
    if global_step >= total:
        tqdm.write("[controller] already finished")
        return {"reason": "done", "update": update, "global_step": global_step}

    vec = RemoteDrivingVecEnv(N, rt["num_workers"], EnvConfig.from_dict(cfg["env"]), k_buf,
                              cfg["seeds"]["track_start"], delivery_seed, episode_counters)
    if ckpt is not None:
        vec.rng.bit_generator.state = ckpt["delivery_rng"]

    buf = RolloutBuffer(T, N, k_buf, FRAME_SHAPE, model.hidden_size)
    to = lambda x: torch.as_tensor(x, device=device)  # noqa: E731
    gamma, rscale = ppo_cfg["gamma"], ppo_cfg["reward_scale"]
    cur = cfg["curriculum"]

    reason = "done"
    t_start = time.time()
    pbar = tqdm(total=total, initial=global_step, unit="step", dynamic_ncols=True, desc="controller")
    try:
        obs = vec.reset()
        h = model.initial_state(N, device)
        ep_return = np.zeros(N)
        ep_len = np.zeros(N, dtype=np.int64)
        while global_step < total:
            t0 = time.time()
            p_s = delivery_prob(global_step, total, cur["p_start"], cur["p_end"], cur["ramp_fraction"])
            buf.h0 = h.detach().cpu().clone()
            completed, deliveries, frames_seen = [], 0, 0
            for t in range(T):
                frames, delta, prev_a, start = obs
                buf.frames[t], buf.delta[t], buf.prev_action[t], buf.episode_start[t] = frames, delta, prev_a, start
                a, u, logp, v, h = model.act(to(frames), to(delta), to(prev_a), h, to(start),
                                             generator=sample_gen)
                a_np = a.cpu().numpy()
                obs, r, term, trunc, finals = vec.step(a_np, p_s)
                deliveries += int((obs[1] == 0).sum() - obs[3].sum())
                frames_seen += N - int(obs[3].sum())

                ep_return += r
                ep_len += 1
                r_scaled = (r * rscale).astype(np.float32)
                boot = [f for f in finals if trunc[f[0]] and not term[f[0]]]
                if boot:
                    idx = [f[0] for f in boot]
                    with torch.no_grad():
                        _, _, v_final, _ = model.step(
                            to(np.stack([f[1] for f in boot])), to(np.array([f[2] for f in boot])),
                            to(a_np[idx]), h[idx], torch.zeros(len(idx), device=device))
                    r_scaled[idx] += gamma * v_final.cpu().numpy()
                for i, _, _, info in finals:
                    completed.append((ep_return[i], ep_len[i], info["track_fraction"], info["lap_finished"]))
                    ep_return[i], ep_len[i] = 0.0, 0

                buf.u[t], buf.log_prob[t], buf.value[t] = u.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()
                buf.reward[t], buf.done[t] = r_scaled, term | trunc

            with torch.no_grad():
                frames, delta, prev_a, start = obs
                _, _, last_v, _ = model.step(to(frames), to(delta), to(prev_a), h, to(start))
            buf.advantage, buf.returns = compute_gae(buf.reward, buf.value, buf.done,
                                                     last_v.cpu().numpy(), gamma, ppo_cfg["gae_lambda"])

            stats, lr = ppo_update(model, optimizer, buf, ppo_cfg, lr, device, mb_gen)
            global_step += T * N
            update += 1
            pbar.update(T * N)

            ep = np.array(completed, dtype=np.float64).reshape(-1, 4)
            n_ep = len(ep)
            row = {
                "update": update, "global_step": global_step, "p_s": round(p_s, 4), "lr": stats.lr,
                "episodes": n_ep,
                "ep_return": ep[:, 0].mean() if n_ep else float("nan"),
                "ep_return_se": ep[:, 0].std(ddof=1) / np.sqrt(n_ep) if n_ep > 1 else float("nan"),
                "ep_len": ep[:, 1].mean() if n_ep else float("nan"),
                "track_fraction": ep[:, 2].mean() if n_ep else float("nan"),
                "lap_rate": ep[:, 3].mean() if n_ep else float("nan"),
                "delivery_rate": deliveries / max(frames_seen, 1),
                "policy_loss": stats.policy_loss, "value_loss": stats.value_loss,
                "entropy": stats.entropy, "kl": stats.kl, "clip_frac": stats.clip_frac,
                "epochs_done": stats.epochs_done, "early_stopped": int(stats.early_stopped),
                "sps": T * N / (time.time() - t0), "wall_time": time.time() - t_start,
            }
            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, METRIC_FIELDS).writerow(row)
            pbar.set_postfix(ret=f"{row['ep_return']:.1f}", p_s=f"{p_s:.2f}", kl=f"{stats.kl:.4f}", lr=f"{stats.lr:.1e}")
            tqdm.write(
                f"[update {update:5d} | step {global_step:>11,}] p_s={p_s:.3f} lr={stats.lr:.2e}->{lr:.2e} "
                f"eps={n_ep} return={row['ep_return']:.1f}±{row['ep_return_se']:.1f} "
                f"track={row['track_fraction']:.3f} kl={stats.kl:.4f} epochs={stats.epochs_done:.2f}"
                f"{' (KL stop)' if stats.early_stopped else ''} vloss={stats.value_loss:.3f} "
                f"sps={row['sps']:.0f}")

            finished = global_step >= total
            if update % rt["checkpoint_every"] == 0 or finished or stop_flag.stop:
                state = {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "lr": lr,
                    "global_step": global_step, "update": update, "num_envs": N,
                    "episode_counters": vec.episode_counters.copy(),
                    "torch_rng": torch.get_rng_state(), "sample_gen": sample_gen.get_state(),
                    "mb_gen": mb_gen.get_state(), "delivery_rng": vec.rng.bit_generator.state,
                    "config": cfg,
                }
                path = save_checkpoint(state, out_dir, update, keep_milestone=not stop_flag.stop)
                tqdm.write(f"[checkpoint] update {update} -> {path}")
            if stop_flag.stop and not finished:
                reason = "interrupted"
                tqdm.write("[controller] stop signal received; checkpoint saved, exiting")
                break
    finally:
        pbar.close()
        vec.close()
    return {"reason": reason, "update": update, "global_step": global_step}
