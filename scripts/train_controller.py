"""Train the frozen remote-driving controller with recurrent PPO.

Resumes automatically from <out-dir>/ckpt_latest.pt. Exit code 0 when training is finished,
3 when stopped by SIGTERM/SIGUSR1 after checkpointing (the sbatch script requeues on 3).
"""

from __future__ import annotations

import argparse
import sys

from carexp.config import CONFIG_DIR, load_stage_config

EXIT_INTERRUPTED = 3


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(CONFIG_DIR / "controller.yaml"))
    p.add_argument("--out-dir", required=True, help="run directory: checkpoints, metrics.csv, config")
    p.add_argument("--smoke", action="store_true", help="tiny run (few envs, few steps)")
    p.add_argument("--seed", type=int)
    p.add_argument("--total-steps", type=int)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--device")
    args = p.parse_args(argv)

    cfg = load_stage_config(args.config, smoke=args.smoke)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.total_steps is not None:
        cfg["ppo"]["total_steps"] = args.total_steps
    if args.num_envs is not None:
        cfg["ppo"]["num_envs"] = args.num_envs
    if args.num_workers is not None:
        cfg["runtime"]["num_workers"] = args.num_workers
    if args.device is not None:
        cfg["runtime"]["device"] = args.device

    # imported here so spawned env workers, which re-import this module, do not load torch
    from carexp.controller.train import train

    result = train(cfg, args.out_dir)
    return EXIT_INTERRUPTED if result["reason"] == "interrupted" else 0


if __name__ == "__main__":
    sys.exit(main())
