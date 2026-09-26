"""Table III: evaluate a trained controller under periodic transmission (p_s x m grid).

Each cell is saved to <out-dir>/cells/ when it finishes and printed; a restart skips finished
cells. Exit code 0 when all cells are done, 3 when stopped by SIGTERM/SIGUSR1.
"""

from __future__ import annotations

import argparse
import sys

from carexp.config import CONFIG_DIR, load_stage_config

EXIT_INTERRUPTED = 3


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="controller checkpoint (.pt) from train_controller.py")
    p.add_argument("--out-dir", required=True, help="results directory")
    p.add_argument("--config", default=str(CONFIG_DIR / "eval_periodic.yaml"))
    p.add_argument("--smoke", action="store_true", help="tiny run (2x2 grid, 3 tracks, short episodes)")
    p.add_argument("--n-tracks", type=int)
    p.add_argument("--p-s", type=float, nargs="+")
    p.add_argument("--m", type=int, nargs="+")
    p.add_argument("--num-workers", type=int)
    p.add_argument("--num-slots", type=int)
    p.add_argument("--device")
    args = p.parse_args(argv)

    cfg = load_stage_config(args.config, smoke=args.smoke)
    if args.n_tracks is not None:
        cfg["tracks"]["n"] = args.n_tracks
    if args.p_s is not None:
        cfg["grid"]["p_s"] = args.p_s
    if args.m is not None:
        cfg["grid"]["m"] = args.m
    for name in ("num_workers", "num_slots", "device"):
        if getattr(args, name) is not None:
            cfg["runtime"][name] = getattr(args, name)

    # imported here so spawned env workers, which re-import this module, do not load torch
    from carexp.eval.periodic import run_periodic_table

    result = run_periodic_table(cfg, args.checkpoint, args.out_dir)
    return EXIT_INTERRUPTED if result["reason"] == "interrupted" else 0


if __name__ == "__main__":
    sys.exit(main())
