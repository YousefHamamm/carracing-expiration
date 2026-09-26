# carracing-expiration

Expiration-aware scheduling for remote driving on Gymnasium CarRacing-v3. See `CLAUDE.md` for the project spec.

## Setup

```
pip install -e ".[test]"
```

## Tests

```
pytest --smoke   # tiny version, ~2 minutes
pytest           # full version (env tests use 5 track seeds x 300 control steps; ~10 minutes)
```

## Layout so far

- `src/carexp/env/carracing.py`: CarRacing wrapper (2-D action map, action repeat, grayscale frames, off-track termination, time limit)
- `src/carexp/env/features.py`: the sender's state vector s_n (31 dims; layout in the module docstring)
- `src/carexp/env/replay.py`: rebuilds a state by seed + action replay
- `src/carexp/channel/channel.py`: Bernoulli(p_s) channel with per-attempt cost
- `src/carexp/channel/receiver.py`: receiver buffer of the last K_BUF delivered frames plus Delta_n
- `configs/base.yaml`: env and receiver settings
- `src/carexp/controller/model.py`: IMPALA CNN + GRU controller, tanh-Gaussian policy, value head
- `src/carexp/controller/vec_env.py`: parallel envs (worker processes) with Bernoulli delivery and a receiver buffer per env
- `src/carexp/controller/ppo.py`: recurrent PPO, GAE, KL-adaptive learning rate, KL early stop
- `src/carexp/controller/curriculum.py`: p_s schedule
- `src/carexp/controller/train.py`: training loop, metrics.csv, checkpoint/resume, SIGUSR1/SIGTERM handling
- `configs/controller.yaml`: controller + PPO settings (`smoke:` section used with --smoke)
- `src/carexp/env/pool.py`: env worker pool with per-slot resets (evaluation)
- `src/carexp/schedulers/`: scheduler interface and the periodic baseline
- `src/carexp/eval/closed_loop.py`: env -> scheduler -> channel -> receiver -> controller loop over many episodes/cells
- `src/carexp/eval/store.py`, `stats.py`: per-cell result files with resume; mean ± s.e., paired differences
- `src/carexp/eval/periodic.py`, `configs/eval_periodic.yaml`: Table III

## Controller training

```
python scripts/train_controller.py --smoke --out-dir runs/controller/smoke        # ~20 s
mkdir -p logs && OUT_DIR=runs/controller/seed0 sbatch slurm/train_controller.sbatch
```

The run directory holds `config.yaml`, `seeds.json`, `metrics.csv` (one row per update) and
checkpoints (`ckpt_latest.pt` plus `ckpt_NNNNNN.pt` every `checkpoint_every` updates).
Re-running with the same `--out-dir` resumes from `ckpt_latest.pt`.

## Table III (periodic transmission)

```
python scripts/eval_periodic.py --smoke --checkpoint runs/controller/smoke/ckpt_latest.pt --out-dir results/table3_smoke
mkdir -p logs && CKPT=runs/controller/seed0/ckpt_latest.pt OUT_DIR=results/table3 sbatch slurm/eval_periodic.sbatch
```

Every cell (p_s, m) runs the same 200 evaluation tracks (seeds 0-199) with the same channel draws
per track. Each finished cell is written to `cells/ps<p>_m<m>.csv` (one row per track) and `.json`
(summary), printed, and folded into `table_iii.csv` / `table_iii.md`. A restart skips finished
cells; `meta.json` pins the checkpoint hash and setup, and a restart with a different setup is refused.
