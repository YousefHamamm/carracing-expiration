#!/bin/bash
# Stage 1: train the controller (recurrent PPO, 512 envs, 15M steps).
#
# Usage (from the repo root; logs/ must exist before submitting):
#   mkdir -p logs && OUT_DIR=runs/controller/seed0 sbatch slurm/train_controller.sbatch
#   OUT_DIR=runs/controller/smoke sbatch slurm/train_controller.sbatch --smoke
# Extra arguments are passed to scripts/train_controller.py.
#
# Resume: the script restarts from $OUT_DIR/ckpt_latest.pt. Ten minutes before the time limit
# SLURM sends USR1; training finishes the current update, checkpoints, exits with code 3, and
# this script requeues the job, which resumes on restart.
#
#SBATCH --job-name=ctrl-train
#SBATCH --partition=TODO            # TODO(CLAUDE.md): partition
#SBATCH --account=TODO              # TODO(CLAUDE.md): account
#SBATCH --gres=gpu:1                # TODO(CLAUDE.md): GPU type, e.g. gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64          # env workers = cpus - 2; Box2D + rendering is ~60 ms per control step per core
#SBATCH --mem=64G                   # rollout buffer alone is ~2.4 GB (128 x 512 x K_BUF x 96 x 96 uint8)
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
source slurm/common.sh

OUT_DIR=${OUT_DIR:?set OUT_DIR, e.g. OUT_DIR=runs/controller/seed0}
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))

srun python -u scripts/train_controller.py --out-dir "$OUT_DIR" --num-workers "$WORKERS" "$@"
rc=$?

if [ "$rc" -eq 3 ]; then
    echo "[slurm] checkpointed before the time limit; requeueing ${SLURM_JOB_ID}"
    scontrol requeue "$SLURM_JOB_ID"
fi
exit "$rc"
