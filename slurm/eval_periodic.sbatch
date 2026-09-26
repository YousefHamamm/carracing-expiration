#!/bin/bash
# Table III: periodic evaluation of a trained controller (25 cells x 200 paired tracks).
#
# Usage (from the repo root; logs/ must exist before submitting):
#   mkdir -p logs && CKPT=runs/controller/seed0/ckpt_latest.pt OUT_DIR=results/table3 sbatch slurm/eval_periodic.sbatch
# Extra arguments are passed to scripts/eval_periodic.py (e.g. --smoke, --p-s 0.2 0.4).
#
# Finished cells are saved as they complete. On USR1 (10 min before the limit) the job exits
# with code 3 and requeues itself; the restart skips finished cells.
#
#SBATCH --job-name=eval-table3
#SBATCH --partition=TODO            # TODO(CLAUDE.md): partition
#SBATCH --account=TODO              # TODO(CLAUDE.md): account
#SBATCH --gres=gpu:1                # TODO(CLAUDE.md): GPU type; the controller runs fine on CPU too (drop this line)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
source slurm/common.sh

CKPT=${CKPT:?set CKPT to the controller checkpoint}
OUT_DIR=${OUT_DIR:?set OUT_DIR, e.g. OUT_DIR=results/table3}
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))

srun python -u scripts/eval_periodic.py --checkpoint "$CKPT" --out-dir "$OUT_DIR" \
    --num-workers "$WORKERS" --num-slots $(( WORKERS * 4 )) "$@"
rc=$?

if [ "$rc" -eq 3 ]; then
    echo "[slurm] interrupted; requeueing ${SLURM_JOB_ID}"
    scontrol requeue "$SLURM_JOB_ID"
fi
exit "$rc"
