# Sourced by every sbatch script. Fill in the cluster-specific parts.
# TODO(CLAUDE.md): module loads and conda env name.
# module load cuda/12.x
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate carexp

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=30   # one progress line every 30 s in the log file instead of a carriage-return stream
export OMP_NUM_THREADS=1     # env workers are single-threaded; torch in the main process sets its own threads
echo "[slurm] job=${SLURM_JOB_ID:-local} node=$(hostname) restarts=${SLURM_RESTART_COUNT:-0} start=$(date -Is)"
