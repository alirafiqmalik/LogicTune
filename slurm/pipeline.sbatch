#!/bin/bash -l
#SBATCH --job-name=logictune-pipeline
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
# Usage: RUN_NAME=tinyllama-v1 sbatch -p <partition> --constraint=<gpus> slurm/pipeline.sbatch [pipeline flags]
# The pipeline skips finished stages, so a preempted and requeued job resumes.
set -euo pipefail
source "${LOGICTUNE_SRC:-$PWD}/slurm/env.sh"
source "$LOGICTUNE_VENV/bin/activate"
cd "$LOGICTUNE_SRC"
RUN_DIR="$LOGICTUNE_WORK/runs/${RUN_NAME:?set RUN_NAME}"
mkdir -p "$RUN_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$RUN_DIR/gpu-${SLURM_JOB_ID}.csv"
(git -C "$LOGICTUNE_SRC" rev-parse HEAD 2>/dev/null || cat "$LOGICTUNE_SRC/SOURCE_COMMIT" 2>/dev/null) > "$RUN_DIR/source_commit.txt" || true
python main.py pipeline --run-dir "$RUN_DIR" "$@"
