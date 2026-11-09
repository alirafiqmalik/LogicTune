#!/bin/bash -l
#SBATCH --job-name=logictune-probe
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
# Usage: RUN_NAME=probe MODEL=<hf id or path> sbatch slurm/probe.sbatch
# Evaluates one model on the evaluation prompts (no training).
set -euo pipefail
source "${LOGICTUNE_SRC:-$PWD}/slurm/env.sh"
source "$LOGICTUNE_VENV/bin/activate"
cd "$LOGICTUNE_SRC"
OUT="$LOGICTUNE_WORK/runs/${RUN_NAME:?set RUN_NAME}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
python main.py evaluate --model "${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" --tag "${TAG:-base}" --out "$OUT" --n-samples "${N_SAMPLES:-4}"
