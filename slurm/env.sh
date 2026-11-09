# Shared environment for LogicTune jobs on a Slurm cluster (sourced by the other scripts).
# Override any of these before sourcing, e.g. `export LOGICTUNE_WORK=/work/<group>/<user>/logictune`.
: "${LOGICTUNE_WORK:=$HOME/logictune}"           # venv, HF cache, runs, logs
: "${LOGICTUNE_SRC:=$LOGICTUNE_WORK/src}"         # synced copy of this repository
: "${LOGICTUNE_VENV:=$LOGICTUNE_WORK/env/venv}"
export LOGICTUNE_WORK LOGICTUNE_SRC LOGICTUNE_VENV
export HF_HOME="${HF_HOME:-$LOGICTUNE_WORK/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$LOGICTUNE_WORK"/{runs,logs} "$HF_HOME"
