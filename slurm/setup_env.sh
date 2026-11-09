#!/bin/bash -l
# Create the pinned Python environment for LogicTune (run once on a login node).
set -euo pipefail
source "$(dirname "$0")/env.sh"
module load uv/latest python/3.12.3 2>/dev/null || true
export UV_CACHE_DIR="$LOGICTUNE_WORK/uv_cache"
uv venv --python "$(which python3)" "$LOGICTUNE_VENV"
source "$LOGICTUNE_VENV/bin/activate"
# PyTorch wheels come from the official index (it publishes no upload dates,
# so a site-wide UV_EXCLUDE_NEWER policy must be lifted for this exact pin only).
env -u UV_EXCLUDE_NEWER uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
uv pip install -r "$LOGICTUNE_SRC/requirements.txt"
python -c "import torch, transformers, trl, peft; print(torch.__version__, transformers.__version__, trl.__version__, peft.__version__)"
