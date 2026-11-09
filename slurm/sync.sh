#!/bin/bash
# Copy this repository to the cluster (run from your workstation).
# Usage: UNITY_HOST=unity LOGICTUNE_WORK=/work/<group>/<user>/logictune slurm/sync.sh
set -euo pipefail
: "${UNITY_HOST:?set UNITY_HOST (an ssh host alias)}"
: "${LOGICTUNE_WORK:?set LOGICTUNE_WORK (remote work directory)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rsync -az --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude 'runs/' \
  --exclude 'results/' "$ROOT/" "$UNITY_HOST:$LOGICTUNE_WORK/src/"
ssh "$UNITY_HOST" "cd $LOGICTUNE_WORK/src && echo '$(git -C "$ROOT" rev-parse HEAD)' > SOURCE_COMMIT"
echo "synced to $UNITY_HOST:$LOGICTUNE_WORK/src"
