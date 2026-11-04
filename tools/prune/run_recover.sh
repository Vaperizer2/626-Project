#!/usr/bin/env bash
# carlpc2: recover a pruned candidate by feature distillation (GPU part under the shared lock), then write the
# recovered candidate next to it.
#   setsid nohup tools/prune/run_recover.sh NAME CANDIDATE_DIR [recover.py train args] > LOG 2>&1 < /dev/null &
set -euo pipefail
NAME=$1
CAND=$2
shift 2
R=/home/carl/626-runs/prune
PY=/home/carl/Projects/626-Project/.venv/bin/python
export PRUNE_BASE=$R/base/baseline_fp32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$R/code"
echo "waiting for /tmp/626-gpu.lock $(date -Is)"
flock /tmp/626-gpu.lock bash -c "echo 'lock acquired' \$(date -Is); nvidia-smi --query-gpu=memory.used --format=csv;
  $PY recover.py train --candidate '$CAND' --out '$R/runs/$NAME' $*"
$PY build.py load "$R/candidates/$NAME" --base "$CAND" --state "$R/runs/$NAME/state.npz"
echo "done $(date -Is)"
