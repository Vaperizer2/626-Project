#!/usr/bin/env bash
# Run the frozen harness on each candidate: eval, time, then compare against the deployed baseline (acceptance)
# and against the dense TensorRT fp16 baseline (the effect of pruning at equal runtime).
#   tools/prune/harness_queue.sh CANDIDATE_DIR...      logs: artifacts/prune/logs/<id>.<step>.log
#   RETIME=1 tools/prune/harness_queue.sh ...          time + compare only (re-measure in another GPU state)
set -uo pipefail
# The head fit's GridSearchCV(n_jobs=-1) spawns up to 15 loky workers; 4 keep eval inside the 6 GB scope
# (an uncapped eval was OOM-killed there) and give identical fits.
export LOKY_MAX_CPU_COUNT=${LOKY_MAX_CPU_COUNT:-4}
H=/home/vaper/Work/Prototypes/626-wt/prune/tools/speed
REF=/home/vaper/Work/Prototypes/626-artifacts/harness/candidates
LOGS=/home/vaper/Work/Prototypes/626-artifacts/prune/logs
mkdir -p "$LOGS"
for c in "$@"; do
  id=$(basename "$c")
  sfx=${RETIME:+.retime}
  if [ -z "${RETIME:-}" ]; then
    "$H/speed" eval "$c" > "$LOGS/$id.eval.log" 2>&1 || { echo "$id eval FAILED"; continue; }
  fi
  "$H/speed" time "$c" > "$LOGS/$id.time$sfx.log" 2>&1 || { echo "$id time FAILED"; continue; }
  "$H/speed" compare "$REF/baseline_fp32" "$c" > "$LOGS/$id.compare_fp32$sfx.log" 2>&1
  "$H/speed" compare "$REF/baseline_trt_fp16" "$c" > "$LOGS/$id.compare_trt$sfx.log" 2>&1
  echo "$id done"
done
