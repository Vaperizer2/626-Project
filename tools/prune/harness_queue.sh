#!/usr/bin/env bash
# Run the frozen harness on each candidate: eval, time, then compare against the deployed baseline (acceptance)
# and against the dense TensorRT fp16 baseline (the effect of pruning at equal runtime).
#   tools/prune/harness_queue.sh CANDIDATE_DIR...      logs: artifacts/prune/logs/<id>.<step>.log
set -uo pipefail
H=/home/vaper/Work/Prototypes/626-wt/prune/tools/speed
REF=/home/vaper/Work/Prototypes/626-artifacts/harness/candidates
LOGS=/home/vaper/Work/Prototypes/626-artifacts/prune/logs
mkdir -p "$LOGS"
for c in "$@"; do
  id=$(basename "$c")
  "$H/speed" eval "$c" > "$LOGS/$id.eval.log" 2>&1 || { echo "$id eval FAILED"; continue; }
  "$H/speed" time "$c" > "$LOGS/$id.time.log" 2>&1 || { echo "$id time FAILED"; continue; }
  "$H/speed" compare "$REF/baseline_fp32" "$c" > "$LOGS/$id.compare_fp32.log" 2>&1
  "$H/speed" compare "$REF/baseline_trt_fp16" "$c" > "$LOGS/$id.compare_trt.log" 2>&1
  echo "$id done"
done
