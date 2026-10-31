#!/usr/bin/env bash
# Harness pass over candidates: eval each, then time them interleaved for R rounds, then compare with the baseline.
#   run.sh [-r ROUNDS] [-t] <candidate_dir>...    (-t: time only, reuse the latest eval)
set -uo pipefail
H=/home/vaper/Work/Prototypes/626-wt/harness/tools/speed
BASE=/home/vaper/Work/Prototypes/626-artifacts/harness/candidates/baseline_fp32
ROUNDS=1 TIME_ONLY=0
while getopts "r:t" o; do case $o in r) ROUNDS=$OPTARG ;; t) TIME_ONLY=1 ;; *) exit 2 ;; esac; done
shift $((OPTIND - 1))
if [ "$TIME_ONLY" = 0 ]; then
  for c in "$@"; do echo "=== eval $c"; "$H/speed" eval "$c" || echo "EVAL FAILED $c"; done
fi
for r in $(seq "$ROUNDS"); do
  for c in "$@"; do echo "=== time $c (round $r)"; "$H/speed" time "$c" || echo "TIME FAILED $c"; done
done
for c in "$@"; do echo "=== compare $c"; "$H/speed" compare "$BASE" "$c" || echo "COMPARE FAILED $c"; done
