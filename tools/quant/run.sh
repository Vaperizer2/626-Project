#!/usr/bin/env bash
# Harness pass over candidates: eval each, then time them for R rounds, then compare with the baseline.
#   run.sh [-r ROUNDS] [-t] [-c] <candidate_dir>...    (-t: time only, -c: compare only)
# Every harness call runs alone in a memory-capped scope, started only when legion has >= 4 GB available.
# The harness fits heads with joblib n_jobs=-1; 20 loky workers overflow the 6 GB cap, so cap the pool at 4
# (lbfgs fits are deterministic, so the worker count does not change any result). The scope's peak is logged.
set -uo pipefail
H=/home/vaper/Work/Prototypes/626-wt/harness/tools/speed
BASE=/home/vaper/Work/Prototypes/626-artifacts/harness/candidates/baseline_fp32
ROUNDS=1 EVAL=1 TIME=1
while getopts "r:tc" o; do case $o in r) ROUNDS=$OPTARG ;; t) EVAL=0 ;; c) EVAL=0 TIME=0 ;; *) exit 2 ;; esac; done
shift $((OPTIND - 1))

guarded() {
  while [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -lt 4 ]; do
    echo "$(date +%T) waiting for 4 GB available"; sleep 30
  done
  LOKY_MAX_CPU_COUNT=4 systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=2G bash -c \
    '"$@"; rc=$?; cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup); echo "scope peak $(($(cat $cg/memory.peak) >> 20)) MiB, swap peak $(($(cat $cg/memory.swap.peak) >> 20)) MiB"; exit $rc' _ "$@"
}

if [ "$EVAL" = 1 ]; then
  for c in "$@"; do echo "=== eval $c $(date +%T)"; guarded "$H/speed" eval "$c" || echo "EVAL FAILED $c"; done
fi
if [ "$TIME" = 1 ]; then
  for r in $(seq "$ROUNDS"); do
    for c in "$@"; do echo "=== time $c (round $r) $(date +%T)"; guarded "$H/speed" time "$c" || echo "TIME FAILED $c"; done
  done
fi
for c in "$@"; do echo "=== compare $c"; "$H/speed" compare "$BASE" "$c" || echo "COMPARE FAILED $c"; done
