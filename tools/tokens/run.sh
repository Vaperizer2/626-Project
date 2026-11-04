#!/usr/bin/env bash
# eval, time and compare candidates against the deployed baseline with the frozen harness.
#   run.sh <id>...            run these ids in order
#   run.sh --queue <file>     pop ids from the top of <file> until it is empty (edit the file to reorder)
# Logs to 626-artifacts/tokens/logs/<id>.log; the harness appends its records to results.jsonl.
# legion memory rule (COMMON.md): wait while available memory is under 4 GB, and run each step in a 6 GB scope.
set -uo pipefail
H=/home/vaper/Work/Prototypes/626-wt/harness/tools/speed
ART=/home/vaper/Work/Prototypes/626-artifacts
mkdir -p "$ART/tokens/logs"
# fit_probe grid-searches with n_jobs=-1 (15 loky workers on 20 cores), which alone overflows 6 GB. Four workers
# give the same fits (lbfgs is deterministic; checked on the res336 embeddings) in about 1 min and 1.5 GB.
export LOKY_MAX_CPU_COUNT=4
step() {
  until [ "$(free -g | awk '/^Mem:/ {print $7}')" -ge 4 ]; do sleep 20; done
  systemd-run --user --scope --quiet -p MemoryMax=6G -p MemorySwapMax=2G "$H/speed" "$@"
}
one() {
  local d="$ART/tokens/candidates/$1"
  { step eval "$d" && step time "$d" && step compare "$ART/harness/candidates/baseline_fp32" "$d"; } \
    >>"$ART/tokens/logs/$1.log" 2>&1
  echo "$1 exit $? $(grep -E 'verdict|CV acc .* vs|ratio B/A' "$ART/tokens/logs/$1.log" | tail -3 | tr '\n' ' ')"
}
if [ "${1:-}" = --queue ]; then
  while id=$(sed -n '1{s/[[:space:]]//g;p}' "$2") && [ -n "$id" ]; do
    sed -i 1d "$2"
    one "$id"
  done
else
  for id in "$@"; do one "$id"; done
fi
