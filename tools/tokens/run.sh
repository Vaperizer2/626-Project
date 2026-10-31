#!/usr/bin/env bash
# eval, time and compare each candidate id against the deployed baseline with the frozen harness.
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
for id in "$@"; do
  d="$ART/tokens/candidates/$id"
  { step eval "$d" && step time "$d" && step compare "$ART/harness/candidates/baseline_fp32" "$d"; } \
    >>"$ART/tokens/logs/$id.log" 2>&1
  echo "$id exit $? $(grep -E 'verdict|CV acc .* vs|ratio B/A' "$ART/tokens/logs/$id.log" | tail -3 | tr '\n' ' ')"
done
