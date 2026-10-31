#!/usr/bin/env bash
# eval, time and compare each candidate id against the deployed baseline with the frozen harness.
# Logs to 626-artifacts/tokens/logs/<id>.log; the harness appends its records to results.jsonl.
set -uo pipefail
H=/home/vaper/Work/Prototypes/626-wt/harness/tools/speed
ART=/home/vaper/Work/Prototypes/626-artifacts
mkdir -p "$ART/tokens/logs"
for id in "$@"; do
  d="$ART/tokens/candidates/$id"
  { "$H/speed" eval "$d" && "$H/speed" time "$d" && "$H/speed" compare "$ART/harness/candidates/baseline_fp32" "$d"; } \
    >"$ART/tokens/logs/$id.log" 2>&1
  echo "$id exit $? $(grep -E 'verdict|CV acc .* vs|ratio B/A' "$ART/tokens/logs/$id.log" | tr '\n' ' ')"
done
