#!/usr/bin/env bash
# Frozen-harness steps for speed4 candidates, from this checkout's tools/speed (byte-identical to 01c721d).
#   judge.sh time <candidate_dir>...     # batch-1 latency vs the fp32 baseline: label-free, run on every build
#   judge.sh eval <candidate_dir>...     # dev-fold CV: the selection budget, finalists only
#   judge.sh compare <candidate_dir>...  # vs the deployed fp32 baseline (acceptance) and the reference candidates
# Each harness call waits for 4 GB available and runs in a 6 GB scope (COMMON.md); the head fit's joblib pool is
# capped at 4 workers (lbfgs fits are deterministic, so the result does not change). Logs to speed4/logs/<id>.log.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
H=$here/../speed/speed
ART=/home/vaper/Work/Prototypes/626-artifacts
BASE=$ART/harness/candidates/baseline_fp32
REFS=("$ART/compose/candidates/compose_int8_res308" "$ART/trtbugs/candidates/trtbugs_int8_tome308"
      "$ART/speed2/candidates/speed2_int8_res294_mlp3840")
guarded() {
  until [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 4 ]; do sleep 30; done
  LOKY_MAX_CPU_COUNT=4 systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=2G "$H" "$@"
}
mode=$1
shift
for c in "$@"; do
  c=$(cd "$c" && pwd)
  log=$ART/speed4/logs/$(basename "$c").log
  {
    echo "=== $mode $(date -u +%FT%TZ)"
    case $mode in
      time|eval) guarded "$mode" "$c" || echo "$mode FAILED" ;;
      compare) "$H" compare "$BASE" "$c"; for r in "${REFS[@]}"; do "$H" compare "$r" "$c"; done ;;
      *) echo "usage: judge.sh time|eval|compare <dir>..."; exit 2 ;;
    esac
  } 2>&1 | tee -a "$log"
done
