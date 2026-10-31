#!/usr/bin/env bash
# legion: run harness_queue.sh on CANDIDATEs once PID has exited and at least 4 GB RAM is available, inside a
# 6 GB memory scope (COMMON.md memory rule).
#   setsid nohup tools/prune/queue_after.sh PID CANDIDATE_DIR... > LOG 2>&1 < /dev/null &
set -uo pipefail
PID=$1
shift
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 4 ]; do sleep 20; done
exec systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=2G \
  "$(dirname "$(readlink -f "$0")")/harness_queue.sh" "$@"
