#!/usr/bin/env bash
# legion: run harness_queue.sh on CANDIDATEs one queue at a time, once at least 4 GB RAM is available, inside a
# 6 GB memory scope (COMMON.md memory rule). Queues serialise on artifacts/prune/queue.lock.
#   [RETIME=1] setsid nohup tools/prune/queue_after.sh CANDIDATE_DIR... > LOG 2>&1 < /dev/null &
set -uo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
exec flock /home/vaper/Work/Prototypes/626-artifacts/prune/queue.lock bash -c '
  while pgrep -f "^(/usr/bin/)?bash /[^ ]*harness_queue[.]sh" > /dev/null; do sleep 20; done
  until [ "$(awk "/MemAvailable/ {print int(\$2 / 1048576)}" /proc/meminfo)" -ge 4 ]; do sleep 20; done
  exec systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=2G "$0/harness_queue.sh" "$@"' "$HERE" "$@"
