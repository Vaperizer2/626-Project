#!/usr/bin/env bash
# carlpc2's GPU is shared with tracks that do not take /tmp/626-gpu.lock. Run <cmd> once <MiB> are free for 30 s,
# and run it again (up to 6 tries) when it dies of a CUDA or onnxruntime out-of-memory error. Commands must be
# safe to rerun from scratch.
#   gpuwait.sh <MiB> <cmd...>
set -uo pipefail
need=$1
shift
for try in 1 2 3 4 5 6; do
  ok=0
  while [ "$ok" -lt 3 ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$need" ]; then ok=$((ok + 1)); else ok=0; fi
    sleep 10
  done
  log=$(mktemp)
  "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] || ! grep -qE "OutOfMemoryError|out of memory|BFCArena|CUDA_ERROR_OUT_OF_MEMORY" "$log"; then
    rm -f "$log"
    exit "$rc"
  fi
  rm -f "$log"
  echo "gpuwait: out of GPU memory on try $try; waiting for $need MiB again"
done
exit 1
