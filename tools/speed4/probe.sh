#!/usr/bin/env bash
# TensorRT layer profile of a candidate graph on legion (tools/quant/trtprobe.py: plain TensorRT, fp16 + int8, takes
# the GPU bench lock itself), then the per-role split by token count (profile_blocks.py). Diagnostic, not a harness
# number. Waits for 4 GB available and runs in a 6 GB scope (COMMON.md memory rule).
#   probe.sh <candidate_dir> [...]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
W=/home/vaper/Work/Prototypes/626-artifacts/speed4/work
for c in "$@"; do
  c=$(cd "$c" && pwd) out=$W/probe_$(basename "$c")
  until [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 4 ]; do sleep 30; done
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=2G \
    "$here/../quant/py" "$here/../quant/trtprobe.py" "$c/model.onnx" "$out" --fp16 --int8 --iters 200
  python3 "$here/profile_blocks.py" "$out" --json "$out/split.json"
done
