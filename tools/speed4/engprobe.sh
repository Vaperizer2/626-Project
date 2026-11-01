#!/usr/bin/env bash
# Layer profile of the TensorRT engine the harness itself built and timed for a candidate (its onnxruntime engine
# cache), so a harness number can be explained kernel by kernel. trtprobe.py reuses <out>/engine.plan when present.
#   engprobe.sh <candidate_id> [...]          -> 626-artifacts/speed4/work/eng_<candidate_id>/
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ART=/home/vaper/Work/Prototypes/626-artifacts
for id in "$@"; do
  eng=$(ls -t "$ART"/harness/work/"$id"/*/trt_cache/*.engine | head -1)
  out=$ART/speed4/work/eng_$id
  mkdir -p "$out"
  cp --reflink=auto "$eng" "$out/engine.plan"
  until [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 4 ]; do sleep 30; done
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=2G \
    "$here/../quant/py" "$here/../quant/trtprobe.py" /dev/null "$out" --iters 200
  python3 "$here/profile_blocks.py" "$out" --json "$out/split.json" | grep -E "ms$|per block"
done
