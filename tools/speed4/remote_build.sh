#!/usr/bin/env bash
# carlpc2 half of build.sh: ToMe export of one schedule -> static batch 1 -> pool calibration statistics ->
# SmoothQuant 0.5 -> INT8 QDQ on every GEMM (fc2 inputs at p99.999): the compose winner's recipe, unchanged.
#   remote_build.sh <schedule spec> <candidate_id> [calib images, default 512]
# A probe build (timing only) passes a small image count; its accuracy means nothing.
# The GPU steps need about 3 GB, not most of the 8 GB, so they wait for free VRAM instead of /tmp/626-gpu.lock
# (COMMON.md asks for the lock only for jobs that need most of it; speed3 holds it for training).
set -euo pipefail
vram() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 3500 ]; do sleep 20; done; }
spec=$1 id=$2 n=${3:-512}
R=/home/carl/626-runs/speed4
PY=/home/carl/Projects/626-Project/.venv/bin/python
export QUANT_ART=$R/art QUANT_BASELINE=/home/carl/626-runs/harness/candidates/baseline_fp32 \
  QUANT_VENV=/home/carl/626-runs/quant/venv
W=$R/work/$id
rm -rf "$W" "$R/candidates/$id"
mkdir -p "$W/stats"
TOKENS_BASELINE=$QUANT_BASELINE TOKENS_SHA=$SPEED4_SHA $PY "$R/repo/tools/speed4/tokgraph.py" "$spec" "$W/tok"
src=$(ls -d "$W"/tok/*/)
cd "$R/repo/tools/quant"
./py candidate.py static1 "$W/static1" "$(cat "$src/runtime.json")" --src "$src"
vram; ./py calib_stats.py "$W/static1/model.onnx" "$W/stats/stats.npz" --n "$n"
./py qdq.py smooth "$W/static1" "$W/stats/stats.npz" "$W/smooth05" --alpha 0.5
./py qdq.py qdq "$W/smooth05" "$W/smooth05/stats_max.npz" "$R/candidates/$id" --calib max \
  --calib-override mlp.fc2=p99.999 --exclude ""
vram; ./py ../compose/sanity.py "$W/static1" "$W/smooth05" "$R/candidates/$id"
jq -n --arg spec "$spec" --arg sha "$SPEED4_SHA" --arg n "$n" --slurpfile pre "$src/preprocess.json" \
  '{schedule: $spec, calib_images: ($n | tonumber), tools_sha: $sha, tokens_per_block: $pre[0].tokens_per_block}' \
  > "$R/candidates/$id/recipe.json"
rm -f "$W/stats/calib_stats.onnx" "$W/stats/model.onnx.data" "$W"/tok/*/model.onnx.data
echo done
