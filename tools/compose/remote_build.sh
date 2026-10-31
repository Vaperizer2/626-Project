#!/usr/bin/env bash
# carlpc2 half of build.sh: static batch 1 -> pool calibration statistics -> SmoothQuant 0.5 -> INT8 QDQ.
#   remote_build.sh <source_candidate_dir> <candidate_id>
set -euo pipefail
src=$1 id=$2
R=/home/carl/626-runs/compose
export QUANT_ART=$R/art QUANT_BASELINE=/home/carl/626-runs/harness/candidates/baseline_fp32 \
  QUANT_VENV=/home/carl/626-runs/quant/venv
W=$R/work/$id
cd "$R/repo/tools/quant"
rm -rf "$W" "$R/candidates/$id"
mkdir -p "$W/stats"
./py candidate.py static1 "$W/static1" "$(cat "$src/runtime.json")" --src "$src"
flock /tmp/626-gpu.lock ./py calib_stats.py "$W/static1/model.onnx" "$W/stats/stats.npz"
./py qdq.py smooth "$W/static1" "$W/stats/stats.npz" "$W/smooth05" --alpha 0.5
./py qdq.py qdq "$W/smooth05" "$W/smooth05/stats_max.npz" "$R/candidates/$id" --calib max \
  --calib-override mlp.fc2=p99.999 --exclude ""
flock /tmp/626-gpu.lock ./py ../compose/sanity.py "$W/static1" "$W/smooth05" "$R/candidates/$id"
rm -f "$W/stats/calib_stats.onnx" "$W/stats/model.onnx.data"
echo done
