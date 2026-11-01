#!/usr/bin/env bash
# carlpc2, under the GPU lock (run.sh int8): a trained block state -> the s294 graph (speed2's res294 + MLP 3840,
# same nodes and shapes) -> the speed2 winner's INT8 recipe unchanged (static batch 1, pool calibration statistics,
# SmoothQuant 0.5, INT8 QDQ on all 112 GEMMs, fc2 inputs at p99.999) -> sanity vs its own fp32 graph.
#   remote_int8.sh <candidate_id> <state.npz>
set -euo pipefail
id=$1 state=$(readlink -f "$2")
R=/home/carl/626-runs/speed3
PY=/home/carl/Projects/626-Project/.venv/bin/python
S294=$R/base/s294_mlp3840
export QUANT_ART=$R/art QUANT_BASELINE=/home/carl/626-runs/harness/candidates/baseline_fp32 \
  QUANT_VENV=/home/carl/626-runs/quant/venv QUANT_GIT_SHA=$SPEED3_SHA
mkdir -p "$R/art" && ln -sfn /home/carl/626-runs/quant/art/calib "$R/art/calib"
G="bash $R/repo/tools/speed3/gpuwait.sh 3500"
W=$R/work/$id
rm -rf "$W" "$R/candidates/$id"
mkdir -p "$W/stats"
PRUNE_BASE=$S294 $PY "$R/repo/tools/prune/build.py" load "$W/fp32" --base "$S294" --state "$state"
cd "$R/repo/tools/quant"
./py candidate.py static1 "$W/static1" "$(cat "$W/fp32/runtime.json")" --src "$W/fp32"
$G ./py calib_stats.py "$W/static1/model.onnx" "$W/stats/stats.npz"
./py qdq.py smooth "$W/static1" "$W/stats/stats.npz" "$W/smooth05" --alpha 0.5
./py qdq.py qdq "$W/smooth05" "$W/smooth05/stats_max.npz" "$R/candidates/$id" --calib max \
  --calib-override mlp.fc2=p99.999 --exclude ""
$G ./py ../compose/sanity.py "$W/static1" "$W/smooth05" "$R/candidates/$id"
jq -n --arg state "$state" --arg sha "$SPEED3_SHA" --slurpfile pre "$W/fp32/preprocess.json" \
  '{state: $state, base: "s294_mlp3840 (speed2 res294 + MLP 3840, FLAP fluct, bias-comp)", tools_sha: $sha,
    recipe: "speed2 INT8: SmoothQuant 0.5, max calib, mlp.fc2 p99.999, 112/112 GEMMs", fp32_built_from: $pre[0].built_from}' \
  > "$R/candidates/$id/recipe.json"
[ ! -f "${state%.npz}.json" ] || cp "${state%.npz}.json" "$R/candidates/$id/state_meta.json"
[ ! -f "$(dirname "$state")/config.json" ] || cp "$(dirname "$state")/config.json" "$R/candidates/$id/train_config.json"
rm -f "$W/stats/calib_stats.onnx" "$W/stats/model.onnx.data"
echo done
