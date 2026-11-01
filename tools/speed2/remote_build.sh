#!/usr/bin/env bash
# carlpc2 half of build.sh: optional prune of the deployed fp32 graph -> resolution surgery -> the winner's INT8
# recipe (static batch 1, pool calibration statistics, SmoothQuant 0.5, INT8 QDQ on every GEMM, fc2 inputs at
# p99.999) -> fidelity screen against the winner's own fp32 and INT8 graphs.
#   remote_build.sh <candidate_id> <tokens_recipe> <prune build.py args | none>
# QDQ_EXCLUDE (comma-separated GEMM roles kept in fp16) defaults to none, as in the winner.
set -euo pipefail
id=$1 tok=$2
shift 2
R=/home/carl/626-runs/speed2
PY=/home/carl/Projects/626-Project/.venv/bin/python
BASE=/home/carl/626-runs/harness/candidates/baseline_fp32
WIN=/home/carl/626-runs/compose/work/compose_int8_res308
export QUANT_ART=$R/art QUANT_BASELINE=$BASE QUANT_VENV=/home/carl/626-runs/quant/venv
W=$R/work/$id
rm -rf "$W" "$R/candidates/$id"
mkdir -p "$W/stats"
if [ "$1" = none ]; then
  fp=$BASE
else
  PRUNE_BASE=$BASE $PY "$R/repo/tools/prune/build.py" "$1" "$W/fp32_384" "${@:2}"
  fp=$W/fp32_384
fi
TOKENS_BASELINE=$fp TOKENS_OUT=$W/tok TOKENS_SHA=$SPEED2_SHA $PY "$R/repo/tools/tokens/build.py" "$tok"
src=$W/tok/$tok
cd "$R/repo/tools/quant"
./py candidate.py static1 "$W/static1" "$(cat "$src/runtime.json")" --src "$src"
flock /tmp/626-gpu.lock ./py calib_stats.py "$W/static1/model.onnx" "$W/stats/stats.npz"
./py qdq.py smooth "$W/static1" "$W/stats/stats.npz" "$W/smooth05" --alpha 0.5
./py qdq.py qdq "$W/smooth05" "$W/smooth05/stats_max.npz" "$R/candidates/$id" --calib max \
  --calib-override mlp.fc2=p99.999 --exclude "${QDQ_EXCLUDE:-}"
flock /tmp/626-gpu.lock ./py ../compose/sanity.py "$W/static1" "$W/smooth05" "$R/candidates/$id"
if [ -f "$WIN/static1/preprocess.json" ] && cmp -s <(jq -S '{input_size, crop_pct}' "$WIN/static1/preprocess.json") \
    <(jq -S '{input_size, crop_pct}' "$src/preprocess.json"); then
  flock /tmp/626-gpu.lock ./py ../speed2/screen.py "$WIN/static1" "$W/static1" \
    /home/carl/626-runs/compose/candidates/compose_int8_res308 "$R/candidates/$id"
fi
jq -n --arg prune "$*" --arg tok "$tok" --arg sha "$SPEED2_SHA" --slurpfile pre "$src/preprocess.json" \
  '{prune_args: $prune, tokens_recipe: $tok, tools_sha: $sha, tokens_built_from: $pre[0].built_from}' \
  > "$R/candidates/$id/recipe.json"
[ -f "$W/fp32_384/prune.json" ] && cp "$W/fp32_384/prune.json" "$R/candidates/$id/"
rm -f "$W/stats/calib_stats.onnx" "$W/stats/model.onnx.data"
echo done
