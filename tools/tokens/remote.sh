#!/usr/bin/env bash
# Build torch-export recipes on carlpc2 (RAM), bring back only the graph and metadata, and reflink the baseline
# weights next to it on legion. build.py relinks every weight to the baseline data file, so the bytes match.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ART=/home/vaper/Work/Prototypes/626-artifacts
R=${TOKENS_REMOTE:-/home/carl/626-runs/tokens}
LOCAL=${TOKENS_LOCAL:-$ART/tokens/candidates}
sha=$(git -C "$here" rev-parse --short HEAD)$(git -C "$here" diff --quiet HEAD -- . || echo -dirty)
ssh carlpc2 "mkdir -p $R/tools $R/check_inputs $R/candidates"
scp -q "$here"/build.py "$here"/siglip.py "carlpc2:$R/tools/"
rsync -a "$ART/tokens/check_inputs/" "carlpc2:$R/check_inputs/"
for id in "$@"; do
  ssh carlpc2 "cd $R && TOKENS_SHA=$sha TOKENS_BASELINE=/home/carl/626-runs/harness/candidates/baseline_fp32 \
    TOKENS_OUT=$R/candidates /home/carl/Projects/626-Project/.venv/bin/python tools/build.py $id --check" 2>&1 \
    | grep -v -e Warning -e 'warnings.warn' -e 'DeprecationWarning'
  d="$LOCAL/$id"
  rm -rf "$d" && mkdir -p "$d"
  rsync -a --exclude model.onnx.data "carlpc2:$R/candidates/$id/" "$d/"
  cp --reflink=auto "$ART/harness/candidates/baseline_fp32/model.onnx.data" "$d/model.onnx.data"
  ssh carlpc2 "rm -f $R/candidates/$id/model.onnx.data"
done
