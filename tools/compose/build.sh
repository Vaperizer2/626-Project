#!/usr/bin/env bash
# Apply the quant track's best INT8 recipe (SmoothQuant 0.5, all 112 GEMMs INT8, fc2 inputs at p99.999, static
# batch 1) to a token-track graph. Heavy work runs on carlpc2 (remote_build.sh); legion only syncs and fetches.
#   build.sh start <source_candidate_dir> <candidate_id>  # sync code and the source graph, launch the remote build
#   build.sh fetch <candidate_id> [ORT_level]             # once the remote log ends in "done"; ORT_level sets
#                                                          runtime.json graph_optimization_level (ToMe INT8 needs
#                                                          ORT_DISABLE_ALL: logs/trtcheck_tome.log)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$here/../.." && pwd)
ART=/home/vaper/Work/Prototypes/626-artifacts
R=/home/carl/626-runs/compose
case $1 in
start)
  srcdir=$(cd "$2" && pwd) id=$3 src=$(basename "$2")
  sha=$(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD -- tools src || echo -dirty)
  ssh carlpc2 "mkdir -p $R/repo/tools $R/repo/src $R/src $R/work $R/candidates $R/logs $R/art \
    && ln -sfn /home/carl/626-runs/quant/art/calib $R/art/calib"
  for d in tools/quant tools/speed tools/compose src/compclass; do
    rsync -a --delete --exclude __pycache__ "$REPO/$d/" "carlpc2:$R/repo/$d/"
  done
  rsync -a --delete --exclude model.onnx.data "$srcdir/" "carlpc2:$R/src/$src/"
  # token candidates point every weight at the baseline data file (tools/tokens/build.py relink)
  ssh carlpc2 "cp --reflink=auto /home/carl/626-runs/harness/candidates/baseline_fp32/model.onnx.data $R/src/$src/ \
    && { QUANT_GIT_SHA=$sha setsid nohup bash $R/repo/tools/compose/remote_build.sh $R/src/$src $id \
       > $R/logs/$id.log 2>&1 < /dev/null & }"
  echo "started $id from $src @ $sha; log carlpc2:$R/logs/$id.log"
  ;;
fetch)
  id=$2
  ssh carlpc2 "tail -1 $R/logs/$id.log" | grep -qx done || { echo "$id not done"; exit 1; }
  mkdir -p "$ART/compose/candidates" "$ART/compose/logs"
  rsync -a --delete "carlpc2:$R/candidates/$id/" "$ART/compose/candidates/$id/"
  rsync -a "carlpc2:$R/logs/$id.log" "$ART/compose/logs/$id.build.log"
  if [ -n "${3:-}" ]; then
    python3 - "$ART/compose/candidates/$id/runtime.json" "$3" <<'PY'
import json, sys
p, level = sys.argv[1:]
r = json.load(open(p))
r["graph_optimization_level"] = level
json.dump(r, open(p, "w"), indent=1)
PY
  fi
  echo "fetched $ART/compose/candidates/$id"
  ;;
*) echo "usage: build.sh start <src> <id> | fetch <id>"; exit 2 ;;
esac
