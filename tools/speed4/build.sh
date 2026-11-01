#!/usr/bin/env bash
# Build speed4 INT8 ToMe candidates on carlpc2 and fetch them to legion with the TensorRT session-config fix.
#   build.sh start <schedule spec> [candidate_id] [calib images]   # sync code, launch remote_build.sh (nohup)
#   build.sh fetch <candidate_id>                                  # once the remote log ends in "done"
# The default id is speed4_int8_<schedule id>. Every fetched runtime.json keeps ORT_ENABLE_ALL and disables the two
# level-1 rewrites TensorRT compiles wrongly (626-artifacts/trtbugs/NOTES.md).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$here/../.." && pwd)
ART=/home/vaper/Work/Prototypes/626-artifacts
R=/home/carl/626-runs/speed4
case $1 in
start)
  spec=$2
  id=${3:-speed4_int8_$(python3 "$here/sched.py" id "$spec")}
  sha=$(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD -- tools src || echo -dirty)
  ssh carlpc2 "mkdir -p $R/repo/tools $R/repo/src $R/work $R/candidates $R/logs $R/art \
    && ln -sfn /home/carl/626-runs/quant/art/calib $R/art/calib"
  for d in tools/quant tools/speed tools/compose tools/tokens tools/speed4 src/compclass; do
    rsync -a --delete --exclude __pycache__ "$REPO/$d/" "carlpc2:$R/repo/$d/"
  done
  ssh carlpc2 "SPEED4_SHA=$sha setsid nohup bash $R/repo/tools/speed4/remote_build.sh '$spec' $id ${4:-512} \
    > $R/logs/$id.log 2>&1 < /dev/null &"
  echo "started $id ($spec) @ $sha; log carlpc2:$R/logs/$id.log"
  ;;
fetch)
  id=$2
  ssh carlpc2 "tail -1 $R/logs/$id.log" | grep -qx done || { echo "$id not done"; exit 1; }
  d=$ART/speed4/candidates/$id
  mkdir -p "$d" "$ART/speed4/logs"
  rsync -a --delete "carlpc2:$R/candidates/$id/" "$d/"
  rsync -a "carlpc2:$R/logs/$id.log" "$ART/speed4/logs/$id.build.log"
  jq '.graph_optimization_level = "ORT_ENABLE_ALL"
      | .session_config = {"optimization.disable_specified_optimizers": "MatMulAddFusion;TransposeOptimizer"}' \
    "$d/runtime.json" > "$d/runtime.json.new" && mv "$d/runtime.json.new" "$d/runtime.json"
  echo "fetched $d"
  ;;
*) echo "usage: build.sh start <spec> [id] [n] | fetch <id>"; exit 2 ;;
esac
