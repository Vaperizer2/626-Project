#!/usr/bin/env bash
# Drive speed3 from legion. carlpc2 did the torch and INT8 work and is gone (2026-11-03): sync, bg, log, int8 and fetch
# no longer work; judge runs the frozen harness here.
#   run.sh sync                          # code -> carlpc2:$R/repo
#   run.sh bg <name> <cmd...>            # carlpc2, detached, under the GPU lock: cd repo; <cmd> > logs/<name>.log
#   run.sh log <name>                    # tail of that log
#   run.sh int8 <id> <state.npz>         # carlpc2 (bg): state -> s294 graph -> the winner's INT8 recipe -> candidates/<id>
#   run.sh fetch <id>                    # carlpc2 candidate -> 626-artifacts/speed3/candidates/<id>
#   run.sh judge <id>                    # legion: harness eval, time, compare vs fp32 and vs the speed2 winner
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$here/../.." && pwd)
ART=/home/vaper/Work/Prototypes/626-artifacts
R=/home/carl/626-runs/speed3
PY=/home/carl/Projects/626-Project/.venv/bin/python
H=$REPO/tools/speed/speed
BASE=$ART/harness/candidates/baseline_fp32
WINNER=${SPEED3_WINNER:-$ART/speed2/candidates/speed2_int8_res294_mlp3840}
case $1 in
sync)
  for d in tools/quant tools/speed tools/compose tools/prune tools/tokens tools/speed2 tools/speed3 src/compclass; do
    rsync -a --delete --exclude __pycache__ "$REPO/$d/" "carlpc2:$R/repo/$d/"
  done
  sha=$(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD -- tools src || echo -dirty)
  ssh carlpc2 "echo $sha > $R/repo/SHA"
  echo "synced @ $sha"
  ;;
bg)
  name=$2
  shift 2
  ssh carlpc2 "cd $R/repo || exit 1; SPEED3_SHA=\$(cat SHA) setsid nohup flock /tmp/626-gpu.lock $(printf '%q ' "$@") \
    > $R/logs/$name.log 2>&1 < /dev/null &"
  echo "started $name; log carlpc2:$R/logs/$name.log"
  ;;
log)
  ssh carlpc2 "tail -${3:-5} $R/logs/$2.log"
  ;;
int8)
  "$0" bg "$2.build" bash tools/speed3/remote_int8.sh "$2" "$3"
  ;;
fetch)
  id=$2
  ssh carlpc2 "tail -1 $R/logs/$id.build.log" | grep -qx done || { echo "$id not done"; exit 1; }
  mkdir -p "$ART/speed3/candidates" "$ART/speed3/logs"
  rsync -a --delete "carlpc2:$R/candidates/$id/" "$ART/speed3/candidates/$id/"
  rsync -a "carlpc2:$R/logs/$id.build.log" "$ART/speed3/logs/$id.build.log"
  echo "fetched $ART/speed3/candidates/$id"
  ;;
judge)
  d=$(cd "$2" 2>/dev/null && pwd || echo "$ART/speed3/candidates/$2")
  log=$ART/speed3/logs/$(basename "$d").log
  run() { LOKY_MAX_CPU_COUNT=4 systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=2G "$H" "$@"; }
  { run eval "$d" && run time "$d" && run compare "$BASE" "$d" && run compare "$WINNER" "$d"; } 2>&1 | tee -a "$log"
  ;;
*) sed -n 2,9p "$0"; exit 2 ;;
esac
