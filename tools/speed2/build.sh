#!/usr/bin/env bash
# Build speed2 candidates on carlpc2 and judge them with the frozen harness on legion.
#   build.sh start <id> <tokens_recipe> <prune build.py args | none>   # sync code, launch remote_build.sh
#   build.sh fetch <id>                                                # once the remote log ends in "done"
#   build.sh judge <id>                                                # harness eval, time, compare x2 (legion)
# judge compares against the deployed baseline (acceptance) and the current winner (speed).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$here/../.." && pwd)
ART=/home/vaper/Work/Prototypes/626-artifacts
R=/home/carl/626-runs/speed2
H=$REPO/tools/speed/speed
BASE=$ART/harness/candidates/baseline_fp32
WINNER=${SPEED2_WINNER:-$ART/compose/candidates/compose_int8_res308}
case $1 in
start)
  id=$2
  shift 2
  sha=$(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD -- tools src || echo -dirty)
  ssh carlpc2 "mkdir -p $R/repo/tools $R/repo/src $R/work $R/candidates $R/logs $R/art \
    && ln -sfn /home/carl/626-runs/quant/art/calib $R/art/calib"
  for d in tools/quant tools/speed tools/compose tools/prune tools/tokens tools/speed2 src/compclass; do
    rsync -a --delete --exclude __pycache__ "$REPO/$d/" "carlpc2:$R/repo/$d/"
  done
  ssh carlpc2 "SPEED2_SHA=$sha QUANT_GIT_SHA=$sha setsid nohup bash $R/repo/tools/speed2/remote_build.sh $id $(printf '%q ' "$@") \
    > $R/logs/$id.log 2>&1 < /dev/null &"
  echo "started $id @ $sha; log carlpc2:$R/logs/$id.log"
  ;;
fetch)
  id=$2
  ssh carlpc2 "tail -1 $R/logs/$id.log" | grep -qx done || { echo "$id not done"; exit 1; }
  mkdir -p "$ART/speed2/candidates" "$ART/speed2/logs"
  rsync -a --delete "carlpc2:$R/candidates/$id/" "$ART/speed2/candidates/$id/"
  rsync -a "carlpc2:$R/logs/$id.log" "$ART/speed2/logs/$id.build.log"
  echo "fetched $ART/speed2/candidates/$id"
  ;;
judge)
  d=$(cd "$2" 2>/dev/null && pwd || echo "$ART/speed2/candidates/$2")
  log=$ART/speed2/logs/$(basename "$d").log
  run() { LOKY_MAX_CPU_COUNT=4 systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=2G "$H" "$@"; }
  { run eval "$d" && run time "$d" && run compare "$BASE" "$d" && run compare "$WINNER" "$d"; } 2>&1 | tee -a "$log"
  ;;
*) echo "usage: build.sh start <id> <tokens_recipe> <prune args|none> | fetch <id> | judge <id|dir>"; exit 2 ;;
esac
