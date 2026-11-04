#!/usr/bin/env bash
# legion: time A and B in alternating harness sessions (A B A B ...), N rounds, then print each session's paired
# ratio to baseline_fp32, its SM clock, and B/A per adjacent pair, so a TRT-vs-TRT gain is measured at like clocks.
#   tools/prune/alternate.sh A_DIR B_DIR N > LOG
set -uo pipefail
H=/home/vaper/Work/Prototypes/626-wt/prune/tools/speed
A=$1
B=$2
N=$3
since=$(date -u +%Y-%m-%dT%H:%M:%S)
for i in $(seq "$N"); do
  "$H/speed" time "$A" > /dev/null 2>&1 || echo "time $A failed"
  "$H/speed" time "$B" > /dev/null 2>&1 || echo "time $B failed"
done
jq -rs --arg a "$(basename "$A")" --arg b "$(basename "$B")" --arg since "$since" '
  [.[] | select(.kind == "time" and .time_utc >= $since and (.candidate == $a or .candidate == $b))
       | {c: .candidate, r: .paired.ratio_median, mhz: .gpu_during.sm_clock_mhz_median, load: .loadavg_1m}] as $s
  | ($s[] | "\(.c)\tratio \(.r)\tSM \(.mhz) MHz\tload \(.load)"),
    ([range(0; ($s | length) / 2 | floor)] | map($s[2 * . + 1].r / $s[2 * .].r) as $p
     | "B/A per pair: \($p)  median \($p | sort | .[(length - 1) / 2 | floor])")' \
  /home/vaper/Work/Prototypes/626-artifacts/harness/results.jsonl
