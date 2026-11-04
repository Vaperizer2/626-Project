"""Decision table of the pruning candidates from the harness results (latest records per candidate).

    table.py [ID_PREFIX]      # default prune_
"""
from __future__ import annotations

import json
import sys

RESULTS = "/home/vaper/Work/Prototypes/626-artifacts/harness/results.jsonl"


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "prune_"
    recs = [json.loads(ln) for ln in open(RESULTS)]
    evals = {r["candidate"]: r for r in recs if r["kind"] == "eval"}
    times = {r["candidate"]: r for r in recs if r["kind"] == "time"}
    cmp = {(r["a"], r["b"]): r for r in recs if r["kind"] == "compare"}
    ids = sorted({r["candidate"] for r in recs if r["kind"] == "eval" and r["candidate"].startswith(prefix)})
    print("| id | CV acc ± sd | Δ vs fp32 (pp) | McNemar b-worse/b-better (p) | ratio vs fp32 | ratio vs trt_fp16 "
          "(beyond noise) | batch-1 ms | errors | verdict |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in ids:
        e, t = evals[c], times.get(c)
        f, g = cmp.get(("baseline_fp32", c)), cmp.get(("baseline_trt_fp16", c))
        mc = f["mcnemar"] if f else None
        row = [c, f"{e['cv_acc_mean']:.4f} ± {e['cv_acc_sd']:.4f}",
               f"{f['d_cv_acc'] * 100:+.2f}" if f else "",
               f"{mc['b_wrong_a_right']}/{mc['b_right_a_wrong']} ({mc['p_two_sided']:.2g})" if mc else "",
               f"{f['latency']['ratio']:.4f}" if f else "",
               f"{g['latency']['ratio']:.3f} ({g['latency']['beyond_noise']})" if g else "",
               f"{t['batch1']['median_ms']:.1f}" if t else "",
               f"{e['n_errors']}/{t['n_errors'] if t else '-'}",
               f["verdict"] if f else ""]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
