"""Decision table for the token-count candidates, read from the frozen harness's results.jsonl.

    python tools/tokens/table.py [--tsv]

One row per candidate in build.RECIPES that has a compare record against baseline_fp32 (the latest one whose
candidate hash matches the latest eval). Numbers are the harness's own; nothing here measures anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build import RECIPES

RESULTS = Path("/home/vaper/Work/Prototypes/626-artifacts/harness/results.jsonl")


def rows() -> list[dict]:
    recs = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    out = []
    for cid, r in RECIPES.items():
        cmp = [x for x in recs if x["kind"] == "compare" and x["b"] == cid and x["a"] == "baseline_fp32"]
        if not cmp:
            continue
        c = cmp[-1]
        ev = [x for x in recs if x["kind"] == "eval" and x["candidate"] == cid and x["candidate_sha"] == c["b_sha"]][-1]
        t = r.tokens()
        lat = c["latency"] or {}
        out.append({"id": cid, "hypothesis": r.hypothesis, "size": r.size, "pos": r.pos,
                    "tokens": f"{r.grid ** 2}" if t[0] == t[-1] == r.grid ** 2 else f"{r.grid ** 2}->{t[-1]}",
                    "mean_tokens": round(sum(t) / len(t)), "cv_acc": ev["cv_acc_mean"], "cv_sd": ev["cv_acc_sd"],
                    "d_pp": 100 * c["d_cv_acc"], "mcnemar": c["mcnemar"], "ratio": lat.get("ratio"),
                    "ratio_range": lat.get("ratio_range"), "beyond_noise": lat.get("beyond_noise"),
                    "b1_ms": lat.get("b_batch1_median_ms"), "verdict": c["verdict"],
                    "ep": ev["ep_assignment"], "time": c["time_utc"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", action="store_true")
    a = ap.parse_args()
    head = ["id", "size", "tokens", "mean", "CV acc ± sd", "Δ pp", "McNemar B-/B+ p", "ratio (rounds)", "b1 ms",
            "EP nodes", "verdict"]
    lines = []
    for r in rows():
        m = r["mcnemar"]
        rr = r["ratio_range"] or [float("nan")] * 2
        ep = " ".join(f"{k.replace('ExecutionProvider', '')}:{v['nodes']}" for k, v in r["ep"].items())
        lines.append([r["id"], str(r["size"]), r["tokens"], str(r["mean_tokens"]),
                      f"{r['cv_acc']:.4f} ± {r['cv_sd']:.4f}", f"{r['d_pp']:+.2f}",
                      f"{m['b_wrong_a_right']}/{m['b_right_a_wrong']} {m['p_two_sided']:.3f}",
                      f"{r['ratio']:.4f} ({rr[0]:.4f}-{rr[1]:.4f})" if r["ratio"] else "n/a",
                      f"{r['b1_ms']:.2f}" if r["b1_ms"] else "n/a", ep, r["verdict"]])
    if a.tsv:
        print("\t".join(head))
        print("\n".join("\t".join(x) for x in lines))
        return
    w = [max(len(x[i]) for x in [head, *lines]) for i in range(len(head))]
    for x in [head, *lines]:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(x)))


if __name__ == "__main__":
    main()
