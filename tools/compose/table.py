"""Decision table for the compose track, read from the frozen harness's results.jsonl (nothing here measures).

    table.py [--tsv]

One row per candidate below that has a compare record against baseline_fp32: the latest compare, and the eval
whose candidate hash it used. The first rows are the starting points from the token and quant tracks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path("/home/vaper/Work/Prototypes/626-artifacts/harness/results.jsonl")
ROWS = {
    "baseline_trt_fp16": ("precision", "TensorRT fp16, 384 px, 729 tokens"),
    "tokens_res308_trtfp16": ("tokens", "308 px (484 tokens), interpolated pos-emb, TRT fp16"),
    "tokens_tome308_to384b13_to256b20_trtfp16": ("tokens", "308 px + ToMe 484->384 after b13, 384->256 after b20, TRT fp16"),
    "quant_int8_sq05_all_fc2p99999": ("int8", "384 px, SmoothQuant 0.5, all 112 GEMMs INT8 QDQ, fc2 p99.999"),
    "compose_int8_res308": ("tokens + int8", "INT8 recipe on res308, calibrated on the 308 px graph"),
    "compose_int8_tome308": ("tokens + int8", "INT8 recipe on the 308 px ToMe graph, calibrated on it"),
    "compose_int8_tome308_noopt": ("tokens + int8", "compose_int8_tome308 with ORT graph optimizations off"),
}


def rows() -> list[dict]:
    recs = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    out = []
    for cid, (mech, change) in ROWS.items():
        cmp = [x for x in recs if x["kind"] == "compare" and x["b"] == cid and x["a"] == "baseline_fp32"]
        if not cmp:
            continue
        c = cmp[-1]
        ev = [x for x in recs if x["kind"] == "eval" and x["candidate"] == cid and x["candidate_sha"] == c["b_sha"]][-1]
        lat = c["latency"] or {}
        out.append({"id": cid, "mechanism": mech, "change": change, "cv_acc": ev["cv_acc_mean"],
                    "cv_sd": ev["cv_acc_sd"], "d_pp": 100 * c["d_cv_acc"], "mcnemar": c["mcnemar"],
                    "ratio": lat.get("ratio"), "ratio_range": lat.get("ratio_range"),
                    "b1_ms": lat.get("b_batch1_median_ms"), "verdict": c["verdict"],
                    "ep": {k: v["nodes"] for k, v in ev["ep_assignment"].items()}, "time": c["time_utc"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", action="store_true")
    a = ap.parse_args()
    head = ["id", "mechanism", "change", "CV acc ± sd", "Δ pp", "McNemar B-/B+ p", "ratio (rounds)", "b1 ms",
            "EP nodes", "verdict"]
    lines = []
    for r in rows():
        m = r["mcnemar"]
        rr = r["ratio_range"] or [float("nan")] * 2
        lines.append([r["id"], r["mechanism"], r["change"], f"{r['cv_acc']:.4f} ± {r['cv_sd']:.4f}",
                      f"{r['d_pp']:+.2f}", f"{m['b_wrong_a_right']}/{m['b_right_a_wrong']} {m['p_two_sided']:.3f}",
                      f"{r['ratio']:.4f} ({rr[0]:.4f}-{rr[1]:.4f})" if r["ratio"] else "n/a",
                      f"{r['b1_ms']:.2f}" if r["b1_ms"] else "n/a",
                      " ".join(f"{k.replace('ExecutionProvider', '')}:{v}" for k, v in r["ep"].items()), r["verdict"]])
    if a.tsv:
        print("\n".join("\t".join(x) for x in [head, *lines]))
        return
    w = [max(len(x[i]) for x in [head, *lines]) for i in range(len(head))]
    for x in [head, *lines]:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(x)))


if __name__ == "__main__":
    main()
