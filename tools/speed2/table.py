"""Decision table for the speed2 track, read from the frozen harness's results.jsonl (nothing here measures).

    table.py [--tsv]

One row per candidate below with a compare record against baseline_fp32 (acceptance) and, where present, against
the winner it tried to beat (speed). Each column comes from the latest compare of that pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path("/home/vaper/Work/Prototypes/626-artifacts/harness/results.jsonl")
BASE = "baseline_fp32"
ROWS = {
    "compose_int8_res308": ("compose_int8_res308", "starting winner: 308 px + INT8 SmoothQuant 0.5, all 112 GEMMs"),
    "speed2_int8_res308_opt5": ("compose_int8_res308", "TRT builder optimisation level 5"),
    "speed2_int8_res308_mlp3840": ("compose_int8_res308", "MLP 4304 -> 3840 (FLAP fluct, bias-comp): fc1 in 2 waves"),
    "speed2_int8_res294": ("compose_int8_res308", "294 px (441 tokens), INT8 recalibrated: attention in 2 waves"),
    "speed2_int8_res308_heads15": ("compose_int8_res308", "attention heads 16 -> 15 per block (FLAP fluct, bias-comp)"),
    "speed2_int8_res308_mlp3840s308": ("compose_int8_res308", "MLP 3840 as above, importance and bias-comp from 308 px stats"),
    "speed2_int8_res294_mlp3840": ("speed2_int8_res294", "res294 + MLP 3840 (294 px stats)"),
    "speed2_int8_res294_drop7": ("speed2_int8_res294", "res294 with block 7 removed (least damaging in the pool screen)"),
}


def latest(recs: list[dict], a: str, b: str) -> dict | None:
    c = [x for x in recs if x["kind"] == "compare" and x["a"] == a and x["b"] == b]
    return c[-1] if c else None


def rows() -> list[dict]:
    recs = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    out = []
    for cid, (ref, change) in ROWS.items():
        c = latest(recs, BASE, cid)
        if c is None:
            continue
        ev = [x for x in recs if x["kind"] == "eval" and x["candidate"] == cid and x["candidate_sha"] == c["b_sha"]][-1]
        w = latest(recs, ref, cid) if ref != cid else None
        out.append({"id": cid, "change": change, "cv": ev["cv_acc_mean"], "sd": ev["cv_acc_sd"], "d_pp": 100 * c["d_cv_acc"],
                    "mcnemar": c["mcnemar"], "ratio": c["latency"]["ratio"], "range": c["latency"]["ratio_range"],
                    "b1": c["latency"]["b_batch1_median_ms"], "verdict": c["verdict"], "ref": ref,
                    "vs_ref": w and w["latency"]["ratio"], "vs_ref_range": w and w["latency"]["ratio_range"],
                    "vs_ref_beyond": w and w["latency"]["beyond_noise"], "vs_ref_mcnemar": w and w["mcnemar"],
                    "ep": {k: v["nodes"] for k, v in ev["ep_assignment"].items()}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", action="store_true")
    a = ap.parse_args()
    head = ["id", "change", "CV acc ± sd", "Δ pp", "McNemar B-/B+ p", "ratio vs fp32 (rounds)", "b1 ms",
            "vs winner (rounds, beyond noise)", "EP nodes", "verdict vs fp32"]
    lines = []
    for r in rows():
        m, rr = r["mcnemar"], r["range"]
        vs = (f"{r['vs_ref']:.4f} ({r['vs_ref_range'][0]:.4f}-{r['vs_ref_range'][1]:.4f}, {r['vs_ref_beyond']})"
              if r["vs_ref"] else "-")
        lines.append([r["id"], r["change"], f"{r['cv']:.4f} ± {r['sd']:.4f}", f"{r['d_pp']:+.2f}",
                      f"{m['b_wrong_a_right']}/{m['b_right_a_wrong']} {m['p_two_sided']:.3f}",
                      f"{r['ratio']:.4f} ({rr[0]:.4f}-{rr[1]:.4f})", f"{r['b1']:.2f}", vs,
                      " ".join(f"{k.replace('ExecutionProvider', '')}:{v}" for k, v in r["ep"].items()), r["verdict"]])
    if a.tsv:
        print("\n".join("\t".join(x) for x in [head, *lines]))
        return
    w = [max(len(x[i]) for x in [head, *lines]) for i in range(len(head))]
    for x in [head, *lines]:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(x)))


if __name__ == "__main__":
    main()
