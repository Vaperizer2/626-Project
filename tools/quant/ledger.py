"""Append one attempt to decision.tsv from the harness's latest compare of <candidate> against baseline_fp32.

    ledger.py <candidate_id> "<hypothesis>" "<change>" ["<note>"]
    ledger.py --failed <candidate_id> "<hypothesis>" "<change>" "<why>"   # no harness compare to read
"""

from __future__ import annotations

import argparse
import json

from common import ART

RESULTS = "/home/vaper/Work/Prototypes/626-artifacts/harness/results.jsonl"
COLS = ["id", "hypothesis", "change", "cv_acc", "sd", "d_vs_fp32_pp", "mcnemar", "ratio_vs_fp32", "verdict", "note"]


def latest(kind: str, cand: str, **match) -> dict | None:
    key = "b" if kind == "compare" else "candidate"
    recs = [r for r in map(json.loads, open(RESULTS)) if r.get("kind") == kind and r.get(key) == cand
            and all(r.get(k) == v for k, v in match.items())]
    return recs[-1] if recs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--failed", action="store_true")
    ap.add_argument("cand")
    ap.add_argument("hypothesis")
    ap.add_argument("change")
    ap.add_argument("note", nargs="?", default="")
    a = ap.parse_args()
    row = dict.fromkeys(COLS, "")
    row.update(id=a.cand, hypothesis=a.hypothesis, change=a.change, note=a.note)
    if a.failed:
        row["verdict"] = "FAILED"
    else:
        c = latest("compare", a.cand, a="baseline_fp32")
        e = latest("eval", a.cand, candidate_sha=c["b_sha"])
        m, lat = c["mcnemar"], c["latency"]
        row.update(cv_acc=f"{c['b_cv_acc']:.4f}", sd=f"{e['cv_acc_sd']:.4f}",
                   d_vs_fp32_pp=f"{100 * c['d_cv_acc']:+.2f}",
                   mcnemar=f"{m['b_wrong_a_right']}/{m['b_right_a_wrong']} p={m['p_two_sided']:.2f}",
                   ratio_vs_fp32=(f"{lat['ratio']:.4f} [{lat['ratio_range'][0]:.4f}-{lat['ratio_range'][1]:.4f}]"
                                  if lat.get("ratio") else "n/a"),
                   verdict=c["verdict"])
    path = ART / "decision.tsv"
    if not path.exists() or not path.read_text().strip():
        path.write_text("\t".join(COLS) + "\n")
    with path.open("a") as f:
        f.write("\t".join(str(row[k]).replace("\t", " ") for k in COLS) + "\n")
    print("\t".join(str(row[k]) for k in COLS))


if __name__ == "__main__":
    main()
