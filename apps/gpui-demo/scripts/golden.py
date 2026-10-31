"""Compare a `gpui-demo classify --json` output against the golden baseline in results/demo/golden/.

    python3 apps/gpui-demo/scripts/golden.py results/demo/golden/classify_all.json NEW.json [--heads a,b] [--tol 1e-6]

The baseline was made at 07e569c with onnxruntime 1.31.1 on the CPU EP, from apps/gpui-demo:
    gpui-demo classify --model all --json demo_stack/*.jpg   (classify_all.json, 112 rows)
    gpui-demo classify --json demo_stack/*.jpg               (classify_default.json, 28 rows of model.onnx)
On the CPU EP two runs are bit-identical. Rows are matched on (image file name, model; a row without one is
"deployed"). A row passes when the label, the abstain flag and the top-3 labels are equal and every probability is
within --tol. Exit status 1 on any failure or missing row.
"""
import argparse
import json
import sys
from pathlib import Path


def rows(path, heads):
    out = {}
    for r in json.load(open(path)):
        key = (Path(r["source"]).name, r.get("model") or "deployed")
        if heads is None or key[1] in heads:
            out[key] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--heads", default=None)
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    heads = set(a.heads.split(",")) if a.heads else None
    base, cand = rows(a.baseline, heads), rows(a.candidate, heads)
    fails, worst = [], 0.0
    for key, b in sorted(base.items()):
        c = cand.get(key)
        if c is None:
            fails.append(f"{key}: missing")
            continue
        if b["error"] or c["error"]:
            fails.append(f"{key}: error {b['error']!r} vs {c['error']!r}")
            continue
        d = max(abs(x - y) for x, y in zip(b["probs"], c["probs"])) if len(b["probs"]) == len(c["probs"]) else 1.0
        worst = max(worst, d)
        why = []
        if b["label"] != c["label"]:
            why.append(f"label {b['label']} vs {c['label']}")
        if b["unsure"] != c["unsure"]:
            why.append(f"unsure {b['unsure']} vs {c['unsure']}")
        if [t[0] for t in b["top3"]] != [t[0] for t in c["top3"]]:
            why.append("top3 order")
        if d > a.tol:
            why.append(f"max|dp| {d:.3g}")
        if why:
            fails.append(f"{key}: " + ", ".join(why))
    extra = sorted(set(cand) - set(base))
    per_head = {}
    for (_, m) in base:
        per_head[m] = per_head.get(m, 0) + 1
    print(f"rows {len(base)} ({', '.join(f'{k}={v}' for k, v in sorted(per_head.items()))}); "
          f"extra in candidate {len(extra)}; max|dp| {worst:.3g} (tol {a.tol:g}); failures {len(fails)}")
    for f in fails[:40]:
        print("  FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
