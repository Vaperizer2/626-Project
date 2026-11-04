"""Per-block cost table for sched.py from a ladder probe: one INT8 graph whose token count steps down every two blocks,
profiled layer by layer (probe.sh), so one engine measures every count.

    fit_costs.py <probe_dir> <spec> <out.json> [--fixed <probe_dir>=<spec>] --ref <spec>=<harness ratio> ...

From the profile's per-layer (block, role, ms):
- A(T): qkv + fused attention + proj of blocks whose attention sees T tokens.
- M(T): the rest of a block with no merge at T tokens (MLP, LayerNorms, quantise, QKV shuffles).
- G(T): what a merge block costs beyond A(t_in) + M(t_out), by the count before the merge (the ToMe ops).
- fixed: layers outside the blocks (patch embedding, final norm, attention pool), linear in the final token count
  through this probe and the --fixed probes (the pool's kv projection runs over the final tokens).
The harness line maps predicted probe ms to the frozen harness's paired ratio through reference candidates whose
ratio is known: least squares over (predicted ms, ratio); with one reference, a line through the origin.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from profile_blocks import split  # noqa: E402
from sched import Costs, Schedule  # noqa: E402

ATTN = {"attn.qkv", "mha", "attn.proj"}


def fit(probe: Path, s: Schedule) -> dict:
    rows = split(probe)["layers"]
    ti, to = s.attn_tokens(), s.mlp_tokens()
    attn, total = defaultdict(float), defaultdict(float)
    fixed = 0.0
    for r in rows:
        b = r["block"]
        if b is None:
            fixed += r["ms"]
            continue
        total[b] += r["ms"]
        if r["role"] in ATTN:
            attn[b] += r["ms"]
    A, M, G = defaultdict(list), defaultdict(list), defaultdict(list)
    for b in range(len(ti)):
        A[ti[b]].append(attn[b])
    a = {t: float(np.mean(v)) for t, v in A.items()}
    for b in range(len(ti)):
        if ti[b] == to[b]:
            M[to[b]].append(total[b] - a[ti[b]])
    m = {t: float(np.mean(v)) for t, v in M.items()}
    for b in range(len(ti)):
        if ti[b] != to[b]:
            G[ti[b]].append(total[b] - a[ti[b]] - m[to[b]])
    g = {t: float(np.mean(v)) for t, v in G.items()}
    return {"A": a, "M": m, "G": g, "fixed": fixed, "probe": str(probe), "spec": s.spec,
            "per_block": {b: round(total[b], 4) for b in sorted(total)}, "final_tokens": to[-1]}


def fixed_of(probe: Path) -> float:
    return sum(r["ms"] for r in split(probe)["layers"] if r["block"] is None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", type=Path)
    ap.add_argument("spec")
    ap.add_argument("out", type=Path)
    ap.add_argument("--ref", action="append", default=[], help="<spec>=<harness paired ratio>")
    ap.add_argument("--fixed", action="append", default=[], help="<probe_dir>=<its spec>")
    a = ap.parse_args()
    d = fit(a.probe, Schedule.parse(a.spec))
    pts = [(d["final_tokens"], d["fixed"])] + [(Schedule.parse(f.partition("=")[2]).mlp_tokens()[-1],
                                                 fixed_of(Path(f.partition("=")[0]))) for f in a.fixed]
    f1, f0 = np.polyfit(*zip(*pts), 1) if len(pts) > 1 else (0.0, d["fixed"])
    d["fixed"] = {"f0": float(f0), "f1": float(f1), "points": pts}
    d["harness"] = {"slope": 1.0, "intercept": 0.0}
    a.out.write_text(json.dumps(d, indent=1))
    c = Costs.load(a.out)
    refs = [(Schedule.parse(r.rpartition("=")[0]), float(r.rpartition("=")[2])) for r in a.ref]
    x = np.array([c.ms(s) for s, _ in refs])
    y = np.array([r for _, r in refs])
    if len(refs) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = (y / x).item(), 0.0
    d["harness"] = {"slope": float(slope), "intercept": float(intercept),
                    "refs": [{"spec": s.spec, "ratio": r, "ms": float(m)} for (s, r), m in zip(refs, x)]}
    a.out.write_text(json.dumps(d, indent=1))
    for k in "AMG":
        print(k, " ".join(f"{t}:{v:.4f}" for t, v in sorted(d[k].items(), key=lambda kv: -kv[0])))
    print(f"fixed {f0:.3f} + {f1:.6f} * T_final ms from {pts}; harness ratio = {slope:.5f} * ms + {intercept:.5f}")
    for (s, r), m in zip(refs, x):
        print(f"  ref {s.id}: {m:.3f} ms, ratio {r} (line gives {slope * m + intercept:.4f})")


if __name__ == "__main__":
    main()
