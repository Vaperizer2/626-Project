"""Unofficial accuracy screen on carlpc2: dev embeddings of a torch trunk, fidelity to the teachers, and the harness
head recipe on them.

    screen.py <name> <graph: t384|t294|s294|DIR> [--state STATE.npz] [--fp32]

Embeds the 2,512 dev images through view 0 (the exact harness transform at the graph's input size), saves
R/screens/<name>.npy, and appends one JSON line to R/screens/screens.jsonl with the cosine to the t294 and t384
teacher embeddings (when those exist) and the CV of tools/speed/evaluate.py's `cv_head` (fused and probe-only). This
is the frozen harness's head code on torch embeddings, not the INT8 TensorRT graph the harness scores, so it ranks
variants but is not a result.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import GRAPHS, R, REPO, dev_rows, graph_size, load_block_state, loader, source, trunk

import sys

sys.path.insert(0, str(REPO / "tools/speed"))
import evaluate  # noqa: E402

evaluate.DEPLOYED = Path("/home/carl/626-runs/distill/v2meta")  # byte-identical copy of exports/final/head.npz


@torch.no_grad()
def embed(m: torch.nn.Module, size: int, fp32: bool, batch: int = 32) -> np.ndarray:
    items = source("dev")
    out = []
    for (x,) in loader(items, np.zeros(len(items), int), (size,), batch, shuffle=False):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32):
            out.append(m(x.cuda(non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("graph")
    ap.add_argument("--state", type=Path)
    ap.add_argument("--fp32", action="store_true", help="fp32 (TF32 matmuls) instead of bf16 autocast")
    ap.add_argument("--fq", action="store_true", help="fake-quantise with the speed2 INT8 recipe (fakequant.py)")
    ap.add_argument("--keep-float", default="", help="with --fq: comma-separated GEMM roles left in floating point")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    d = GRAPHS.get(a.graph, Path(a.graph))
    m = trunk(d)
    if a.state:
        load_block_state(m, a.state)
    m.cuda()
    if a.fq:
        import fakequant

        cal = fakequant.calibrate(m)
        fakequant.quantize(m, cal)
        for r, mod in fakequant.gemms(m).items():
            mod.enabled = r not in a.keep_float.split(",")
        (R / "screens" / f"{a.name}.amax.json").write_text(json.dumps(fakequant.amax_table(cal), indent=0))
    t0 = time.time()
    F = embed(m, graph_size(d), a.fp32)
    (R / "screens").mkdir(exist_ok=True)
    np.save(R / "screens" / f"{a.name}.npy", F)
    mf, rows = dev_rows()
    fold = np.array([r["fold"] for r in rows])
    rec = {"name": a.name, "graph": str(d), "state": str(a.state) if a.state else None, "fp32": a.fp32, "fq": a.fq, "keep_float": a.keep_float,
           "embed_s": round(time.time() - t0)}
    for ref in ("t294", "t384"):
        p = R / "screens" / f"{ref}.npy"
        if p.exists() and a.name != ref:
            c = cos(F, np.load(p))
            rec[f"cos_{ref}"] = {"mean": round(float(c.mean()), 5), "p05": round(float(np.quantile(c, 0.05)), 5),
                                 "by_fold": [round(float(c[fold == k].mean()), 5) for k in range(mf.n_folds)]}
    res, pred = evaluate.cv_head(F, mf, rows, fusion=True)
    y = np.array([mf.class_to_idx[r["label"]] for r in rows])
    np.save(R / "screens" / f"{a.name}.correct.npy", pred == y)
    rec |= {k: round(res[k], 5) for k in ("cv_acc_mean", "probe_only_cv_acc_mean", "oof_acc")}
    rec["fold_acc"] = [round(f["accuracy"], 4) for f in res["folds"]]
    rec["fold_C"] = [f["C"] for f in res["folds"]]
    with open(R / "screens" / "screens.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
