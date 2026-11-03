"""Where the INT8 error comes from: fake-quantised s294 (or a state) with one GEMM group left in floating point at a
time, cosine to the fp32 t294 embeddings of the same dev images (view 0).

    sens.py [--state STATE.npz] [--n 384]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import fakequant
from common import GRAPHS, R, load_block_state, loader, source, trunk

GROUPS = {"none": (), "attn_pool": ("attn_pool.",), "qkv": ("attn.qkv",), "proj": ("attn.proj",),
          "fc1": ("blocks.", "mlp.fc1"), "fc2": ("blocks.", "mlp.fc2"), "blocks0-8": tuple(f"blocks.{i}." for i in range(9)),
          "blocks9-17": tuple(f"blocks.{i}." for i in range(9, 18)), "blocks18-26": tuple(f"blocks.{i}." for i in range(18, 27))}


def selected(role: str, pats: tuple[str, ...]) -> bool:
    if pats[:1] == ("blocks.",) and len(pats) == 2:
        return role.startswith("blocks.") and role.endswith(pats[1])
    return any(p in role for p in pats)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path)
    ap.add_argument("--n", type=int, default=384)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    m = trunk(GRAPHS["s294"])
    if a.state:
        load_block_state(m, a.state)
    m.cuda()
    fakequant.quantize(m, fakequant.calibrate(m))
    items = source("dev")
    idx = np.linspace(0, len(items) - 1, a.n).round().astype(int)
    X = torch.cat([x for (x,) in loader([items[i] for i in idx], np.zeros(a.n, int), (294,), 32)])
    T = torch.from_numpy(np.load(R / "screens/t294.npy")[idx])
    mods = fakequant.gemms(m)
    for name, pats in GROUPS.items():
        for r, mod in mods.items():
            mod.enabled = not (pats and selected(r, pats))
        E = torch.cat([m(X[j:j + 32].cuda()).float().cpu() for j in range(0, len(X), 32)])
        c = torch.nn.functional.cosine_similarity(E, T)
        print(json.dumps({"fp_group": name, "n_fp": sum(not mod.enabled for mod in mods.values()),
                          "cos_t294": round(float(c.mean()), 5), "p05": round(float(c.quantile(0.05)), 5)}), flush=True)
    for mod in mods.values():
        mod.enabled = False
    E = torch.cat([m(X[j:j + 32].cuda()).float().cpu() for j in range(0, len(X), 32)])
    print(json.dumps({"fp_group": "all (fp32 s294)", "cos_t294": round(float(torch.nn.functional.cosine_similarity(E, T).mean()), 5)}))


if __name__ == "__main__":
    main()
