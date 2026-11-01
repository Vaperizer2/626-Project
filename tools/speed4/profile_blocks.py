"""Split a trtprobe run (tools/quant/trtprobe.py: layers.json + profile.json) into per-role costs by token count.

    profile_blocks.py <probe_dir> [--json out.json]

Each layer gets the first role its ONNX metadata names (``blocks.<i>.attn.qkv``, ``...mlp.fc1``, ...; a fused
attention kernel gets ``mha``) and the token count of its first [1, T, C] or [1, H, T, d] output. Layers without a
role are charged to the block whose name they carry, as ``other``. Prints ms per (role, tokens): the empirical side
of the wave model.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

ROLES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def tokens_of(layer: dict) -> int | None:
    outs = layer.get("Outputs", [])
    if isinstance(outs, str):
        outs = ast.literal_eval(outs)
    for o in outs:
        d = o.get("Dimensions", [])
        if len(d) == 3 and d[0] == 1 and d[1] > 1:
            return d[1]
        if len(d) == 4 and d[0] == 1 and d[1] == 16:
            return d[2]
    return None


def classify(layer: dict) -> tuple[str, int | None]:
    meta = layer.get("Metadata", "")
    name = layer["Name"]
    tac = layer.get("TacticName", "")
    if "mha" in name.lower() or "mha" in tac.lower():
        role = "mha"
    else:
        m = re.search(r"q\.blocks\.(\d+)\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)\.(?:xq|xdq|wdq)", meta)
        role = m.group(2) if m else ("attn_pool" if "attn_pool" in meta else "other")
    b = re.search(r"blocks\.(\d+)\.", meta)
    return role, int(b.group(1)) if b else None


def split(d: Path) -> dict:
    layers = json.loads((d / "layers.json").read_text())["Layers"]
    prof = json.loads((d / "profile.json").read_text())
    by = defaultdict(lambda: [0.0, 0])
    per_block = defaultdict(float)
    for layer in layers:
        ms = prof.get(layer["Name"])
        if ms is None:
            continue
        role, blk = classify(layer)
        t = tokens_of(layer)
        by[(role, t)][0] += ms
        by[(role, t)][1] += 1
        per_block[blk] += ms
    return {"total_ms": sum(prof.values()),
            "roles": [{"role": r, "tokens": t, "ms": round(ms, 4), "n": n, "ms_each": round(ms / n, 4)}
                      for (r, t), (ms, n) in sorted(by.items(), key=lambda kv: (kv[0][0], -(kv[0][1] or 0)))],
            "per_block": {str(k): round(v, 4) for k, v in sorted(per_block.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    s = split(a.probe)
    print(f"{a.probe.name}: {s['total_ms']:.3f} ms")
    for r in s["roles"]:
        print(f"  {r['role']:10s} T={str(r['tokens']):5s} n={r['n']:3d} total {r['ms']:7.3f} ms  each {r['ms_each']:.4f}")
    print("  per block:", " ".join(f"{k}:{v:.3f}" for k, v in s["per_block"].items()))
    if a.json:
        a.json.write_text(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
