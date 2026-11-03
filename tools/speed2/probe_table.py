"""Group a trtprobe run (tools/quant/trtprobe.py: layers.json + profile.json) by GEMM role and kernel.

    probe_table.py <probe_dir> [<probe_dir> ...]

Diagnostic only. A layer's role is the first ``q.blocks.<i>.<role>`` (or ``attn_pool``) its ONNX metadata names;
layers without one are grouped by their Myelin fusion name with the block numbers stripped.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def role_of(layer: dict) -> str:
    meta = layer.get("Metadata", "")
    m = re.search(r"q\.blocks\.\d+\.([a-z_.0-9]+?)\.(?:xdq|wdq)", meta)
    if m:
        return m.group(1)
    if "attn_pool" in meta:
        return "attn_pool"
    if "mha" in layer["Name"].lower() or "mha" in layer.get("TacticName", "").lower():
        return "mha"
    return re.sub(r"_myl\d+_\d+$", "", layer["Name"])


def table(d: Path) -> None:
    layers = json.loads((d / "layers.json").read_text())["Layers"]
    prof = json.loads((d / "profile.json").read_text())
    total = sum(prof.values())
    rows = defaultdict(lambda: [0.0, 0, set(), set()])
    for layer in layers:
        ms = prof.get(layer["Name"])
        if ms is None:
            continue
        r = rows[role_of(layer)]
        r[0] += ms
        r[1] += 1
        r[2].add(layer.get("TacticName", "")[:90])
        r[3].add(tuple(tuple(o["Dimensions"]) for o in layer.get("Outputs", [])))
    print(f"{d.name}: {total:.3f} ms over {len(prof)} layers")
    for k, (ms, n, tac, shp) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
        print(f"  {ms:7.3f} ms {100 * ms / total:5.1f}%  n={n:3d}  {k[:60]:60s} {sorted(shp)[:1]} {sorted(tac)[:2]}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        table(Path(p))
