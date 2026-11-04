"""Diagnostic: TensorRT's fused attention time against the head count, at the trunk's 484 tokens and head dim 72.

Writes softmax(q k^T / sqrt(72)) v over [1, H, T, 72] inputs, in the exporter's op pattern, for each H, and runs
tools/quant/trtprobe.py --fp16 on it. Not a result: it only tells how attention time steps with H.

    tools/quant/py tools/speed2/mha_probe.py <out_dir> [--heads 12,13,14,15,16] [--tokens 484] [--dim 72]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

QUANT = Path(__file__).resolve().parents[1] / "quant"


def graph(h: int, t: int, d: int) -> onnx.ModelProto:
    n = helper.make_node
    nodes = [n("Mul", ["q", "scale"], ["qs"]), n("Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2]),
             n("MatMul", ["qs", "kt"], ["s"]), n("Softmax", ["s"], ["p"], axis=-1), n("MatMul", ["p", "v"], ["o"])]
    io = [helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, h, t, d]) for x in "qkv"]
    g = helper.make_graph(nodes, f"mha{h}", io, [helper.make_tensor_value_info("o", TensorProto.FLOAT, [1, h, t, d])],
                          [numpy_helper.from_array(np.array(d ** -0.5, np.float32), "scale")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--heads", default="12,13,14,15,16")
    ap.add_argument("--tokens", type=int, default=484)
    ap.add_argument("--dim", type=int, default=72)
    a = ap.parse_args()
    res = {}
    for h in map(int, a.heads.split(",")):
        d = a.out / f"h{h}_t{a.tokens}_d{a.dim}"
        d.mkdir(parents=True, exist_ok=True)
        onnx.save(graph(h, a.tokens, a.dim), d / "mha.onnx")
        r = subprocess.run([sys.executable, str(QUANT / "trtprobe.py"), str(d / "mha.onnx"), str(d / "probe"), "--fp16",
                            "--iters", "200"], capture_output=True, text=True, cwd=QUANT)
        if r.returncode:
            raise SystemExit(r.stderr[-2000:])
        prof = json.loads((d / "probe" / "profile.json").read_text())
        res[h] = {"layer_ms": round(sum(prof.values()), 4), "layers": {k[:60]: round(v, 4) for k, v in prof.items()},
                  "execute_line": next((ln for ln in r.stdout.splitlines() if "execute_v2" in ln), r.stderr[-300:])}
        print(h, json.dumps(res[h]), flush=True)
    (a.out / f"mha_t{a.tokens}_d{a.dim}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
