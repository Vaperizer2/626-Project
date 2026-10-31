"""Write quant-track candidates: a graph (baseline or QDQ) plus preprocess.json and runtime.json.

    candidate.py static1 <dst> '<runtime.json as JSON>'   # baseline graph with batch fixed to 1
    candidate.py padheads <dst> '<runtime.json as JSON>'  # + attention head dim zero-padded 72 -> PAD (exact)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from common import BASELINE, REPO

TRT_FP16 = {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": "trt_cache",
            "trt_timing_cache_enable": True, "trt_timing_cache_path": "trt_cache",
            "trt_max_workspace_size": 2147483648}


def git_sha() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def fix_batch(m: onnx.ModelProto) -> onnx.ModelProto:
    for v in (m.graph.input[0], m.graph.output[0]):
        d = v.type.tensor_type.shape.dim[0]
        d.ClearField("dim_param")
        d.dim_value = 1
    return m


def write(dst: Path, runtime: dict, note: str) -> None:
    pre = json.loads((BASELINE / "preprocess.json").read_text())
    pre["built_from"] = f"{BASELINE} ({note}; tools/quant @ {git_sha()})"
    (dst / "preprocess.json").write_text(json.dumps(pre, indent=1))
    (dst / "runtime.json").write_text(json.dumps(runtime, indent=1))
    subprocess.run(["cp", str(BASELINE / "labels.json"), str(dst / "labels.json")], check=True)


def static1(dst: Path, runtime: dict) -> None:
    dst.mkdir(parents=True, exist_ok=False)
    m = fix_batch(onnx.load(BASELINE / "model.onnx", load_external_data=False))
    onnx.save(m, dst / "model.onnx")
    subprocess.run(["cp", "--reflink=auto", str(BASELINE / "model.onnx.data"), str(dst / "model.onnx.data")],
                   check=True)
    write(dst, runtime, "batch fixed to 1, runtime options")


def padheads(dst: Path, runtime: dict, pad: int = 80) -> None:
    """Zero-pad every block's attention head dim from 72 to ``pad``. Padded q/k columns add 0 to q.k, padded v
    columns produce output columns that meet zero rows of the proj weight, so the math is unchanged; the softmax
    scale stays 1/sqrt(72). The attention pool keeps its own reshape constants and is not padded."""
    from qdq import load, roles, set_init

    m, w = load(BASELINE)
    g = m.graph
    heads, hd = 16, 72
    new = {"val_15": [-1, 729, 3, heads, pad], "val_47": [-1, 729, pad], "val_50": [-1, heads, pad, 729],
           "val_63": [-1, 729, heads * pad]}
    for k, v in new.items():
        g.initializer.append(numpy_helper.from_array(np.array(v, np.int64), f"pad.{k}"))
    r = roles(m)
    blocks = sorted({int(k.split(".")[1]) for k in r if k.startswith("blocks.")})
    pool_nodes = {n.name for n in g.node if n.op_type == "Reshape" and n.output[0] in
                  {"val_1974", "val_1977"}}
    for n in g.node:
        if n.op_type == "Reshape" and n.input[1] in new and n.name not in pool_nodes:
            n.input[1] = f"pad.{n.input[1]}"
    for b in blocks:
        qkv, proj = r[f"blocks.{b}.attn.qkv"], r[f"blocks.{b}.attn.proj"]
        W = w[qkv.input[1]].reshape(-1, 3, heads, hd)
        set_init(m, qkv.input[1], np.pad(W, ((0, 0), (0, 0), (0, 0), (0, pad - hd))).reshape(len(W), -1))
        bias = f"trunk.blocks.{b}.attn.qkv.bias"
        B = w[bias].reshape(3, heads, hd)
        set_init(m, bias, np.pad(B, ((0, 0), (0, 0), (0, pad - hd))).reshape(-1))
        P = w[proj.input[1]].reshape(heads, hd, -1)
        set_init(m, proj.input[1], np.pad(P, ((0, 0), (0, pad - hd), (0, 0))).reshape(heads * pad, -1))
    fix_batch(m)
    dst.mkdir(parents=True, exist_ok=False)
    onnx.save(m, dst / "model.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
              location="model.onnx.data", size_threshold=1024)
    write(dst, runtime, f"batch fixed to 1, attention head dim zero-padded 72 -> {pad}, runtime options")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("static1", "padheads"):
        p = sub.add_parser(name)
        p.add_argument("dst", type=Path)
        p.add_argument("runtime_json")
    a = ap.parse_args()
    {"static1": static1, "padheads": padheads}[a.cmd](a.dst, json.loads(a.runtime_json))


if __name__ == "__main__":
    main()
