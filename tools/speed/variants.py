"""Build variants of a candidate directory. Each result is an ordinary candidate: eval and time it.

    variants.py runtime  <src> <dst> '<runtime.json as JSON>'   # same graph, other EP / options
    variants.py fp16     <src> <dst>                             # float16 weights and math, float32 IO
    variants.py truncate <src> <dst> --drop K                    # remove the last K SigLIP blocks

Needs only onnx (+ onnxruntime for fp16), so it also runs on carlpc2 where RAM is plentiful.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import onnx


def reflink(src: Path, dst: Path) -> None:
    subprocess.run(["cp", "--reflink=auto", str(src), str(dst)], check=True)


def start(src: Path, dst: Path, note: str, runtime: dict | None = None) -> dict:
    dst.mkdir(parents=True, exist_ok=False)
    pre = json.loads((src / "preprocess.json").read_text())
    pre["built_from"] = f"{src} ({note}; tools/speed/variants.py)"
    (dst / "preprocess.json").write_text(json.dumps(pre, indent=1))
    rt = runtime or (json.loads((src / "runtime.json").read_text()) if (src / "runtime.json").exists() else {})
    (dst / "runtime.json").write_text(json.dumps(rt, indent=1))
    if (src / "labels.json").exists():
        shutil.copy(src / "labels.json", dst / "labels.json")
    return pre


def copy_graph(src: Path, dst: Path) -> None:
    for f in src.iterdir():
        if f.name == "model.onnx" or f.name.startswith("model.onnx."):
            reflink(f, dst / f.name)


def prune(g: onnx.GraphProto) -> None:
    producer = {o: n for n in g.node for o in n.output}
    need, stack = set(), [o.name for o in g.output]
    while stack:
        t = stack.pop()
        if t in producer and id(producer[t]) not in need:
            need.add(id(producer[t]))
            stack.extend(producer[t].input)
    nodes = [n for n in g.node if id(n) in need]
    used = {i for n in nodes for i in n.input}
    inits = [t for t in g.initializer if t.name in used]
    del g.node[:], g.initializer[:], g.value_info[:]
    g.node.extend(nodes)
    g.initializer.extend(inits)


def truncate(src: Path, dst: Path, drop: int) -> None:
    m = onnx.load(src / "model.onnx", load_external_data=False)
    ln = {n.input[1]: n for n in m.graph.node if n.op_type == "LayerNormalization"}
    depth = sum(1 for k in ln if k.endswith(".norm1.weight") and k.startswith("trunk.blocks."))
    keep = depth - drop
    ln["trunk.norm.weight"].input[0] = ln[f"trunk.blocks.{keep}.norm1.weight"].input[0]
    prune(m.graph)
    start(src, dst, f"blocks {keep}..{depth - 1} removed")
    onnx.save(m, dst / "model.onnx")
    for f in src.glob("model.onnx.*"):
        reflink(f, dst / f.name)


def fp16(src: Path, dst: Path) -> None:
    from onnxruntime.transformers.float16 import convert_float_to_float16

    m = onnx.load(src / "model.onnx")
    m = convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True)
    start(src, dst, "float16 via onnxruntime.transformers.float16, keep_io_types")
    big = m.ByteSize() > 2**31 - 2**24
    onnx.save(m, dst / "model.onnx", save_as_external_data=big, location="model.onnx.data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("runtime", "fp16", "truncate"):
        p = sub.add_parser(name)
        p.add_argument("src", type=Path)
        p.add_argument("dst", type=Path)
    sub.choices["runtime"].add_argument("runtime_json")
    sub.choices["truncate"].add_argument("--drop", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "runtime":
        start(a.src, a.dst, "runtime options only", json.loads(a.runtime_json))
        copy_graph(a.src, a.dst)
    elif a.cmd == "fp16":
        fp16(a.src, a.dst)
    else:
        truncate(a.src, a.dst, a.drop)


if __name__ == "__main__":
    main()
