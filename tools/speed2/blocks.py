"""Remove whole transformer blocks from a So400m trunk graph, anywhere in the stack.

    blocks.py drop   <src_dir> <dst_dir> --blocks 20[,21]   # block i's consumers read block i's input instead
    blocks.py screen <src_dir> <work_dir> [--n 64]          # every single-block drop, cosine to <src_dir> on pool
                                                             # sens images (CUDA EP, TF32 off); never dev images

A pre-LN block maps x to x + attn(norm1(x)) + mlp(...), so dropping it is the identity on the residual stream.
Its input is the tensor its norm1 LayerNormalization reads; its output is what the next block's norm1 (or the
final norm, for the last block) reads.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "speed"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quant"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compose"))

from variants import prune  # noqa: E402


def block_io(m: onnx.ModelProto) -> list[str]:
    """Residual-stream tensor entering block i, for i = 0..depth; the last entry is what the final norm reads."""
    ln = {n.input[1]: n for n in m.graph.node if n.op_type == "LayerNormalization"}
    depth = sum(1 for k in ln if k.startswith("trunk.blocks.") and k.endswith(".norm1.weight"))
    return [ln[f"trunk.blocks.{i}.norm1.weight"].input[0] for i in range(depth)] + [ln["trunk.norm.weight"].input[0]]


def drop(src: Path, dst: Path, blocks: list[int]) -> None:
    m = onnx.load(src / "model.onnx", load_external_data=False)
    io = block_io(m)
    rename = {}
    for b in sorted(blocks):
        rename[io[b + 1]] = rename.get(io[b], io[b])
    for n in m.graph.node:
        n.input[:] = [rename.get(i, i) for i in n.input]
    prune(m.graph)
    dst.mkdir(parents=True, exist_ok=False)
    onnx.save(m, dst / "model.onnx")
    for f in src.glob("model.onnx.*"):
        subprocess.run(["cp", "--reflink=auto", str(f), str(dst / f.name)], check=True)
    pre = json.loads((src / "preprocess.json").read_text())
    pre["built_from"] = f"{pre.get('built_from')}; blocks {blocks} removed (tools/speed2/blocks.py)"
    (dst / "preprocess.json").write_text(json.dumps(pre, indent=1))
    for f in ("runtime.json", "labels.json", "prune.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)


def screen(src: Path, work: Path, n: int) -> None:
    import onnxruntime as ort

    from common import pool_images, preprocess, spec_of
    from sanity import agreement, embed

    ort.preload_dlls()
    X = preprocess(pool_images("sens", n), spec_of(src))
    ref = embed(src, X)
    depth = len(block_io(onnx.load(src / "model.onnx", load_external_data=False))) - 1
    rows = []
    for b in range(depth):
        d = work / f"drop{b}"
        shutil.rmtree(d, ignore_errors=True)
        drop(src, d, [b])
        rows.append({"block": b, **agreement(ref, embed(d, X))})
        shutil.rmtree(d)
        print(json.dumps(rows[-1]), flush=True)
    order = sorted(rows, key=lambda r: -r["cos_median"])
    (work / "screen.json").write_text(json.dumps({"src": str(src), "n": len(X), "rows": rows}, indent=1))
    print("least damaging:", [(r["block"], round(r["cos_median"], 4), round(r["cos_min"], 4)) for r in order[:8]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drop")
    d.add_argument("src", type=Path)
    d.add_argument("dst", type=Path)
    d.add_argument("--blocks", required=True)
    s = sub.add_parser("screen")
    s.add_argument("src", type=Path)
    s.add_argument("work", type=Path)
    s.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "drop":
        drop(a.src, a.dst, [int(b) for b in a.blocks.split(",")])
    else:
        a.work.mkdir(parents=True, exist_ok=True)
        screen(a.src, a.work, a.n)


if __name__ == "__main__":
    main()
