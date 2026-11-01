"""Structured attention-head pruning of a So400m trunk graph (FLAP-style, no training).

    heads.py stats <model_dir> <out.npz> [--n 256]           # per block: sum and sum of squares over tokens of
                                                             # every attn.proj input channel, pool calib images
    heads.py prune <src_dir> <stats.npz> <dst_dir> --keep 14  # keep the 14 highest-scoring heads in every block

A head's score is the sum over its 72 channels of std(channel) x ||proj weight row||, the fluctuation criterion the
prune track found best for MLP neurons. The mean output of each dropped channel is folded into the proj bias. The
qkv columns and proj rows of dropped heads are removed and the block's attention Reshape targets get the new head
count; the attention pool is left alone. Statistics run on the CUDA EP with TF32 off, never on development images.
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
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quant"))

from qdq import gemm_table  # noqa: E402

HEADS, HEAD_DIM = 16, 72


def blocks_of(m: onnx.ModelProto) -> list[int]:
    return sorted({int(k.split(".")[1]) for k in gemm_table(m) if k.startswith("blocks.")})


def stats(src: Path, out: Path, n: int) -> None:
    import onnxruntime as ort

    from common import pool_images, preprocess, spec_of

    m = onnx.load(src / "model.onnx", load_external_data=False)
    t = gemm_table(m)
    g = m.graph
    g.initializer.append(numpy_helper.from_array(np.array([0, 1], np.int64), "hs.axes01"))
    names = []
    for b in blocks_of(m):
        x = t[f"blocks.{b}.attn.proj"][0].input[0]
        g.node.extend([helper.make_node("ReduceSum", [x, "hs.axes01"], [f"hs.sum{b}"], keepdims=0),
                       helper.make_node("Mul", [x, x], [f"hs.sq{b}"]),
                       helper.make_node("ReduceSum", [f"hs.sq{b}", "hs.axes01"], [f"hs.sumsq{b}"], keepdims=0)])
        g.output.extend(helper.make_tensor_value_info(f"hs.{k}{b}", TensorProto.FLOAT, None) for k in ("sum", "sumsq"))
        names += [f"hs.sum{b}", f"hs.sumsq{b}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / "heads_stats.onnx"
    onnx.save(m, tmp)
    for f in src.glob("model.onnx.*"):
        subprocess.run(["cp", "--reflink=auto", str(f), str(out.parent / f.name)], check=True)
    X = preprocess(pool_images("calib", n), spec_of(src))
    ort.preload_dlls()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(str(tmp), so, providers=[("CUDAExecutionProvider", {"use_tf32": 0}),
                                                      "CPUExecutionProvider"])
    acc = None
    for i in range(len(X)):
        r = s.run(names, {"pixel_values": X[i:i + 1]})
        acc = [a.astype(np.float64) for a in r] if acc is None else [a + b for a, b in zip(acc, r)]
    del s
    tokens = (spec_of(src).input_size // 14) ** 2 * len(X)
    np.savez(out, mean=np.stack(acc[0::2]) / tokens, meansq=np.stack(acc[1::2]) / tokens,
             blocks=np.array(blocks_of(m)), n_images=len(X))
    tmp.unlink()
    for f in src.glob("model.onnx.*"):
        (out.parent / f.name).unlink()
    print(f"wrote {out} ({len(X)} pool calib images)")


def attention_nodes(m: onnx.ModelProto, qkv: onnx.NodeProto, proj: onnx.NodeProto) -> list[onnx.NodeProto]:
    """Every node on the paths from the qkv MatMul to the proj MatMul (exclusive)."""
    cons: dict[str, list[onnx.NodeProto]] = {}
    for n in m.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    seen, out, stack = set(), [], list(qkv.output)
    while stack:
        for n in cons.get(stack.pop(), []):
            if n is proj or id(n) in seen:
                continue
            seen.add(id(n))
            out.append(n)
            stack.extend(n.output)
    return out


def prune(src: Path, stats_path: Path, dst: Path, keep: int) -> None:
    m = onnx.load(src / "model.onnx")
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    w = lambda name: numpy_helper.to_array(inits[name])  # noqa: E731

    def put(name: str, a: np.ndarray) -> None:
        inits[name].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(a, np.float32), name))

    st = np.load(stats_path)
    table = gemm_table(m)
    new_shapes: dict[str, str] = {}
    kept = {}
    for j, b in enumerate(st["blocks"]):
        (qkv, qkv_b), (proj, proj_b) = table[f"blocks.{b}.attn.qkv"], table[f"blocks.{b}.attn.proj"]
        Wp = w(proj.input[1])
        std = np.sqrt(np.maximum(st["meansq"][j] - st["mean"][j] ** 2, 0))
        score = (std * np.linalg.norm(Wp, axis=1)).reshape(HEADS, HEAD_DIM).sum(1)
        heads = np.sort(np.argsort(-score, kind="stable")[:keep])
        drop = np.setdiff1d(np.arange(HEADS), heads)
        kept[int(b)] = {"heads": heads.tolist(), "score": score.round(4).tolist()}
        rows = (drop[:, None] * HEAD_DIM + np.arange(HEAD_DIM)).ravel()
        put(proj_b, w(proj_b) + st["mean"][j][rows] @ Wp[rows])
        put(proj.input[1], Wp.reshape(HEADS, HEAD_DIM, -1)[heads].reshape(keep * HEAD_DIM, -1))
        Wq = w(qkv.input[1])
        put(qkv.input[1], Wq.reshape(len(Wq), 3, HEADS, HEAD_DIM)[:, :, heads].reshape(len(Wq), -1))
        put(qkv_b, w(qkv_b).reshape(3, HEADS, HEAD_DIM)[:, heads].reshape(-1))
        for n in attention_nodes(m, qkv, proj):
            if n.op_type != "Reshape":
                continue
            old = n.input[1]
            if old not in new_shapes:
                shape = w(old).copy()
                shape[shape == HEADS] = keep
                shape[shape == HEADS * HEAD_DIM] = keep * HEAD_DIM
                new_shapes[old] = f"heads{keep}.{old}"
                g.initializer.append(numpy_helper.from_array(shape, new_shapes[old]))
            n.input[1] = new_shapes[old]
    del g.value_info[:]
    m.ir_version = 10
    dst.mkdir(parents=True, exist_ok=False)
    onnx.save(m, dst / "model.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
              location="model.onnx.data", size_threshold=1024)
    onnx.checker.check_model(str(dst / "model.onnx"))
    pre = json.loads((src / "preprocess.json").read_text())
    pre["built_from"] = (f"{pre.get('built_from')}; attention heads {HEADS} -> {keep} per block by std x ||proj row||, "
                         f"bias-compensated (tools/speed2/heads.py)")
    (dst / "preprocess.json").write_text(json.dumps(pre, indent=1))
    for f in ("runtime.json", "labels.json"):
        shutil.copy(src / f, dst / f)
    (dst / "heads.json").write_text(json.dumps({"keep": keep, "stats": str(stats_path), "blocks": kept}))
    print(f"{dst}: {keep}/{HEADS} heads per block, reshape targets {sorted(new_shapes)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("src", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--n", type=int, default=256)
    p = sub.add_parser("prune")
    p.add_argument("src", type=Path)
    p.add_argument("stats", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--keep", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "stats":
        stats(a.src, a.out, a.n)
    else:
        prune(a.src, a.stats, a.dst, a.keep)


if __name__ == "__main__":
    main()
