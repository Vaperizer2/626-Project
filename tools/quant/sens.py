"""Per-GEMM INT8 sensitivity: fake-quantise chosen GEMMs in fp32 and measure the embedding error they cause.

One CUDA-EP session (strict fp32, TF32 off) holds every GEMM's INT8 fake quantisation (per-channel int8 weights,
per-tensor activation scale from the pool statistics), each behind a 0/1 gate input:
x' = x + g (DQ(Q(x)) - x), W' = W + g (DQ(Wq) - W). Each configuration is just a different gate vector, so one
session measures single-block, per-GEMM-type and leave-one-block-out configurations on the pool ``sens`` images.

    sens.py <src_dir> <stats.npz> <out.json> [--calib max] [--n 64]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from common import ART, gpu_lock, pool_images, preprocess
from fidelity import report
from qdq import amax, load, roles, tokens


def gated_graph(src: Path, stats, calib: str, dst: Path) -> list[str]:
    m, w = load(src)
    g = m.graph
    r = roles(m)
    order = sorted(r, key=lambda k: (int(k.split(".")[1]) if k.startswith("blocks.") else 99, k))
    new_nodes = {}
    for role in order:
        n = r[role]
        x, wname = n.input[0], n.input[1]
        p = f"s.{role}"
        W = w[wname]
        sw = np.maximum(np.abs(W).max(0), 1e-8) / 127
        g.initializer.extend([
            numpy_helper.from_array(np.array(amax(stats, x, calib, tokens(role)) / 127, np.float32), f"{p}.sa"),
            numpy_helper.from_array(np.array(0, np.int8), f"{p}.zp"),
            numpy_helper.from_array(np.clip(np.round(W / sw), -127, 127).astype(np.int8), f"{p}.wq"),
            numpy_helper.from_array(sw.astype(np.float32), f"{p}.sw"),
            numpy_helper.from_array(np.zeros(len(sw), np.int8), f"{p}.wzp")])
        g.input.append(helper.make_tensor_value_info(f"gate.{role}", TensorProto.FLOAT, []))
        mk = helper.make_node
        new_nodes[id(n)] = [
            mk("QuantizeLinear", [x, f"{p}.sa", f"{p}.zp"], [f"{p}.xq"]),
            mk("DequantizeLinear", [f"{p}.xq", f"{p}.sa", f"{p}.zp"], [f"{p}.xdq"]),
            mk("Sub", [f"{p}.xdq", x], [f"{p}.xd"]), mk("Mul", [f"{p}.xd", f"gate.{role}"], [f"{p}.xg"]),
            mk("Add", [x, f"{p}.xg"], [f"{p}.x"]),
            mk("DequantizeLinear", [f"{p}.wq", f"{p}.sw", f"{p}.wzp"], [f"{p}.wdq"], axis=1),
            mk("Sub", [f"{p}.wdq", wname], [f"{p}.wd"]), mk("Mul", [f"{p}.wd", f"gate.{role}"], [f"{p}.wg"]),
            mk("Add", [wname, f"{p}.wg"], [f"{p}.w"])]
        n.input[0], n.input[1] = f"{p}.x", f"{p}.w"
    nodes = []
    for n in g.node:
        nodes.extend(new_nodes.get(id(n), []))
        nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=dst.name + ".data", size_threshold=1024)
    return order


def configs(order: list[str]) -> dict[str, set[str]]:
    blocks = sorted({int(k.split(".")[1]) for k in order if k.startswith("blocks.")})
    cfg = {"none": set(), "all": set(order)}
    for t in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
        cfg[f"type:{t}"] = {k for k in order if k.endswith(t) and k.startswith("blocks.")}
    cfg["head"] = {k for k in order if k.startswith("attn_pool.")}
    for b in blocks:
        cfg[f"only:{b}"] = {k for k in order if k.startswith(f"blocks.{b}.")}
    for b in blocks:
        cfg[f"allbut:{b}"] = set(order) - cfg[f"only:{b}"]
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("stats", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--calib", default="max")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--only", default="", help="comma list of config names to run")
    a = ap.parse_args()
    import onnxruntime as ort

    work = ART / "work" / f"sens_{a.src.name}_{a.calib}"
    graph = work / "gated.onnx"
    order = gated_graph(a.src, np.load(a.stats), a.calib, graph)
    X = preprocess(pool_images("sens", a.n))
    cfg = configs(order)
    if a.only:
        cfg = {k: v for k, v in cfg.items() if k in a.only.split(",")}
    ort.preload_dlls()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    res = {}
    with gpu_lock():
        sess = ort.InferenceSession(str(graph), so, providers=[("CUDAExecutionProvider", {"use_tf32": 0})])
        t0 = time.perf_counter()

        def run(on: set[str]) -> np.ndarray:
            gates = {f"gate.{k}": np.array(float(k in on), np.float32) for k in order}
            return np.concatenate([sess.run(["embedding"], {"pixel_values": X[i:i + 1], **gates})[0]
                                   for i in range(len(X))])

        R = run(set())
        for name, on in cfg.items():
            res[name] = {**report(run(on), R), "n_gemms": len(on)}
            print(f"{name:14s} 1-cos mean {1 - res[name]['cos_mean']:.5f}  p05 cos {res[name]['cos_p05']:.4f}  "
                  f"agree {res[name]['head_argmax_agree']:.3f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        del sess
    a.out.write_text(json.dumps({"src": str(a.src), "stats": str(a.stats), "calib": a.calib, "n": a.n,
                                 "results": res}, indent=1))
    subprocess.run(["rm", "-rf", str(work)], check=True)


if __name__ == "__main__":
    main()
