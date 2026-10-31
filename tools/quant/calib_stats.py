"""Activation statistics at every weight-GEMM input, collected on pool images for INT8 calibration.

The graph gets extra outputs per GEMM activation x [1, T, C]: per-channel max |x| over tokens (for SmoothQuant and
the max calibrator) and the TOPK largest |x| over the whole tensor (for percentile calibrators). Reductions run on
the GPU inside the session, so only a few KB per tensor leave the device. Images are the pool calibration split
(disjoint from every development and test image; see calib_set.py).

    calib_stats.py <model.onnx> <out.npz> [--split calib] [--n 512]

The augmented graph and a reflink copy of the external weights go next to <out.npz>, never next to the source.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from common import gpu_lock, pool_images, preprocess

TOPK = 1024


def gemms(m: onnx.ModelProto) -> list[onnx.NodeProto]:
    """MatMul nodes with a 2-D weight initializer and a computed (non-constant) activation input."""
    inits = {t.name: t for t in m.graph.initializer}
    return [n for n in m.graph.node if n.op_type == "MatMul" and n.input[1] in inits
            and len(inits[n.input[1]].dims) == 2 and n.input[0] not in inits]


def stats_graph(src: Path, dst: Path) -> list[str]:
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    acts = list(dict.fromkeys(n.input[0] for n in gemms(m)))
    g.initializer.extend([numpy_helper.from_array(np.array([0, 1], np.int64), "qs.axes01"),
                          numpy_helper.from_array(np.array([-1], np.int64), "qs.flat"),
                          numpy_helper.from_array(np.array([TOPK], np.int64), "qs.k")])
    for i, x in enumerate(acts):
        a = f"qs.abs{i}"
        g.node.extend([helper.make_node("Abs", [x], [a]),
                       helper.make_node("ReduceMax", [a, "qs.axes01"], [f"cmax/{x}"], keepdims=0),
                       helper.make_node("Reshape", [a, "qs.flat"], [f"qs.flat{i}"]),
                       helper.make_node("TopK", [f"qs.flat{i}", "qs.k"], [f"top/{x}", f"qs.idx{i}"])])
        g.output.extend([helper.make_tensor_value_info(f"cmax/{x}", TensorProto.FLOAT, None),
                         helper.make_tensor_value_info(f"top/{x}", TensorProto.FLOAT, None)])
    for loc in {e.value for t in g.initializer if t.data_location == TensorProto.EXTERNAL for e in t.external_data
                if e.key == "location"}:
        subprocess.run(["cp", "--reflink=auto", str(src.parent / loc), str(dst.parent / loc)], check=True)
    onnx.save(m, dst)
    return acts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--split", default="calib")
    ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()
    import onnxruntime as ort

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.parent / "calib_stats.onnx"
    acts = stats_graph(a.model, tmp)
    X = preprocess(pool_images(a.split, a.n))
    print(f"{len(acts)} GEMM activations, {len(X)} images", flush=True)
    ort.preload_dlls()
    names = [f"cmax/{x}" for x in acts] + [f"top/{x}" for x in acts]
    with gpu_lock():
        sess = ort.InferenceSession(str(tmp), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        t0 = time.perf_counter()
        cmax = {x: [] for x in acts}
        top = {x: [] for x in acts}
        for i in range(len(X)):
            out = sess.run(names, {"pixel_values": X[i:i + 1]})
            for j, x in enumerate(acts):
                cmax[x].append(out[j])
                top[x].append(out[len(acts) + j])
        print(f"{time.perf_counter() - t0:.0f}s", flush=True)
        del sess
    np.savez(a.out, acts=np.array(acts), **{f"cmax/{x}": np.stack(v) for x, v in cmax.items()},
             **{f"top/{x}": np.stack(v) for x, v in top.items()})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
