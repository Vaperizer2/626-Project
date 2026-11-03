"""Check that common.trunk(DIR) computes DIR's ONNX graph: torch fp32 vs onnxruntime CPU fp32 on dev view-0 inputs.

    parity.py <graph: t384|t294|s294|DIR> [--state STATE.npz --onnx-dir DIR2] [--n 4]

With --state, the torch side loads STATE.npz and the ONNX side is DIR2 (the graph `build.py load` wrote it into).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common import GRAPHS, graph_size, load_block_state, loader, source, trunk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--state", type=Path)
    ap.add_argument("--onnx-dir", type=Path)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    d = GRAPHS.get(a.graph, Path(a.graph))
    torch.backends.cuda.matmul.allow_tf32 = False
    m = trunk(d)
    if a.state:
        load_block_state(m, a.state)
    items = source("dev")[:: 2512 // a.n][: a.n]
    (x,) = next(iter(loader(items, np.zeros(len(items), int), (graph_size(d),), a.n, workers=2)))
    with torch.no_grad():
        t = m(x).numpy()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str((a.onnx_dir or d) / "model.onnx"), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    o = np.concatenate([sess.run(None, {name: x[i:i + 1].numpy()})[0] for i in range(len(x))])
    c = (t * o).sum(1) / np.linalg.norm(t, axis=1) / np.linalg.norm(o, axis=1)
    print(json.dumps({"graph": str(d), "state": str(a.state), "n": len(x), "cos_min": float(c.min()),
                      "max_abs_diff": float(np.abs(t - o).max()), "max_abs": float(np.abs(o).max())}))


if __name__ == "__main__":
    main()
