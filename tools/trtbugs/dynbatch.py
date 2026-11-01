"""A batch-serving copy of a static batch-1 candidate or export that runs on TensorRT (bug 1, see NOTES.md).

    dynbatch.py <src_dir> <dst_dir> [--max-batch 8]

Graph: the batch dimension of every input and output becomes the symbolic ``batch``. The speed-track graphs already
carry -1 as the batch in every Reshape target (tools/speed/baseline.py, tools/tokens), so nothing else changes.
runtime.json: the session config entry ``optimization.disable_specified_optimizers`` turns off the two level-1
onnxruntime transformers whose output TensorRT 10.16 compiles wrongly. TransposeOptimizer moves the 5-D Transpose of
each packed q/k/v (or k/v) projection below the Slices that split it (wrong at batch > 1, bug 1). MatMulAddFusion
does not fire on a batch-symbolic graph; it is listed so every TensorRT runtime.json carries the same entry (bug 2).
Every other optimisation stays on. With --max-batch the TensorRT profile is 1..N at opt 1 (TensorRT tunes for
batch 1); without it the harness default 1..8 applies. Weights are reflinked.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import onnx
from onnx import TensorProto

DISABLE = {"optimization.disable_specified_optimizers": "MatMulAddFusion;TransposeOptimizer"}


def dynamic_batch(m: onnx.ModelProto) -> onnx.ModelProto:
    for v in (*m.graph.input, *m.graph.output):
        d = v.type.tensor_type.shape.dim[0]
        d.ClearField("dim_value")
        d.dim_param = "batch"
    del m.graph.value_info[:]
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--max-batch", type=int)
    a = ap.parse_args()
    a.dst.mkdir(parents=True, exist_ok=False)
    m = onnx.load(a.src / "model.onnx", load_external_data=False)
    locs = {e.value for t in m.graph.initializer if t.data_location == TensorProto.EXTERNAL for e in t.external_data
            if e.key == "location"}
    for loc in locs:
        subprocess.run(["cp", "--reflink=auto", str(a.src / loc), str(a.dst / loc)], check=True)
    onnx.save(dynamic_batch(m), a.dst / "model.onnx")
    onnx.checker.check_model(str(a.dst / "model.onnx"))
    for f in a.src.iterdir():
        if f.suffix in (".json", ".npz") and f.name != "runtime.json":
            shutil.copy(f, a.dst / f.name)
    rt = json.loads((a.src / "runtime.json").read_text())
    assert rt["provider"] == "TensorrtExecutionProvider", rt["provider"]
    rt["session_config"] = rt.get("session_config", {}) | DISABLE
    if rt.get("static_batch") is not None:
        rt["static_batch"] = None
    if a.max_batch:
        inp = m.graph.input[0].name
        shape = "x".join(str(d.dim_value) for d in m.graph.input[0].type.tensor_type.shape.dim[1:])
        rt["options"] |= {"trt_profile_min_shapes": f"{inp}:1x{shape}", "trt_profile_opt_shapes": f"{inp}:1x{shape}",
                          "trt_profile_max_shapes": f"{inp}:{a.max_batch}x{shape}"}
    (a.dst / "runtime.json").write_text(json.dumps(rt, indent=1))
    pre = a.dst / "preprocess.json"
    p = json.loads(pre.read_text())
    p["built_from"] = f"{p.get('built_from')}; batch symbolic, MatMulAddFusion and TransposeOptimizer off " \
                      "(tools/trtbugs/dynbatch.py)"
    pre.write_text(json.dumps(p, indent=1))
    print(f"{a.dst}: batch symbolic, session_config {rt['session_config']}")


if __name__ == "__main__":
    main()
