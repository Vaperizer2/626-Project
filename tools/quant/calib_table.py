"""INT8 calibration table for TensorRT implicit quantisation through the ORT TensorRT EP.

Every float tensor of the source graph gets a ReduceMax(Abs(x)) output; the max over pool calibration images
(never development images; see calib_set.py) is that tensor's symmetric range. ORT's own MinMax calibrator was
OOM-killed on this graph (12 GB + 11 GB swap). The table is calibration.flatbuffers, the format the EP reads
through ``trt_int8_calibration_table_name``. The candidate is the source graph unchanged plus that table and a
runtime.json with ``trt_int8_enable``; TensorRT then picks INT8 or fp16 per layer by timing.

    calib_table.py <src_dir> <dst_dir> [--n 64] [--cuda] [--table-dir <dir the runtime.json names>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from candidate import TRT_FP16, git_sha
from common import pool_images, preprocess


def amax_graph(src: Path, dst: Path) -> list[str]:
    m = onnx.shape_inference.infer_shapes(onnx.load(src, load_external_data=False))
    g = m.graph
    floats = {v.name for v in [*g.value_info, *g.input, *g.output]
              if v.type.tensor_type.elem_type == TensorProto.FLOAT}
    tensors = [t for t in dict.fromkeys([g.input[0].name, *(o for n in g.node for o in n.output)]) if t in floats]
    for i, t in enumerate(tensors):
        g.node.extend([helper.make_node("Abs", [t], [f"ct.abs{i}"]),
                       helper.make_node("ReduceMax", [f"ct.abs{i}"], [f"amax/{t}"], keepdims=0)])
        g.output.append(helper.make_tensor_value_info(f"amax/{t}", TensorProto.FLOAT, []))
    del g.value_info[:]
    for loc in {e.value for t in g.initializer if t.data_location == TensorProto.EXTERNAL for e in t.external_data
                if e.key == "location"}:
        subprocess.run(["cp", "--reflink=auto", str(src.parent / loc), str(dst.parent / loc)], check=True)
    onnx.save(m, dst)
    return tensors


def write_table(ranges: dict[str, float], path: Path) -> None:
    import flatbuffers
    from onnxruntime.quantization.CalTableFlatBuffers import KeyValue, TrtTable

    b = flatbuffers.Builder(1024)
    kvs = []
    for k in sorted(ranges):
        key, val = b.CreateString(k), b.CreateString(repr(ranges[k]))
        KeyValue.KeyValueStart(b)
        KeyValue.KeyValueAddKey(b, key)
        KeyValue.KeyValueAddValue(b, val)
        kvs.append(KeyValue.KeyValueEnd(b))
    TrtTable.TrtTableStartDictVector(b, len(kvs))
    for kv in kvs:
        b.PrependUOffsetTRelative(kv)
    d = b.EndVector()
    TrtTable.TrtTableStart(b)
    TrtTable.TrtTableAddDict(b, d)
    b.Finish(TrtTable.TrtTableEnd(b))
    path.write_bytes(b.Output())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--cuda", action="store_true", help="CUDA EP (TF32 off) instead of CPU")
    ap.add_argument("--table-dir", type=Path, default=None,
                    help="absolute directory the deployed runtime.json points at (default: dst)")
    a = ap.parse_args()
    import onnxruntime as ort

    a.dst.mkdir(parents=True, exist_ok=False)
    work = a.dst / "calib_work"
    work.mkdir()
    tensors = amax_graph(a.src / "model.onnx", work / "amax.onnx")
    X = preprocess(pool_images("calib", a.n))
    ort.preload_dlls()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    ep = [("CUDAExecutionProvider", {"use_tf32": 0})] if a.cuda else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(work / "amax.onnx"), so, providers=ep)
    names = [f"amax/{t}" for t in tensors]
    t0 = time.perf_counter()
    amax = np.max([sess.run(names, {"pixel_values": X[i:i + 1]}) for i in range(len(X))], axis=0)
    print(f"{len(tensors)} tensors, {len(X)} images in {time.perf_counter() - t0:.0f}s", flush=True)
    del sess
    shutil.rmtree(work)
    ranges = {t: float(v) for t, v in zip(tensors, amax)}
    write_table(ranges, a.dst / "calibration.flatbuffers")
    (a.dst / "calibration.json").write_text(json.dumps(ranges, indent=1))
    shutil.copy(a.src / "labels.json", a.dst / "labels.json")
    pre = json.loads((a.src / "preprocess.json").read_text())
    pre["built_from"] = (f"{a.src} (implicit INT8: max |x| table over {a.n} pool calibration images; "
                         f"tools/quant/calib_table.py @ {git_sha()})")
    (a.dst / "preprocess.json").write_text(json.dumps(pre, indent=1))
    table = (a.table_dir or a.dst.resolve()) / "calibration.flatbuffers"
    rt = {"provider": "TensorrtExecutionProvider",
          "options": {**TRT_FP16, "trt_int8_enable": True, "trt_int8_calibration_table_name": str(table),
                      "trt_int8_use_native_calibration_table": False}}
    (a.dst / "runtime.json").write_text(json.dumps(rt, indent=1))
    print(f"wrote {a.dst} without the graph: copy the source model.onnx(+data) in next to the table")


if __name__ == "__main__":
    main()
