"""INT8 calibration table for TensorRT implicit quantisation through the ORT TensorRT EP.

ORT's MinMax calibrator records every tensor's range on pool calibration images (never development images; see
calib_set.py). The table is calibration.flatbuffers, the format the EP reads through
``trt_int8_calibration_table_name``, holding the symmetric range max(|min|, |max|) per tensor. It is written here
rather than by ORT's write_calibration_table, which stores max(min, max) and so clips negative-dominant tensors.
The candidate is the source graph unchanged plus that table and a runtime.json with ``trt_int8_enable``; TensorRT
then picks INT8 or fp16 per layer by timing.

    calib_table.py <src_dir> <dst_dir> [--n 64] [--table-dir <dir the runtime.json names>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from candidate import TRT_FP16, git_sha
from common import pool_images, preprocess


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
    ap.add_argument("--table-dir", type=Path, default=None,
                    help="absolute directory the deployed runtime.json points at (default: dst)")
    a = ap.parse_args()
    from onnxruntime.quantization import CalibrationMethod, create_calibrator
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    X = preprocess(pool_images("calib", a.n))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(X)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"pixel_values": x[None]}

    a.dst.mkdir(parents=True, exist_ok=False)
    work = a.dst / "calib_work"
    work.mkdir()
    cal = create_calibrator(a.src / "model.onnx", [], augmented_model_path=str(work / "augmented.onnx"),
                            calibrate_method=CalibrationMethod.MinMax, use_external_data_format=True)
    cal.set_execution_providers(["CPUExecutionProvider"])
    t0 = time.perf_counter()
    cal.collect_data(Reader())
    print(f"{a.n} images in {time.perf_counter() - t0:.0f}s", flush=True)
    ranges = {k: max(abs(float(lo)), abs(float(hi)))
              for k, (lo, hi) in ((k, v.range_value) for k, v in cal.compute_data().data.items())}
    write_table(ranges, a.dst / "calibration.flatbuffers")
    (a.dst / "calibration.json").write_text(json.dumps(ranges, indent=1))
    shutil.rmtree(work)
    for f in ("model.onnx", "model.onnx.data"):
        subprocess.run(["cp", "--reflink=auto", str(a.src / f), str(a.dst / f)], check=True)
    shutil.copy(a.src / "labels.json", a.dst / "labels.json")
    pre = json.loads((a.src / "preprocess.json").read_text())
    pre["built_from"] = (f"{a.src} (implicit INT8: ORT MinMax table on {a.n} pool calibration images; "
                         f"tools/quant/calib_table.py @ {git_sha()})")
    (a.dst / "preprocess.json").write_text(json.dumps(pre, indent=1))
    table = (a.table_dir or a.dst.resolve()) / "calibration.flatbuffers"
    rt = {"provider": "TensorrtExecutionProvider",
          "options": {**TRT_FP16, "trt_int8_enable": True, "trt_int8_calibration_table_name": str(table),
                      "trt_int8_use_native_calibration_table": False}}
    (a.dst / "runtime.json").write_text(json.dumps(rt, indent=1))
    print(f"wrote {a.dst}, {len(ranges)} tensor ranges")


if __name__ == "__main__":
    main()
