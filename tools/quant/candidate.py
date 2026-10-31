"""Write quant-track candidates: a graph (baseline or QDQ) plus preprocess.json and runtime.json.

    candidate.py static1 <dst> '<runtime.json as JSON>'   # baseline graph with batch fixed to 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import onnx

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


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("static1")
    p.add_argument("dst", type=Path)
    p.add_argument("runtime_json")
    a = ap.parse_args()
    static1(a.dst, json.loads(a.runtime_json))


if __name__ == "__main__":
    main()
