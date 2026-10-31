"""Diagnostic: does a graph's TensorRT engine compute what the graph means? Not a result; harness numbers count.

Runs a few preprocessed images through <graph_dir>/model.onnx on the TensorRT EP (options from --runtime or the
dir's runtime.json, caches under compose/work/trtcheck/) and on the CUDA EP with TF32 off, and prints the cosine
between the two embeddings per image, plus the cosine of each to a reference graph's CUDA fp32 embedding.

    trtcheck.py <inputs.npy> <ref_graph_dir> <graph_dir> [--runtime JSON] [--tag NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quant"))

from common import gpu_lock  # noqa: E402

WORK = Path("/home/vaper/Work/Prototypes/626-artifacts/compose/work/trtcheck")


def embed(model: Path, providers: list, X: np.ndarray, level: str = "ORT_ENABLE_ALL") -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
    s = ort.InferenceSession(str(model), so, providers=providers)
    out = np.concatenate([s.run(["embedding"], {"pixel_values": X[i:i + 1]})[0] for i in range(len(X))])
    del s
    return out


def cos(a: np.ndarray, b: np.ndarray) -> list[float]:
    return [round(float(x), 4) for x in (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("graph", type=Path)
    ap.add_argument("--runtime", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import onnxruntime as ort

    ort.preload_dlls()
    X = np.load(a.inputs)
    rt = json.loads(a.runtime) if a.runtime else json.loads((a.graph / "runtime.json").read_text())
    opts = dict(rt["options"])
    cache = WORK / (a.tag or a.graph.name)
    cache.mkdir(parents=True, exist_ok=True)
    for k in ("trt_engine_cache_path", "trt_timing_cache_path"):
        if k in opts:
            opts[k] = str(cache)
    s = X.shape[-1]
    for k in ("min", "opt", "max"):
        opts.setdefault(f"trt_profile_{k}_shapes", f"pixel_values:1x3x{s}x{s}")
    cuda = [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
    with gpu_lock():
        ref = embed(a.ref / "model.onnx", cuda, X)
        fp32 = embed(a.graph / "model.onnx", cuda, X)
        trt = embed(a.graph / "model.onnx", [("TensorrtExecutionProvider", opts), "CUDAExecutionProvider"], X,
                    rt.get("graph_optimization_level", "ORT_ENABLE_ALL"))
    rep = {"graph": str(a.graph), "tag": a.tag, "trt_options": {k: v for k, v in rt["options"].items() if "cache" not in k},
           "graph_optimization_level": rt.get("graph_optimization_level", "ORT_ENABLE_ALL"),
           "trt_vs_cuda_fp32": cos(trt, fp32), "cuda_fp32_vs_ref": cos(fp32, ref), "trt_vs_ref": cos(trt, ref)}
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
