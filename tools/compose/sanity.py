"""Embedding agreement between the stages of one INT8 build, on pool ``sens`` images (never development images).

CUDA EP with TF32 off, so the float graphs run in fp32 and QuantizeLinear/DequantizeLinear simulate INT8. Folding
SmoothQuant is exact algebra, so the smoothed graph must match the static-1 source to fp32 rounding on the median
image (a folding bug moves every image; on ToMe graphs that rounding can flip a near-tie merge on a few images); the
QDQ graph's cosine to the source is its INT8 error before TensorRT picks kernels. A screen, not a result.

    sanity.py <static1_dir> <smooth_dir> <qdq_dir> [--n 16]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quant"))

from common import pool_images, preprocess, spec_of  # noqa: E402


def embed(model_dir: Path, X: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(str(model_dir / "model.onnx"), so,
                             providers=[("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"])
    return np.concatenate([s.run(["embedding"], {"pixel_values": X[i:i + 1]})[0] for i in range(len(X))])


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {"cos_min": float(cos.min()), "cos_median": float(np.median(cos)),
            "max_abs_diff": float(np.abs(a - b).max()), "max_abs": float(np.abs(a).max()),
            "finite": bool(np.isfinite(b).all())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("static1", type=Path)
    ap.add_argument("smooth", type=Path)
    ap.add_argument("qdq", type=Path)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    import onnxruntime as ort

    ort.preload_dlls()
    X = preprocess(pool_images("sens", a.n), spec_of(a.static1))
    ref = embed(a.static1, X)
    rep = {"n": len(X), "input_size": X.shape[-1], "smooth_vs_static1": agreement(ref, embed(a.smooth, X)),
           "qdq_vs_static1": agreement(ref, embed(a.qdq, X))}
    print(json.dumps(rep, indent=1))
    assert rep["smooth_vs_static1"]["cos_median"] > 0.99999, "SmoothQuant folding changed the fp32 function"


if __name__ == "__main__":
    main()
