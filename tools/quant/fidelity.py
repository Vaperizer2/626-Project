"""Label-free fidelity proxy: a candidate's embeddings on held-aside pool images against strict-fp32 baseline ones.

Runs the candidate's own graph on its runtime.json provider (TensorRT caches under artifacts/quant/work/<id>/),
batch 1, on the pool ``sens`` split (never development images). Reports cosine to the strict-fp32 reference and
how often the deployed fused head's argmax agrees with the reference's. A screen before the harness, not a result.

    fidelity.py <candidate_dir> [--n 256]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import ART, BASELINE, gpu_lock, pool_images, preprocess

HEAD = Path("/home/vaper/Work/Prototypes/626-Project/exports/final/head.npz")


def session(cdir: Path, strict_fp32: bool = False):
    import onnxruntime as ort

    ort.preload_dlls()
    rt = json.loads((cdir / "runtime.json").read_text()) if (cdir / "runtime.json").exists() else {}
    ep, opts = rt.get("provider", "CUDAExecutionProvider"), dict(rt.get("options", {}))
    if strict_fp32:
        ep, opts = "CUDAExecutionProvider", {"use_tf32": 0}
    if ep == "TensorrtExecutionProvider":
        cache = ART / "work" / cdir.name / "trt_cache"
        cache.mkdir(parents=True, exist_ok=True)
        for k in ("trt_engine_cache_path", "trt_timing_cache_path"):
            if k in opts:
                opts[k] = str(cache)
        s = "pixel_values:1x3x384x384"
        for k in ("min", "opt", "max"):
            opts.setdefault(f"trt_profile_{k}_shapes", s)
        providers = [(ep, opts), "CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = [(ep, opts), "CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(cdir / "model.onnx"), so, providers=providers)
    print(f"[{cdir.name}] session {sess.get_providers()[0]} in {time.perf_counter() - t0:.0f}s", flush=True)
    return sess


def embed(sess, X: np.ndarray) -> np.ndarray:
    return np.concatenate([sess.run(["embedding"], {"pixel_values": X[i:i + 1]})[0] for i in range(len(X))])


def reference(X: np.ndarray, n: int) -> np.ndarray:
    path = ART / "work" / f"ref_sens_strictfp32_{n}.npy"
    if not path.exists():
        with gpu_lock():
            np.save(path, embed(session(BASELINE, strict_fp32=True), X))
    return np.load(path)


def report(F: np.ndarray, R: np.ndarray) -> dict:
    from compclass.final import fused_scores

    head = dict(np.load(HEAD))
    cos = (F * R).sum(1) / (np.linalg.norm(F, axis=1) * np.linalg.norm(R, axis=1))
    agree = (fused_scores(F, head).argmax(1) == fused_scores(R, head).argmax(1)).mean()
    return {"n": len(F), "cos_median": float(np.median(cos)), "cos_mean": float(cos.mean()),
            "cos_p05": float(np.percentile(cos, 5)), "cos_min": float(cos.min()), "head_argmax_agree": float(agree),
            "n_nonfinite": int((~np.isfinite(F).all(1)).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    X = preprocess(pool_images("sens", a.n))
    R = reference(X, a.n)
    with gpu_lock():
        F = embed(session(a.candidate.resolve()), X)
    rep = {"candidate": a.candidate.name, **report(F, R)}
    out = ART / "work" / a.candidate.name / "fidelity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
