"""Build a candidate's TensorRT engine with verbose logs and report what TensorRT says about sparse weights.

    flock /tmp/626-gpu-bench.lock speed-venv-python trt_probe.py CANDIDATE [--log OUT.log]

Evidence for the mechanism only (did TensorRT see 2:4 weights, and did it pick sparse kernels?). Timing numbers
come from the harness. Uses its own engine cache under artifacts/prune/probe/, never the harness work dir.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROBE = Path("/home/vaper/Work/Prototypes/626-artifacts/prune/probe")


def child(cand: Path, cache: Path) -> None:
    import numpy as np
    import onnxruntime as ort

    ort.preload_dlls()
    ort.set_default_logger_severity(0)
    so = ort.SessionOptions()
    so.log_severity_level = 0
    opts = dict(json.loads((cand / "runtime.json").read_text())["options"])
    opts.update(trt_engine_cache_path=str(cache), trt_timing_cache_path=str(cache), trt_detailed_build_log=True,
                trt_profile_min_shapes="pixel_values:1x3x384x384", trt_profile_opt_shapes="pixel_values:1x3x384x384",
                trt_profile_max_shapes="pixel_values:8x3x384x384")
    sess = ort.InferenceSession(str(cand / "model.onnx"), so,
                                providers=[("TensorrtExecutionProvider", opts), "CUDAExecutionProvider"])
    y = sess.run(None, {"pixel_values": np.zeros((1, 3, 384, 384), np.float32)})[0]
    print(f"PROBE providers={sess.get_providers()} out={y.shape} finite={bool(np.isfinite(y).all())}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    cand = a.candidate.resolve()
    cache = PROBE / cand.name / "trt_cache"
    if a.child:
        child(cand, cache)
        return
    cache.mkdir(parents=True, exist_ok=True)
    log = PROBE / cand.name / "build.log"
    with open(log, "w") as f:
        subprocess.run([sys.executable, __file__, str(cand), "--child"], stdout=f, stderr=subprocess.STDOUT,
                       env=os.environ, check=False)
    pat = re.compile(r"sparse|sparsity|PROBE|error", re.I)
    lines = [ln.rstrip() for ln in open(log, errors="replace") if pat.search(ln)]
    print(f"{log}: {sum(1 for _ in open(log, errors='replace'))} lines; {len(lines)} match sparse|sparsity|error")
    for ln in lines[:80]:
        print(ln[:400])


if __name__ == "__main__":
    main()
