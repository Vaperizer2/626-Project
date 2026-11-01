"""How much does a TensorRT engine's speed vary between builds of one graph? Builds the harness's own graph for a
candidate (its with_head.onnx, same provider options and session config) N times into fresh engine caches through
onnxruntime, then profiles each engine with tools/quant/trtprobe.py. Diagnostic only; harness numbers count.

    tools/quant/py tools/speed4/buildvar.py <candidate_dir> <out_dir> [--n 3] [--harness-path] [--nvml]

--opt KEY=VALUE adds a provider option (e.g. trt_builder_optimization_level=5) to every build.
--harness-path builds through the harness's own make_session (a copy of the candidate under a throw-away hash, so
its work dir and engine cache are fresh); --nvml first initialises the NVML sampler, as ``speed time`` does.

Per build: execute_v2 median, profiled layer sum, and the count of stand-alone residual-add + LayerNorm kernels
(``__myl_CastAddAddCastMean...``), the de-fusion that made trtbugs_int8_tome308's harness engine 5% slower.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "speed"))

import dataclasses  # noqa: E402

from harness import deployed_graph, gpu_lock, load_candidate, make_session  # noqa: E402


def build(c, graph: Path, cache: Path) -> Path:
    import onnxruntime as ort

    ort.preload_dlls()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel,
                                          c.runtime.get("graph_optimization_level", "ORT_ENABLE_ALL"))
    for k, v in c.runtime.get("session_config", {}).items():
        so.add_session_config_entry(k, str(v))
    opts = dict(c.runtime["options"])
    opts["trt_engine_cache_path"] = opts["trt_timing_cache_path"] = str(cache)
    cache.mkdir(parents=True)
    s = ort.InferenceSession(str(graph), so, providers=[("TensorrtExecutionProvider", opts), "CUDAExecutionProvider"])
    assert s.get_providers()[0] == "TensorrtExecutionProvider"
    del s
    return next(cache.glob("*.engine"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--harness-path", action="store_true")
    ap.add_argument("--nvml", action="store_true")
    ap.add_argument("--opt", action="append", default=[], help="extra provider option KEY=VALUE for the builds")
    a = ap.parse_args()
    c = load_candidate(a.candidate)
    if a.opt:
        extra = {k: (int(v) if v.isdigit() else v) for k, _, v in (o.partition("=") for o in a.opt)}
        c = dataclasses.replace(c, runtime={**c.runtime, "options": {**c.runtime["options"], **extra}})
    graph = deployed_graph(c, 20)
    rows = []
    tag = ("h" if a.harness_path else "b") + ("n" if a.nvml else "") + "".join(
        f"_{o.split('=')[0].removeprefix('trt_')}{o.split('=')[1]}" for o in a.opt)
    for i in range(a.n):
        d = a.out / f"build{tag}{i}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        with gpu_lock():
            if a.nvml:
                from timing import GpuSampler

                GpuSampler().static()
            if a.harness_path:
                cc = dataclasses.replace(c, sha=f"buildvar{tag}{i}{c.sha}")
                shutil.rmtree(cc.work, ignore_errors=True)
                s = make_session(cc, deployed_graph(cc, 20))
                del s
                eng = next((cc.work / "trt_cache").glob("*.engine"))
            else:
                eng = build(c, graph, d / "cache")
        shutil.move(str(eng), d / "engine.plan")
        shutil.rmtree(d / "cache", ignore_errors=True)
        if a.harness_path:
            shutil.rmtree(cc.work)
        r = subprocess.run([sys.executable, str(HERE.parent / "quant" / "trtprobe.py"), "/dev/null", str(d), "--iters", "200"],
                           capture_output=True, text=True, cwd=HERE.parent / "quant")
        line = next(ln for ln in r.stdout.splitlines() if "execute_v2" in ln)
        prof = json.loads((d / "profile.json").read_text())
        rec = {"build": f"{tag}{i}", "execute_v2_median_ms": float(re.search(r"median ([\d.]+)", line).group(1)),
               "layer_sum_ms": round(sum(prof.values()), 3),
               "separate_add_ln": sum(1 for k in prof if k.startswith("__myl_CastAddAddCastMean"))}
        (d / "engine.plan").unlink()
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    with open(a.out / "buildvar.jsonl", "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
