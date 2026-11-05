"""Efficiency study, step 4: batch-1 latency of exported ONNX graphs under onnxruntime's CUDA execution provider.

    <venv with onnxruntime-gpu>/python tools/views/ort_latency.py --graph exports/candidates/mobileclip2_s3/model_fp16.onnx ...

Needs only onnxruntime-gpu and numpy (no project imports), so it runs from a throw-away environment and leaves
uv.lock alone; point ORT_DLL_DIR at a folder with the CUDA 12 / cuDNN 9 DLLs if no nvidia pip packages are present. Method as in the team lead's tools/speed/timing.py: input and outputs bound on the GPU (IO binding),
warm-up until 20 runs and 10 s have passed, then 5 rounds x 60 runs interleaved across the graphs; median and p90
of the 300 per-graph runs. Session options: ORT_ENABLE_ALL, CUDA EP defaults.
"""
import argparse, json, os, statistics, time
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROUNDS, RUNS, WARMUP, WARMUP_S = 5, 60, 20, 10.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", action="append", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    try:  # CUDA / cuDNN DLLs: from the nvidia pip packages, or from ORT_DLL_DIR (e.g. the project venv's torch/lib)
        ort.preload_dlls(directory=os.environ.get("ORT_DLL_DIR"))
    except Exception as e:
        print(f"preload_dlls: {e}")
    assert "CUDAExecutionProvider" in ort.get_available_providers(), ort.get_available_providers()
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess, binds = {}, {}
    for g in a.graph:
        s = ort.InferenceSession(g, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert s.get_providers()[0] == "CUDAExecutionProvider", s.get_providers()
        inp = s.get_inputs()[0]; shape = [1 if not isinstance(d, int) else d for d in inp.shape]
        x = ort.OrtValue.ortvalue_from_numpy(np.random.randn(*shape).astype(np.float32), "cuda", 0)
        b = s.io_binding(); b.bind_ortvalue_input(inp.name, x)
        for o in s.get_outputs():
            b.bind_output(o.name, "cuda", 0)
        sess[g], binds[g] = s, b
        print(f"{g}: input {shape}, providers {s.get_providers()[0]}", flush=True)

    def once(g):
        t0 = time.perf_counter_ns(); sess[g].run_with_iobinding(binds[g]); binds[g].synchronize_outputs()
        return (time.perf_counter_ns() - t0) / 1e6

    for g in sess:
        t0 = time.perf_counter(); n = 0
        while n < WARMUP or time.perf_counter() - t0 < WARMUP_S:
            once(g); n += 1
    lat = {g: [] for g in sess}
    for r in range(ROUNDS):
        for g in sess:
            lat[g] += [once(g) for _ in range(RUNS)]
        print(f"round {r+1}/{ROUNDS} " + "  ".join(f"{Path(g).parent.name}/{Path(g).name} {statistics.median(lat[g]):.2f} ms" for g in sess), flush=True)
    rows = []
    print(f"\nonnxruntime {ort.__version__} CUDA EP, batch 1, IO binding, {ROUNDS}x{RUNS} runs interleaved\n")
    print("| Graph | file MB | median ms | p90 ms | min ms |")
    print("|---|---:|---:|---:|---:|")
    for g in sess:
        v = {"graph": g, "mb": Path(g).stat().st_size / 2**20, "median_ms": float(np.median(lat[g])),
             "p90_ms": float(np.percentile(lat[g], 90)), "min_ms": float(min(lat[g]))}
        rows.append(v)
        print(f"| {Path(g).parent.name}/{Path(g).name} | {v['mb']:.0f} | {v['median_ms']:.2f} | {v['p90_ms']:.2f} | {v['min_ms']:.2f} |")
    if a.json:
        json.dump({"onnxruntime": ort.__version__, "rows": rows}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
