"""Build a TensorRT engine from an ONNX graph with the Python API and report where its batch-1 time goes.

Diagnostic only: harness numbers are the ones that count. Mirrors the ORT TensorRT EP defaults the candidates use
(builder optimisation level 3, 2 GB workspace) and adds detailed profiling verbosity, so the engine inspector lists
every kernel (Myelin regions included) and an IProfiler gives per-layer times.

    trtprobe.py <model.onnx> <out_dir> [--fp16] [--int8] [--opt-level 3] [--iters 50]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorrt_bindings as trt

from common import gpu_lock


def build(onnx_path: Path, a) -> bytes:
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, log)
    if not parser.parse_from_file(str(onnx_path)):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    cfg.builder_optimization_level = a.opt_level
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if a.fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if a.int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
    inp = net.get_input(0)
    if -1 in tuple(inp.shape):
        prof = builder.create_optimization_profile()
        s = tuple(1 if d == -1 else d for d in inp.shape)
        prof.set_shape(inp.name, s, s, s)
        cfg.add_optimization_profile(prof)
    t0 = time.perf_counter()
    eng = builder.build_serialized_network(net, cfg)
    print(f"built in {time.perf_counter() - t0:.0f}s", flush=True)
    return bytes(eng)


class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.ms: dict[str, list[float]] = defaultdict(list)

    def report_layer_time(self, name, ms):
        self.ms[name].append(ms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    ort.preload_dlls()
    a.out.mkdir(parents=True, exist_ok=True)
    with gpu_lock():
        eng_path = a.out / "engine.plan"
        if not eng_path.exists():
            eng_path.write_bytes(build(a.onnx, a))
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        eng = rt.deserialize_cuda_engine(eng_path.read_bytes())
        ctx = eng.create_execution_context()
        names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        bufs = []
        for n in names:
            shape = tuple(1 if d == -1 else d for d in eng.get_tensor_shape(n))
            if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                ctx.set_input_shape(n, shape)
            dt = trt.nptype(eng.get_tensor_dtype(n))
            v = ort.OrtValue.ortvalue_from_numpy(np.random.default_rng(0).normal(size=shape).astype(dt), "cuda", 0)
            bufs.append(v)
        ptrs = [v.data_ptr() for v in bufs]
        for _ in range(20):
            ctx.execute_v2(ptrs)
        lat = []
        for _ in range(a.iters):
            t = time.perf_counter_ns()
            ctx.execute_v2(ptrs)
            lat.append((time.perf_counter_ns() - t) / 1e6)
        prof = Profiler()
        ctx.profiler = prof
        for _ in range(10):
            ctx.execute_v2(ptrs)
        info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    (a.out / "layers.json").write_text(json.dumps(info, indent=1))
    per = {k: float(np.median(v)) for k, v in prof.ms.items()}
    (a.out / "profile.json").write_text(json.dumps(per, indent=1))
    total = sum(per.values())
    print(f"execute_v2 median {np.median(lat):.2f} ms (p10 {np.percentile(lat, 10):.2f}, p90 "
          f"{np.percentile(lat, 90):.2f}); profiled layer sum {total:.2f} ms over {len(per)} layers")
    for k, v in sorted(per.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {v:7.3f} ms {100 * v / total:5.1f}%  {k[:150]}")


if __name__ == "__main__":
    main()
