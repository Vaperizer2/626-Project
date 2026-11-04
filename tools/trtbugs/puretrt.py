"""The same TRT-vs-CUDA row check as probe.py run, but with the engine built by the TensorRT Python API straight
from the ONNX file (no onnxruntime in the build), plus the engine's layer list. Tells a TensorRT fault from an
onnxruntime TensorRT EP fault.

    puretrt.py <model.onnx> <inputs.npy> <out_dir> [--batch 2] [--fp16] [--int8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorrt_bindings as trt


def build(path: Path, shape: tuple, batch: int, a) -> bytes:
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, log)
    if not parser.parse_from_file(str(path)):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if a.fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if a.int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
    prof = builder.create_optimization_profile()
    prof.set_shape(net.get_input(0).name, (1, *shape), (1, *shape), (batch, *shape))
    cfg.add_optimization_profile(prof)
    return bytes(builder.build_serialized_network(net, cfg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("inputs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args()
    ort.preload_dlls()
    a.out.mkdir(parents=True, exist_ok=True)
    X = np.load(a.inputs)[:a.batch]
    s = ort.InferenceSession(str(a.model), providers=[("CUDAExecutionProvider", {"use_tf32": 0})])
    oname = s.get_outputs()[0].name
    ref = np.concatenate([s.run([oname], {"pixel_values": X[i:i + 1]})[0] for i in range(len(X))])
    del s
    eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(build(a.model, X.shape[1:], a.batch, a))
    ctx = eng.create_execution_context()
    res = {}
    for b in (1, a.batch):
        outs = []
        for i in range(0, len(X), b):
            x = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(X[i:i + b]), "cuda", 0)
            ctx.set_input_shape("pixel_values", x.shape())
            y = ort.OrtValue.ortvalue_from_shape_and_type(list(ctx.get_tensor_shape(oname)), np.float32, "cuda", 0)
            ctx.set_tensor_address("pixel_values", x.data_ptr())
            ctx.set_tensor_address(oname, y.data_ptr())
            ctx.execute_async_v3(0)
            outs.append(y.numpy())
        Y = np.concatenate(outs).reshape(len(X), -1).astype(np.float64)
        R = ref.reshape(len(X), -1).astype(np.float64)
        c = (Y * R).sum(1) / np.linalg.norm(Y, axis=1) / np.linalg.norm(R, axis=1)
        res[f"batch{b}"] = [round(float(v), 4) for v in c]
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    (a.out / "layers.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({"model": str(a.model), "cos_trt_vs_cuda": res, "layers": len(info["Layers"])}))


if __name__ == "__main__":
    main()
