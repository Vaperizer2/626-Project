"""TensorRT EP against CUDA EP fp32, per output row, at several batch sizes: the feedback loop for both TRT bugs.

    probe.py cut <src_dir> <name> (--block K | --tensor T)
        Writes <src_dir>/<name>.onnx: pixel_values -> the residual stream after block K (0-based; -1 = patch
        embedding plus position embedding), or -> tensor T. Weights stay in <src_dir>'s external data file.
    probe.py tail <src_dir> <name> --start S --shape D1,D2 [--tensor T]
        As cut, but tensor S becomes the graph input (renamed pixel_values): only the nodes between S and T.
    probe.py outputs <model.onnx> <dst.onnx> T1,T2,...
        A copy with the named tensors as extra graph outputs; run compares every output.
    probe.py level1 <model.onnx> <out.onnx> [--disable NAME,...]
        The graph onnxruntime's level-1 transformers hand an execution provider (CPU session at ORT_ENABLE_BASIC;
        level 1 runs before partitioning, so TensorRT EP parses this graph), weights in <out.onnx>.data. Prints
        the op counts that changed.
    probe.py acts <model.onnx> <inputs.npy> <out.npy>
        The model's output for each input on CUDA EP fp32, as inputs for a tail graph.
    probe.py run <model.onnx> <inputs.npy> [--batches 1,2,8] [--rows N] [--trt JSON] [--level ORT_ENABLE_ALL]
                 [--profile dynamic|static] [--disable NAME,...] [--config JSON]
        Reference: CUDA EP with TF32 off at the same --level, one image per run. Candidate: TensorRT EP (fp32
        unless --trt sets options), no engine cache, profile 1..max(batches) (dynamic) or one static engine per
        batch size; --disable goes to disabled_optimizers, --config to session config entries. Prints one JSON
        line per batch size and graph output: cosine of every TRT output row to its reference row and, when the
        batches start with 1, the lowest cosine of a batch-b row to the same session's batch-1 row.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def block_output(m: onnx.ModelProto, k: int) -> str:
    """The residual stream entering block k+1's norm1 (or the final norm after the last block)."""
    lns = [n for n in m.graph.node if n.op_type == "LayerNormalization"]

    def gamma(n):
        return n.input[1].replace("__", ".")

    nxt = [n for n in lns if gamma(n).endswith(f"blocks.{k + 1}.norm1.weight")]
    if not nxt:
        nxt = [n for n in lns if gamma(n).removeprefix("trunk.") == "norm.weight"]
    return nxt[0].input[0]


def cut(src: Path, name: str, block: int | None, tensor: str | None, start: str | None = None,
        shape: list | None = None) -> Path:
    m = onnx.load(src / "model.onnx", load_external_data=False)
    g = m.graph
    target = tensor or (block_output(m, block) if block is not None else g.output[0].name)
    producer = {o: n for n in g.node for o in n.output}
    need, stack = set(), [target]
    while stack:
        t = stack.pop()
        n = producer.get(t)
        if n is not None and id(n) not in need and t != start:
            need.add(id(n))
            stack.extend(n.input)
    nodes = [n for n in g.node if id(n) in need]
    used = {i for n in nodes for i in n.input}
    keep = [t for t in g.initializer if t.name in used]
    del g.node[:]
    g.node.extend(nodes)
    del g.initializer[:]
    g.initializer.extend(keep)
    del g.value_info[:]
    del g.output[:]
    g.output.append(helper.make_tensor_value_info(target, TensorProto.FLOAT, None))
    if start:
        for n in g.node:
            n.input[:] = ["pixel_values" if i == start else i for i in n.input]
        del g.input[:]
        g.input.append(helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["batch", *shape]))
    out = src / f"{name}.onnx"
    onnx.save(m, out)
    print(f"{out}: {len(nodes)} nodes -> {target}")
    return out


def session(model: Path, providers: list, level: str, disable: list[str] = (), config: dict | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    for k, v in (config or {}).items():
        so.add_session_config_entry(k, v)
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
    return ort.InferenceSession(str(model), so, providers=providers, disabled_optimizers=disable or None)


def rows_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.reshape(len(a), -1).astype(np.float64), b.reshape(len(b), -1).astype(np.float64)
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def run(model: Path, X: np.ndarray, batches: list[int], trt: dict, level: str, profile: str, disable: list[str],
        config: dict, rows: int = 0) -> None:
    import onnxruntime as ort

    ort.preload_dlls()
    inp = "pixel_values"
    outs = [o.name for o in onnx.load(model, load_external_data=False).graph.output]
    big = max(max(batches), rows)
    X = np.concatenate([X] * -(-big // len(X)))[:big]
    s = session(model, [("CUDAExecutionProvider", {"use_tf32": 0})], level)
    ref = [np.concatenate(r) for r in zip(*(s.run(outs, {inp: X[i:i + 1]}) for i in range(big)))]
    del s
    shape = "x".join(map(str, X.shape[1:]))

    def trt_session(lo: int, hi: int):
        opts = {**trt, "trt_profile_min_shapes": f"{inp}:{lo}x{shape}",
                "trt_profile_opt_shapes": f"{inp}:{lo}x{shape}", "trt_profile_max_shapes": f"{inp}:{hi}x{shape}"}
        t0 = time.perf_counter()
        s = session(model, [("TensorrtExecutionProvider", opts), "CUDAExecutionProvider"], level, disable, config)
        assert s.get_providers()[0] == "TensorrtExecutionProvider", s.get_providers()
        return s, time.perf_counter() - t0

    s, build = trt_session(1, big) if profile == "dynamic" else (None, 0.0)
    one = None
    for b in batches:
        if profile == "static":
            del s
            s, build = trt_session(b, b)
        ys = [np.concatenate(r) for r in zip(*(s.run(outs, {inp: X[i:i + b]})
                                                for i in range(0, big - big % b or b, b)))]
        one = ys if b == 1 else one
        for j, (name, y, r) in enumerate(zip(outs, ys, ref)):
            c = rows_cos(y, r[:len(y)])
            rec = {"batch": b, "output": name, "profile": profile, "level": level, "disable": disable,
                   "config": config, "trt": {k: v for k, v in trt.items() if "cache" not in k},
                   "build_s": round(build, 1), "rows": len(y), "cos_min": round(float(c.min()), 5),
                   "cos_median": round(float(np.median(c)), 5), "cos_rows": [round(float(v), 4) for v in c[:8]]}
            if one is not None and b != 1:
                rec["cos_vs_trt_batch1_min"] = round(float(rows_cos(y, one[j][:len(y)]).min()), 5)
            print(json.dumps(rec), flush=True)


def add_outputs(model: Path, dst: Path, tensors: list[str]) -> None:
    m = onnx.load(model, load_external_data=False)
    m.graph.output.extend(helper.make_tensor_value_info(t, TensorProto.FLOAT, None) for t in tensors)
    onnx.save(m, dst)


def level1(model: Path, out: Path, disable: list[str]) -> None:
    import collections

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(out)
    so.add_session_config_entry("session.optimized_model_external_initializers_file_name", out.name + ".data")
    ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"], disabled_optimizers=disable or None)
    a, b = (collections.Counter(n.op_type for n in onnx.load(f, load_external_data=False).graph.node)
            for f in (model, out))
    print(json.dumps({"model": str(model), "level1": str(out), "disabled": disable,
                      "changed_ops": {k: [a[k], b[k]] for k in sorted(a | b) if a[k] != b[k]}}))


def acts(model: Path, X: np.ndarray, out: Path) -> None:
    import onnxruntime as ort

    ort.preload_dlls()
    s = session(model, [("CUDAExecutionProvider", {"use_tf32": 0})], "ORT_DISABLE_ALL")
    name = s.get_outputs()[0].name
    np.save(out, np.concatenate([s.run([name], {"pixel_values": X[i:i + 1]})[0] for i in range(len(X))]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut")
    c.add_argument("src", type=Path)
    c.add_argument("name")
    c.add_argument("--block", type=int)
    c.add_argument("--tensor")
    t = sub.add_parser("tail")
    t.add_argument("src", type=Path)
    t.add_argument("name")
    t.add_argument("--start", required=True)
    t.add_argument("--tensor")
    t.add_argument("--shape", required=True, help="input dims after the batch, e.g. 729,1152")
    o = sub.add_parser("outputs")
    o.add_argument("model", type=Path)
    o.add_argument("dst", type=Path)
    o.add_argument("tensors")
    lv = sub.add_parser("level1")
    lv.add_argument("model", type=Path)
    lv.add_argument("out", type=Path)
    lv.add_argument("--disable", default="")
    ac = sub.add_parser("acts")
    ac.add_argument("model", type=Path)
    ac.add_argument("inputs", type=Path)
    ac.add_argument("out", type=Path)
    r = sub.add_parser("run")
    r.add_argument("model", type=Path)
    r.add_argument("inputs", type=Path)
    r.add_argument("--batches", default="1,2,8")
    r.add_argument("--trt", default="{}")
    r.add_argument("--level", default="ORT_ENABLE_ALL")
    r.add_argument("--profile", choices=("dynamic", "static"), default="dynamic")
    r.add_argument("--disable", default="")
    r.add_argument("--config", default="{}", help="session config entries for the TRT session, as JSON")
    r.add_argument("--rows", type=int, default=0, help="images to run (default: the largest batch)")
    a = ap.parse_args()
    if a.cmd == "cut":
        cut(a.src, a.name, a.block, a.tensor)
    elif a.cmd == "tail":
        cut(a.src, a.name, None, a.tensor, a.start, [int(d) for d in a.shape.split(",")])
    elif a.cmd == "outputs":
        add_outputs(a.model, a.dst, a.tensors.split(","))
    elif a.cmd == "level1":
        level1(a.model, a.out, [d for d in a.disable.split(",") if d])
    elif a.cmd == "acts":
        acts(a.model, np.load(a.inputs), a.out)
    else:
        run(a.model, np.load(a.inputs), [int(b) for b in a.batches.split(",")], json.loads(a.trt), a.level,
            a.profile, [d for d in a.disable.split(",") if d], json.loads(a.config), a.rows)


if __name__ == "__main__":
    main()
