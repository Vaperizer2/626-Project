"""Batch throughput of a compose-layout export (input ``input`` -> ``logits``, ``probs``) on its TensorRT runtime,
with every batch's outputs checked against the same images run one at a time.

    tools/compose/py tools/trtbugs/batchbench.py <export_dir> <out.json> [--batches 1,4,8,16,28] [--opt 1]
                                                 [--cache DIR] [--rounds 5] [--runs 40] [--check-only]

<export_dir> holds a batch-symbolic model.onnx (batchfix.py --dynamic-batch) and runtime.json. One session serves
every batch size: TensorRT profile 1..max(batches) at opt --opt, the runtime's options and session_config.
Inputs are the first max(batches) development images, preprocessed as the export's preprocess.json says.
Timing follows tools/speed/timing.py: inputs and outputs bound on the GPU, run_with_iobinding plus
synchronize_outputs, at least 20 runs and 10 s of warmup per batch size, then rounds x runs timed runs
(--check-only skips the timing). Run on legion under flock /tmp/626-gpu-bench.lock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/speed"))

from evaluate import dev_rows, preprocessed  # noqa: E402
from harness import Candidate, ep_assignment  # noqa: E402
from timing import GpuSampler  # noqa: E402

WARMUP_RUNS, WARMUP_S = 20, 10


def make_session(d: Path, rt: dict, lo: int, opt: int, hi: int, size: int, cache: Path):
    import onnxruntime as ort

    ort.preload_dlls()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, rt.get("graph_optimization_level",
                                                                            "ORT_ENABLE_ALL"))
    so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    for k, v in rt.get("session_config", {}).items():
        so.add_session_config_entry(k, str(v))
    opts = dict(rt["options"])
    cache.mkdir(parents=True, exist_ok=True)
    for k in ("trt_engine_cache_path", "trt_timing_cache_path"):
        if k in opts:
            opts[k] = str(cache)
    shape = f"3x{size}x{size}"
    opts |= {"trt_profile_min_shapes": f"input:{lo}x{shape}", "trt_profile_opt_shapes": f"input:{opt}x{shape}",
             "trt_profile_max_shapes": f"input:{hi}x{shape}"}
    t0 = time.perf_counter()
    s = ort.InferenceSession(str(d / "model.onnx"), so, providers=[
        ("TensorrtExecutionProvider", opts), "CUDAExecutionProvider", "CPUExecutionProvider"])
    if s.get_providers()[0] != "TensorrtExecutionProvider":
        raise SystemExit(f"TensorRT did not load: {s.get_providers()}")
    ep = {k: v["nodes"] for k, v in ep_assignment(s).items()}
    if set(ep) != {"TensorrtExecutionProvider"}:
        raise SystemExit(f"nodes off TensorRT: {ep}")
    return s, time.perf_counter() - t0, ep


class Bound:
    def __init__(self, s, x: np.ndarray, n_classes: int):
        import onnxruntime as ort

        self.s, self.io = s, s.io_binding()
        self.io.bind_ortvalue_input("input", ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(x), "cuda", 0))
        self.out = {k: ort.OrtValue.ortvalue_from_shape_and_type([len(x), n_classes], np.float32, "cuda", 0)
                    for k in ("logits", "probs")}
        for k, v in self.out.items():
            self.io.bind_ortvalue_output(k, v)

    def once(self) -> float:
        t = time.perf_counter_ns()
        self.s.run_with_iobinding(self.io)
        self.io.synchronize_outputs()
        return (time.perf_counter_ns() - t) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--batches", default="1,4,8,16,28")
    ap.add_argument("--opt", type=int, default=1)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    batches = [int(b) for b in a.batches.split(",")]
    big = max(batches)
    rt = json.loads((a.export / "runtime.json").read_text())
    pre = json.loads((a.export / "preprocess.json").read_text())
    labels = json.loads((a.export / "labels.json").read_text())
    n_classes = len(labels)
    m, rows = dev_rows()
    c = Candidate(a.export, a.export.name, "", pre, rt, 0, None)
    X = preprocessed(c, m, rows[:big])
    cache = a.cache or a.export / f"trt_cache_opt{a.opt}_max{big}"
    rep = {"export": str(a.export), "runtime": rt, "profile": {"min": 1, "opt": a.opt, "max": big},
           "images": f"first {big} development rows", "warmup": {"runs": WARMUP_RUNS, "s": WARMUP_S},
           "rounds": a.rounds, "runs_per_round": a.runs, "load_before": os.getloadavg()}
    s, build, ep = make_session(a.export, rt, 1, a.opt, big, pre["input_size"], cache)
    rep |= {"session_s": round(build, 1), "ep_nodes": ep}
    single = {k: np.concatenate(v) for k, v in zip(("logits", "probs"), zip(
        *(s.run(["logits", "probs"], {"input": X[i:i + 1]}) for i in range(big))))}
    gpu = GpuSampler()
    rep["gpu_static"] = gpu.static()
    res = []
    with gpu:
        for b in batches:
            outs = {"logits": [], "probs": []}
            for i in range(0, big - big % b, b):
                for k, v in zip(("logits", "probs"), s.run(["logits", "probs"], {"input": X[i:i + b]})):
                    outs[k].append(v)
            L, P = (np.concatenate(outs[k]) for k in ("logits", "probs"))
            n = len(P)
            Lr, Pr = single["logits"][:n], single["probs"][:n]
            cos = (L * Lr).sum(1) / np.linalg.norm(L, axis=1) / np.linalg.norm(Lr, axis=1)
            check = {"rows": n, "argmax_agree": float((P.argmax(1) == Pr.argmax(1)).mean()),
                     "max_abs_prob": float(np.abs(P - Pr).max()), "min_logit_cos": float(cos.min()),
                     "finite": bool(np.isfinite(P).all() and np.isfinite(L).all())}
            if a.check_only:
                res.append({"batch": b, "check": check})
                print(json.dumps(res[-1]), flush=True)
                continue
            r = Bound(s, X[:b], n_classes)
            t_end, warm = time.monotonic() + WARMUP_S, 0
            while warm < WARMUP_RUNS or time.monotonic() < t_end:
                r.once()
                warm += 1
            lat = np.array([[r.once() for _ in range(a.runs)] for _ in range(a.rounds)])
            med = float(np.median(lat))
            row = {"batch": b, "median_ms": med, "p10_ms": float(np.percentile(lat, 10)),
                   "p90_ms": float(np.percentile(lat, 90)), "round_medians_ms": np.median(lat, 1).round(3).tolist(),
                   "img_per_s": b * 1000 / med, "warmup_runs": warm, "check": check}
            res.append(row)
            print(json.dumps(row), flush=True)
    rep |= {"results": res, "gpu_during": gpu.summary(), "load_after": os.getloadavg()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    ok = all(r["check"]["argmax_agree"] == 1 and r["check"]["min_logit_cos"] > 0.999 and r["check"]["finite"]
             for r in res)
    print(f"{'all batches agree with batch 1' if ok else 'BATCH OUTPUTS DISAGREE WITH BATCH 1'}; wrote {a.out}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
