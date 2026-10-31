"""Shared pieces of the speed harness: paths, the candidate contract, ORT sessions, the GPU lock, results."""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

MAIN = Path("/home/vaper/Work/Prototypes/626-Project")
MANIFEST = MAIN / "data/v2/manifest.json"
FINGERPRINT = "431315e87d2612cb"
DEPLOYED = MAIN / "exports/final"
ART = Path("/home/vaper/Work/Prototypes/626-artifacts/harness")
RESULTS = ART / "results.jsonl"
BASELINE = ART / "candidates" / "baseline_fp32"
LOCK = "/tmp/626-gpu-bench.lock"
INPUT, OUTPUT, PROBS = "pixel_values", "embedding", "probs"
SIGLIP_DIM = 1152
MAX_BATCH = 8
PREPROCESS_KEYS = ("input_size", "crop_pct", "interpolation", "mean", "std", "zero_shot_fusion")


@dataclass(frozen=True)
class Candidate:
    dir: Path
    id: str
    sha: str
    preprocess: dict
    runtime: dict
    dim: int
    fixed_batch: int | None  # None when the batch dimension is symbolic

    @property
    def spec(self) -> SimpleNamespace:
        """The fields ``compclass.data.build_transforms`` reads from a model's PreprocessSpec."""
        return SimpleNamespace(**{k: self.preprocess[k] for k in PREPROCESS_KEYS[:5]})

    @property
    def fusion(self) -> bool:
        return bool(self.preprocess["zero_shot_fusion"])

    @property
    def work(self) -> Path:
        return ART / "work" / self.id / self.sha[:12]


def _external_files(model: onnx.ModelProto) -> list[str]:
    locs = {e.value for t in model.graph.initializer if t.data_location == TensorProto.EXTERNAL
            for e in t.external_data if e.key == "location"}
    return sorted(locs)


def load_candidate(path: str | Path) -> Candidate:
    """Parse and validate a candidate directory against the contract in README.md."""
    d = Path(path).resolve()
    if not (d / "model.onnx").exists() or not (d / "preprocess.json").exists():
        raise SystemExit(f"{d}: a candidate needs model.onnx and preprocess.json")
    pre = json.loads((d / "preprocess.json").read_text())
    missing = [k for k in PREPROCESS_KEYS if k not in pre]
    if missing:
        raise SystemExit(f"{d}/preprocess.json: missing {missing}")
    rt_path = d / "runtime.json"
    rt = json.loads(rt_path.read_text()) if rt_path.exists() else {}
    rt.setdefault("provider", "CUDAExecutionProvider")
    rt.setdefault("options", {})
    m = onnx.load(d / "model.onnx", load_external_data=False)
    ins = [i for i in m.graph.input if i.name not in {t.name for t in m.graph.initializer}]
    if [i.name for i in ins] != [INPUT] or [o.name for o in m.graph.output] != [OUTPUT]:
        raise SystemExit(f"{d}: graph must map input {INPUT!r} to output {OUTPUT!r}; has "
                         f"{[i.name for i in ins]} -> {[o.name for o in m.graph.output]}")
    for v in (ins[0], m.graph.output[0]):
        if v.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise SystemExit(f"{d}: {v.name} must be float32 (convert with keep_io_types=True)")
    idim = ins[0].type.tensor_type.shape.dim
    odim = m.graph.output[0].type.tensor_type.shape.dim
    if len(idim) != 4 or [x.dim_value for x in idim[1:]] != [3, pre["input_size"], pre["input_size"]]:
        raise SystemExit(f"{d}: input shape {[x.dim_value or x.dim_param for x in idim]} does not match "
                         f"[N,3,{pre['input_size']},{pre['input_size']}]")
    if idim[0].dim_value not in (0, 1):
        raise SystemExit(f"{d}: batch dimension must be symbolic (or a static 1)")
    dim = odim[1].dim_value
    if not dim:
        raise SystemExit(f"{d}: embedding dimension must be static")
    if pre["zero_shot_fusion"] and dim != SIGLIP_DIM:
        raise SystemExit(f"{d}: zero_shot_fusion needs the {SIGLIP_DIM}-d SigLIP So400m embedding, got {dim}")
    h = hashlib.sha256()
    for name in ["model.onnx", *_external_files(m), "preprocess.json", "runtime.json"]:
        if (d / name).exists():
            h.update(name.encode())
            with open(d / name, "rb") as f:
                h.update(hashlib.file_digest(f, "sha256").digest())
    return Candidate(d, d.name, h.hexdigest(), pre, rt, dim, idim[0].dim_value or None)


def head_graph(model: onnx.ModelProto, n_classes: int, fusion: bool) -> onnx.ModelProto:
    """Append the deployed head (probe + optional zero-shot fusion + temperature + softmax) to a backbone graph.

    Mirrors ``compclass.final.FusedClassifier.forward``. Only opset-stable ops are used (LpNormalization for the
    L2 norm) so the head attaches to any candidate opset. Outputs: ``embedding`` and ``probs``. The weights are
    seeded random values of the deployed shapes: timing does not depend on them and eval reads only ``embedding``.
    """
    g = model.graph
    d = g.output[0].type.tensor_type.shape.dim[1].dim_value
    rng = np.random.default_rng(0)
    weights = {"mu": rng.normal(size=d), "sd": rng.uniform(0.5, 1.5, d), "coefT": rng.normal(size=(d, n_classes)),
               "intercept": rng.normal(size=n_classes), "temperature": 0.9}
    if fusion:
        weights |= {"WT": rng.normal(size=(d, n_classes)), "t_zs": 50.0, "w_zs": 0.25}
    for k, v in weights.items():
        g.initializer.append(numpy_helper.from_array(np.asarray(v, np.float32), f"head.{k}"))
    n = helper.make_node
    nodes = [n("Sub", [OUTPUT, "head.mu"], ["head.c"]), n("Div", ["head.c", "head.sd"], ["head.z"]),
             n("MatMul", ["head.z", "head.coefT"], ["head.zl"]), n("Add", ["head.zl", "head.intercept"], ["head.pl"]),
             n("LogSoftmax", ["head.pl"], ["head.lp"], axis=1)]
    fused = "head.lp"
    if fusion:
        nodes += [n("LpNormalization", [OUTPUT], ["head.fn"], axis=1, p=2),
                  n("MatMul", ["head.fn", "head.WT"], ["head.cos"]), n("Mul", ["head.cos", "head.t_zs"], ["head.zz"]),
                  n("LogSoftmax", ["head.zz"], ["head.zs"], axis=1), n("Mul", ["head.zs", "head.w_zs"], ["head.wz"]),
                  n("Add", ["head.lp", "head.wz"], ["head.fused"])]
        fused = "head.fused"
    nodes += [n("Div", [fused, "head.temperature"], ["head.logits"]), n("Softmax", ["head.logits"], [PROBS], axis=1)]
    g.node.extend(nodes)
    batch = g.output[0].type.tensor_type.shape.dim[0]
    g.output.append(helper.make_tensor_value_info(PROBS, TensorProto.FLOAT,
                                                  [batch.dim_param or batch.dim_value, n_classes]))
    return model


def deployed_graph(c: Candidate, n_classes: int) -> Path:
    """The graph the harness runs for a candidate: its backbone + the deployed head, outputs (embedding, probs).

    Written once per candidate content hash under ``work/``. onnxruntime refuses external data outside the model's
    directory (symlinks included), so weight files are reflink-copied (instant on btrfs, a plain copy elsewhere).
    eval and time both run this file, so a TensorRT engine is built once and eval measures the timed graph.
    """
    out = c.work / "with_head.onnx"
    if out.exists():
        return out
    c.work.mkdir(parents=True, exist_ok=True)
    m = onnx.load(c.dir / "model.onnx", load_external_data=False)
    for name in _external_files(m):
        subprocess.run(["cp", "--reflink=auto", str(c.dir / name), str(c.work / name)], check=True)
    head_graph(m, n_classes, c.fusion)
    tmp = out.with_suffix(".tmp")
    onnx.save(m, tmp)
    tmp.rename(out)
    return out


def make_session(c: Candidate, path: Path):
    """onnxruntime session on the candidate's execution provider; fails instead of silently falling back."""
    import onnxruntime as ort

    ort.preload_dlls()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    level = c.runtime.get("graph_optimization_level", "ORT_ENABLE_ALL")
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level)
    so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    for k, v in c.runtime.get("session_config", {}).items():
        so.add_session_config_entry(k, str(v))
    ep, opts = c.runtime["provider"], dict(c.runtime["options"])
    if ep == "TensorrtExecutionProvider":
        for k in ("trt_engine_cache_path", "trt_timing_cache_path"):
            if k in opts:
                opts[k] = str(c.work / opts[k])
                Path(opts[k]).mkdir(parents=True, exist_ok=True)
        s = c.preprocess["input_size"]
        if c.fixed_batch is None:
            opts.setdefault("trt_profile_min_shapes", f"{INPUT}:1x3x{s}x{s}")
            opts.setdefault("trt_profile_opt_shapes", f"{INPUT}:1x3x{s}x{s}")
            opts.setdefault("trt_profile_max_shapes", f"{INPUT}:{MAX_BATCH}x3x{s}x{s}")
        providers = [(ep, opts), ("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    elif ep == "CUDAExecutionProvider":
        providers = [(ep, opts), "CPUExecutionProvider"]
    else:
        raise SystemExit(f"unsupported provider {ep!r}: use CUDAExecutionProvider or TensorrtExecutionProvider")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(path), so, providers=providers)
    if sess.get_providers()[0] != ep:
        raise SystemExit(f"{ep} did not load (session providers: {sess.get_providers()})")
    nodes = {ep: e["nodes"] for ep, e in ep_assignment(sess).items()}
    print(f"[{c.id}] session on {sess.get_providers()} in {time.perf_counter() - t0:.1f}s; nodes per EP {nodes}",
          flush=True)
    return sess


def ep_assignment(sess) -> dict:
    """Subgraphs and nodes each provider received, so a silent partial fallback is visible in every record."""
    out: dict = {}
    for sg in sess.get_provider_graph_assignment_info():
        ops = [n.op_type for n in sg.get_nodes()]
        e = out.setdefault(sg.ep_name, {"subgraphs": 0, "nodes": 0, "ops": {}})
        e["subgraphs"] += 1
        e["nodes"] += len(ops)
        for op in ops:
            e["ops"][op] = e["ops"].get(op, 0) + 1
    return out


def provider_info(sess) -> dict:
    import onnxruntime as ort

    opts = sess.get_provider_options()
    return {"providers": sess.get_providers(), "provider_options": {k: dict(v) for k, v in opts.items()},
            "ep_assignment": ep_assignment(sess), "ort_version": ort.__version__}


@contextmanager
def gpu_lock():
    """Hold /tmp/626-gpu-bench.lock (flock), the legion rule for GPU timing.

    When the harness already runs under ``flock /tmp/626-gpu-bench.lock ...`` it inherits the locked descriptor;
    locking a fresh descriptor would then wait on our own parent forever, so reuse the inherited one.
    """
    def target(fd: str) -> str:
        try:
            return os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            return ""

    if any(target(fd) == LOCK for fd in os.listdir("/proc/self/fd")):
        yield
        return
    with open(LOCK, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"waiting for {LOCK} ...", flush=True)
            t0 = time.monotonic()
            fcntl.flock(f, fcntl.LOCK_EX)
            print(f"got {LOCK} after {time.monotonic() - t0:.0f}s", flush=True)
        yield


def git_state() -> dict:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()

    return {"git_sha": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", "tools/speed", "src/compclass"))}


def append_result(rec: dict) -> dict:
    rec = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **git_state(), **rec}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec, default=float) + "\n")
    return rec


def latest(kind: str, c: Candidate) -> dict | None:
    """Most recent ``kind`` record for this candidate's exact content hash."""
    if not RESULTS.exists():
        return None
    hits = [r for r in map(json.loads, RESULTS.read_text().splitlines())
            if r.get("kind") == kind and r.get("candidate") == c.id and r.get("candidate_sha") == c.sha]
    return hits[-1] if hits else None
