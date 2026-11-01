"""One deployable model file for a speed-track trunk candidate: the trunk plus the deployed head refitted on it.

    tools/compose/py tools/compose/export_fast.py <candidate_dir> <out_dir> [--n-parity 314] [--n-exact 32]

Head: ``compclass.final.fit_head`` (the code behind exports/final, checked by check_fit.py) on the candidate's own
2,512 development embeddings, the ones the frozen harness's latest eval of this exact candidate content extracted on
its execution provider, with the deployed text embeddings W from exports/final/head.npz.

Graph: the candidate graph with the head appended as ``compclass.final.FusedClassifier`` computes it; input renamed
``input``, outputs ``logits`` and ``probs``, as exports/final/model.onnx. Weights stay in model.onnx.data.

Parity (legion, under the GPU lock):
  runtime  the export on the candidate's runtime.json provider against the trunk alone on the same provider plus the
           head in numpy, on every 8th development image. Both are separate engine builds, so this includes the head
           running inside the engine at reduced precision and build-to-build tactic differences.
  exact    the same comparison on the CUDA EP with TF32 off (fp32 math, QDQ simulated), on a few images: the
           appended head must equal the numpy head.
  ep       nodes per provider in the runtime session; none may land on the CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/speed"))

from evaluate import dataset, dev_rows  # noqa: E402
from harness import DEPLOYED, INPUT, OUTPUT, ep_assignment, gpu_lock, latest, load_candidate  # noqa: E402

WORK = Path(os.environ.get("EXPORT_WORK", "/home/vaper/Work/Prototypes/626-artifacts/compose/work"))
TOL = {"runtime": {"max_abs_prob": 0.05, "argmax_agree": 0.99}, "exact": {"max_abs_prob": 1e-5, "argmax_agree": 1.0}}
HEAD_KEYS = ("mu", "sd", "coef", "intercept", "W", "t_zs", "w_zs", "temperature")


def harness_embeddings(c) -> tuple[np.ndarray, dict]:
    rec = latest("eval", c)
    if rec is None:
        raise SystemExit(f"{c.dir}: no harness eval of this content (sha {c.sha[:12]}); run tools/speed eval first")
    if rec["n_errors"]:
        raise SystemExit(f"{c.id}: harness eval had {rec['n_errors']} non-finite embeddings")
    return np.load(rec["embeddings"]), rec


def append_head(m: onnx.ModelProto, head: dict) -> onnx.ModelProto:
    """FusedClassifier.forward on the trunk output: log_softmax(standardise(f) @ coef.T + b) + w_zs *
    log_softmax(t_zs * normalise(f) @ W.T), divided by the temperature, then softmax."""
    g = m.graph
    w = {"mu": head["mu"], "sd": head["sd"], "coefT": head["coef"].T, "intercept": head["intercept"],
         "WT": head["W"].T, "t_zs": head["t_zs"], "w_zs": head["w_zs"], "temperature": head["temperature"]}
    g.initializer.extend(numpy_helper.from_array(np.ascontiguousarray(v, np.float32), f"head.{k}")
                         for k, v in w.items())
    n = helper.make_node
    g.node.extend([
        n("Sub", [OUTPUT, "head.mu"], ["head.c"]), n("Div", ["head.c", "head.sd"], ["head.z"]),
        n("MatMul", ["head.z", "head.coefT"], ["head.zl"]), n("Add", ["head.zl", "head.intercept"], ["head.pl"]),
        n("LogSoftmax", ["head.pl"], ["head.lp"], axis=1),
        n("LpNormalization", [OUTPUT], ["head.fn"], axis=1, p=2),
        n("MatMul", ["head.fn", "head.WT"], ["head.cos"]), n("Mul", ["head.cos", "head.t_zs"], ["head.zz"]),
        n("LogSoftmax", ["head.zz"], ["head.zs"], axis=1), n("Mul", ["head.zs", "head.w_zs"], ["head.wz"]),
        n("Add", ["head.lp", "head.wz"], ["head.fused"]),
        n("Div", ["head.fused", "head.temperature"], ["logits"]), n("Softmax", ["logits"], ["probs"], axis=1)])
    for node in g.node:
        node.input[:] = ["input" if i == INPUT else i for i in node.input]
    g.input[0].name = "input"
    batch = g.output[0].type.tensor_type.shape.dim[0]
    classes = head["coef"].shape[0]
    del g.output[:]
    g.output.extend(helper.make_tensor_value_info(o, TensorProto.FLOAT, [batch.dim_param or batch.dim_value, classes])
                    for o in ("logits", "probs"))
    return m


def python_probs(F: np.ndarray, head: dict) -> np.ndarray:
    from compclass.final import fused_scores, log_softmax

    return np.exp(log_softmax(fused_scores(F, head) / head["temperature"]))


def runtime_for_app(c) -> dict:
    """The candidate's provider and options; cache paths stay relative (the app resolves them, see report)."""
    return {"provider": c.runtime["provider"], "options": c.runtime["options"],
            "graph_optimization_level": c.runtime.get("graph_optimization_level", "ORT_ENABLE_ALL"),
            "provider_order": [c.runtime["provider"], "CUDAExecutionProvider", "CPUExecutionProvider"],
            "static_batch": c.fixed_batch}


def write_export(c, head: dict, report: dict, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    m = onnx.load(c.dir / "model.onnx", load_external_data=False)
    locs = {e.value for t in m.graph.initializer if t.data_location == TensorProto.EXTERNAL for e in t.external_data
            if e.key == "location"}
    for loc in locs:
        subprocess.run(["cp", "--reflink=auto", str(c.dir / loc), str(out / loc)], check=True)
    onnx.save(append_head(m, head), out / "model.onnx")
    onnx.checker.check_model(str(out / "model.onnx"))
    np.savez(out / "head.npz", **head)
    shutil.copy(DEPLOYED / "labels.json", out / "labels.json")
    final = json.loads((DEPLOYED / "preprocess.json").read_text())
    size, crop = c.preprocess["input_size"], c.preprocess["crop_pct"]
    pre = final | {"input_size": size, "crop_pct": crop, "resize_short_side": round(size / crop),
                   "abstain_threshold": float(head["abstain_threshold"]),
                   "temperature": float(head["temperature"]), "trunk": c.id,
                   "built_from": f"{c.preprocess.get('built_from')}; head refitted by tools/compose/export_fast.py"}
    (out / "preprocess.json").write_text(json.dumps(pre, indent=1))
    (out / "runtime.json").write_text(json.dumps(runtime_for_app(c), indent=1))
    (out / "fit_report.json").write_text(json.dumps(report, indent=1, default=float))


def session(model: Path, provider: str, options: dict, cache: Path | None, runtime: dict):
    import onnxruntime as ort

    ort.preload_dlls()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel,
                                          runtime.get("graph_optimization_level", "ORT_ENABLE_ALL"))
    so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    for k, v in runtime.get("session_config", {}).items():
        so.add_session_config_entry(k, str(v))
    opts = dict(options)
    if provider == "TensorrtExecutionProvider":
        cache.mkdir(parents=True, exist_ok=True)
        for k in ("trt_engine_cache_path", "trt_timing_cache_path"):
            if k in opts:
                opts[k] = str(cache)
        providers = [(provider, opts), "CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = [(provider, opts), "CPUExecutionProvider"]
    t0 = time.perf_counter()
    s = ort.InferenceSession(str(model), so, providers=providers)
    if s.get_providers()[0] != provider:
        raise SystemExit(f"{provider} did not load (session providers {s.get_providers()}); run through "
                         "tools/compose/py so the TensorRT libraries are on the library path")
    print(f"  session {model.parent.name}/{model.name} on {s.get_providers()[0]} in {time.perf_counter() - t0:.0f}s",
          flush=True)
    return s


def run(s, name: str, out: str, X: np.ndarray) -> np.ndarray:
    return np.concatenate([s.run([out], {name: X[i:i + 1]})[0] for i in range(len(X))])


def agreement(p: np.ndarray, q: np.ndarray) -> dict:
    d = np.abs(p - q)
    return {"n": len(p), "max_abs_prob": float(d.max()), "mean_abs_prob": float(d.mean()),
            "max_abs_top1_prob": float(np.abs(p.max(1) - q.max(1)).max()),
            "argmax_agree": float((p.argmax(1) == q.argmax(1)).mean())}


def parity(c, out: Path, head: dict, F_harness: np.ndarray, n: int, n_exact: int) -> dict:
    m, rows = dev_rows()
    idx = np.arange(0, len(rows), max(1, len(rows) // n))[:n]
    ds = dataset(c, m, [rows[i] for i in idx])
    X = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    work = WORK / f"export_{c.id}"
    rep = {"images": f"dev rows {idx[0]}:{idx[-1] + 1}:{idx[1] - idx[0]} ({len(idx)})", "tolerance": TOL}
    ep, opts = c.runtime["provider"], c.runtime["options"]
    with gpu_lock():
        s = session(out / "model.onnx", ep, opts, work / "trt_cache_export", c.runtime)
        rep["ep_assignment"] = {k: {"nodes": v["nodes"], "subgraphs": v["subgraphs"]}
                                for k, v in ep_assignment(s).items()}
        P = run(s, "input", "probs", X)
        L = run(s, "input", "logits", X)
        del s
        s = session(c.dir / "model.onnx", ep, opts, work / "trt_cache_trunk", c.runtime)
        F = run(s, INPUT, OUTPUT, X)
        del s
        cuda = {"use_tf32": 0}
        Xe = X[:n_exact]
        s = session(out / "model.onnx", "CUDAExecutionProvider", cuda, None, c.runtime)
        Pe = run(s, "input", "probs", Xe)
        del s
        s = session(c.dir / "model.onnx", "CUDAExecutionProvider", cuda, None, c.runtime)
        Fe = run(s, INPUT, OUTPUT, Xe)
        del s
    rep["runtime"] = agreement(P, python_probs(F, head))
    rep["runtime_vs_harness_embeddings"] = agreement(P, python_probs(F_harness[idx], head))
    rep["runtime_finite"] = bool(np.isfinite(P).all() and np.isfinite(L).all())
    rep["runtime_probs_sum_max_err"] = float(np.abs(P.sum(1) - 1).max())
    rep["exact"] = agreement(Pe, python_probs(Fe, head))
    cpu = rep["ep_assignment"].get("CPUExecutionProvider", {}).get("nodes", 0)
    rep["nodes_off_stated_provider"] = sum(v["nodes"] for k, v in rep["ep_assignment"].items() if k != ep)
    rep["checks"] = {
        "runtime": rep["runtime"]["max_abs_prob"] <= TOL["runtime"]["max_abs_prob"]
        and rep["runtime"]["argmax_agree"] >= TOL["runtime"]["argmax_agree"] and rep["runtime_finite"],
        "exact": rep["exact"]["max_abs_prob"] <= TOL["exact"]["max_abs_prob"]
        and rep["exact"]["argmax_agree"] >= TOL["exact"]["argmax_agree"],
        "no_cpu_nodes": cpu == 0}
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n-parity", type=int, default=314)
    ap.add_argument("--n-exact", type=int, default=32)
    a = ap.parse_args()
    from compclass.final import fit_head

    c = load_candidate(a.candidate)
    if not c.fusion:
        raise SystemExit(f"{c.id}: the deployed head fuses the zero-shot text head; the trunk must declare it")
    F, ev = harness_embeddings(c)
    m, rows = dev_rows()
    if not np.array_equal(np.load(ev["flags"])["ids"], [r["id"] for r in rows]):
        raise SystemExit("harness embeddings are not in development-row order")
    W = np.load(DEPLOYED / "head.npz")["W"]
    print(f"[{c.id}] fitting the deployed head on {len(F)} harness embeddings", flush=True)
    head, report = fit_head(F, m, rows, W)
    report |= {"trunk_candidate": c.id, "trunk_candidate_sha": c.sha, "trunk_built_from": c.preprocess.get("built_from"),
               "embeddings": ev["embeddings"], "harness_eval": {k: ev[k] for k in (
                   "time_utc", "git_sha", "cv_acc_mean", "cv_acc_sd", "oof_acc", "temperature", "abstain_threshold")},
               "text_embeddings": str(DEPLOYED / "head.npz"), "exporter": f"tools/compose/export_fast.py @ {git_sha()}"}
    write_export(c, head, report, a.out)
    print(json.dumps({k: v for k, v in report.items() if k != "abstain_table"}, indent=1, default=float), flush=True)
    report["parity"] = parity(c, a.out, {k: head[k] for k in HEAD_KEYS}, F, a.n_parity, a.n_exact)
    (a.out / "fit_report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report["parity"], indent=1))
    if not all(report["parity"]["checks"].values()):
        raise SystemExit(f"parity failed: {report['parity']['checks']}")


def git_sha() -> str:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain", "--", "tools", "src") else "")


if __name__ == "__main__":
    main()
