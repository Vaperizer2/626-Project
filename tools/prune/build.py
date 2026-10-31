"""Build pruned So400m candidates from the deployed graph (harness `baseline_fp32`), TensorRT fp16 runtime.

    build.py unstructured OUT --sparsity 0.5 [--targets mlp,attn]   # per-matrix lowest-|w| zeroing
    build.py semi24       OUT [--targets mlp] [--no-sparse-flag]     # 2:4 lowest-|w| along the input dim
    build.py structured   OUT --keep 3072 [--importance magnitude|ACT.npz]   # drop MLP neurons, shrink matrices
    build.py load         OUT --base CANDIDATE --state STATE.npz     # write recovered weights into CANDIDATE's graph

Targets are the Linear layers of all 27 blocks: `mlp` = fc1, fc2; `attn` = qkv, proj. The attention-pool head is
never pruned. Each OUT gets prune.json with the achieved sparsity per matrix.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so400m  # noqa: E402

TARGETS = {"mlp": ("mlp.fc1", "mlp.fc2"), "attn": ("attn.qkv", "attn.proj")}


def target_linears(targets: str) -> list[str]:
    names = [n for t in targets.split(",") for n in TARGETS[t]]
    return [so400m.block_linear(i, n) for i in range(so400m.DEPTH) for n in names]


def magnitude_keep(w: np.ndarray, sparsity: float) -> np.ndarray:
    k = int(round(w.size * sparsity))
    thr = np.partition(np.abs(w).ravel(), k)[k]
    return np.abs(w) >= thr


def keep_24(w: np.ndarray) -> np.ndarray:
    """Keep the 2 largest |w| of every 4 consecutive weights along the input (reduction) dim; w is [in, out]."""
    k, n = w.shape
    g = np.abs(w).reshape(k // 4, 4, n)
    order = np.argsort(g, axis=1, kind="stable")
    keep = np.zeros_like(g, dtype=bool)
    np.put_along_axis(keep, order[:, 2:, :], True, axis=1)
    return keep.reshape(k, n)


def zero_out(W: so400m.Weights, lins: list[str], keep_fn) -> dict:
    stats = {}
    for lin in lins:
        w = W.w(lin)
        keep = keep_fn(w)
        W.set_w(lin, np.where(keep, w, 0.0))
        stats[lin] = round(1 - keep.mean(), 4)
    return stats


def neuron_importance(W: so400m.Weights, how: str) -> np.ndarray:
    """[DEPTH, hidden] score per MLP neuron: ||fc2 row|| times ||fc1 column|| or the mean |GELU output|."""
    out_norm = np.stack([np.linalg.norm(W.w(f"blocks.{i}.mlp.fc2"), axis=1) for i in range(so400m.DEPTH)])
    if how == "magnitude":
        return out_norm * np.stack([np.linalg.norm(W.w(f"blocks.{i}.mlp.fc1"), axis=0)
                                    for i in range(so400m.DEPTH)])
    return out_norm * np.load(how)["mean_abs_act"]


def shrink_mlp(W: so400m.Weights, keep: int, score: np.ndarray) -> dict:
    kept = {}
    for i in range(so400m.DEPTH):
        idx = np.sort(np.argsort(-score[i], kind="stable")[:keep])
        fc1, fc2 = f"blocks.{i}.mlp.fc1", f"blocks.{i}.mlp.fc2"
        W.set_w(fc1, W.w(fc1)[:, idx])
        W.set_b(fc1, W.b(fc1)[idx])
        W.set_w(fc2, W.w(fc2)[idx, :])
        kept[fc1] = idx.tolist()
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("unstructured", "semi24", "structured", "load"):
        p = sub.add_parser(name)
        p.add_argument("out", type=Path)
    sub.choices["unstructured"].add_argument("--sparsity", type=float, required=True)
    sub.choices["unstructured"].add_argument("--targets", default="mlp,attn")
    sub.choices["semi24"].add_argument("--targets", default="mlp")
    sub.choices["semi24"].add_argument("--no-sparse-flag", action="store_true")
    sub.choices["structured"].add_argument("--keep", type=int, required=True)
    sub.choices["structured"].add_argument("--importance", default="magnitude")
    sub.choices["load"].add_argument("--base", type=Path, required=True)
    sub.choices["load"].add_argument("--state", type=Path, required=True)
    sub.choices["load"].add_argument("--prefixes", default="blocks.")
    a = ap.parse_args()

    base = a.base if a.cmd == "load" else so400m.BASE
    W = so400m.Weights(so400m.load(base))
    if a.cmd == "unstructured":
        stats = zero_out(W, target_linears(a.targets), lambda w: magnitude_keep(w, a.sparsity))
        note, rt = f"unstructured lowest-|w| {a.sparsity:.0%} per matrix on {a.targets}", so400m.trt_runtime()
    elif a.cmd == "semi24":
        stats = zero_out(W, target_linears(a.targets), keep_24)
        note = f"2:4 lowest-|w| along input dim on {a.targets}"
        rt = so400m.trt_runtime(sparse=not a.no_sparse_flag)
    elif a.cmd == "structured":
        assert a.keep % 64 == 0, "keep a tensor-core friendly hidden width"
        stats = shrink_mlp(W, a.keep, neuron_importance(W, a.importance))
        note = f"MLP hidden 4304 -> {a.keep} per block by {Path(a.importance).name} importance"
        rt = so400m.trt_runtime()
    else:
        sd = dict(np.load(a.state))
        stats = {"loaded": W.load_torch_state(sd, tuple(a.prefixes.split(",")))}
        base_note = json.loads((a.base / "preprocess.json").read_text())["built_from"]
        note = f"recovered weights {a.state} into {a.base.name} [{base_note}]"
        rt = json.loads((a.base / "runtime.json").read_text())
    so400m.save_candidate(W.model, a.out, note, rt)
    (a.out / "prune.json").write_text(json.dumps({"cmd": vars(a) | {"out": str(a.out)}, "note": note,
                                                  "stats": stats}, default=str))
    print(f"{a.out}: {note}")


if __name__ == "__main__":
    main()
