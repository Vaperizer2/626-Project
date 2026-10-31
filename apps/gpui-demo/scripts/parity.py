"""Parity of the Rust front end (gpui-demo) with the Python/PyTorch pipeline on dev images.

    python apps/gpui-demo/scripts/parity.py --bin <path to gpui-demo> [--n 64] [--out results/demo/gpui_parity.json]

Needs numpy + Pillow (same Pillow as the project, 13.0.0) and the repo checkout. Two checks:
 1. input tensors: `gpui-demo preprocess` vs compclass.demo.preprocess(open_image(path)) (Pillow reference)
 2. probabilities: `gpui-demo classify --json` vs results/kaggle_original/row_probs.npz (PyTorch reference)
Only split == 'dev' rows of data/v2/manifest.json are used (never 'test', never the web holdout).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def load_demo_module():
    spec = importlib.util.spec_from_file_location("compclass_demo", REPO / "src/compclass/demo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up while decorating
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default=str(REPO / "results/demo/gpui_parity.json"))
    ap.add_argument("--ort-lib", default=None)
    args = ap.parse_args()

    model_dir = REPO / "exports/final"
    man = json.loads((REPO / "data/v2/manifest.json").read_text())
    dev = sorted((r for r in man["rows"] if r["split"] == "dev"), key=lambda r: r["id"])
    step = len(dev) // args.n
    rows = dev[::step][: args.n]
    assert len(rows) == args.n and all(r["split"] == "dev" for r in rows)
    paths = [str(REPO / "data/v2" / r["path"]) for r in rows]  # manifest paths start with images_root

    demo = load_demo_module()
    pre, labels = demo.load_contract(model_dir)

    # 1. input tensors vs the Pillow reference
    tdiff, nbad, nel = [], 0, 0
    with tempfile.TemporaryDirectory() as td:
        for i, p in enumerate(paths):
            out = Path(td) / f"{i}.f32"
            subprocess.run([args.bin, "preprocess", p, str(out), "--models", str(REPO / "exports"), "--model", "deployed"],
                           check=True, capture_output=True)
            rust = np.fromfile(out, dtype="<f4").reshape(1, 3, pre["input_size"], pre["input_size"])
            ref = demo.preprocess(demo.open_image(Path(p)), pre)
            d = np.abs(rust - ref)
            tdiff.append(float(d.max()))
            nbad += int((d > 0).sum())
            nel += d.size

    # 2. probabilities vs the PyTorch reference (one process: the model is loaded once)
    cmd = [args.bin, "classify", "--json", "--models", str(REPO / "exports"), "--model", "deployed"]
    if args.ort_lib:
        cmd += ["--ort-lib", args.ort_lib]
    t0 = time.time()
    res = subprocess.run(cmd + paths, check=True, capture_output=True, text=True)
    wall = time.time() - t0
    preds = json.loads(res.stdout)
    ref = np.load(REPO / "results/kaggle_original/row_probs.npz")
    idx = {str(k): i for i, k in enumerate(ref["ids"])}
    thr = pre["abstain_threshold"]
    per = []
    for r, pr in zip(rows, preds):
        assert pr["error"] is None, pr
        q = np.asarray(pr["probs"], np.float64)
        p = ref["probs"][idx[r["id"]]].astype(np.float64)
        per.append({"id": r["id"], "label": r["label"], "rust_argmax": labels[int(q.argmax())],
                    "ref_argmax": labels[int(p.argmax())], "max_abs_dprob": float(np.abs(q - p).max()),
                    "rust_top_p": float(q.max()), "ref_top_p": float(p.max()), "rust_ms": pr["ms"],
                    "unsure_agree": bool((q.max() < thr) == (p.max() < thr))})
    agree = sum(x["rust_argmax"] == x["ref_argmax"] for x in per)
    ms = [x["rust_ms"] for x in per]
    load_line = [ln for ln in res.stderr.splitlines() if "loaded in" in ln]
    out = {
        "binary": subprocess.run([args.bin, "--version"], capture_output=True, text=True).stdout.strip(),
        "what": "gpui-demo (Rust, ort + Pillow-exact preprocessing) vs Python reference on dev images",
        "n_images": len(per),
        "selection": f"data/v2 manifest rows with split=='dev', sorted by id, every {step}th, first {args.n}",
        "reference_probs": "results/kaggle_original/row_probs.npz (PyTorch pipeline)",
        "reference_tensors": "compclass.demo.preprocess(compclass.demo.open_image(path)), Pillow",
        "argmax_agree": agree,
        "argmax_agree_frac": agree / len(per),
        "unsure_decision_agree": sum(x["unsure_agree"] for x in per),
        "max_abs_dprob": max(x["max_abs_dprob"] for x in per),
        "mean_max_abs_dprob": float(np.mean([x["max_abs_dprob"] for x in per])),
        "tensor_max_abs_diff": max(tdiff),
        "tensor_elements_differing": nbad,
        "tensor_elements_total": nel,
        "images_with_identical_tensor": sum(d == 0 for d in tdiff),
        "accuracy_vs_label_rust": sum(x["rust_argmax"] == x["label"] for x in per) / len(per),
        "accuracy_vs_label_ref": sum(x["ref_argmax"] == x["label"] for x in per) / len(per),
        "latency_ms_median": float(np.median(ms)),
        "latency_ms_mean": float(np.mean(ms)),
        "latency_ms_p90": float(np.percentile(ms, 90)),
        "classify_wall_s": wall,
        "model_load_log": load_line[0] if load_line else None,
        "per_image": per,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_image"}, indent=1))


if __name__ == "__main__":
    main()
