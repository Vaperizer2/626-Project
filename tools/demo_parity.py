"""Parity check: the demo's pipeline (Pillow preprocessing + onnxruntime) vs the training-side reference
(compclass.final.load_final in PyTorch on build_transforms(train=False) tensors), on DEV images only.

    uv run python tools/demo_parity.py --n 64 --out results/demo/demo_parity.json [--copy-demo 24 DIR]

Rows are picked round-robin over classes (sorted by id) so every class is represented. The sealed test split is
never read. Note: the final head was refit on ALL dev rows, so agreement with the *labels* on these rows is
training-set accuracy and is reported only as a sanity check, not as a generalisation estimate.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from compclass import demo
from compclass.data import ImageDataset, build_transforms, get_dev, load_manifest
from compclass.final import load_final


def spread(rows: list[dict], classes: list[str], n: int, offset: int = 0) -> list[dict]:
    by = {c: sorted((r for r in rows if r["label"] == c), key=lambda r: r["id"]) for c in classes}
    out, k = [], offset
    while len(out) < n and any(len(v) > k for v in by.values()):
        out += [by[c][k] for c in classes if len(by[c]) > k][: n - len(out)]
        k += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/v2/manifest.json")
    ap.add_argument("--model-dir", default="exports/final")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="results/demo/demo_parity.json")
    ap.add_argument("--copy-demo", nargs=2, metavar=("N", "DIR"), default=None,
                    help="also copy N other dev images (spread over classes) to DIR as <label>__<id>.jpg")
    a = ap.parse_args()
    m = load_manifest(a.manifest)
    dev = get_dev(m)
    assert all(r["split"] == "dev" for r in dev)
    rows = spread(dev, m.classes, a.n)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    # reference: compclass.final on CPU (fp32, no TF32), eval transform from compclass.data
    t0 = time.perf_counter()
    model, spec = load_final(Path(a.model_dir) / "head.npz")
    ds = ImageDataset(rows, m.root, m.class_to_idx, build_transforms(spec, None, train=False))
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    with torch.no_grad():
        ref = torch.cat([model(X[i:i + 8])[1] for i in range(0, len(X), 8)]).numpy()
    t_ref = time.perf_counter() - t0
    del model
    # demo pipeline: decode with demo.open_image, preprocess with demo.preprocess, run onnxruntime (CPU)
    clf = demo.Classifier(a.model_dir, "onnx", device="cpu")
    got, in_diff, ms = [], [], []
    for i, r in enumerate(rows):
        t1 = time.perf_counter()
        x = demo.preprocess(demo.open_image(m.root / r["path"]), clf.pre)
        got.append(clf.backend(x))
        ms.append((time.perf_counter() - t1) * 1000)
        in_diff.append(float(np.abs(x[0] - X[i].numpy()).max()))
    got = np.stack(got)
    y = np.array([m.class_to_idx[r["label"]] for r in rows])
    thr = clf.threshold
    rep = {
        "n_images": len(rows), "split": "dev", "n_classes_covered": int(len({r["label"] for r in rows})),
        "reference": "compclass.final.load_final (torch CPU fp32) on compclass.data.build_transforms(train=False)",
        "candidate": f"compclass.demo.open_image + preprocess + {clf.backend.name}",
        "max_abs_input_diff": float(max(in_diff)),
        "max_abs_prob_diff": float(np.abs(ref - got).max()),
        "mean_abs_prob_diff": float(np.abs(ref - got).mean()),
        "argmax_agreement": float((ref.argmax(1) == got.argmax(1)).mean()),
        "abstain_decision_agreement": float(((ref.max(1) < thr) == (got.max(1) < thr)).mean()),
        "onnx_cpu_ms_per_image_median": float(np.median(ms[1:] if len(ms) > 1 else ms)),
        "onnx_cpu_ms_per_image_first": float(ms[0]),
        "torch_ref_total_s_including_load": round(t_ref, 1),
        "sanity_label_accuracy_on_training_rows": float((got.argmax(1) == y).mean()),
        "note": "rows are dev = the final head's training rows; label accuracy here is NOT a generalisation estimate",
        "manifest_fingerprint": m.fingerprint, "ids": [r["id"] for r in rows],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "ids"}, indent=1))
    if a.copy_demo:
        n, d = int(a.copy_demo[0]), Path(a.copy_demo[1])
        d.mkdir(parents=True, exist_ok=True)
        used = {r["id"] for r in rows}
        extra = spread([r for r in dev if r["id"] not in used and not r.get("flags")], m.classes, n)
        for r in extra:
            shutil.copy(m.root / r["path"], d / f"{r['label'].replace(' ', '_')}__{r['id']}.jpg")
        print(f"copied {len(extra)} dev images to {d}")


if __name__ == "__main__":
    main()
