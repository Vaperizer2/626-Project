"""Efficiency study, step 3: MobileCLIP2-S3 fine-tunes against its frozen probe and the fused recipe, per fold.

    uv run python tools/views/finetune_summary.py [--results results/runs_win.jsonl] [--name mobileclip2_s3_full ...]

For each run name with five fold records in the Windows run log: CV macro-F1 / accuracy, paired per-fold deltas
against (a) the frozen MobileCLIP2-S3 probe at crop 1.0 (results/probes_win.jsonl), (b) the fused probe +
zero-shot recipe recomputed OOF from the feature cache and heads (same code path as tools/views/fusion_table.py),
and (c) the SigLIP So400m fused reference at crop 1.0; plus trainable parameters, epochs, training time and peak
VRAM from the records. Deltas are mean ± SE over the five folds with the count of folds in favour.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
from compclass.data import load_manifest, get_dev, get_fold
from compclass.features import fit_probe
from compclass.evaluate import compute_metrics

CACHE, HEADS = Path("results/cache/features"), Path("results/cache/zeroshot")
FP = "431315e87d2612cb-1fdfce4c916e"
T_ZS, W_ZS = 50.0, 0.25


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def fused_oof(key, F, E, W, y, cl, folds, classes):
    cos = (E / np.linalg.norm(E, axis=1, keepdims=True)) @ (W / np.linalg.norm(W, axis=1, keepdims=True)).T
    LP = np.full((len(y), len(classes)), np.nan)
    for itr, iva in folds:
        LP[iva] = fit_probe(F[itr], y[itr], cl[itr]).predict_log_proba(F[iva])
    L = LP + W_ZS * log_softmax(T_ZS * cos)
    return np.array([compute_metrics(y[i], L[i].argmax(1), classes)["macro_f1"] for _, i in folds])


def pm(a, b):
    d = a - b
    return f"{d.mean()*100:+.2f} ± {d.std(ddof=1)/5**.5*100:.2f} ({(d>0).sum()}/5)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/runs_win.jsonl")
    ap.add_argument("--name", action="append", default=None)
    a = ap.parse_args()
    names = a.name or ["mobileclip2_s3_full", "mobileclip2_s3_partial"]
    m = load_manifest("data/v2/manifest.json")
    dev = get_dev(m)
    pos = {r["id"]: i for i, r in enumerate(dev)}
    y = np.array([m.class_to_idx[r["label"]] for r in dev]); cl = np.array([r["cluster"] for r in dev])
    folds = [([pos[r["id"]] for r in tr], [pos[r["id"]] for r in va]) for tr, va in (get_fold(m, k) for k in range(5))]
    # references
    probes = [json.loads(l) for l in open("results/probes_win.jsonl") if l.strip()]
    pr = {}
    for r in probes:
        if r.get("method") == "linear_probe" and r["model"] == "mobileclip2_s3":
            pr[r["fold"]] = r["macro_f1"]
    probe = np.array([pr[i] for i in range(5)])
    F = np.load(CACHE / f"mobileclip2_s3-{FP}.npy").astype(np.float32)
    p = np.load(HEADS / "mobileclip2_s3_proj.npz")
    fused = fused_oof("mobileclip2_s3", F, F @ p["weight"].T + p["bias"], np.load(HEADS / "mobileclip2_s3_W.npy"), y, cl, folds, m.classes)
    Fs = np.load(CACHE / f"siglip_so400m_384-{FP}-crop100.npy").astype(np.float32)
    sig = fused_oof("siglip_so400m_384", Fs, Fs, np.load("exports/final/head.npz")["W"].astype(np.float32), y, cl, folds, m.classes)
    recs = [json.loads(l) for l in open(a.results, encoding="utf-8") if l.strip()]
    print(f"References (5-fold CV macro-F1): MobileCLIP2-S3 probe {probe.mean():.4f}, MobileCLIP2-S3 fused {fused.mean():.4f}, "
          f"SigLIP So400m fused {sig.mean():.4f}\n")
    print("| Run | trainable | epochs (best) | min/fold | peak VRAM | CV macro-F1 (± SD) | CV acc | vs S3 probe | vs S3 fused | vs SigLIP fused | folds |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for name in names:
        per = {}
        for r in recs:
            if r.get("name") == name and r.get("method") == "finetune":
                per[r["fold"]] = r  # latest record per fold wins
        if len(per) < 5:
            print(f"(skipping {name}: {len(per)} folds)"); continue
        f1 = np.array([per[i]["macro_f1"] for i in range(5)]); acc = np.array([per[i]["accuracy"] for i in range(5)])
        ep = np.mean([per[i]["epochs_run"] for i in range(5)]); be = np.mean([per[i]["best_epoch"] for i in range(5)])
        tt = np.mean([per[i]["train_time_s"] for i in range(5)]) / 60; vr = max(per[i]["peak_vram_mb"] for i in range(5))
        print(f"| {name} | {per[0]['trainable_params']/1e6:.1f} M | {ep:.1f} ({be:.1f}) | {tt:.1f} | {vr:.0f} MB | {f1.mean():.4f} ± {f1.std(ddof=1):.4f} | "
              f"{acc.mean():.4f} | {pm(f1, probe)} | {pm(f1, fused)} | {pm(f1, sig)} | {np.round(f1, 3).tolist()} |")


if __name__ == "__main__":
    main()
