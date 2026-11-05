"""The deployed recipe (probe + zero-shot head, fused with T 50 / w 0.25) on several backbones, out-of-fold on the
development folds, paired per fold against SigLIP So400m at crop 1.0 and against each backbone's own probe.

    uv run python tools/views/fusion_table.py --model mobileclip2_s3 --model mobileclip2_b --model pe_core_b16_336

Needs each key's feature cache (compclass.features), its heads (tools/views/heads.py) and, for the cross-check,
its compclass.zeroshot OOF file. SigLIP's crop-1.0 cache comes from tools/views/extract_views.py.
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
from compclass.data import load_manifest, get_dev, get_fold
from compclass.features import fit_probe
from compclass.evaluate import compute_metrics

CACHE, HEADS, OOF = Path("results/cache/features"), Path("results/cache/zeroshot"), Path("results/oof")
FP = "431315e87d2612cb-1fdfce4c916e"
T_ZS, W_ZS = 50.0, 0.25


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    args = ap.parse_args()
    m = load_manifest("data/v2/manifest.json")
    dev = get_dev(m)
    pos = {r["id"]: i for i, r in enumerate(dev)}
    y = np.array([m.class_to_idx[r["label"]] for r in dev])
    cl = np.array([r["cluster"] for r in dev])
    folds = [([pos[r["id"]] for r in tr], [pos[r["id"]] for r in va]) for tr, va in (get_fold(m, k) for k in range(5))]

    def cv(L):
        return (np.array([compute_metrics(y[i], L[i].argmax(1), m.classes)["macro_f1"] for _, i in folds]),
                np.array([compute_metrics(y[i], L[i].argmax(1), m.classes)["accuracy"] for _, i in folds]))

    def score(F, E, W):
        cos = (E / np.linalg.norm(E, axis=1, keepdims=True)) @ (W / np.linalg.norm(W, axis=1, keepdims=True)).T
        LP = np.full((len(dev), len(m.classes)), np.nan)
        for itr, iva in folds:
            LP[iva] = fit_probe(F[itr], y[itr], cl[itr]).predict_log_proba(F[iva])
        return {"probe": cv(LP), "zero-shot": cv(cos), "fused": cv(LP + W_ZS * log_softmax(T_ZS * cos)), "cos": cos}

    hz = np.load("exports/final/head.npz")
    Fs = np.load(CACHE / f"siglip_so400m_384-{FP}-crop100.npy").astype(np.float32)
    ref = score(Fs, Fs, hz["W"].astype(np.float32))
    rows = [("SigLIP So400m@384, crop 1.0 (reference)", ref)]
    for key in args.model:
        F = np.load(CACHE / f"{key}-{FP}.npy").astype(np.float32)
        pr = np.load(HEADS / f"{key}_proj.npz")
        E = F @ pr["weight"].T + pr["bias"]
        r = score(F, E, np.load(HEADS / f"{key}_W.npy").astype(np.float32))
        z = OOF / f"zeroshot_{key}.npz"
        if z.exists():
            zz = np.load(z, allow_pickle=True)
            sim = np.full_like(r["cos"], np.nan); sim[[pos[i] for i in zz["ids"]]] = zz["sim"]
            r["open_clip zero-shot"] = cv(sim)
            r["agree"] = float((sim.argmax(1) == r["cos"].argmax(1)).mean())
        rows.append((key, r))

    def pm(a, b):
        d = a - b
        return f"{d.mean()*100:+.2f} ± {d.std(ddof=1)/5**.5*100:.2f} pt ({(d>0).sum()}/5)"

    print("| Backbone | probe F1 | zero-shot F1 (timm+proj) | zero-shot F1 (open_clip) | top-1 agreement | fused F1 | fused acc | fused vs own probe | fused vs SigLIP fused |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in rows:
        oc = f"{r['open_clip zero-shot'][0].mean():.4f}" if "open_clip zero-shot" in r else "–"
        ag = f"{r['agree']:.3f}" if "agree" in r else "–"
        print(f"| {name} | {r['probe'][0].mean():.4f} | {r['zero-shot'][0].mean():.4f} | {oc} | {ag} | {r['fused'][0].mean():.4f} | {r['fused'][1].mean():.4f} | "
              f"{pm(r['fused'][0], r['probe'][0])} | {pm(r['fused'][0], ref['fused'][0])} |")


if __name__ == "__main__":
    main()
