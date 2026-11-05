"""2 x 2 (x views): backbone x input view, probe / zero-shot / fused (deployed constants T 50, w 0.25), OOF on the dev folds.
Paired comparisons per fold. Needs the view caches (tools/views/extract_views.py) and the PE-Core heads
(tools/views/pe_core_heads.py).   uv run python tools/views/view_fusion.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from compclass.data import load_manifest, get_dev, get_fold
from compclass.features import fit_probe
from compclass.evaluate import compute_metrics

S = Path("results/cache/zeroshot")
m = load_manifest("data/v2/manifest.json")
dev = get_dev(m)
pos = {r["id"]: i for i, r in enumerate(dev)}
y = np.array([m.class_to_idx[r["label"]] for r in dev])
CACHE = Path("results/cache/features"); FP = "431315e87d2612cb-1fdfce4c916e"
TR = [[pos[r["id"]] for r in get_fold(m, k)[0]] for k in range(5)]
VA = [[pos[r["id"]] for r in get_fold(m, k)[1]] for k in range(5)]
clusters = np.array([r["cluster"] for r in dev])
T_ZS, W_ZS = 50.0, 0.25


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def cv(L):
    return (np.array([compute_metrics(y[i], L[i].argmax(1), m.classes)["macro_f1"] for i in VA]),
            np.array([compute_metrics(y[i], L[i].argmax(1), m.classes)["accuracy"] for i in VA]))


hz = np.load("exports/final/head.npz"); proj = np.load(S / "pe_core_l14_336_proj.npz")
BACKBONES = {
    "SigLIP So400m@384": ("siglip_so400m_384", hz["W"].astype(np.float32), None),
    "PE-Core L@336": ("pe_core_l14_336", np.load(S / "pe_core_l14_336_W.npy").astype(np.float32),
                      (proj["weight"].astype(np.float32), proj["bias"].astype(np.float32))),
}
VIEWS = {"default": "", "crop100": "-crop100", "squash": "-squash"}  # PE-Core's default IS crop100
res = {}
for bname, (key, W, pr) in BACKBONES.items():
    for vname, tag in VIEWS.items():
        f = CACHE / f"{key}-{FP}{tag}.npy"
        if not f.exists():
            continue
        F = np.load(f).astype(np.float32)
        E = F if pr is None else F @ pr[0].T + pr[1]
        cos = (E / np.linalg.norm(E, axis=1, keepdims=True)) @ (W / np.linalg.norm(W, axis=1, keepdims=True)).T
        LP = np.full((len(dev), len(m.classes)), np.nan)
        for itr, iva in zip(TR, VA):
            LP[iva] = fit_probe(F[itr], y[itr], clusters[itr]).predict_log_proba(F[iva])
        r = {"probe": cv(LP), "zero-shot": cv(cos), "fused": cv(LP + W_ZS * log_softmax(T_ZS * cos))}
        res[(bname, vname)] = r
        print(f"{bname:18s} {vname:8s} | probe F1 {r['probe'][0].mean():.4f} acc {r['probe'][1].mean():.4f} | "
              f"zero-shot F1 {r['zero-shot'][0].mean():.4f} | fused F1 {r['fused'][0].mean():.4f} acc {r['fused'][1].mean():.4f} "
              f"folds {r['fused'][0].round(3)}", flush=True)


def pair(label, a, b):
    d = a["fused"][0] - b["fused"][0]; da = a["fused"][1] - b["fused"][1]
    print(f"{label:70s} F1 {d.mean()*100:+.2f} +- {d.std(ddof=1)/5**.5*100:.2f} pt ({(d>0).sum()}/5)   acc {da.mean()*100:+.2f} +- {da.std(ddof=1)/5**.5*100:.2f} pt")


print("\nPaired, fused models (T 50, w 0.25):")
sg, pe = "SigLIP So400m@384", "PE-Core L@336"
pairs = [("SigLIP crop100 vs SigLIP default (border cut)", (sg, "crop100"), (sg, "default")),
         ("SigLIP squash vs SigLIP default", (sg, "squash"), (sg, "default")),
         ("PE-Core squash vs PE-Core default (= crop100)", (pe, "squash"), (pe, "default")),
         ("PE-Core default vs SigLIP crop100  [same view]", (pe, "default"), (sg, "crop100")),
         ("PE-Core squash vs SigLIP squash    [same view]", (pe, "squash"), (sg, "squash")),
         ("PE-Core default vs SigLIP default  [as deployed]", (pe, "default"), (sg, "default")),
         ("PE-Core best view vs SigLIP best view", None, None)]
for label, a, b in pairs:
    if a is None:
        best = {bn: max((k for k in res if k[0] == bn), key=lambda k: res[k]["fused"][0].mean()) for bn in (sg, pe)}
        label = f"PE-Core {best[pe][1]} vs SigLIP {best[sg][1]}  [best view each]"; a, b = best[pe], best[sg]
    if a in res and b in res:
        pair(label, res[a], res[b])
