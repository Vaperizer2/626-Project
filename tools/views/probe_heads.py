"""Per-fold linear-probe heads of a backbone, folded into one Linear layer, for ``train.head_init`` (LP-FT).

    uv run python tools/views/probe_heads.py --model mobileclip2_s3

For each development fold the probe (compclass.features.fit_probe: StandardScaler + multinomial logistic regression,
C by grouped CV) is fitted on the fold's TRAINING rows from the feature cache, then folded:
    coef . (f - mu) / sd + b  =  (coef / sd) . f + (b - coef . mu / sd)
and written to results/cache/heads/<key>_fold<k>.npz (weight [C, D], bias [C]); git-ignored. The folded layer's
macro-F1 on the fold's validation rows is printed next to the probe's, and must match.
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
from compclass.data import load_manifest, get_dev, get_fold
from compclass.features import fit_probe
from compclass.evaluate import compute_metrics

CACHE, OUT = Path("results/cache/features"), Path("results/cache/heads")
FP = "431315e87d2612cb-1fdfce4c916e"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--view", default="", help="cache view tag, e.g. crop100")
    a = ap.parse_args()
    m = load_manifest("data/v2/manifest.json")
    dev = get_dev(m)
    pos = {r["id"]: i for i, r in enumerate(dev)}
    y = np.array([m.class_to_idx[r["label"]] for r in dev]); cl = np.array([r["cluster"] for r in dev])
    OUT.mkdir(parents=True, exist_ok=True)
    for key in a.model:
        F = np.load(CACHE / f"{key}-{FP}{'-' + a.view if a.view else ''}.npy").astype(np.float32)
        for k in range(5):
            tr, va = get_fold(m, k)
            itr, iva = [pos[r["id"]] for r in tr], [pos[r["id"]] for r in va]
            s = fit_probe(F[itr], y[itr], cl[itr])
            sc, lr = s.best_estimator_.named_steps["standardscaler"], s.best_estimator_.named_steps["logisticregression"]
            W = lr.coef_ / sc.scale_; b = lr.intercept_ - W @ sc.mean_
            f1_probe = compute_metrics(y[iva], s.predict(F[iva]), m.classes)["macro_f1"]
            f1_fold = compute_metrics(y[iva], (F[iva] @ W.T + b).argmax(1), m.classes)["macro_f1"]
            np.savez(OUT / f"{key}_fold{k}.npz", weight=W.astype(np.float32), bias=b.astype(np.float32))
            print(f"{key} fold {k}: C {s.best_params_['logisticregression__C']}, probe F1 {f1_probe:.4f}, folded {f1_fold:.4f}")


if __name__ == "__main__":
    main()
