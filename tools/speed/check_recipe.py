"""Prove eval's head recipe reproduces the recorded development numbers on the recorded features.

Input: the So400m-384 dev features cached by ``compclass.features.extract_features`` (torch, fp16 autocast) on the
GPU PC, the exact features behind results/benchmark.csv and exports/final/fit_report.json:
    scp carlpc2:/home/carl/Projects/626-Project/results/cache/features/siglip_so400m_384-431315e87d2612cb-1fdfce4c916e.npy \\
        /home/vaper/Work/Prototypes/626-artifacts/harness/reference/

benchmark.csv's 0.8252 predates commit cbf1877, which widened fit_probe's C grid from (0.01, 0.1, 1, 10) to
(0.001, 0.003, 0.01, 0.1, 1); the deployed head and eval use the wider grid. The fused out-of-fold numbers match
fit_report.json exactly on carlpc2; on legion fold 3's inner-CV choice between C=0.01 and C=0.1 (macro-F1 margin
~2e-4) resolves the other way, which moves the pooled accuracy by one image.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from evaluate import cv_head, dev_rows, labels
from harness import ART, DEPLOYED, MAIN

REF = ART / "reference" / "siglip_so400m_384-431315e87d2612cb-1fdfce4c916e.npy"
OLD_GRID = (0.01, 0.1, 1.0, 10.0)


def old_grid_fold_accs(F, m, rows) -> list[float]:
    from compclass.features import fit_probe

    y, fold, groups = labels(m, rows)
    accs = []
    for k in range(m.n_folds):
        tr, va = fold != k, fold == k
        s = fit_probe(F[tr], y[tr], groups[tr], Cs=OLD_GRID)
        accs.append(round(float((s.predict(F[va]) == y[va]).mean()), 4))
    return accs


def main() -> None:
    m, rows = dev_rows()
    F = np.load(REF)
    assert F.shape == (len(rows), 1152), F.shape
    fit = json.loads((DEPLOYED / "fit_report.json").read_text())
    with open(MAIN / "results/phase1/fusion_ablation.csv") as f:
        bench = next(r for r in csv.DictReader(f) if r["T"] == "50" and r["w"] == "0.0")
    old = old_grid_fold_accs(F, m, rows)
    probe, _ = cv_head(F, m, rows, fusion=False)
    fused, pred = cv_head(F, m, rows, fusion=True)
    n = len(rows)
    checks = {
        "probe, old C grid: fold accs (fusion_ablation.csv)": (old, [float(bench[f"fold{k}_acc"]) for k in range(5)], 0),
        "probe, old C grid: CV acc (benchmark.csv 0.8252)": (round(float(np.mean(old)), 4), 0.8252, 0),
        "probe: pooled OOF acc (fit_report)": (probe["oof_acc"], fit["oof_probe_only_accuracy"], 0),
        "fused: pooled OOF acc within 2 images (fit_report)": (fused["oof_acc"], fit["oof_accuracy"], 2 / n),
    }
    ok = True
    for name, (got, want, tol) in checks.items():
        good = got == want if tol == 0 else abs(got - want) <= tol + 1e-12
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {name}: got {got}, recorded {want}")
    print(f"fused per-fold C {[f['C'] for f in fused['folds']]}; CV acc {fused['cv_acc_mean']:.4f} ± "
          f"{fused['cv_acc_sd']:.4f}, macro-F1 {fused['cv_f1_mean']:.4f} ± {fused['cv_f1_sd']:.4f}, "
          f"pooled OOF {fused['oof_acc']:.4f} (recorded {fit['oof_accuracy']:.4f})")
    np.savez(ART / "reference" / "reference_flags.npz", ids=np.array([r["id"] for r in rows]),
             correct=pred == labels(m, rows)[0])
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
