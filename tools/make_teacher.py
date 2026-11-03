"""Per-fold teacher logits for distilling a frozen-backbone linear probe (model C) into the scratch CNN.

For every development fold k a logistic-regression probe is fitted on the SigLIP features of that fold's
*training* rows only (same procedure as ``compclass.features.run_probe``: standardise, multinomial logistic
regression, C chosen by grouped 3-fold CV inside the training rows). Its logits for every dev row are written
to ``<out_dir>/<teacher>_fold<k>.npz`` (``ids``, ``logits``, ``is_train``). The student's trainer reads only
the training rows' logits; the validation rows' logits are stored so the teacher's own fold score can be
reported, which is the usual check that the teacher is worth distilling. The sealed test split is not scored.

    uv run python tools/make_teacher.py --model siglip_so400m_384 --folds 0 1 2 3 4 [--cross-fit]

``--cross-fit``: the training rows' logits come from an inner 5-fold cross-fit (grouped by cluster) inside the
training split, so every training row is scored by a probe that did not see it; the regularisation C is the one
the full-split probe chose, so the logit scale is the same across rows. Output name gets a ``_xfit`` suffix.
Validation rows are still scored by the full-split probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compclass.data import get_dev, get_fold, load_manifest
from compclass.evaluate import compute_metrics
from compclass.features import extract_features, fit_probe


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/v2/manifest.json")
    ap.add_argument("--model", default="siglip_so400m_384")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out-dir", default="results/cache/teacher")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cross-fit", action="store_true")
    args = ap.parse_args(argv)
    m = load_manifest(args.manifest)
    dev = get_dev(m)
    X = extract_features(args.model, dev, m, batch_size=args.batch_size)
    ids = np.array([r["id"] for r in dev])
    y = np.array([m.class_to_idx[r["label"]] for r in dev])
    pos = {r["id"]: i for i, r in enumerate(dev)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for k in args.folds:
        tr, va = get_fold(m, k)
        itr = np.array([pos[r["id"]] for r in tr])
        iva = np.array([pos[r["id"]] for r in va])
        search = fit_probe(X[itr], y[itr], np.array([r["cluster"] for r in tr]))
        logits = search.decision_function(X).astype(np.float32)  # [N_dev, C], log-odds of the fitted probe
        if args.cross_fit:
            C = search.best_params_["logisticregression__C"]
            groups = np.array([r["cluster"] for r in tr])
            for inner_tr, inner_va in GroupKFold(5).split(itr, y[itr], groups):
                pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
                pipe.fit(X[itr[inner_tr]], y[itr[inner_tr]])
                logits[itr[inner_va]] = pipe.decision_function(X[itr[inner_va]]).astype(np.float32)
        is_train = np.zeros(len(dev), dtype=bool)
        is_train[itr] = True
        tr_m = compute_metrics(y[itr], logits[itr].argmax(1), m.classes)
        va_m = compute_metrics(y[iva], logits[iva].argmax(1), m.classes)
        suffix = "_xfit" if args.cross_fit else ""
        path = out_dir / f"probe_{args.model}{suffix}_fold{k}.npz"
        np.savez(path, ids=ids, logits=logits, is_train=is_train, y=y, classes=np.array(m.classes),
                 manifest=m.fingerprint, fold=k, best_C=search.best_params_["logisticregression__C"])
        rec = {"fold": k, "best_C": search.best_params_["logisticregression__C"],
               "train_acc": round(tr_m["accuracy"], 4), "train_macro_f1": round(tr_m["macro_f1"], 4),
               "val_acc": round(va_m["accuracy"], 4), "val_macro_f1": round(va_m["macro_f1"], 4), "file": str(path)}
        summary.append(rec)
        print(f"[teacher {args.model} f{k}] C={rec['best_C']}  train acc {rec['train_acc']:.4f}  "
              f"val acc {rec['val_acc']:.4f}  val macro-F1 {rec['val_macro_f1']:.4f}  -> {path}", flush=True)
    (out_dir / f"probe_{args.model}{suffix}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
