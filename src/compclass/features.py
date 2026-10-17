"""Frozen features + logistic-regression linear probe.

A linear probe asks: *how linearly separable are our classes in this backbone's feature space, without
changing the backbone?* It is cheap (one forward pass per image, cached to .npy) and isolates the quality of
pretraining from fine-tuning tricks -- the fair way to compare DINOv2 vs CLIP vs SigLIP vs supervised ImageNet.

    uv run python -m compclass.features --model dinov2_vitb14 --folds 0 1 2 3 4

The regularisation strength C is chosen *inside* each training fold (3-fold GroupKFold on clusters), so the
validation fold stays untouched and the score is an honest estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .data import ImageDataset, Manifest, Row, build_transforms, get_dev, get_fold, load_manifest
from .evaluate import compute_metrics, log_run
from .models import build_model


@torch.no_grad()
def extract_features(key: str, rows: list[Row], m: Manifest, cache_dir: str | Path = "results/cache/features",
                     batch_size: int = 64, num_workers: int = 4, pretrained: bool = True) -> np.ndarray:
    """Pooled backbone features [N, D] for ``rows`` (eval preprocessing), cached by model + row ids."""
    ids_hash = hashlib.sha256("".join(r["id"] for r in rows).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"{key}-{m.fingerprint}-{ids_hash}.npy"
    if cache.exists():
        return np.load(cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, spec = build_model(key, num_classes=0, pretrained=pretrained)  # num_classes=0 -> features
    model.eval().to(device)
    dl = DataLoader(ImageDataset(rows, m.root, m.class_to_idx, build_transforms(spec, None, train=False)),
                    batch_size=batch_size, num_workers=num_workers)
    feats = []
    for x, _ in dl:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            feats.append(model(x.to(device)).float().cpu())
    out = torch.cat(feats).numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def fit_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, Cs=(0.01, 0.1, 1.0, 10.0)) -> GridSearchCV:
    """Standardise -> multinomial logistic regression; C chosen by grouped 3-fold CV on the training rows."""
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    search = GridSearchCV(pipe, {"logisticregression__C": list(Cs)}, cv=GroupKFold(3), scoring="f1_macro",
                          n_jobs=-1)
    return search.fit(X, y, groups=groups)


def run_probe(manifest_path: str, key: str, folds: list[int], results: str = "results/runs.jsonl",
              pretrained: bool = True, **kw) -> list[dict]:
    m = load_manifest(manifest_path)
    dev = get_dev(m)
    t0 = time.perf_counter()
    X = extract_features(key, dev, m, pretrained=pretrained, **kw)
    extract_s = time.perf_counter() - t0
    pos = {r["id"]: i for i, r in enumerate(dev)}
    # Out-of-fold class probabilities (each dev row scored by the probe that did not train on it);
    # used for label-noise auditing and ensembling.
    oof = np.full((len(dev), len(m.classes)), np.nan, dtype=np.float32)
    records = []
    for k in folds:
        tr, va = get_fold(m, k)
        itr, iva = [pos[r["id"]] for r in tr], [pos[r["id"]] for r in va]
        ytr = np.array([m.class_to_idx[r["label"]] for r in tr])
        yva = np.array([m.class_to_idx[r["label"]] for r in va])
        search = fit_probe(X[itr], ytr, np.array([r["cluster"] for r in tr]))
        oof[iva] = search.predict_proba(X[iva])
        metrics = compute_metrics(yva, search.predict(X[iva]), m.classes)
        rec = {"method": "linear_probe", "name": f"probe_{key}", "model": key, "fold": k, "manifest": m.fingerprint,
               "feature_dim": int(X.shape[1]), "best_C": search.best_params_["logisticregression__C"],
               "extract_time_s": round(extract_s, 1), **metrics}
        log_run(rec, results)
        print(f"[probe {key} f{k}] C={rec['best_C']}  acc {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}")
        records.append(rec)
    out = Path(results).parent / "oof" / f"probe_{key}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.array([r["id"] for r in dev]), y=np.array([m.class_to_idx[r["label"]] for r in dev]),
             proba=oof, classes=np.array(m.classes))
    return records


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/v2/manifest.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--results", default="results/runs.jsonl")
    args = ap.parse_args(argv)
    run_probe(args.manifest, args.model, args.folds, args.results)


if __name__ == "__main__":
    main()
