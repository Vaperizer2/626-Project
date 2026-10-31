"""eval: development-fold CV accuracy of the deployed head recipe on a candidate's own embeddings.

Embeddings come from the candidate's ONNX graph on its own execution provider, one image per run (the deployed
batch size, and the configuration ``time`` measures), so precision and quantisation error reach the score. Per
fold, the head is fitted exactly as ``compclass.final.cmd_fit`` does for its out-of-fold check:
``features.fit_probe`` (standardise + logistic regression, C by grouped 3-fold CV on the training rows), fused
with the fixed zero-shot text head (T=50, w=0.25, the deployed text embeddings) when the candidate declares
``zero_shot_fusion``. Temperature and the abstain threshold do not change the argmax; they
are refitted on the out-of-fold scores and reported for information.
"""

from __future__ import annotations

import time

import numpy as np

from harness import (DEPLOYED, FINGERPRINT, INPUT, MANIFEST, MAX_BATCH, OUTPUT, Candidate, append_result,
                     deployed_graph, gpu_lock, load_candidate, make_session, provider_info)


def dev_rows():
    from compclass.data import get_dev, load_manifest

    m = load_manifest(MANIFEST)
    if m.fingerprint != FINGERPRINT:
        raise SystemExit(f"manifest fingerprint {m.fingerprint} != {FINGERPRINT}")
    return m, get_dev(m)


def dataset(c: Candidate, m, rows):
    from compclass.data import ImageDataset, build_transforms

    return ImageDataset(rows, m.root, m.class_to_idx, build_transforms(c.spec, None, train=False))


def preprocessed(c: Candidate, m, rows) -> np.ndarray:
    ds = dataset(c, m, rows)
    return np.stack([ds[i][0].numpy() for i in range(len(ds))])


def labels(m, rows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class index, development fold and near-duplicate cluster per row."""
    return (np.array([m.class_to_idx[r["label"]] for r in rows]), np.array([r["fold"] for r in rows]),
            np.array([r["cluster"] for r in rows]))


def cv_head(F: np.ndarray, m, rows, fusion: bool) -> tuple[dict, np.ndarray]:
    """Per-fold fit of the deployed head on embeddings ``F``; returns the summary and out-of-fold predictions."""
    from compclass.evaluate import compute_metrics
    from compclass.features import fit_probe
    from compclass.final import fit_temperature, fused_scores, head_from_search, log_softmax

    W = np.load(DEPLOYED / "head.npz")["W"] if fusion else None
    y, fold, groups = labels(m, rows)
    scores = np.zeros((len(rows), len(m.classes)), np.float32)
    probe = np.zeros(len(rows), int)
    folds = []
    for k in range(m.n_folds):
        tr, va = fold != k, fold == k
        s = fit_probe(F[tr], y[tr], groups[tr])
        scores[va] = fused_scores(F[va], head_from_search(s, W)) if fusion else log_softmax(s.decision_function(F[va]))
        probe[va] = s.predict(F[va])
        met = compute_metrics(y[va], scores[va].argmax(1), m.classes)
        folds.append({"fold": k, "n": int(va.sum()), "accuracy": met["accuracy"], "macro_f1": met["macro_f1"],
                      "C": float(s.best_params_["logisticregression__C"]),
                      "probe_only_accuracy": float((probe[va] == y[va]).mean())})
        print(f"  fold {k}: acc {met['accuracy']:.4f}  macro-F1 {met['macro_f1']:.4f}  C={folds[-1]['C']}", flush=True)
    pred = scores.argmax(1)
    pooled = compute_metrics(y, pred, m.classes)
    temp = fit_temperature(scores, y)
    conf = np.exp(log_softmax(scores / temp)).max(1)
    thr = next((t for t in np.round(np.arange(0.0, 0.96, 0.05), 2)
                if (conf >= t).any() and (pred[conf >= t] == y[conf >= t]).mean() >= 0.90), 0.5)
    acc = [f["accuracy"] for f in folds]
    f1 = [f["macro_f1"] for f in folds]
    return {"cv_acc_mean": float(np.mean(acc)), "cv_acc_sd": float(np.std(acc, ddof=1)),
            "cv_f1_mean": float(np.mean(f1)), "cv_f1_sd": float(np.std(f1, ddof=1)),
            "oof_acc": pooled["accuracy"], "oof_f1": pooled["macro_f1"],
            "probe_only_cv_acc_mean": float(np.mean([f["probe_only_accuracy"] for f in folds])),
            "folds": folds, "temperature": temp, "abstain_threshold": float(thr),
            "coverage": float((conf >= thr).mean()),
            "accepted_accuracy": float((pred[conf >= thr] == y[conf >= thr]).mean())}, pred


def extract(c: Candidate, m, rows) -> tuple[np.ndarray, dict, float]:
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset(c, m, rows), batch_size=MAX_BATCH, num_workers=4)
    with gpu_lock():
        sess = make_session(c, deployed_graph(c, len(m.classes)))
        info = provider_info(sess)
        t0 = time.perf_counter()
        F = [sess.run([OUTPUT], {INPUT: x[i:i + 1].numpy()})[0] for x, _ in dl for i in range(len(x))]
        seconds = time.perf_counter() - t0
        del sess
    return np.concatenate(F).astype(np.float32), info, seconds


def run_eval(path: str) -> dict:
    c = load_candidate(path)
    m, rows = dev_rows()
    print(f"[{c.id}] extracting {len(rows)} dev embeddings ({c.runtime['provider']})", flush=True)
    F, info, seconds = extract(c, m, rows)
    bad = ~np.isfinite(F).all(1)
    F[bad] = 0.0
    print(f"[{c.id}] {seconds:.0f}s, {int(bad.sum())} non-finite embeddings; fitting heads "
          f"({'probe + zero-shot fusion' if c.fusion else 'probe only'})", flush=True)
    res, pred = cv_head(F, m, rows, c.fusion)
    y, fold, _ = labels(m, rows)
    np.savez(c.work / "flags.npz", ids=np.array([r["id"] for r in rows]), y=y, pred=pred, correct=pred == y,
             fold=fold)
    np.save(c.work / "embeddings.npy", F)
    rec = append_result({"kind": "eval", "candidate": c.id, "candidate_sha": c.sha, "candidate_dir": str(c.dir),
                         "built_from": c.preprocess.get("built_from"), "n_images": len(rows),
                         "n_errors": int(bad.sum()), "n_timed_runs": 0, "embedding_dim": c.dim,
                         "input_size": c.preprocess["input_size"], "zero_shot_fusion": c.fusion,
                         "extract_s": round(seconds, 1), **info, **res, "manifest": m.fingerprint,
                         "flags": str(c.work / "flags.npz"), "embeddings": str(c.work / "embeddings.npy")})
    print(f"[{c.id}] CV acc {rec['cv_acc_mean']:.4f} ± {rec['cv_acc_sd']:.4f}  macro-F1 {rec['cv_f1_mean']:.4f} ± "
          f"{rec['cv_f1_sd']:.4f}  (pooled OOF acc {rec['oof_acc']:.4f}), n={len(rows)}, errors={rec['n_errors']}")
    return rec
