"""Regenerate the analysis tables behind the report from raw run logs, OOF predictions and cached features.

    uv run python scripts/analysis_tables.py [--features-dir results/cache/features]

Outputs (results/):
  benchmark.csv                         every method: mean +- SD over grouped CV folds, in-scope scores, cost
  schedule_ablation_folds012.csv        old vs fixed fine-tune recipe, paired on folds 0-2 (NOTE: the "fixed" commit
                                        changed BOTH the LR warm-up restart AND BatchNorm freezing -> confounded)
  phase1/label_issues_v2.csv            confident-learning label issues (OOF ensemble of 3 SigLIP probes)
  phase1/cleaning_ablation_siglip_so400m.csv   train-time exclusion of OOD-flagged / label-issue rows
  phase1/fusion_ablation.csv            probe + zero-shot fusion grid (zero-shot scores from open_clip preprocessing)
  leakage_simulation.csv                legacy-style split vs grouped CV, from frozen-feature probes
Requires the cached dev features written by `compclass.features` (results/cache/features/<model>-<fp>-<ids>.npy,
present on the GPU PC) for the cleaning and leakage tables; the other tables only need files tracked in git.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R = Path(__file__).resolve().parents[1]
RES = R / "results"


def load_dev():
    man = json.loads((R / "data/v2/manifest.json").read_text())
    flag = {r["id"]: "ood_suspect" in r["flags"] for r in man["rows"]}
    dev = [r for r in man["rows"] if r["split"] == "dev"]
    return man, flag, dev


def inscope(ids, y, pred, flag):
    ii = np.array([not flag[i] for i in ids])
    return accuracy_score(y[ii], pred[ii]), f1_score(y[ii], pred[ii], average="macro")


def scores(z):
    return z["proba"] if "proba" in z.files else z["sim"] if "sim" in z.files else z["probs"]


def probe_zeroshot_table(runs_path: Path, oof_dir: Path, flag) -> pd.DataFrame:
    runs = pd.read_json(runs_path, lines=True)
    t = runs.groupby(["method", "name"]).agg(acc=("accuracy", "mean"), acc_sd=("accuracy", "std"), f1=("macro_f1", "mean"),
                                             f1_sd=("macro_f1", "std"), folds=("fold", "count")).reset_index()
    t = t.rename(columns={"method": "regime"})
    for i, r in t.iterrows():
        f = oof_dir / f"{r['name']}.npz"
        if f.exists():
            z = np.load(f)
            t.loc[i, ["inscope_acc", "inscope_f1"]] = inscope(z["ids"], z["y"], scores(z).argmax(1), flag)
    t["schedule"] = "-"
    return t


def finetune_table(df: pd.DataFrame, schedule: str, pred_dirs: list[Path], flag) -> pd.DataFrame:
    out = []
    for name, g in df.groupby("name"):
        files = sorted(f for d in pred_dirs for f in glob.glob(str(d / f"{name}-*_fold*.npz"))
                       if int(f.split("_fold")[-1][0]) in set(g.fold))
        ia = ifl = np.nan
        if files:
            zs = [np.load(f) for f in files]
            ids = np.concatenate([z["ids"] for z in zs]); y = np.concatenate([z["y"] for z in zs])
            p = np.concatenate([z["probs"].argmax(1) for z in zs]); ia, ifl = inscope(ids, y, p, flag)
        tp = g["trainable_params"].iloc[0] if "trainable_params" in g and g["trainable_params"].notna().all() else g["params"].iloc[0]
        out.append(dict(name=name, regime="fine-tune", schedule=schedule, folds=len(g), acc=g.accuracy.mean(),
                        acc_sd=g.accuracy.std(), f1=g.macro_f1.mean(), f1_sd=g.macro_f1.std(), inscope_acc=ia, inscope_f1=ifl,
                        params_M=g.params.iloc[0] / 1e6, trainable_M=tp / 1e6, train_min_per_fold=g.train_time_s.mean() / 60,
                        peak_vram_gb=g.peak_vram_mb.mean() / 1024, best_epoch=g.best_epoch.mean()))
    return pd.DataFrame(out)


def label_issues(oof_dir: Path, dev, flag) -> tuple[pd.DataFrame, np.ndarray]:
    """Confident learning (Northcutt et al. 2021), simplified: a row is an issue if another class's OOF probability
    reaches that class's average self-confidence while the given label's probability is below its own average."""
    top = ["probe_siglip_so400m_384", "probe_siglip2_so400m_384", "probe_siglip2_b16"]
    Z = [np.load(oof_dir / f"{k}.npz") for k in top]
    ids, y = Z[0]["ids"], Z[0]["y"]
    P = np.mean([z["proba"] for z in Z], 0); classes = list(Z[0]["classes"])
    t = np.array([P[y == j, j].mean() for j in range(len(classes))])
    selfc = P[np.arange(len(y)), y]
    Q = P.copy(); Q[np.arange(len(y)), y] = -1
    alt, altp = Q.argmax(1), Q.max(1)
    issue = (altp >= t[alt]) & (selfc < t[y])
    path = {r["id"]: r["path"] for r in dev}
    df = pd.DataFrame(dict(id=ids, label=[classes[i] for i in y], pred=[classes[i] for i in alt], self_conf=selfc,
                           alt_conf=altp, issue=issue, ood=[flag[i] for i in ids]))
    df = df[df.issue].sort_values("self_conf"); df["path"] = df.id.map(path)
    return df, issue


def load_feat(features_dir: Path, key: str) -> np.ndarray:
    f = [x for x in glob.glob(str(features_dir / f"{key}-*.npy")) if "-tta" not in x and "-random" not in x]
    assert len(f) == 1, (key, f)
    return np.load(f[0])


def cv_probe(X, y, fold, drop, ood, C):
    pred = np.zeros(len(y), int)
    for k in range(5):
        tr, va = (fold != k) & ~drop, fold == k
        pred[va] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[tr], y[tr]).predict(X[va])
    ins = ~ood
    return dict(all_acc=accuracy_score(y, pred), all_f1=f1_score(y, pred, average="macro"),
                in_acc=accuracy_score(y[ins], pred[ins]), in_f1=f1_score(y[ins], pred[ins], average="macro"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", default=str(RES / "cache/features"))
    a = ap.parse_args()
    man, flag, dev = load_dev()
    p1, p1b, p2 = RES / "phase1/raw", RES / "phase1b", RES / "phase2"

    # 1. benchmark table
    pr = probe_zeroshot_table(p1 / "runs.jsonl", p1 / "oof", flag)
    tz = probe_zeroshot_table(p1b / "runs.jsonl", p1b / "oof", flag)
    fx = pd.read_json(p2 / "runs_fixed.jsonl", lines=True)
    old = pd.concat([pd.read_json(p2 / "raw_oldschedule" / f, lines=True) for f in ("runs_A.jsonl", "runs_B.jsonl")])
    bench = pd.concat([pr, tz, finetune_table(fx, "fixed", [p2 / "preds_fixed"], flag),
                       finetune_table(old, "old (bug)", [p2 / "raw_oldschedule/preds"], flag)], ignore_index=True)
    cols = ["name", "regime", "schedule", "folds", "acc", "acc_sd", "f1", "f1_sd", "inscope_acc", "inscope_f1", "params_M",
            "trainable_M", "train_min_per_fold", "peak_vram_gb", "best_epoch"]
    bench[cols].sort_values(["schedule", "f1"], ascending=[True, False]).round(4).to_csv(RES / "benchmark.csv", index=False)

    # 2. schedule ablation, paired on folds 0-2
    pp = old[["name", "fold", "macro_f1"]].rename(columns={"macro_f1": "old"}).merge(
        fx[fx.fold <= 2][["name", "fold", "macro_f1"]].rename(columns={"macro_f1": "fixed"}), on=["name", "fold"])
    pp["d"] = pp.fixed - pp.old
    pp.groupby("name").agg(old=("old", "mean"), fixed=("fixed", "mean"), delta=("d", "mean"),
                           folds_improved=("d", lambda s: int((s > 0).sum())), n=("d", "count")) \
        .sort_values("delta", ascending=False).round(4).to_csv(RES / "schedule_ablation_folds012.csv")

    # 3. label issues
    issues_df, issue = label_issues(p1 / "oof", dev, flag)
    issues_df.round(4).to_csv(RES / "phase1/label_issues_v2.csv", index=False)

    # 4. fusion grid (Phase-1 analysis; zero-shot similarities came from open_clip's own preprocessing)
    pz, zz = np.load(p1 / "oof/probe_siglip_so400m_384.npz"), np.load(p1 / "oof/zeroshot_siglip_so400m_384.npz")
    order = {i: k for k, i in enumerate(zz["ids"])}; S = zz["sim"][[order[i] for i in pz["ids"]]]
    yy, lp = pz["y"], np.log(np.clip(pz["proba"], 1e-9, 1))
    fo = np.array([r["fold"] for r in dev]); ins = np.array([not flag[i] for i in pz["ids"]])
    rows = []
    for T in (50, 100):
        lz = log_softmax(S * T, axis=1)
        for w in (0, 0.25, 0.5, 1.0, 2.0):
            p = (lp + w * lz).argmax(1)
            rows.append(dict(T=T, w=w, acc=accuracy_score(yy, p), f1=f1_score(yy, p, average="macro"),
                             inscope_acc=accuracy_score(yy[ins], p[ins]),
                             **{f"fold{k}_acc": accuracy_score(yy[fo == k], p[fo == k]) for k in range(5)}))
    pd.DataFrame(rows).round(4).to_csv(RES / "phase1/fusion_ablation.csv", index=False)

    # 5./6. need cached features
    fdir = Path(a.features_dir)
    if not glob.glob(str(fdir / "*.npy")):
        print(f"no cached features in {fdir}: skipping cleaning ablation and leakage simulation"); return
    cls = {c: i for i, c in enumerate(man["classes"])}
    y = np.array([cls[r["label"]] for r in dev]); fold = np.array([r["fold"] for r in dev])
    ood = np.array([flag[r["id"]] for r in dev])
    X = load_feat(fdir, "siglip_so400m_384")
    res = {nm: cv_probe(X, y, fold, mask, ood, C=0.01) for nm, mask in
           [("keep_all", np.zeros(len(y), bool)), ("drop_issues", issue), ("drop_ood", ood), ("drop_ood+issues", ood | issue)]}
    pd.DataFrame([dict(variant=k, **v) for k, v in res.items()]).round(4) \
        .to_csv(RES / "phase1/cleaning_ablation_siglip_so400m.csv", index=False)

    # Leakage simulation. The legacy 70/15/15 split was drawn over raw FILES, so byte-identical copies of one image
    # could land in train and in val/test. In v2 such copies are collapsed into one row whose src_paths list every
    # copy. A dev row is "in legacy train" if any copy was in legacy train, and "in legacy eval" if any copy was in
    # legacy val or test; rows in both are exactly the leaked images. Train a probe on legacy-train rows and score the
    # legacy-eval rows, then compare with the grouped-CV accuracy of the same probe.
    leg = json.loads((R / "data/splits/split_manifest.json").read_text())
    leg_of: dict[str, set] = {}
    for sp in ("train", "val", "test"):
        for e in leg[sp]:
            leg_of.setdefault(e["path"].split("electronic-component-recognizer/", 1)[1], set()).add(sp)
    sets = [set().union(*[leg_of.get(s, set()) for s in r["src_paths"]]) for r in dev]
    in_tr = np.array(["train" in s for s in sets]); in_ev = np.array([bool(s & {"val", "test"}) for s in sets])
    leak = in_tr & in_ev
    grouped = pr.set_index("name")["acc"]
    out = []
    for key, C in (("siglip_so400m_384", 0.003), ("convnext_tiny", 0.01), ("resnet50", 0.01)):
        Xk = load_feat(fdir, key)
        p = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Xk[in_tr], y[in_tr]).predict(Xk[in_ev])
        el = leak[in_ev]; legacy = accuracy_score(y[in_ev], p)
        out.append(dict(model=key, legacy_style_acc=legacy, legacy_eval_rows=int(in_ev.sum()), leaked_rows=int(el.sum()),
                        acc_on_leaked=accuracy_score(y[in_ev][el], p[el]), acc_on_clean=accuracy_score(y[in_ev][~el], p[~el]),
                        grouped_cv_acc=grouped[f"probe_{key}"], inflation_pts=100 * (legacy - grouped[f"probe_{key}"])))
    pd.DataFrame(out).round(4).to_csv(RES / "leakage_simulation.csv", index=False)
    print("wrote", [str(p.relative_to(R)) for p in (RES / "benchmark.csv", RES / "leakage_simulation.csv")])


if __name__ == "__main__":
    main()
