"""Run the frozen final model over EVERY file of the original Kaggle download and break the score down honestly.

    uv run python scripts/eval_original_kaggle.py        # on the GPU PC, from the repo root

Every one of the 3,530 original files is scored against its original folder label. Files are mapped to their data-v2
row (exact duplicates share one row/image); the 27 files of the 13 conflicting-label groups that v2 excluded are
scored from data/kaggle_eval_extra/. The headline "accuracy on the original dataset" is NOT a generalisation
estimate: 2,512 of the rows were used to fit the final head. The breakdown separates
  * test   - v2 sealed-test rows (never used for training or selection)  -> honest
  * dev    - rows the final head was trained on                           -> optimistic (training accuracy)
  * excluded - conflicting-label duplicates (one image, two folder labels) -> at most one label can be right
It also replays the LEGACY 70/15/15 split (data/splits/split_manifest.json) used by the old notebooks:
  * as_deployed  - the final model on legacy-test files (many were in its training data)
  * replay       - the same backbone + fused head, but the probe re-fitted ONLY on rows that contain a legacy-train
                   file, then scored on legacy-test files; directly comparable to the old notebooks' protocol
                   (old best: ConvNeXt-Base 0.7132), including its duplicate leakage, which is reported separately.
Outputs: results/kaggle_original/{summary.json, per_file.csv, row_probs.npz}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from compclass.data import Manifest, get_dev, get_test, load_manifest
from compclass.features import extract_features, fit_probe
from compclass.final import KEY, OUT, fused_scores, head_from_search, log_softmax

RES = Path("results/kaggle_original")


def legacy_membership() -> dict[str, str]:
    leg = json.loads(Path("data/splits/split_manifest.json").read_text())
    return {e["path"].split("electronic-component-recognizer/", 1)[1]: sp for sp in ("train", "val", "test") for e in leg[sp]}


def acc_block(df: pd.DataFrame, classes: list[str]) -> dict:
    if len(df) == 0:
        return {"n": 0}
    y, p = df.label_idx.to_numpy(), df.pred_idx.to_numpy()
    acc_ = df.conf >= df.thr
    return {"n": int(len(df)), "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=list(range(len(classes))), average="macro", zero_division=0)),
            "coverage": float(acc_.mean()), "accepted_accuracy": float((y[acc_] == p[acc_]).mean()) if acc_.any() else None}


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    m = load_manifest("data/v2/manifest.json")
    head = dict(np.load(OUT / "head.npz"))
    thr, temp = float(head["abstain_threshold"]), float(head["temperature"])
    dev, test = get_dev(m), get_test(m, confirm=True)  # test already spent once; this is reporting, not selection
    rows = dev + test
    F = np.concatenate([extract_features(KEY, dev, m), extract_features(KEY, test, m)])
    ex_rows = json.loads(Path("data/kaggle_eval_extra/excluded_rows.json").read_text())["rows"]
    mx = Manifest(m.classes, ex_rows, Path("data/kaggle_eval_extra"), "excluded-conflicts", 5, {})
    Fx = extract_features(KEY, ex_rows, mx)
    P = np.exp(log_softmax(fused_scores(np.concatenate([F, Fx]), head) / temp))
    np.savez(RES / "row_probs.npz", ids=np.array([r["id"] for r in rows]), probs=P[: len(rows)])

    legacy = legacy_membership()
    cls = m.class_to_idx
    recs = []
    for i, r in enumerate(rows + ex_rows):
        split = r.get("split", "excluded")
        ood = "ood_suspect" in r.get("flags", [])
        for sp in r["src_paths"]:
            lab = sp.split("/")[0]
            recs.append(dict(file=sp, label=lab, label_idx=cls[lab], row=r["id"], v2_split=split, ood_suspect=ood,
                             legacy_split=legacy.get(sp, "not_in_legacy"), pred=m.classes[int(P[i].argmax())],
                             pred_idx=int(P[i].argmax()), conf=float(P[i].max()), thr=thr))
    df = pd.DataFrame(recs)
    df["correct"] = df.label_idx == df.pred_idx
    df.drop(columns=["thr"]).to_csv(RES / "per_file.csv", index=False)

    summary: dict = {"n_files": int(len(df)), "n_rows": len(rows), "n_excluded_groups": len(ex_rows),
                     "abstain_threshold": thr, "all_files": acc_block(df, m.classes)}
    summary["by_v2_split"] = {s: acc_block(df[df.v2_split == s], m.classes) for s in ("test", "dev", "excluded")}
    summary["test_inscope"] = acc_block(df[(df.v2_split == "test") & ~df.ood_suspect], m.classes)
    summary["by_ood_flag"] = {"in_scope": acc_block(df[~df.ood_suspect & (df.v2_split != "excluded")], m.classes),
                              "ood_suspect": acc_block(df[df.ood_suspect], m.classes)}

    # legacy split replays
    lt = df[df.legacy_split == "test"]
    summary["legacy_test_as_deployed"] = acc_block(lt, m.classes)
    summary["legacy_test_as_deployed_unseen_rows_only"] = acc_block(lt[lt.v2_split == "test"], m.classes)
    row_legacy = df.groupby("row").legacy_split.agg(set)
    pos = {r["id"]: i for i, r in enumerate(rows)}
    tr_rows = [r for r in rows if "train" in row_legacy.get(r["id"], set())]
    itr = [pos[r["id"]] for r in tr_rows]
    search = fit_probe(F[itr], np.array([cls[r["label"]] for r in tr_rows]), np.array([r["cluster"] for r in tr_rows]))
    h2 = {**head_from_search(search, head["W"]), "temperature": head["temperature"]}
    P2 = np.exp(log_softmax(fused_scores(F, h2) / temp))
    lt_rows = lt[lt.v2_split != "excluded"].copy()
    lt_rows["pred_idx"] = [int(P2[pos[r]].argmax()) for r in lt_rows.row]
    lt_rows["conf"] = [float(P2[pos[r]].max()) for r in lt_rows.row]
    leaked = lt_rows.row.map(lambda r: "train" in row_legacy[r])
    summary["legacy_test_replay"] = {
        "description": "probe refit on rows containing a legacy-train file; scored on legacy-test files (old protocol)",
        "n_train_rows": len(tr_rows), "C": float(search.best_params_["logisticregression__C"]),
        "all": acc_block(lt_rows, m.classes),
        "leaked_copy_of_train": acc_block(lt_rows[leaked], m.classes),
        "clean_not_in_train": acc_block(lt_rows[~leaked], m.classes),
        "excluded_conflict_files_not_scored": int((lt.v2_split == "excluded").sum()),
        "old_best_for_reference": {"model": "ConvNeXt-Base strong recipe", "accuracy": 0.7132, "macro_f1": 0.6969,
                                   "source": "legacy/docs/EXPERIMENT_LOG.md"}}
    per_class = df[df.v2_split == "test"].groupby("label").correct.mean().round(4).to_dict()
    summary["test_per_class_file_accuracy"] = per_class
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "test_per_class_file_accuracy"}, indent=1))


if __name__ == "__main__":
    main()
