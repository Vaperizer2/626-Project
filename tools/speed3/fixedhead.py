"""Diagnostic without the refit noise: the head fitted on the fp32 teacher's embeddings, applied to each candidate's.

    fixedhead.py <teacher.npy> <name>=<embeddings.npy> ...

Per development fold k, `compclass.features.fit_probe` + `compclass.final.fused_scores` (the harness recipe) is fitted
once on the teacher's embeddings of the other folds, then scores fold k of the teacher and of every candidate. Prints
per candidate the pooled accuracy and the agreement with the teacher's own predictions. The harness instead refits
the head on each candidate, so C-selection flips move it by up to 0.6 pp per fold; this fixes the head and measures
only how the embeddings move the decisions. Not a result.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common import REPO, dev_rows

sys.path.insert(0, str(REPO / "tools/speed"))
import evaluate  # noqa: E402


def main() -> None:
    from compclass.features import fit_probe
    from compclass.final import fused_scores, head_from_search

    T = np.load(sys.argv[1])
    cands = {k: np.load(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])}
    m, rows = dev_rows()
    y, fold, groups = evaluate.labels(m, rows)
    W = np.load(evaluate.DEPLOYED / "head.npz")["W"]
    pred = {k: np.zeros(len(y), int) for k in ["teacher", *cands]}
    for k in range(m.n_folds):
        tr, va = fold != k, fold == k
        head = head_from_search(fit_probe(T[tr], y[tr], groups[tr]), W)
        pred["teacher"][va] = fused_scores(T[va], head).argmax(1)
        for n, F in cands.items():
            pred[n][va] = fused_scores(F[va], head).argmax(1)
    for n, p in pred.items():
        print(json.dumps({"name": n, "acc": round(float((p == y).mean()), 5),
                          "agree_teacher": round(float((p == pred["teacher"]).mean()), 5),
                          "n_flip_vs_teacher": int((p != pred["teacher"]).sum())}), flush=True)


if __name__ == "__main__":
    main()
