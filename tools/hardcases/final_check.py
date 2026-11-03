"""The one look at the demo stack, after the final candidate was picked on dev CV alone.

    python tools/hardcases/final_check.py fit H17       # final heads on ALL dev rows (no demo pictures read)
    python tools/hardcases/final_check.py once H17      # baseline vs candidate on apps/gpui-demo/demo_stack

``once`` refuses to run twice for the same candidate (a marker file records the look). Embeddings of the demo
pictures come from extract_views.py --manifest (same trunk, ORT CUDA EP, batch 1), run beforehand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ART, DEPLOYED, REPO, W_ZS, deployed_W, dev, embeddings, log_softmax, zs_logp

DEMO = REPO / "apps/gpui-demo/demo_stack"
ONCE = ART / "demo_once"
VIEWS5 = ("squash", "center", "crop100", "letterbox", "hflip")

# Final candidates: test view, training views, kNN blend. Picked on dev CV (decisions.tsv) before any look.
CANDIDATES = {
    "H11": {"test": "squash", "train": ("squash",), "knn": False},
    "H16": {"test": "squash", "train": VIEWS5, "knn": False},
    "H17": {"test": "squash", "train": ("squash",), "knn": True},
}
KNN_GRID = [0.0, 0.1, 0.25, 0.5, 1.0]


def knn_all(F_bank: np.ndarray, y_bank: np.ndarray, Q: np.ndarray, n_classes: int, k: int = 10,
            tau: float = 0.05) -> np.ndarray:
    Bn = F_bank / np.linalg.norm(F_bank, axis=1, keepdims=True)
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    S = Qn @ Bn.T
    top = np.argpartition(-S, k, axis=1)[:, :k]
    w = np.exp(np.take_along_axis(S, top, 1) / tau)
    votes = np.zeros((len(Q), n_classes))
    np.add.at(votes, (np.arange(len(top))[:, None], y_bank[top]), w)
    return np.log(votes / votes.sum(1, keepdims=True) + 1e-4)


def fit(hid: str) -> None:
    """Head on all dev rows as compclass.final.fit_head does: probe refit on every dev row, temperature and
    abstain threshold from the candidate's OOF scores, kNN weight by dev OOF accuracy over the same grid."""
    from compclass.features import fit_probe
    from compclass.final import fit_temperature, head_from_search

    c, d = CANDIDATES[hid], dev()
    views = [embeddings(v) for v in c["train"]]
    X = np.concatenate(views)
    s = fit_probe(X, np.concatenate([d.y] * len(views)), np.concatenate([d.groups] * len(views)))
    head = head_from_search(s, deployed_W())
    oof = np.load(ART / "oof" / f"scores_{hid}.npz")["scores"]
    temp = fit_temperature(oof, d.y)
    probs = np.exp(log_softmax(oof / temp))
    pred, conf = probs.argmax(1), probs.max(1)
    thr = next((float(t) for t in np.round(np.arange(0.0, 0.96, 0.05), 2)
                if (conf >= t).any() and (pred[conf >= t] == d.y[conf >= t]).mean() >= 0.90), 0.5)
    head |= {"temperature": np.float32(temp), "abstain_threshold": np.float32(thr), "knn_weight": np.float32(0.0)}
    if c["knn"]:
        from common import probe_oof
        from experiments import knn_logp

        F = embeddings(c["test"])
        lp, _ = probe_oof(F, c["test"])
        zs = W_ZS * zs_logp(F, deployed_W())
        knn = np.zeros((len(d.y), d.n_classes))
        for k in range(5):
            knn[d.fold == k] = knn_logp(F, d.fold != k, d.fold == k)[d.fold == k]
        accs = [float(((lp + zs + g * knn).argmax(1) == d.y).mean()) for g in KNN_GRID]
        head["knn_weight"] = np.float32(KNN_GRID[int(np.argmax(accs))])
        head["knn_bank"], head["knn_labels"] = F.astype(np.float32), d.y.astype(np.int64)
    out = ONCE / f"head_{hid}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **head)
    print(json.dumps({"candidate": hid, "C": head["C"], "temperature": temp, "abstain_threshold": thr,
                      "knn_weight": float(head["knn_weight"]), "train_rows": len(X)}))


def predict(head: dict, F: np.ndarray, n_classes: int) -> np.ndarray:
    z = (F - head["mu"]) / head["sd"]
    s = log_softmax(z @ head["coef"].T + head["intercept"]) + float(head["w_zs"]) * zs_logp(F, head["W"],
                                                                                           float(head["t_zs"]))
    if float(head.get("knn_weight", 0.0)):
        s = s + float(head["knn_weight"]) * knn_all(head["knn_bank"], head["knn_labels"], F, n_classes)
    return np.exp(log_softmax(s / float(head["temperature"])))


def once(hid: str) -> None:
    marker = ONCE / f"LOOKED_{hid}"
    if marker.exists():
        raise SystemExit(f"{marker} exists: the demo stack was already looked at for {hid}")
    meta = json.loads((DEMO / "demo_stack.json").read_text())
    files = json.loads((ONCE / "emb" / "ids.json").read_text())
    classes = dev().classes
    base_head = dict(np.load(DEPLOYED / "head.npz"))
    cand_head = dict(np.load(ONCE / f"head_{hid}.npz"))
    pb = predict(base_head, np.load(ONCE / "emb" / "center.npy"), len(classes))
    pc = predict(cand_head, np.load(ONCE / "emb" / f"{CANDIDATES[hid]['test']}.npy"), len(classes))
    by_file = {Path(f).name: i for i, f in enumerate(files)}

    def show(p: np.ndarray, head: dict, i: int) -> dict:
        j = int(p[i].argmax())
        return {"pred": classes[j], "conf": round(float(p[i, j]), 4),
                "unsure": float(p[i, j]) < float(head["abstain_threshold"])}

    def correct(r: dict, truth: str | None) -> bool:
        return (r["pred"] == truth and not r["unsure"]) if truth else r["unsure"]

    rows = []
    for im in meta["images"]:
        i, truth = by_file[im["file"]], im["true_label"]
        b, c = show(pb, base_head, i), show(pc, cand_head, i)
        rows.append({"file": im["file"], "section": im["section"], "truth": truth,
                     "recorded": {"pred": im["model_pred"], "conf": im["model_conf"], "unsure": im["model_unsure"]},
                     "baseline": b, "candidate": c, "baseline_ok": correct(b, truth),
                     "candidate_ok": correct(c, truth),
                     "baseline_true_prob": round(float(pb[i, classes.index(truth)]), 4) if truth else None,
                     "candidate_true_prob": round(float(pc[i, classes.index(truth)]), 4) if truth else None})
    marker.write_text("looked\n")
    out = ONCE / f"demo_once_{hid}.json"
    out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['file']:32s} {r['truth']!s:22s} recorded {r['recorded']['pred']:>21s} "
              f"base {r['baseline']['pred']:>21s} {r['baseline']['conf']:.3f}{' U' if r['baseline']['unsure'] else '  '} "
              f"cand {r['candidate']['pred']:>21s} {r['candidate']['conf']:.3f}{' U' if r['candidate']['unsure'] else '  '}")
    agree = sum(r["baseline"]["pred"] == r["recorded"]["pred"] for r in rows)
    print(f"baseline pipeline agrees with the app's recorded prediction on {agree}/{len(rows)}")
    for sec in meta["sections"]:
        rs = [r for r in rows if r["section"] == sec]
        print(f"{sec}: baseline {sum(r['baseline_ok'] for r in rs)}/{len(rs)}, "
              f"candidate {sum(r['candidate_ok'] for r in rs)}/{len(rs)}")


if __name__ == "__main__":
    {"fit": fit, "once": once}[sys.argv[1]](sys.argv[2])
