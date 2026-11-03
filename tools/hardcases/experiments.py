"""Hypotheses for the hard-cases study, one function each, scored on dev CV and logged to decisions.tsv.

    python tools/hardcases/experiments.py H0 [H1 ...]
"""

from __future__ import annotations

import sys

import numpy as np
from common import (
    ART,
    BASELINE_CV,
    PAIRS,
    REPO,
    T_ZS,
    W_ZS,
    baseline_pred,
    deployed_W,
    dev,
    embeddings,
    fit_probe_balanced,
    inner_probe_oof,
    log_softmax,
    nested,
    probe_oof,
    record,
    zs_logp,
)

HYPOTHESES = {}


def hypothesis(fn):
    HYPOTHESES[fn.__name__] = fn
    return fn


def fused(F: np.ndarray, name: str, W: np.ndarray | None = None, w: float = W_ZS, t: float = T_ZS) -> np.ndarray:
    lp, _ = probe_oof(F, name)
    return lp + w * zs_logp(F, deployed_W() if W is None else W, t)


@hypothesis
def H0():
    scores = fused(embeddings(), "center")
    if not (scores.argmax(1) == baseline_pred()).all():
        raise SystemExit("H0 does not reproduce the frozen harness's baseline predictions")
    row = record("H0", "baseline: deployed head recipe on the frozen harness's center-crop embeddings",
                 "reproduce 0.8268 and the baseline pair counts before any change", "none", scores,
                 verdict_override="BASELINE")
    assert abs(row["cv_acc"] - BASELINE_CV) < 1e-12, row["cv_acc"]


@hypothesis
def H1():
    """Zero-shot weight: the zero-shot head knows 'LED', 'DIP switch', 'Arduino' from web text; more weight may
    pull web-style pictures towards the right class. Only w * T moves the argmax (log_softmax(T cos) differs from
    T cos by a per-row constant), so T stays 50 and w carries the sweep. Grid rows are oracle (picked on the scored
    folds); H1-nested picks w inside each training split."""
    F, W = embeddings(), deployed_W()
    lp, Cs = probe_oof(F, "center")
    zs = zs_logp(F, W)
    grid = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    for w in grid:
        if w != W_ZS:
            record(f"H1-w{w}", f"zero-shot weight {w} (T 50)", "grid point (oracle)", "free (one scalar)",
                   lp + w * zs, note="grid point, picked on the scored folds")
    inner = inner_probe_oof(F, "center", Cs)
    scores, chosen = nested(lambda w: lp + w * zs, lambda k, w: inner[k] + w * zs, grid)
    record("H1-nested", "zero-shot weight chosen by inner CV per fold", "honest estimate of tuning w",
           "free (one scalar)", scores, note=f"w chosen per fold: {chosen}")


@hypothesis
def H2():
    """Class-balanced probe weights: rarer classes (led 106, transformer 110 rows vs switch 152) get equal pull."""
    F = embeddings()
    lp, _ = probe_oof(F, "center_balanced", fit=fit_probe_balanced)
    record("H2", "class-balanced logistic-regression weights", "led/transformer are among the smallest classes",
           "free (refit head.npz)", lp + W_ZS * zs_logp(F, deployed_W()))


def knn_logp(F: np.ndarray, train: np.ndarray, query: np.ndarray, k: int = 10, tau: float = 0.05) -> np.ndarray:
    """Soft cosine kNN: neighbours from ``train`` rows vote exp(cos / tau) for their class; [N, C], NaN off-query."""
    d = dev()
    Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    S = Fn[query] @ Fn[train].T
    top = np.argpartition(-S, k, axis=1)[:, :k]
    w = np.exp(np.take_along_axis(S, top, 1) / tau)
    votes = np.zeros((query.sum(), d.n_classes))
    np.add.at(votes, (np.arange(len(top))[:, None], d.y[train][top]), w)
    out = np.full((len(d.y), d.n_classes), np.nan)
    out[query] = np.log(votes / votes.sum(1, keepdims=True) + 1e-4)
    return out


@hypothesis
def H3():
    """kNN head blended in: a memory of near-identical training pictures can fix a confusion the linear probe
    smooths over (an SMD inductor that looks like other SMD inductors, not like a diode). Blend weight nested."""
    F, d = embeddings(), dev()
    lp, Cs = probe_oof(F, "center")
    zs = W_ZS * zs_logp(F, deployed_W())
    outer = np.zeros((len(d.y), d.n_classes))
    for k in range(5):
        outer[d.fold == k] = knn_logp(F, d.fold != k, d.fold == k)[d.fold == k]
    inner_lp = inner_probe_oof(F, "center", Cs)
    inner_knn = np.full((5, len(d.y), d.n_classes), np.nan)
    for k in range(5):
        for j in range(5):
            if j != k:
                tr = (d.fold != k) & (d.fold != j)
                inner_knn[k][d.fold == j] = knn_logp(F, tr, d.fold == j)[d.fold == j]
    grid = [0.0, 0.1, 0.25, 0.5, 1.0]
    scores, chosen = nested(lambda g: lp + zs + g * outer, lambda k, g: inner_lp[k] + zs + g * inner_knn[k], grid)
    record("H3", "soft kNN (k 10, tau 0.05) blended into the fused scores, weight by inner CV",
           "instance memory for subtype pictures", "a 2,512 x 1152 memory bank in the graph", scores,
           note=f"weight chosen per fold: {chosen}")
    record("H3-g0.5", "soft kNN blended at weight 0.5 (oracle)", "landscape", "memory bank", lp + zs + 0.5 * outer,
           note="grid point, picked on the scored folds")


def external_logp(kind: str) -> np.ndarray:
    """Calibrated OOF log-probs of an earlier, independently trained dev-CV model, in dev-row order."""
    import glob

    from compclass.final import fit_temperature

    d = dev()
    if kind == "convnext":
        files = sorted(glob.glob(str(REPO / "results/phase2/preds_fixed/convnext_tiny-bb9c72a49446_fold*.npz")))
        z = [np.load(f) for f in files]
        pos = {i: n for n, i in enumerate(d.ids)}
        p = np.zeros((len(d.y), d.n_classes))
        for zz in z:
            p[[pos[i] for i in zz["ids"]]] = zz["probs"]
    else:
        z = np.load(REPO / f"results/phase1/raw/oof/probe_{kind}.npz")
        assert (z["ids"] == d.ids).all()
        p = z["proba"]
    lp = np.log(np.clip(p, 1e-8, 1))
    return log_softmax(lp / fit_temperature(lp, d.y))


@hypothesis
def H4():
    """Ensemble with a fine-tuned ConvNeXt-T (the app's B family; Phase-2 per-fold fine-tunes on the same folds).
    A CNN fine-tuned end to end sees texture and shape cues a frozen ViT probe may weigh differently; the app saw
    B right on the Arduino picture. Product of calibrated experts, weight 1 fixed a priori."""
    d = dev()
    base = np.load(ART / "oof" / "scores_H0.npz")["scores"]
    from compclass.final import fit_temperature

    base = log_softmax(base / fit_temperature(base, d.y))
    cx = external_logp("convnext")
    record("H4", "deployed x fine-tuned ConvNeXt-T, calibrated log-prob sum (1:1)",
           "a second, end-to-end trained architecture", "+1 ConvNeXt-T pass (~5 ms GPU), 110 MB", base + cx)
    for g in (0.25, 0.5):
        record(f"H4-g{g}", f"deployed + {g} x ConvNeXt-T log-probs (oracle)", "landscape", "+1 ConvNeXt-T pass",
               base + g * cx, note="grid point, picked on the scored folds")


@hypothesis
def H5():
    """Ensemble with a SigLIP 2 So400m probe (Phase-1 OOF, same folds): a second web-scale backbone with
    different training data; the screen for a 'different backbone' before paying for one."""
    d = dev()
    from compclass.final import fit_temperature

    base = np.load(ART / "oof" / "scores_H0.npz")["scores"]
    base = log_softmax(base / fit_temperature(base, d.y))
    s2 = external_logp("siglip2_so400m_384")
    record("H5", "deployed x SigLIP 2 So400m probe, calibrated log-prob sum (1:1)",
           "a second web-scale backbone", "+1 So400m pass (2x latency), +1.7 GB", base + s2,
           note="SigLIP 2 OOF from Phase 1 (carlpc2 torch features), not re-extracted")


def specialist_scores(F: np.ndarray, base: np.ndarray, Cs: list[float]) -> np.ndarray:
    """Where a row's top two classes are a hard pair, a binary probe trained only on that pair's rows decides
    their order. Other rows keep their scores."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = dev()
    out = base.copy()
    top2 = np.argsort(-base, 1)[:, :2]
    for a, b in PAIRS:
        ia, ib = d.idx(a), d.idx(b)
        for k in range(5):
            tr = (d.fold != k) & np.isin(d.y, [ia, ib])
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=Cs[k], max_iter=3000)).fit(F[tr], d.y[tr])
            rows = np.where((d.fold == k) & (np.sort(top2, 1) == sorted([ia, ib])).all(1))[0]
            if len(rows):
                win = clf.predict(F[rows])
                hi, lo = np.maximum(base[rows, ia], base[rows, ib]), np.minimum(base[rows, ia], base[rows, ib])
                out[rows, ia] = np.where(win == ia, hi, lo)
                out[rows, ib] = np.where(win == ib, hi, lo)
    return out


@hypothesis
def H6():
    """Specialist pair heads: a binary probe per hard pair, trained only on those two classes, re-orders rows
    whose top two are that pair. A two-class fit can use directions the 20-class fit spends elsewhere."""
    F = embeddings()
    lp, Cs = probe_oof(F, "center")
    base = lp + W_ZS * zs_logp(F, deployed_W())
    record("H6", "five binary pair specialists re-order rows whose top two are a hard pair",
           "pair-specific decision boundaries", "5 extra 1152-d linear heads + routing in the graph",
           specialist_scores(F, base, Cs))


def text_W(name: str) -> tuple[np.ndarray, list[np.ndarray]]:
    z = np.load(ART / "text" / f"{name}.npz")
    return z["W"], [z[f"P_{i}"] for i in range(len(z["W"]))]


def zs_max_logp(F: np.ndarray, P: list[np.ndarray], t: float = T_ZS) -> np.ndarray:
    """Zero-shot head with one vector per prompt: a class scores its best-matching prompt (max cosine)."""
    fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    return log_softmax(t * np.stack([(fn @ p.T).max(1) for p in P], 1))


def zs_only_acc(z: np.ndarray) -> float:
    return float((z.argmax(1) == dev().y).mean())


def prompt_hyp(hid: str, what: str, why: str, zs: np.ndarray) -> None:
    F = embeddings()
    lp, Cs = probe_oof(F, "center")
    inner = inner_probe_oof(F, "center", Cs)
    base_zs = zs_logp(F, deployed_W())
    note = f"zero-shot alone {zs_only_acc(zs):.4f} (deployed prompts {zs_only_acc(base_zs):.4f})"
    record(hid, what + " (w 0.25)", why, "free (new W in the graph)", lp + W_ZS * zs, note=note)
    grid = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    scores, chosen = nested(lambda w: lp + w * zs, lambda k, w: inner[k] + w * zs, grid)
    record(hid + "-nested", what + ", w by inner CV", why, "free (new W in the graph)", scores,
           note=note + f"; w chosen per fold: {chosen}")


@hypothesis
def H7():
    """Sixteen generic templates instead of six: a better-averaged text vector per class."""
    W, _ = text_W("generic")
    prompt_hyp("H7", "16 generic templates, class names as deployed", "prompt ensembling", zs_logp(embeddings(), W))


@hypothesis
def H8():
    """Subtype prompts averaged per class: 'a DIP switch', 'an SMD power inductor', 'an LED with long legs',
    'an Arduino board', ... (4-6 forms for each of the 20 classes x 3 templates)."""
    W, _ = text_W("subtypes")
    prompt_hyp("H8", "subtype prompts, mean per class", "name the physical forms the class name hides",
               zs_logp(embeddings(), W))


@hypothesis
def H9():
    """Subtype prompts, max over prompts: a picture only has to match one form of its class."""
    _, P = text_W("subtypes")
    prompt_hyp("H9", "subtype prompts, max over prompts per class", "multi-prototype zero-shot",
               zs_max_logp(embeddings(), P))


@hypothesis
def H9b():
    """Deployed prompts plus subtype prompts in one mean per class."""
    W, _ = text_W("subtypes_named")
    prompt_hyp("H9b", "deployed + subtype prompts, mean per class", "keep the class-name prompts, add forms",
               zs_logp(embeddings(), W))


def view_hyp(hid: str, what: str, why: str, cost: str, test: np.ndarray, train: list[np.ndarray],
             cache: str, parent: str = "H0") -> None:
    lp, _ = probe_oof(test, cache, train_F=train)
    record(hid, what, why, cost, lp + W_ZS * zs_logp(test, deployed_W()), parent=parent)


@hypothesis
def H10():
    """Horizontal-flip TTA with fusion (Phase 1b tried it for the probe alone: -0.5 pt)."""
    F = 0.5 * (embeddings() + embeddings("hflip"))
    view_hyp("H10", "hflip TTA: train and test on mean(center, flip) embeddings", "pose invariance",
             "2 trunk passes per picture", F, [F], "tta_flip")


@hypothesis
def H10b():
    """Flip as training augmentation only: twice the training rows, one pass at run time."""
    view_hyp("H10b", "train on center + flipped rows, test on center", "pose invariance at no run-time cost",
             "free (refit head.npz)", embeddings(), [embeddings(), embeddings("hflip")], "aug_flip")


@hypothesis
def H11():
    """SigLIP's own preprocessing is a squash resize of the whole picture to 384 x 384; the deployed crop
    (short side 427, center 384) drops about 10% of each border and any off-centre part of a wide picture."""
    F = embeddings("squash")
    view_hyp("H11", "squash: whole picture resized to 384 x 384 (train and test)", "SigLIP's native input, nothing cropped",
             "free at run time; app preprocessing + golden file change", F, [F], "squash")


@hypothesis
def H12():
    F = embeddings("letterbox")
    view_hyp("H12", "letterbox: whole picture, long side 384, grey padding (train and test)",
             "nothing cropped, aspect kept", "free at run time; app preprocessing + golden file change", F, [F],
             "letterbox")


@hypothesis
def H13():
    F = embeddings("crop100")
    view_hyp("H13", "crop_pct 1.0: short side 384, center 384 (train and test)", "less border lost",
             "free at run time; preprocess.json + golden file change", F, [F], "crop100")


@hypothesis
def H14():
    """Two views at test and train: the center crop for detail, the squash for the whole scene."""
    F = 0.5 * (embeddings() + embeddings("squash"))
    view_hyp("H14", "mean(center, squash) embeddings (train and test)", "detail + whole scene",
             "2 trunk passes per picture", F, [F], "center_squash")


@hypothesis
def H15():
    """Zero-shot weight on squash features, by inner CV (the probe got better, so the best blend may move)."""
    F, W = embeddings("squash"), deployed_W()
    lp, Cs = probe_oof(F, "squash")
    zs = zs_logp(F, W)
    inner = inner_probe_oof(F, "squash", Cs)
    grid = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    scores, chosen = nested(lambda w: lp + w * zs, lambda k, w: inner[k] + w * zs, grid)
    record("H15", "squash + zero-shot weight by inner CV", "re-tune fusion on the better features",
           "free (one scalar)", scores, note=f"w chosen per fold: {chosen}", parent="H11")
    record("H15-w1.0", "squash + zero-shot weight 1.0 (oracle)", "landscape", "free (one scalar)", lp + zs,
           note="grid point, picked on the scored folds", parent="H11")


@hypothesis
def H16():
    """Squash at test; train on every whole-or-cropped view already extracted (squash, center, crop100,
    letterbox, hflip) as extra rows: view augmentation for the probe, one pass at run time."""
    views = [embeddings(v) for v in ("squash", "center", "crop100", "letterbox", "hflip")]
    view_hyp("H16", "squash test; probe trained on 5 views per training image", "view augmentation",
             "free at run time; app preprocessing + golden file change", views[0], views, "squash_trainviews5",
             parent="H11")


def knn_nested(hid: str, what: str, F: np.ndarray, lp: np.ndarray, inner_lp: np.ndarray, parent: str,
               cost: str) -> None:
    d = dev()
    zs = W_ZS * zs_logp(F, deployed_W())
    outer = np.zeros((len(d.y), d.n_classes))
    for k in range(5):
        outer[d.fold == k] = knn_logp(F, d.fold != k, d.fold == k)[d.fold == k]
    inner_knn = np.full((5, len(d.y), d.n_classes), np.nan)
    for k in range(5):
        for j in range(5):
            if j != k:
                inner_knn[k][d.fold == j] = knn_logp(F, (d.fold != k) & (d.fold != j), d.fold == j)[d.fold == j]
    grid = [0.0, 0.1, 0.25, 0.5, 1.0]
    scores, chosen = nested(lambda g: lp + zs + g * outer, lambda k, g: inner_lp[k] + zs + g * inner_knn[k], grid)
    record(hid, what, "instance memory on the better view", cost, scores, note=f"weight chosen per fold: {chosen}",
           parent=parent)


@hypothesis
def H19():
    """H16 (squash, 5-view training) + kNN blend, weight by inner CV."""
    views = [embeddings(v) for v in ("squash", "center", "crop100", "letterbox", "hflip")]
    lp, Cs = probe_oof(views[0], "squash_trainviews5", train_F=views)
    inner = inner_probe_oof(views[0], "squash_trainviews5", Cs, train_F=views)
    knn_nested("H19", "H16 + soft kNN blend, weight by inner CV", views[0], lp, inner, "H16",
               "memory bank in the graph; app preprocessing change")


@hypothesis
def H17():
    """kNN blend on squash features (H3's instance memory on the better view), weight by inner CV."""
    F, d = embeddings("squash"), dev()
    lp, Cs = probe_oof(F, "squash")
    zs = W_ZS * zs_logp(F, deployed_W())
    outer = np.zeros((len(d.y), d.n_classes))
    for k in range(5):
        outer[d.fold == k] = knn_logp(F, d.fold != k, d.fold == k)[d.fold == k]
    inner_lp = inner_probe_oof(F, "squash", Cs)
    inner_knn = np.full((5, len(d.y), d.n_classes), np.nan)
    for k in range(5):
        for j in range(5):
            if j != k:
                inner_knn[k][d.fold == j] = knn_logp(F, (d.fold != k) & (d.fold != j), d.fold == j)[d.fold == j]
    grid = [0.0, 0.1, 0.25, 0.5, 1.0]
    scores, chosen = nested(lambda g: lp + zs + g * outer, lambda k, g: inner_lp[k] + zs + g * inner_knn[k], grid)
    record("H17", "squash + soft kNN blend, weight by inner CV", "instance memory on the better view",
           "memory bank in the graph; app preprocessing change", scores, note=f"weight chosen per fold: {chosen}",
           parent="H11")


@hypothesis
def H18():
    """Squash x SigLIP 2 So400m probe (Phase-1 OOF, center-crop features): screen for a second backbone."""
    d = dev()
    from compclass.final import fit_temperature

    base = np.load(ART / "oof" / "scores_H11.npz")["scores"]
    base = log_softmax(base / fit_temperature(base, d.y))
    record("H18", "squash deployed x SigLIP 2 So400m probe, calibrated log-prob sum (1:1)",
           "a second web-scale backbone on top of the better view", "+1 So400m pass (2x latency), +1.7 GB",
           base + external_logp("siglip2_so400m_384"),
           note="SigLIP 2 OOF from Phase 1 (center crop, carlpc2 torch features)", parent="H11")


@hypothesis
def H20():
    """Squash at test; probe trained on squash + four random-resized-crop / flip / colour-jitter views per
    training image (augmentation the frozen probe never saw), one pass at run time."""
    views = [embeddings(v) for v in ("squash", "aug0", "aug1", "aug2", "aug3")]
    view_hyp("H20", "squash test; probe trained on squash + 4 random augmented views", "training-time augmentation",
             "free at run time; app preprocessing + golden file change", views[0], views, "squash_aug4",
             parent="H11")


@hypothesis
def H21():
    """H20 + kNN blend (bank: squash views), weight by inner CV."""
    views = [embeddings(v) for v in ("squash", "aug0", "aug1", "aug2", "aug3")]
    lp, Cs = probe_oof(views[0], "squash_aug4", train_F=views)
    inner = inner_probe_oof(views[0], "squash_aug4", Cs, train_F=views)
    knn_nested("H21", "H20 + soft kNN blend, weight by inner CV", views[0], lp, inner, "H20",
               "memory bank in the graph; app preprocessing change")


if __name__ == "__main__":
    for h in sys.argv[1:]:
        HYPOTHESES[h]()
