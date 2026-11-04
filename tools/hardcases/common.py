"""Shared lever for the hard-cases study: dev rows, cached embeddings, the deployed head recipe per fold,
and one scored decision row per hypothesis.

Selection reads only the dev split of data/v2 (grouped 5-fold CV, as in tools/speed/evaluate.py). The sealed
test, data/web_holdout_v1 and apps/gpui-demo/demo_stack are never opened here.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MAIN = Path("/home/vaper/Work/Prototypes/626-Project")
MANIFEST = MAIN / "data/v2/manifest.json"
FINGERPRINT = "431315e87d2612cb"
DEPLOYED = MAIN / "exports/final"
ART = Path("/home/vaper/Work/Prototypes/626-artifacts/hardcases")
BASELINE_WORK = Path("/home/vaper/Work/Prototypes/626-artifacts/harness/work/baseline_fp32/6aee2bba0660")
BASELINE_CV = 0.8268299974231951
DECISIONS = ART / "decisions.tsv"
T_ZS, W_ZS = 50.0, 0.25

# (true class, wrongly predicted class): the five hard-case failure types, as directed confusions.
PAIRS = [("led", "diode"), ("transformer", "inductor coil"), ("switch", "ic chip"), ("inductor coil", "diode"),
         ("microcontroller board", "breadboard")]
PAIR_KEYS = ["led>diode", "transformer>inductor", "switch>ic", "inductor>diode", "mcu>breadboard"]

COLUMNS = ["id", "hypothesis", "why", "deploy_cost", "cv_acc", "cv_sd", "delta_pp", "cv_f1", "folds",
           "inscope_acc", "ood_acc",
           "mcnemar_b", "mcnemar_c", "mcnemar_p", *PAIR_KEYS, "pair_total", "pair_reverse", "pair_mass",
           "pair_mass_p", "coverage", "accepted_acc", "parent", "vs_parent", "verdict", "note"]


@dataclass(frozen=True)
class Dev:
    ids: np.ndarray
    y: np.ndarray
    fold: np.ndarray
    groups: np.ndarray
    ood: np.ndarray  # CLIP-flagged atypical rows (manifest flag ood_suspect)
    classes: list[str]

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def idx(self, name: str) -> int:
        return self.classes.index(name)


@cache
def dev() -> Dev:
    from compclass.data import get_dev, load_manifest

    m = load_manifest(MANIFEST)
    if m.fingerprint != FINGERPRINT:
        raise SystemExit(f"manifest fingerprint {m.fingerprint} != {FINGERPRINT}")
    rows = get_dev(m)
    return Dev(np.array([r["id"] for r in rows]), np.array([m.class_to_idx[r["label"]] for r in rows]),
               np.array([r["fold"] for r in rows]), np.array([r["cluster"] for r in rows]),
               np.array(["ood_suspect" in r["flags"] for r in rows]), m.classes)


def dev_rows():
    from compclass.data import get_dev, load_manifest

    m = load_manifest(MANIFEST)
    return m, get_dev(m)


def embeddings(view: str = "center") -> np.ndarray:
    """Dev embeddings [N, 1152] for a named view. ``center`` is the frozen harness's own baseline extraction."""
    if view == "center":
        return np.load(BASELINE_WORK / "embeddings.npy")
    return np.load(ART / "emb" / f"{view}.npy")


def deployed_W() -> np.ndarray:
    return np.load(DEPLOYED / "head.npz")["W"]


def baseline_pred() -> np.ndarray:
    f = np.load(BASELINE_WORK / "flags.npz")
    if not (f["ids"] == dev().ids).all():
        raise SystemExit("baseline flags are not in dev-row order")
    return f["pred"]


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def zs_logp(F: np.ndarray, W: np.ndarray, t: float = T_ZS) -> np.ndarray:
    fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    return log_softmax(t * fn @ W.T)


def probe_oof(F: np.ndarray, name: str | None = None, train_F: np.ndarray | list | None = None,
              fit=None) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold probe log-probabilities [N, C] of the deployed probe recipe (compclass.features.fit_probe).

    ``train_F`` replaces the features the probe is fitted on (same rows, for example another view); ``train_F``
    may also be a list of views, stacked as extra training rows of the same images. ``fit`` replaces
    ``compclass.features.fit_probe`` (same signature). Results are cached under ART/oof/probe_<name>.npz when
    ``name`` is given.
    """
    out = ART / "oof" / f"probe_{name}.npz" if name else None
    if out and out.exists():
        z = np.load(out)
        return z["lp"], list(z["C"])
    from compclass.features import fit_probe

    fit = fit or fit_probe
    d = dev()
    views = train_F if isinstance(train_F, list) else [train_F if train_F is not None else F]
    lp = np.zeros((len(d.y), d.n_classes), np.float32)
    Cs = []
    for k in range(5):
        tr, va = d.fold != k, d.fold == k
        X = np.concatenate([v[tr] for v in views])
        yy = np.concatenate([d.y[tr]] * len(views))
        gg = np.concatenate([d.groups[tr]] * len(views))
        s = fit(X, yy, gg)
        lp[va] = log_softmax(s.decision_function(F[va]))
        Cs.append(float(s.best_params_["logisticregression__C"]))
        print(f"  probe {name or ''} fold {k}: C={Cs[-1]} acc {(lp[va].argmax(1) == d.y[va]).mean():.4f}", flush=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, lp=lp, C=np.array(Cs))
    return lp, Cs


def inner_probe_oof(F: np.ndarray, name: str, Cs: list[float], train_F: list | None = None) -> np.ndarray:
    """[5, N, C] inner out-of-fold probe log-probs for nested selection. Slice k scores every row outside fold k
    with a probe fitted on the three other training folds, at the C the outer fit chose for fold k; fold k's
    own rows stay NaN, so a knob tuned on slice k never sees them. ``train_F`` stacks extra training views as in
    ``probe_oof``."""
    out = ART / "oof" / f"inner_{name}.npy"
    if out.exists():
        return np.load(out)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = dev()
    views = train_F or [F]
    inner = np.full((5, len(d.y), d.n_classes), np.nan, np.float32)
    for k in range(5):
        for j in range(5):
            if j == k:
                continue
            tr = (d.fold != k) & (d.fold != j)
            va = d.fold == j
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=Cs[k], max_iter=3000))
            lr.fit(np.concatenate([v[tr] for v in views]), np.concatenate([d.y[tr]] * len(views)))
            inner[k, va] = log_softmax(lr.decision_function(F[va]))
        print(f"  inner {name} outer fold {k} done", flush=True)
    np.save(out, inner)
    return inner


def nested(outer, inner, grid: list) -> tuple[np.ndarray, list]:
    """Nested selection of a knob: for each outer fold k pick the grid value with the best inner-OOF accuracy on
    the training folds (``inner(k, value)`` -> scores for all rows, NaN on fold k), and take fold k's outer
    scores at that value (``outer(value)`` -> OOF scores). Ties go to the earlier grid value."""
    d = dev()
    scores = np.zeros((len(d.y), d.n_classes), np.float32)
    chosen = []
    for k in range(5):
        tr = d.fold != k
        accs = [float((inner(k, g)[tr].argmax(1) == d.y[tr]).mean()) for g in grid]
        g = grid[int(np.argmax(accs))]
        chosen.append(g)
        scores[d.fold == k] = outer(g)[d.fold == k]
    return scores, chosen


def fit_probe_balanced(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """compclass.features.fit_probe with class_weight='balanced' (same C grid, same grouped inner CV)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    search = GridSearchCV(pipe, {"logisticregression__C": [0.001, 0.003, 0.01, 0.1, 1.0]}, cv=GroupKFold(3),
                          scoring="f1_macro", n_jobs=-1)
    return search.fit(X, y, groups=groups)


def mcnemar(base_ok: np.ndarray, cand_ok: np.ndarray) -> tuple[int, int, float]:
    """(candidate wrong where baseline right, the reverse, exact two-sided p)."""
    b = int((~cand_ok & base_ok).sum())
    c = int((cand_ok & ~base_ok).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, float(p)


def pair_errors(pred: np.ndarray) -> np.ndarray:
    """[N, 5] bool: row i made hard-case confusion j (true class a predicted as b)."""
    d = dev()
    return np.stack([(d.y == d.idx(a)) & (pred == d.idx(b)) for a, b in PAIRS], 1)


def reverse_errors(pred: np.ndarray) -> int:
    d = dev()
    return int(sum(((d.y == d.idx(b)) & (pred == d.idx(a))).sum() for a, b in PAIRS))


def calibrated(scores: np.ndarray) -> np.ndarray:
    """Probabilities after the OOF temperature fit the deployed model uses, so mass is comparable across heads."""
    from compclass.final import fit_temperature

    return np.exp(log_softmax(scores / fit_temperature(scores, dev().y)))


def pair_mass(probs: np.ndarray) -> np.ndarray:
    """Per-row calibrated probability on the hard-case partner, for every dev row of a pair's true class (638 rows).

    The directed counts are tiny on dev (9 of 2,512 for the baseline), so this continuous proxy carries the signal."""
    d = dev()
    return np.concatenate([probs[d.y == d.idx(a), d.idx(b)] for a, b in PAIRS])


def coverage(probs: np.ndarray) -> tuple[float, float]:
    """Abstain threshold exactly as the deployed fit derives it from OOF probabilities; (coverage, accepted acc)."""
    d = dev()
    pred, conf = probs.argmax(1), probs.max(1)
    thr = next((t for t in np.round(np.arange(0.0, 0.96, 0.05), 2)
                if (conf >= t).any() and (pred[conf >= t] == d.y[conf >= t]).mean() >= 0.90), 0.5)
    acc = conf >= thr
    return float(acc.mean()), float((pred[acc] == d.y[acc]).mean())


def score(scores: np.ndarray, ref: str = "H0") -> dict:
    """Every number a decision row needs, from OOF log-scores [N, C] (argmax = prediction). McNemar, partner
    mass and delta_pp compare against hypothesis ``ref`` (the baseline H0 unless a stacked change asks)."""
    from compclass.evaluate import compute_metrics

    d = dev()
    pred = scores.argmax(1)
    accs = [float((pred[d.fold == k] == d.y[d.fold == k]).mean()) for k in range(5)]
    f1s = [compute_metrics(d.y[d.fold == k], pred[d.fold == k], d.classes)["macro_f1"] for k in range(5)]
    ref_scores = np.load(ART / "oof" / f"scores_{ref}.npz")["scores"]
    base = ref_scores.argmax(1)
    ref_cv = float(np.mean([(base[d.fold == k] == d.y[d.fold == k]).mean() for k in range(5)]))
    b, c, p = mcnemar(base == d.y, pred == d.y)
    pe = pair_errors(pred)
    probs = calibrated(scores)
    mass, base_mass = pair_mass(probs), pair_mass(calibrated(ref_scores))
    mass_p = float(wilcoxon(mass, base_mass).pvalue) if (mass != base_mass).any() else 1.0
    cov, accepted = coverage(probs)
    return {"cv_acc": float(np.mean(accs)), "cv_sd": float(np.std(accs, ddof=1)),
            "delta_pp": 100 * (float(np.mean(accs)) - ref_cv), "cv_f1": float(np.mean(f1s)),
            "folds": " ".join(f"{a:.4f}" for a in accs),
            "inscope_acc": float((pred[~d.ood] == d.y[~d.ood]).mean()),
            "ood_acc": float((pred[d.ood] == d.y[d.ood]).mean()),
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p,
            **{k: int(v) for k, v in zip(PAIR_KEYS, pe.sum(0))}, "pair_total": int(pe.sum()),
            "pair_reverse": reverse_errors(pred), "pair_mass": 100 * float(mass.mean()),
            "base_pair_mass": 100 * float(base_mass.mean()), "pair_mass_p": mass_p,
            "coverage": cov, "accepted_acc": accepted}


def verdict(s: dict, base_total: int) -> str:
    """KEEP when accuracy holds (>= -0.25 pp, the harness noise band, McNemar not significantly worse) and the
    partner mass falls by at least a tenth with Wilcoxon p < 0.05 and the directed pair errors do not rise; or
    when accuracy rises significantly (McNemar p < 0.05) without the partner mass rising significantly.
    REJECT when accuracy falls past -0.5 pp or significantly, when the partner mass rises significantly, or when
    neither proxy improves. Otherwise INCONCLUSIVE. (The one-tenth floor was added after H0-H6 were scored; it
    only made KEEP stricter.)"""
    acc_worse = s["mcnemar_b"] > s["mcnemar_c"] and s["mcnemar_p"] < 0.05
    acc_better = s["mcnemar_c"] > s["mcnemar_b"] and s["mcnemar_p"] < 0.05
    mass_up = s["pair_mass"] > s["base_pair_mass"] and s["pair_mass_p"] < 0.05
    mass_down = (s["pair_mass"] <= 0.9 * s["base_pair_mass"] and s["pair_mass_p"] < 0.05
                 and s["pair_total"] <= base_total)
    if s["delta_pp"] < -0.5 or acc_worse or mass_up:
        return "REJECT"
    if (mass_down and s["delta_pp"] >= -0.25) or acc_better:
        return "KEEP"
    if s["pair_mass"] >= s["base_pair_mass"] and s["delta_pp"] <= 0:
        return "REJECT"
    return "INCONCLUSIVE"


def vs_parent(scores: np.ndarray, parent: str) -> str:
    """Accuracy delta and McNemar of a stacked change against the hypothesis it builds on."""
    d = dev()
    ref = np.load(ART / "oof" / f"scores_{parent}.npz")["scores"].argmax(1) == d.y
    ok = scores.argmax(1) == d.y
    b, c, p = mcnemar(ref, ok)
    return f"{100 * (ok.mean() - ref.mean()):+.2f} pp pooled, McNemar {b}/{c} p={p:.3g}"


def record(hid: str, hypothesis: str, why: str, deploy_cost: str, scores: np.ndarray, note: str = "",
           verdict_override: str | None = None, parent: str = "H0") -> dict:
    """Score OOF log-scores, save them, print a summary and upsert one row of decisions.tsv (keyed by id)."""
    s = score(scores)
    s["parent"], s["vs_parent"] = parent, vs_parent(scores, parent) if parent != "H0" else ""
    rel = score(scores, parent) if parent != "H0" else s
    ref_total = int(pair_errors(np.load(ART / "oof" / f"scores_{parent}.npz")["scores"].argmax(1)).sum())
    row = {"id": hid, "hypothesis": hypothesis, "why": why, "deploy_cost": deploy_cost, **s,
           "verdict": verdict_override or verdict(rel, ref_total), "note": note}
    np.savez(ART / "oof" / f"scores_{hid}.npz", scores=scores.astype(np.float32), ids=dev().ids)
    rows = []
    if DECISIONS.exists():
        with open(DECISIONS) as f:
            rows = [r for r in csv.DictReader(f, delimiter="\t") if r["id"] != hid]
    rows.append({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    rows.sort(key=lambda r: (int(re.match(r"H(\d+)", r["id"])[1]), r["id"]))
    with open(DECISIONS, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: row[k] for k in ("id", "cv_acc", "delta_pp", "ood_acc", "mcnemar_b", "mcnemar_c",
                                          "mcnemar_p", *PAIR_KEYS, "pair_total", "pair_mass", "pair_mass_p",
                                          "coverage", "vs_parent", "verdict")},
                     default=lambda v: round(v, 4)))
    return row


def rescore() -> None:
    """Recompute every decisions.tsv row from its saved OOF scores (after a change to the scored columns)."""
    with open(DECISIONS) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        sc = np.load(ART / "oof" / f"scores_{r['id']}.npz")["scores"]
        record(r["id"], r["hypothesis"], r["why"], r["deploy_cost"], sc, r["note"],
               verdict_override=r["verdict"] if r["verdict"] == "BASELINE" else None, parent=r["parent"] or "H0")


if __name__ == "__main__":
    rescore()
