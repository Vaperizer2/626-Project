"""Calibrate the label-free proxy against harness CV deltas earlier tracks already measured, then rank schedules.

    proxycal.py <proxy_dir> [--metric zerr] [--exclude <spec> ...]

For each schedule with embeddings in <proxy_dir>/emb (fetched from carlpc2) it computes, against the deployed
384 px reference on the same images: cosine (cos), cosine after removing the reference mean (ccos), cosine and
relative error in the deployed head's standardised feature space (zcos, zerr: what the linear probe sees), KL of
the deployed fused head's probabilities (kl) and 10-nearest-neighbour overlap (knn). It fits harness dCV ~ metric by
least squares over table.KNOWN, prints the fit's residual spread, then every schedule with its predicted harness
ratio (sched.py) and predicted dCV. No label is read: the dev-image proxy uses dev pixels only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sched import Costs, Schedule  # noqa: E402
from table import ART, KNOWN  # noqa: E402

HEAD = Path("/home/vaper/Work/Prototypes/626-Project/exports/final/head.npz")
LOWER_IS_BETTER = {"zerr", "kl"}


def lsm(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


class Scorer:
    def __init__(self, ref: np.ndarray):
        self.h = np.load(HEAD)
        self.ref, self.mu = ref, ref.mean(0)
        self.zr = self.z(ref)
        self.pr = np.exp(self.fused(ref))
        self.pr /= self.pr.sum(1, keepdims=True)
        self.nr = self.knn(ref)

    def z(self, e):
        return (e - self.h["mu"]) / self.h["sd"]

    def fused(self, e):
        h = self.h
        fn = e / np.linalg.norm(e, axis=1, keepdims=True)
        return lsm(self.z(e) @ h["coef"].T + h["intercept"]) + h["w_zs"] * lsm(h["t_zs"] * fn @ h["W"].T)

    @staticmethod
    def knn(e, k: int = 10):
        n = e / np.linalg.norm(e, axis=1, keepdims=True)
        s = n @ n.T
        np.fill_diagonal(s, -9)
        return np.argsort(-s, 1)[:, :k]

    def __call__(self, e: np.ndarray) -> dict[str, float]:
        z = self.z(e)
        p = np.exp(self.fused(e))
        p /= p.sum(1, keepdims=True)
        nn = self.knn(e)
        return {"cos": cos(e, self.ref).mean(), "ccos": cos(e - self.mu, self.ref - self.mu).mean(),
                "zcos": cos(z, self.zr).mean(),
                "zerr": (np.linalg.norm(z - self.zr, axis=1) / np.linalg.norm(self.zr, axis=1)).mean(),
                "kl": (self.pr * (np.log(self.pr + 1e-12) - np.log(p + 1e-12))).sum(1).mean(),
                "knn": float(np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(nn, self.nr)]))}


def known_dcv(v: str) -> float:
    """Mean of the recorded deltas (fp16 and int8 builds of one schedule are both measurements of it)."""
    return float(np.mean([float(x.split()[0]) for x in v.split(",")]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--metric", default="zerr")
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()
    emb = {p.stem: p for p in (a.dir / "emb").glob("*.npy")}
    sc = Scorer(np.load(emb["s384"]).astype(np.float64))
    costs = Costs.load(ART / "costs_ladder308.json")
    rows = {}
    for sid, p in emb.items():
        rows[sid] = sc(np.load(p).astype(np.float64))
    known = {Schedule.parse(k).id: known_dcv(v) for k, v in KNOWN.items()}
    excl = {Schedule.parse(s).id for s in a.exclude}
    from scipy.stats import spearmanr

    kid = [k for k in known if k in rows]
    print("known points:", len(kid))
    for k in sorted(kid, key=known.get):
        m = rows[k]
        print(f"  {k[:30]:30s} dCV {known[k]:+.2f}  " + " ".join(f"{x} {v:.4f}" for x, v in m.items())
              + ("  (excluded from fit)" if k in excl else ""))
    sign = {m: (-1 if m in LOWER_IS_BETTER else 1) for m in rows[kid[0]]}
    print("spearman (all known):", {m: round(spearmanr([known[k] for k in kid],
                                                       [sign[m] * rows[k][m] for k in kid])[0], 3) for m in sign})
    fit_ids = [k for k in kid if k not in excl]
    x = np.array([rows[k][a.metric] for k in fit_ids])
    y = np.array([known[k] for k in fit_ids])
    b, c0 = np.polyfit(x, y, 1)
    res = y - (b * x + c0)
    print(f"fit dCV = {b:.2f} * {a.metric} + {c0:.2f} over {len(fit_ids)} points; residual sd {res.std(ddof=2):.2f} pp")
    out = []
    for sid, m in rows.items():
        s = Schedule.parse(sid)
        try:
            r = costs.ratio(s)
        except KeyError:
            r = float("nan")
        out.append((r, sid, m, b * m[a.metric] + c0))
    print(f"{'schedule':40s} {'pred ratio':>10s} {a.metric:>7s} {'ccos':>7s} {'knn':>6s} {'pred dCV':>8s}  known")
    for r, sid, m, d in sorted(out, key=lambda t: (t[0] != t[0], t[0])):
        name = sid if len(sid) <= 40 else sid[:28] + "..." + sid[-9:]
        print(f"{name:40s} {r:10.4f} {m[a.metric]:7.4f} {m['ccos']:7.4f} {m['knn']:6.3f} {d:+8.2f}  "
              f"{known.get(sid, '')}")


if __name__ == "__main__":
    main()
