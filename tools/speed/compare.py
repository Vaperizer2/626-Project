"""compare: candidate B against reference A (normally the baseline) on the latest eval and time records.

Acceptance rule (accuracy side; the latency ratio is what the hillclimb optimises):
    ACCEPTABLE  when  mean CV accuracy(B) >= mean CV accuracy(A) - 0.5 pp
               and NOT (B has more discordant errors than A and the exact two-sided McNemar p < 0.05)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from harness import append_result, latest, load_candidate

MAX_DROP = 0.005
ALPHA = 0.05


def vs_baseline(t: dict) -> tuple[float, float, float] | None:
    """(median, min, max) of the per-round latency ratio to the baseline measured in the same session."""
    if t.get("is_reference"):
        return 1.0, 1.0, 1.0
    p = t.get("paired")
    return (p["ratio_median"], p["ratio_min"], p["ratio_max"]) if p else None


def latency(ta: dict, tb: dict) -> dict:
    """Batch-1 latency ratio B/A. Both runs paired with the baseline -> ratio of their same-session ratios, and B is
    beyond noise when its round ratios do not overlap A's; otherwise the ratio of independent medians, beyond noise
    when the round medians do not overlap."""
    pa, pb = vs_baseline(ta), vs_baseline(tb)
    if pa and pb:
        ratio, lo, hi, paired = pb[0] / pa[0], pb[1] / pa[0], pb[2] / pa[0], True
        beyond = pb[2] < pa[1] or pb[1] > pa[2]
    else:
        ra, rb = ta["batch1"]["round_medians_ms"], tb["batch1"]["round_medians_ms"]
        ratio, lo, hi, paired = tb["batch1"]["median_ms"] / ta["batch1"]["median_ms"], min(rb) / max(ra), \
            max(rb) / min(ra), False
        beyond = max(rb) < min(ra) or max(ra) < min(rb)

    def img_s(t: dict) -> float | None:
        return round(t["batch8"]["img_per_s"], 1) if t["batch8"] and t["batch8"]["valid"] else None

    return {"a_batch1_median_ms": ta["batch1"]["median_ms"], "b_batch1_median_ms": tb["batch1"]["median_ms"],
            "ratio": ratio, "ratio_range": [lo, hi], "paired": paired, "beyond_noise": beyond,
            "a_batch8_img_s": img_s(ta), "b_batch8_img_s": img_s(tb)}


def run_compare(a_path: str, b_path: str) -> dict:
    a, b = load_candidate(a_path), load_candidate(b_path)
    ea, eb, ta, tb = latest("eval", a), latest("eval", b), latest("time", a), latest("time", b)
    missing = [f"{kind} {c.dir}" for kind, c, r in (("eval", a, ea), ("eval", b, eb), ("time", a, ta),
                                                     ("time", b, tb)) if r is None]
    if ea is None or eb is None:
        raise SystemExit(f"no record for the current content of: {', '.join(missing)} (run eval/time first)")
    fa, fb = np.load(ea["flags"]), np.load(eb["flags"])
    if not np.array_equal(fa["ids"], fb["ids"]):
        raise SystemExit("per-image flags cover different images")
    ca, cb = fa["correct"], fb["correct"]
    b_worse, b_better = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(b_better, b_worse + b_better, 0.5).pvalue if b_worse + b_better else 1.0
    d_acc = eb["cv_acc_mean"] - ea["cv_acc_mean"]
    acc_ok = d_acc >= -MAX_DROP - 1e-12
    sig_worse = b_worse > b_better and p < ALPHA
    verdict = "ACCEPTABLE" if acc_ok and not sig_worse else "REJECT"
    lat = latency(ta, tb) if ta and tb else None
    rec = append_result({"kind": "compare", "a": a.id, "a_sha": a.sha, "b": b.id, "b_sha": b.sha,
                         "n_images": int(len(ca)), "n_errors": ea["n_errors"] + eb["n_errors"], "n_timed_runs": 0,
                         "a_cv_acc": ea["cv_acc_mean"], "b_cv_acc": eb["cv_acc_mean"], "d_cv_acc": d_acc,
                         "d_cv_f1": eb["cv_f1_mean"] - ea["cv_f1_mean"], "d_oof_acc": eb["oof_acc"] - ea["oof_acc"],
                         "mcnemar": {"b_wrong_a_right": b_worse, "b_right_a_wrong": b_better, "p_two_sided": p},
                         "latency": lat, "verdict": verdict,
                         "reasons": {"accuracy_within_0.5pp": acc_ok, "mcnemar_significantly_worse": sig_worse}})
    print(f"{b.id} vs {a.id}: CV acc {eb['cv_acc_mean']:.4f} vs {ea['cv_acc_mean']:.4f} (d {100 * d_acc:+.2f} pp), "
          f"macro-F1 d {100 * rec['d_cv_f1']:+.2f} pp")
    print(f"  McNemar: B wrong/A right {b_worse}, B right/A wrong {b_better}, exact p = {p:.4f}")
    if lat:
        print(f"  batch-1 latency {lat['b_batch1_median_ms']:.2f} vs {lat['a_batch1_median_ms']:.2f} ms; ratio B/A "
              f"{lat['ratio']:.4f} (round range {lat['ratio_range'][0]:.4f}..{lat['ratio_range'][1]:.4f}, "
              f"{'same-session vs baseline' if lat['paired'] else 'separate sessions'}; "
              f"beyond noise: {lat['beyond_noise']})")
        print(f"  batch-8 {lat['b_batch8_img_s'] or 'n/a'} vs {lat['a_batch8_img_s'] or 'n/a'} img/s")
    else:
        print(f"  latency: missing {', '.join(m for m in missing if m.startswith('time'))}")
    print(f"  verdict: {verdict}")
    return rec
