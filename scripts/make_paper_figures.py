#!/usr/bin/env python
"""Column-width (3.45 in) figures for the IEEE conference paper (docs/paper/main.tex).

    uv run --no-project --with numpy --with pandas --with matplotlib python scripts/make_paper_figures.py

The report figures (docs/report/figures/*) are drawn for a 7-in single-column page and become unreadable when shrunk
to one IEEE column, so this script redraws the panels the paper needs from the SAME repository files, reusing the
loaders, colours and style of scripts/make_report_figures.py. It reads only repo files (no model, no GPU) and writes
    docs/paper/figures/fig{1..6}_*.pdf + .png
    docs/paper/figures/paper_figure_numbers.json   (every number drawn, with its source file)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_report_figures as R  # noqa: E402  (main() is guarded; importing only defines helpers)

ROOT = R.ROOT
OUT = ROOT / "docs" / "paper" / "figures"
PAD = 0.02  # savefig pad_inches
COL_W = 3.45  # IEEEtran conference column width is 3.5 in; leave a hair of margin
NUM: dict[str, dict] = {}
FS_ANN, FS_TICK = R.FS_ANN, R.FS_TICK
SHORT_GROUP = {
    "Vision-language (web image-text)": "vision-language",
    "Self-supervised (LVD-142M)": "self-supervised",
    "ImageNet transformer": "ImageNet ViT",
    "ImageNet CNN": "ImageNet CNN",
    "CNN from scratch": "from scratch",
    "Classical (HOG+colour+SVM)": "classical",
}


def note(name: str, value, source: str):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    NUM[name] = {"value": value, "source": source}
    return value


def fit_width(fig, target: float = COL_W) -> None:
    """Resize the figure so its tight bounding box (axes + tick labels + titles) is `target` inches wide; the
    LaTeX source then includes it at \\columnwidth with ~1:1 scaling, so the 8/7/6-pt font ladder stays true."""
    target -= 2 * PAD
    for _ in range(8):
        fig.canvas.draw()
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width
        if abs(w - target) < 0.01:
            break
        fig.set_figwidth(max(1.0, fig.get_figwidth() + 0.9 * (target - w)))


def save(fig, stem: str) -> None:
    fit_width(fig)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", pad_inches=PAD)
    plt.close(fig)
    print("wrote", (OUT / f"{stem}.pdf").relative_to(ROOT))


def short_name(bb: str) -> str:
    return R.BACKBONE[bb][0].replace("/14@384", "").replace("@336", "@336")


# ----------------------------------------------------------------------------------------------- fig 1
def fig_ood(d: dict) -> None:
    rows = pd.DataFrame(d["manifest"]["rows"])
    rows["ood"] = rows["flags"].apply(lambda f: "ood_suspect" in f)
    per = rows.groupby("label").agg(n=("id", "size"), ood=("ood", "sum")).reset_index()
    per["share"] = 100 * per.ood / per.n
    per = per.sort_values("share").reset_index(drop=True)
    overall = 100 * rows.ood.mean()
    note("ood_overall_pct", round(overall, 1), "data/v2/manifest.json (flags)")
    note("ood_per_class", {r.label: f"{int(r.ood)}/{int(r.n)}" for r in per.itertuples()}, "data/v2/manifest.json")
    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    y = np.arange(len(per))
    ax.barh(y, per.share, color="#0072B2", height=0.68)
    for yi, r in zip(y, per.itertuples()):
        if r.label in ("switch", "transformer"):
            ax.text(r.share + 1.0, yi, f"{int(r.ood)}/{int(r.n)}", va="center", fontsize=FS_TICK)
    ax.axvline(overall, color="#333333", lw=0.7, ls="--")
    ax.text(overall + 1.0, 0.5, f"all classes {overall:.1f}%", fontsize=FS_TICK, va="center")
    ax.set_yticks(y, per.label)
    ax.set_ylim(-0.7, len(per) - 0.3)
    ax.set_xlim(0, 80)
    ax.set_xlabel("Rows flagged out-of-scope (%)")
    ax.set_title("Homonym classes are most contaminated")
    save(fig, "fig1_ood_per_class")


# ----------------------------------------------------------------------------------------------- fig 2
def fig_benchmark(d: dict, bm: pd.DataFrame) -> None:
    b = bm[~bm.name.str.endswith("_tta")].copy()
    order = b.groupby("backbone").f1.max().sort_values(ascending=True).index.tolist()
    ypos = {bb: i for i, bb in enumerate(order)}
    fig, ax = plt.subplots(figsize=(COL_W, 3.45))
    for _, r in b.iterrows():
        col = R.GROUP_COLOR[r.group]
        off = {"zeroshot": 0.22, "linear_probe": 0.0, "fine-tune": -0.22, "classical": 0.0}[r.regime]
        yy = ypos[r.backbone] + off
        ax.errorbar(r.f1, yy, xerr=r.f1_sd, fmt="none", ecolor=col, elinewidth=0.8, capsize=0, zorder=2)
        ax.scatter(r.f1, yy, s=16, marker=R.REGIME_MARKER[r.regime], facecolor="white" if r.folds != 5 else col,
                   edgecolor=col, lw=0.9, zorder=3)
    ax.set_yticks(range(len(order)), [short_name(bb) for bb in order])
    for t, bb in zip(ax.get_yticklabels(), order):
        t.set_color(R.GROUP_COLOR[R.BACKBONE[bb][2]])
    ax.set_xlim(0.15, 0.9)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Macro-F1, grouped 5-fold CV (mean ± SD)")
    ax.set_title("Vision-language models lead every regime")
    best = b.iloc[0]
    ax.annotate(f"{best.f1:.3f}", (best.f1, ypos[best.backbone]), xytext=(7, 0), textcoords="offset points",
                fontsize=FS_ANN, va="center")
    h_reg = [Line2D([], [], ls="", marker=R.REGIME_MARKER[k], color="#333333", markersize=4, label=R.REGIME_LABEL[k])
             for k in ("zeroshot", "linear_probe", "fine-tune", "classical")]
    h_reg.append(Line2D([], [], ls="", marker="s", markerfacecolor="white", markeredgecolor="#333333", markersize=4,
                        label="3 folds only"))
    leg = ax.legend(handles=h_reg, loc="upper center", bbox_to_anchor=(0.38, -0.135), ncol=5, fontsize=FS_TICK,
                    handletextpad=0.1, columnspacing=0.6, borderaxespad=0.0)
    ax.add_artist(leg)
    h_grp = [Line2D([], [], ls="", marker="s", color=R.GROUP_COLOR[g], markersize=4, label=SHORT_GROUP[g])
             for g in list(R.GROUP_COLOR)[:4]]
    ax.legend(handles=h_grp, loc="upper center", bbox_to_anchor=(0.38, -0.195), ncol=4, fontsize=FS_TICK,
              handletextpad=0.1, columnspacing=0.6, borderaxespad=0.0)
    ax.text(0.98, 0.02, "higher = better →", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_TICK,
            color="#555555")
    save(fig, "fig2_benchmark")
    note("benchmark_rows_drawn", len(b), "results/benchmark.csv (fixed schedule, TTA rows omitted)")


# ----------------------------------------------------------------------------------------------- fig 3
def fig_probe_vs_ft(d: dict) -> None:
    F = d["perfold"].pivot_table(index="name", columns="fold", values="macro_f1")
    rows = []
    for bb, p, f in R.PAIRS:
        dd = (F.loc[f] - F.loc[p]).dropna()
        rows.append(dict(bb=bb, group=R.BACKBONE[bb][2], delta=dd.mean(), deltas=dd.to_numpy(), wins=int((dd > 0).sum()),
                         n=len(dd), partial="partial" in f))
    df = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    for i, r in df.iterrows():
        col = R.GROUP_COLOR[r.group]
        ax.scatter(100 * r.deltas, np.full(r.n, i) + (np.arange(r.n) - (r.n - 1) / 2) * 0.09, s=8, color=col,
                   alpha=0.6, lw=0, zorder=2)
        ax.plot([100 * r.delta] * 2, [i - 0.32, i + 0.32], color=col, lw=1.8, zorder=3)
        ax.text(11.6, i, f"{r.wins}/{r.n}", va="center", ha="right", fontsize=FS_TICK, color=col)
    ax.axvline(0, color="#333333", lw=0.6)
    ax.set_yticks(range(len(df)), [short_name(r.bb) + (" (partial)" if r.partial else "") for r in df.itertuples()])
    for t, g in zip(ax.get_yticklabels(), df.group):
        t.set_color(R.GROUP_COLOR[g])
    ax.set_xlim(-5.5, 12)
    ax.set_ylim(-0.6, len(df) - 0.1)
    ax.set_xlabel("Fine-tune − frozen probe, macro-F1 (points)")
    ax.text(11.6, len(df) - 0.45, "FT wins", ha="right", va="center", fontsize=FS_TICK, color="#555555")
    ax.set_title("Fine-tuning helps all but SigLIP")
    save(fig, "fig3_probe_vs_finetune")
    note("probe_vs_ft", {r.bb: {"delta_pts": round(100 * r.delta, 2), "wins": f"{r.wins}/{r.n}"} for r in df.itertuples()},
         "results/phase1/raw/runs.jsonl + results/phase2/runs_fixed.jsonl")


# ----------------------------------------------------------------------------------------------- fig 4
def fig_params(d: dict, bm: pd.DataFrame) -> None:
    ft = bm[bm.regime == "fine-tune"]
    pr = bm[(bm.regime == "linear_probe") & ~bm.name.str.endswith("_tta")]
    both = pd.concat([ft, pr], ignore_index=True)
    both["x"] = both.trainable_M_all * 1e6
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    x, y = both.x.to_numpy(float), both.f1.to_numpy()
    keep = R.pareto_mask(x, y)
    o = np.argsort(x[keep])
    ax.step(x[keep][o], y[keep][o], where="post", color="#BBBBBB", lw=0.9, zorder=1)
    for _, r in both.iterrows():
        col = R.GROUP_COLOR[r.group]
        ax.errorbar(r.x, r.f1, yerr=r.f1_sd, fmt="none", ecolor=col, elinewidth=0.6, zorder=2)
        ax.scatter(r.x, r.f1, s=16, marker=R.REGIME_MARKER[r.regime], facecolor="white" if r.folds != 5 else col,
                   edgecolor=col, lw=0.9, zorder=3)
    lab = {("siglip_so400m_384", "linear_probe"): ("SigLIP So400m probe", (6, 3), "left"),
           ("siglip_so400m_384", "fine-tune"): ("So400m partial FT", (-6, 4), "right"),
           ("siglip2_b16", "fine-tune"): ("SigLIP 2 B/16 FT", (-6, 0), "right"),
           ("resnet50", "linear_probe"): ("ResNet-50 probe", (6, -1), "left"),
           ("mobilenetv3_large", "fine-tune"): ("MobileNetV3-L FT", (-6, 0), "right"),
           ("scratch_cnn", "fine-tune"): ("scratch CNN", (-6, 0), "right")}
    for _, r in both.iterrows():
        k = (r.backbone, r.regime)
        if k in lab:
            t, xy, ha = lab[k]
            ax.annotate(t, (r.x, r.f1), xytext=xy, textcoords="offset points", fontsize=FS_TICK, ha=ha, va="center",
                        color=R.GROUP_COLOR[r.group])
    ax.set_xscale("log")
    ax.set_xlim(4e3, 3e8)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else (f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}")))
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Macro-F1 (grouped CV)")
    ax.set_title("A 23k-parameter probe ≈ a 76M-parameter fine-tune")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color="#333333", markersize=4, label="linear probe"),
                       Line2D([], [], ls="", marker="s", color="#333333", markersize=4, label="fine-tune"),
                       Line2D([], [], color="#BBBBBB", label="Pareto front")], loc="lower right", fontsize=FS_TICK)
    save(fig, "fig4_params")


# ----------------------------------------------------------------------------------------------- fig 5
def fig_calibration(d: dict) -> None:
    P = d["preds"]
    y, probs = P["y"], P["probs"]
    conf, pred = probs.max(1), probs.argmax(1)
    corr = pred == y
    e = R.ece(probs, y)
    assert abs(e - d["test"]["ece"]) < 1e-9
    fit = d["fit"]
    thr = fit["abstain_threshold"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.9), gridspec_kw={"wspace": 0.5})
    edges = np.linspace(0, 1, 16)
    mids, accs, confs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            mids.append((lo + hi) / 2), accs.append(corr[m].mean()), confs.append(conf[m].mean())
    a1.bar(mids, accs, width=(edges[1] - edges[0]) * 0.9, color="#0072B2", alpha=0.85)
    a1.plot([0, 1], [0, 1], color="#333333", lw=0.6, ls="--")
    a1.scatter(confs, accs, s=5, color="#08306B", zorder=3)
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1.05)
    a1.set_xlabel("Confidence")
    a1.set_ylabel("Accuracy")
    a1.set_title(f"Test ECE {e:.3f}")
    R.panel_letter(a1, "a")
    tab = pd.DataFrame(fit["abstain_table"])
    tab["tc"] = [(conf >= t).mean() for t in tab.threshold]
    tab["ta"] = [corr[conf >= t].mean() for t in tab.threshold]
    a2.plot(100 * tab.coverage, 100 * tab.accepted_acc, "-o", color="#999999", ms=1.8, lw=0.9, label="dev OOF")
    a2.plot(100 * tab.tc, 100 * tab.ta, "-o", color="#0072B2", ms=1.8, lw=0.9, label="sealed test")
    sel = tab[tab.threshold == thr].iloc[0]
    a2.scatter([100 * sel.coverage, 100 * sel.tc], [100 * sel.accepted_acc, 100 * sel.ta], s=28, facecolor="none",
               edgecolor=["#333333", "#0072B2"], lw=0.8, zorder=4)
    a2.axhline(90, color="#333333", lw=0.5, ls=":")
    a2.set_xlim(45, 103)
    a2.set_xlabel("Coverage (%)")
    a2.set_ylabel("Accepted acc. (%)")
    a2.set_title(f"Abstain below {thr:.1f}")
    a2.legend(loc="lower left", fontsize=FS_TICK, handlelength=1.2, borderaxespad=0.1)
    R.panel_letter(a2, "b")
    save(fig, "fig5_calibration_abstention")
    note("test_ece", round(e, 4), "results/final/final_test_v2_preds.npz")
    note("threshold_0.5", {"oof_cov": round(sel.coverage, 4), "oof_acc": round(sel.accepted_acc, 4),
                           "test_cov": round(sel.tc, 4), "test_acc": round(sel.ta, 4)},
         "exports/final/fit_report.json + results/final/final_test_v2_preds.npz")


# ----------------------------------------------------------------------------------------------- fig 6
def fig_generalisation(d: dict) -> None:
    ks = json.load(open(ROOT / "results/kaggle_original/summary.json"))
    wh = json.load(open(ROOT / "results/final/final_test_web_holdout_v1.json"))
    jt = d["test"]
    rp, v2 = ks["legacy_test_replay"], ks["by_v2_split"]
    ob = rp["old_best_for_reference"]
    n_old = ks["legacy_test_as_deployed"]["n"]  # the legacy test files the old notebooks scored
    blue, light, grey = "#0072B2", "#9ECAE1", "#888888"
    bars = [  # (section, label, acc, n, colour, source)
        ("Sealed sets, final model", "Kaggle sealed test", jt["accuracy"], jt["n"], blue, "results/final/final_test_v2.json"),
        ("Sealed sets, final model", "Web holdout (Wikimedia)", wh["accuracy"], wh["n"], blue,
         "results/final/final_test_web_holdout_v1.json"),
        ("Sealed sets, final model", "Web holdout, probe only", wh["probe_only_accuracy"], wh["n"], light,
         "results/final/final_test_web_holdout_v1.json"),
        ("Original Kaggle files, v2 split", "Never seen (test rows)", v2["test"]["accuracy"], v2["test"]["n"], blue,
         "results/kaggle_original/summary.json"),
        ("Original Kaggle files, v2 split", "Seen in head fit (dev rows)", v2["dev"]["accuracy"], v2["dev"]["n"], light,
         "results/kaggle_original/summary.json"),
        ("Original Kaggle files, v2 split", "Conflicting-label copies", v2["excluded"]["accuracy"], v2["excluded"]["n"],
         light, "results/kaggle_original/summary.json"),
        ("Legacy 70/15/15 file split", "Old best (ConvNeXt-B FT)", ob["accuracy"], n_old, grey, ob["source"]),
        ("Legacy 70/15/15 file split", "Ours, same protocol", rp["all"]["accuracy"], rp["all"]["n"], blue,
         "results/kaggle_original/summary.json"),
        ("Legacy 70/15/15 file split", "– copies of train files", rp["leaked_copy_of_train"]["accuracy"],
         rp["leaked_copy_of_train"]["n"], light, "results/kaggle_original/summary.json"),
        ("Legacy 70/15/15 file split", "– no copy in train", rp["clean_not_in_train"]["accuracy"],
         rp["clean_not_in_train"]["n"], light, "results/kaggle_original/summary.json"),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 3.1))
    ypos, yy, last, heads = [], 0.0, None, []
    for sec, *_ in bars:
        if sec != last:
            yy -= 0.55 if last is not None else 0.0
            heads.append((sec, yy))
            yy -= 0.85
            last = sec
        ypos.append(yy)
        yy -= 1.0
    for (sec, lbl, acc, n, col, src), yv in zip(bars, ypos):
        k = int(round(acc * n))
        lo, hi = R.wilson(k, n)
        ax.barh(yv, 100 * acc, color=col, height=0.72)
        ax.errorbar(100 * acc, yv, xerr=[[100 * (acc - lo)], [100 * max(hi - acc, 0)]], fmt="none", ecolor="#222222",
                    elinewidth=0.6, capsize=1.5)
        ax.text(100 * hi + 1.5, yv, f"{100 * acc:.1f}%  (n={n:,})", va="center", fontsize=FS_TICK)
        note(f"gen::{sec}::{lbl.strip()}", {"acc": round(acc, 4), "n": n, "wilson95": [round(lo, 4), round(hi, 4)]}, src)
    for sec, yv in heads:
        ax.text(0.0, yv, sec, fontsize=FS_ANN, fontweight="bold", ha="left", va="center")
    ax.set_yticks(ypos, [b[1] for b in bars])
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(min(ypos) - 0.7, heads[0][1] + 0.6)
    ax.set_xlabel("Accuracy vs. folder label (%, 95% Wilson CI)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig6_generalisation")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    R.apply_figure_style(R.SIZES)
    d = R.load_all()
    R.check_benchmark(d)
    bm = R.main_table(d)
    fig_ood(d)
    fig_benchmark(d, bm)
    fig_probe_vs_ft(d)
    fig_params(d, bm)
    fig_calibration(d)
    fig_generalisation(d)
    (OUT / "paper_figure_numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
    print("wrote", (OUT / "paper_figure_numbers.json").relative_to(ROOT), f"({len(NUM)} entries)")


if __name__ == "__main__":
    main()
