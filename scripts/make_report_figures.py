#!/usr/bin/env python
"""Build every figure and table of the comparative report (docs/report/REPORT.md).

    uv run python scripts/make_report_figures.py

Reads ONLY files already in the repo (results/, exports/final/, data/v2/ manifests and review CSVs) -- no model
is loaded and no GPU is needed; it runs in a few seconds on a laptop CPU. Writes
    docs/report/figures/fig_*.png + .pdf      (300 dpi PNG, vector PDF)
    docs/report/tables/*.csv + *.tex          (booktabs LaTeX tables)
    docs/report/tables/key_numbers.json       (every number quoted in REPORT.md, with its source file)

Inputs (all relative to the repo root):
    results/benchmark.csv                      grouped 5-fold CV summary, all methods ('schedule' column)
    results/phase1/raw/runs.jsonl              per-fold records: probes, zero-shot, HOG baseline
    results/phase1b/runs.jsonl                 per-fold records: hflip-TTA probes
    results/phase2/runs_fixed.jsonl            per-fold records: fine-tunes, fixed LR schedule (main table)
    results/phase2/raw_oldschedule/runs_{A,B}.jsonl   per-fold records: fine-tunes, old (buggy) schedule
    results/schedule_ablation_folds012.csv     old-vs-fixed schedule summary (folds 0-2)
    results/leakage_simulation.csv             legacy-split vs grouped-CV accuracy
    results/phase1/cleaning_ablation_siglip_so400m.csv
    results/final/final_test_v2.json + final_test_v2_preds.npz   THE single sealed-test evaluation
    exports/final/{fit_report,export_report,preprocess}.json, labels.json
    data/v2/manifest.json, data/v2/leakage_report.json, data/v2/review/ood_suspects.csv
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "report" / "figures"
TAB = ROOT / "docs" / "report" / "tables"
KEY: dict[str, dict] = {}  # name -> {"value": ..., "source": ...}


# ----------------------------------------------------------------------------------------------- style
def apply_figure_style(sizes=(8, 7, 6)) -> None:
    """Publication rcParams: role-mapped 3-size font ladder, open frame, outward ticks, frameless legends,
    300-dpi tight saves, embedded TrueType fonts (mirrors the `figure-style` checklist used for this report)."""
    base, secondary, tick = sizes
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "legend.fontsize": secondary, "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titleweight": "normal",
        "axes.titlelocation": "left", "lines.linewidth": 1.2, "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


SIZES = (8, 7, 6)
FS_ANN = SIZES[1]
FS_TICK = SIZES[2]

# pretraining source -> colour (Okabe-Ito; the vermillion alarm hue is not used for data)
GROUP_COLOR = {
    "Vision-language (web image-text)": "#0072B2",
    "Self-supervised (LVD-142M)": "#009E73",
    "ImageNet transformer": "#CC79A7",
    "ImageNet CNN": "#E69F00",
    "CNN from scratch": "#8C564B",
    "Classical (HOG+colour+SVM)": "#666666",
}
REGIME_MARKER = {"zeroshot": "D", "linear_probe": "o", "fine-tune": "s", "classical": "^"}
REGIME_LABEL = {"zeroshot": "zero-shot", "linear_probe": "linear probe", "fine-tune": "fine-tune",
                "classical": "classical"}

# backbone key -> (display name, family, pretraining group)
BACKBONE = {
    "siglip_so400m_384": ("SigLIP So400m/14@384", "SigLIP", "Vision-language (web image-text)"),
    "siglip2_so400m_384": ("SigLIP 2 So400m/14@384", "SigLIP", "Vision-language (web image-text)"),
    "siglip_b16": ("SigLIP B/16", "SigLIP", "Vision-language (web image-text)"),
    "siglip2_b16": ("SigLIP 2 B/16", "SigLIP", "Vision-language (web image-text)"),
    "clip_vitl14_336": ("CLIP ViT-L/14@336", "CLIP", "Vision-language (web image-text)"),
    "clip_vitl14": ("CLIP ViT-L/14", "CLIP", "Vision-language (web image-text)"),
    "clip_vitb16": ("CLIP ViT-B/16", "CLIP", "Vision-language (web image-text)"),
    "dinov2_vitl14": ("DINOv2 ViT-L/14", "DINOv2", "Self-supervised (LVD-142M)"),
    "dinov2_vitb14": ("DINOv2 ViT-B/14", "DINOv2", "Self-supervised (LVD-142M)"),
    "dinov2_vits14": ("DINOv2 ViT-S/14", "DINOv2", "Self-supervised (LVD-142M)"),
    "vit_b16": ("ViT-B/16", "ViT", "ImageNet transformer"),
    "swin_t": ("Swin-T", "Swin", "ImageNet transformer"),
    "convnext_tiny": ("ConvNeXt-T", "ConvNeXt", "ImageNet CNN"),
    "efficientnet_v2_s": ("EfficientNetV2-S", "EfficientNet", "ImageNet CNN"),
    "resnet50": ("ResNet-50", "ResNet", "ImageNet CNN"),
    "mobilenetv3_large": ("MobileNetV3-L", "MobileNet", "ImageNet CNN"),
    "scratch_cnn": ("Scratch CNN", "Scratch CNN", "CNN from scratch"),
    "hog_hsv_svm": ("HOG+HSV+SVM", "Classical", "Classical (HOG+colour+SVM)"),
}


def backbone_of(name: str) -> str:
    for p in ("probe_", "zeroshot_"):
        if name.startswith(p):
            name = name[len(p):]
    for s in ("_tta", "_partial_v2", "_partial"):
        if name.endswith(s):
            name = name[: -len(s)]
    return name


def method_label(name: str, regime: str, folds: int) -> str:
    disp = BACKBONE[backbone_of(name)][0]
    reg = REGIME_LABEL[regime]
    if name.endswith("_tta"):
        reg += " + hflip TTA"
    if "_partial" in name:
        reg = "partial fine-tune"
    if folds != 5:
        reg += f", {folds} folds"
    return f"{disp} ({reg})"


def key(name: str, value, source: str):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    KEY[name] = {"value": value, "source": source}
    return value


# ----------------------------------------------------------------------------------------------- io helpers
def jl(rel: str) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in open(ROOT / rel) if line.strip()])


def rjson(rel: str):
    return json.load(open(ROOT / rel))


def save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}")
    plt.close(fig)
    print("wrote", (FIG / f"{stem}.png").relative_to(ROOT))


def to_tex(df: pd.DataFrame, path: Path, caption: str, label: str, align: str | None = None) -> None:
    """Minimal booktabs table writer (no jinja2 dependency)."""
    def esc(s) -> str:
        s = str(s)
        for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
            s = s.replace(a, b)
        return s.replace("±", r"$\pm$").replace("→", r"$\rightarrow$").replace("−", "$-$")
    align = align or ("l" + "r" * (df.shape[1] - 1))
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(esc(c) for c in df.columns) + r" \\",
             r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join("" if (isinstance(v, float) and np.isnan(v)) else esc(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def write_table(df: pd.DataFrame, stem: str, caption: str, align: str | None = None) -> None:
    df.to_csv(TAB / f"{stem}.csv", index=False)
    to_tex(df, TAB / f"{stem}.tex", caption, f"tab:{stem}", align)
    print("wrote", (TAB / f"{stem}.csv").relative_to(ROOT), "+ .tex")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact binomial (sign) test of k successes in n under p = 0.5."""
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Identical to compclass.final.ece (15 equal-width bins, (lo, hi])."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def macro_f1(y: np.ndarray, pred: np.ndarray, n_cls: int) -> float:
    f = []
    for c in range(n_cls):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=SIZES[0] + 2, fontweight="bold",
            ha="right", va="bottom")


# ----------------------------------------------------------------------------------------------- load
def load_all() -> dict:
    d: dict = {}
    d["bm"] = pd.read_csv(ROOT / "results/benchmark.csv")
    p1, p1b, fx = jl("results/phase1/raw/runs.jsonl"), jl("results/phase1b/runs.jsonl"), jl(
        "results/phase2/runs_fixed.jsonl")
    old = pd.concat([jl("results/phase2/raw_oldschedule/runs_A.jsonl"),
                     jl("results/phase2/raw_oldschedule/runs_B.jsonl")], ignore_index=True)
    d["p1"], d["p1b"], d["fx"], d["old"] = p1, p1b, fx, old
    cols = ["name", "fold", "macro_f1", "accuracy"]
    d["perfold"] = pd.concat([p1[cols], p1b[cols], fx[cols]], ignore_index=True)
    d["leak"] = pd.read_csv(ROOT / "results/leakage_simulation.csv")
    d["sched"] = pd.read_csv(ROOT / "results/schedule_ablation_folds012.csv")
    d["clean"] = pd.read_csv(ROOT / "results/phase1/cleaning_ablation_siglip_so400m.csv")
    d["test"] = rjson("results/final/final_test_v2.json")
    d["preds"] = np.load(ROOT / "results/final/final_test_v2_preds.npz")
    d["fit"] = rjson("exports/final/fit_report.json")
    d["export"] = rjson("exports/final/export_report.json")
    d["pre"] = rjson("exports/final/preprocess.json")
    d["labels"] = rjson("exports/final/labels.json")
    d["manifest"] = rjson("data/v2/manifest.json")
    d["lr"] = rjson("data/v2/leakage_report.json")
    d["ood"] = pd.read_csv(ROOT / "data/v2/review/ood_suspects.csv")
    return d


def check_benchmark(d: dict) -> None:
    """benchmark.csv must equal the mean/SD of the per-fold run records (guards against stale summaries)."""
    bm = d["bm"][d["bm"].schedule != "old (bug)"].set_index("name")
    agg = d["perfold"].groupby("name").macro_f1.agg(["mean", "std", "count"])
    diff = (bm.f1 - agg.loc[bm.index, "mean"]).abs().max()
    assert diff < 6e-5, f"benchmark.csv disagrees with per-fold runs by {diff}"
    assert (bm.folds == agg.loc[bm.index, "count"]).all()
    key("benchmark_vs_runs_max_abs_diff", round(float(diff), 6), "results/benchmark.csv vs runs*.jsonl")


# ----------------------------------------------------------------------------------------------- (a) data audit
def fig_data_audit(d: dict) -> None:
    c = d["lr"]["counts"]
    src = "data/v2/leakage_report.json"
    for k in ("raw_files", "unique_sha256", "exact_dup_redundant_files", "exact_dup_groups",
              "exact_dup_conflicting_label_groups", "excluded_total", "rows", "clusters", "near_dup_clusters",
              "rows_in_near_dup_clusters", "near_dup_mixed_label_clusters", "ood_suspect", "test_rows", "dev_rows"):
        key(f"data_{k}", c[k], src)
    rows = pd.DataFrame(d["manifest"]["rows"])
    rows["ood"] = rows["flags"].apply(lambda f: "ood_suspect" in f)
    per = rows.groupby("label").agg(n=("id", "size"), ood=("ood", "sum")).reset_index()
    per["share"] = per.ood / per.n
    per = per.sort_values("share")
    key("ood_share_overall", round(c["ood_suspect"] / c["rows"], 4), src)
    key("ood_test_rows", int(rows[(rows.split == "test")].ood.sum()), "data/v2/manifest.json")
    key("ood_dev_rows", int(rows[(rows.split == "dev")].ood.sum()), "data/v2/manifest.json")
    for lab in ("switch", "transformer"):
        r = per[per.label == lab].iloc[0]
        key(f"ood_{lab.replace(' ', '_')}", f"{int(r.ood)}/{int(r.n)}", "data/v2/manifest.json")
    key("ood_top_distractors", d["ood"].top_distractor.value_counts().to_dict(), "data/v2/review/ood_suspects.csv")

    stages = [("Raw image files", c["raw_files"]), ("Unique files (SHA-256)", c["unique_sha256"]),
              ("Rows kept (conflicting-label\nexact duplicates removed)", c["rows"]),
              ("Duplicate clusters (CV groups)", c["clusters"]),
              ("Rows in near-duplicate clusters", c["rows_in_near_dup_clusters"]),
              ("Rows flagged out-of-scope (CLIP)", c["ood_suspect"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw={"width_ratios": [1.0, 1.1], "wspace": 0.75})
    y = np.arange(len(stages))[::-1]
    a1.barh(y, [s[1] for s in stages], color=["#999999"] * 4 + ["#E69F00", "#0072B2"], height=0.62)
    for yi, (_, v) in zip(y, stages):
        a1.text(v + 40, yi, f"{v:,}", va="center", fontsize=FS_ANN)
    a1.set_yticks(y, [s[0] for s in stages])
    a1.set_xlabel("Images")
    a1.set_xlim(0, c["raw_files"] * 1.18)
    a1.set_title(f"{c['exact_dup_redundant_files']} redundant copies and {c['excluded_total']} label conflicts removed")
    panel_letter(a1, "a")

    a2.barh(np.arange(len(per)), per.share * 100, color="#0072B2", height=0.66)
    a2.axvline(100 * c["ood_suspect"] / c["rows"], color="#333333", lw=0.8, ls="--")
    a2.text(100 * c["ood_suspect"] / c["rows"] + 1, -1.4, f"all classes {100 * c['ood_suspect'] / c['rows']:.0f}%",
            fontsize=FS_ANN, va="center")
    a2.set_yticks(np.arange(len(per)), per.label)
    a2.set_ylim(-2, len(per) - 0.4)
    a2.set_xlabel("Rows flagged out-of-scope (%)")
    a2.set_title("Homonym-heavy classes are most contaminated")
    panel_letter(a2, "b")
    save(fig, "fig_a_data_audit")

    audit = pd.DataFrame([(s[0].replace("\n", " "), s[1]) for s in stages], columns=["Stage", "Images"])
    write_table(audit, "data_audit", "Data audit of the Kaggle electronic-component scrape (data/v2/leakage_report.json).")
    per_out = per.sort_values("share", ascending=False).assign(share=lambda x: (100 * x.share).round(1))
    per_out.columns = ["Class", "Rows", "OOD-suspect rows", "OOD-suspect %"]
    write_table(per_out, "ood_per_class", "Per-class share of rows flagged out-of-scope by the CLIP screen.")


# ----------------------------------------------------------------------------------------------- (b) benchmark
def main_table(d: dict) -> pd.DataFrame:
    bm = d["bm"][d["bm"].schedule != "old (bug)"].copy()
    bm["backbone"] = bm.name.map(backbone_of)
    bm["group"] = bm.backbone.map(lambda b: BACKBONE[b][2])
    bm["family"] = bm.backbone.map(lambda b: BACKBONE[b][1])
    bm["label"] = [method_label(n, r, f) for n, r, f in zip(bm.name, bm.regime, bm.folds)]
    probe_dim = d["p1"].groupby("name").feature_dim.first().to_dict()
    probe_dim.update(d["p1b"].groupby("name").feature_dim.first().to_dict())
    n_cls = len(d["labels"])

    def trainable(r):
        if r.regime == "linear_probe":
            return n_cls * (probe_dim[r["name"]] + 1) / 1e6  # logistic-regression head: C x (D + 1)
        if r.regime == "zeroshot":
            return 0.0
        return r.trainable_M
    bm["trainable_M_all"] = bm.apply(trainable, axis=1)
    return bm.sort_values("f1", ascending=False).reset_index(drop=True)


def fig_benchmark(d: dict, bm: pd.DataFrame) -> None:
    pf = d["perfold"]
    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    n = len(bm)
    for i, r in bm.iterrows():
        yy = n - 1 - i
        col = GROUP_COLOR[r.group]
        mk = REGIME_MARKER[r.regime]
        folds = pf[pf.name == r["name"]].macro_f1.to_numpy()
        jit = (np.arange(len(folds)) - (len(folds) - 1) / 2) * 0.09
        ax.scatter(folds, yy + jit, s=6, color=col, alpha=0.45, lw=0, zorder=2)
        ax.errorbar(r.f1, yy, xerr=r.f1_sd, fmt="none", ecolor=col, elinewidth=1.0, capsize=0, zorder=3)
        hollow = r.folds != 5
        ax.scatter(r.f1, yy, s=30, marker=mk, facecolor="white" if hollow else col, edgecolor=col, lw=1.0, zorder=4)
    ax.set_yticks(np.arange(n)[::-1], bm.label)
    for t, g in zip(ax.get_yticklabels(), bm.group):
        t.set_color(GROUP_COLOR[g])
    ax.set_xlabel("Macro-F1, grouped 5-fold CV on 2,512 dev images (mean ± SD; small dots = folds)")
    ax.set_xlim(0.15, 0.88)
    ax.set_ylim(-0.8, n - 0.2)
    ax.text(1.0, -0.06, "higher = better →", transform=ax.transAxes, ha="right", va="top", fontsize=FS_TICK)
    best = bm.iloc[0]
    ax.annotate(f"{best.f1:.3f}", (best.f1 + best.f1_sd, n - 1), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=FS_ANN)
    ax.set_title("Frozen web-scale vision-language features lead every regime")
    h1 = [Patch(color=c, label=g) for g, c in GROUP_COLOR.items()]
    h2 = [Line2D([], [], ls="", marker=m, color="#333333", label=REGIME_LABEL[k], markersize=5)
          for k, m in REGIME_MARKER.items()]
    h2.append(Line2D([], [], ls="", marker="s", markerfacecolor="white", markeredgecolor="#333333",
                     label="3 folds only", markersize=5))
    leg1 = ax.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.0, 0.30), title="Pretraining source",
                     title_fontsize=FS_ANN, alignment="left")
    ax.add_artist(leg1)
    ax.legend(handles=h2, loc="lower left", bbox_to_anchor=(0.0, 0.10), ncol=2, title="Adaptation regime",
              title_fontsize=FS_ANN, alignment="left")
    save(fig, "fig_b_benchmark")

    out = pd.DataFrame({
        "Method": bm.label, "Pretraining": bm.group, "Regime": bm.regime.map(REGIME_LABEL), "Folds": bm.folds,
        "Macro-F1": [f"{a:.3f} ± {b:.3f}" for a, b in zip(bm.f1, bm.f1_sd)],
        "Accuracy": [f"{a:.3f} ± {b:.3f}" for a, b in zip(bm.acc, bm.acc_sd)],
        "In-scope acc.": bm.inscope_acc.map(lambda v: "" if pd.isna(v) else f"{v:.3f}"),
        "Trainable params (M)": bm.trainable_M_all.map(lambda v: f"{v:.3f}" if v < 1 else f"{v:.1f}"),
        "Train min/fold": bm.train_min_per_fold.map(lambda v: "" if pd.isna(v) else f"{v:.1f}"),
        "Peak VRAM (GB)": bm.peak_vram_gb.map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
    })
    write_table(out, "benchmark_main",
                "Main benchmark: grouped 5-fold CV on data v2 (results/benchmark.csv, fixed LR schedule). "
                "Probe trainable parameters are the logistic-regression head (20 x (D+1)).", align="llllrrrrrr")
    src = "results/benchmark.csv"
    for nm in ("probe_siglip_so400m_384", "zeroshot_siglip_so400m_384", "siglip_so400m_384_partial_v2", "siglip2_b16",
               "clip_vitb16", "dinov2_vitb14", "vit_b16", "convnext_tiny", "swin_t", "efficientnet_v2_s",
               "resnet50", "mobilenetv3_large", "scratch_cnn", "hog_hsv_svm", "probe_resnet50"):
        r = bm[bm.name == nm].iloc[0]
        key(f"cv_{nm}", {"f1": r.f1, "f1_sd": r.f1_sd, "acc": r.acc, "inscope_acc": r.inscope_acc,
                         "folds": int(r.folds)}, src)
    imnet = bm[(bm.regime == "fine-tune") & bm.group.isin(["ImageNet CNN", "ImageNet transformer"])]
    key("imagenet_finetune_f1_range", [imnet.f1.min(), imnet.f1.max()], src)
    key("imagenet_finetune_acc_range", [imnet.acc.min(), imnet.acc.max()], src)


# ----------------------------------------------------------------------------------------------- (c) Pareto
def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    best, keep = -np.inf, np.zeros(len(x), bool)
    for i in order:
        if y[i] > best:
            keep[i], best = True, y[i]
    return keep


def fig_pareto(d: dict, bm: pd.DataFrame) -> None:
    ft = bm[bm.regime == "fine-tune"].reset_index(drop=True)
    pr = bm[(bm.regime == "linear_probe") & ~bm.name.str.endswith("_tta")].reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True, gridspec_kw={"wspace": 0.12})
    short = {k: v[0].replace(" So400m/14@384", " So400m").replace("@336", "") for k, v in BACKBONE.items()}

    def draw(ax, df, xcol, logx):
        x, y = df[xcol].to_numpy(float), df.f1.to_numpy()
        keep = pareto_mask(x, y)
        o = np.argsort(x[keep])
        ax.step(x[keep][o], y[keep][o], where="post", color="#BBBBBB", lw=0.9, zorder=1)
        for _, r in df.iterrows():
            col = GROUP_COLOR[r.group]
            hollow = r.folds != 5
            ax.errorbar(r[xcol], r.f1, yerr=r.f1_sd, fmt="none", ecolor=col, elinewidth=0.7, zorder=2)
            ax.scatter(r[xcol], r.f1, s=26, marker=REGIME_MARKER[r.regime], facecolor="white" if hollow else col,
                       edgecolor=col, lw=1.0, zorder=3)
        if logx:
            ax.set_xscale("log")
        return keep

    ft_lab = ft.assign(lbl=ft.backbone.map(short))
    draw(a1, ft_lab, "train_min_per_fold", True)
    offs1 = {"siglip_so400m_384": (-6, 0, "right"), "siglip2_b16": (-6, 3, "right"), "clip_vitb16": (6, -1, "left"),
             "dinov2_vitb14": (6, 0, "left"), "vit_b16": (-6, 8, "right"), "convnext_tiny": (-6, 2, "right"),
             "swin_t": (3, -12, "left"), "efficientnet_v2_s": (6, -3, "left"), "resnet50": (-6, 0, "right"),
             "mobilenetv3_large": (0, -9, "center"), "scratch_cnn": (6, 0, "left")}
    for _, r in ft_lab.iterrows():
        dx, dy, ha = offs1[r.backbone]
        txt = r.lbl + (" (partial, 3 folds)" if r.folds != 5 else "")
        a1.annotate(txt, (r.train_min_per_fold, r.f1), xytext=(dx, dy), textcoords="offset points",
                    fontsize=FS_TICK, ha=ha, va="center", color=GROUP_COLOR[r.group])
    a1.set_xlabel("Fine-tuning time per fold (min, RTX 4060 Ti; log scale)")
    a1.xaxis.set_major_locator(mpl.ticker.FixedLocator([0.5, 1, 2, 5, 10, 20]))
    a1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a1.set_ylabel("Macro-F1 (grouped 5-fold CV)")
    a1.set_xlim(0.4, 30)
    a1.set_title("Accuracy tracks pretraining, not training time")
    panel_letter(a1, "a")

    both = pd.concat([ft, pr], ignore_index=True)
    both["x"] = both.trainable_M_all * 1e6
    keep = draw(a2, both, "x", True)
    offs2 = {("siglip_so400m_384", "linear_probe"): (6, 2, "left"), ("siglip2_b16", "linear_probe"): (6, 1, "left"),
             ("dinov2_vits14", "linear_probe"): (2, -9, "left"), ("convnext_tiny", "linear_probe"): (6, 3, "left"),
             ("resnet50", "linear_probe"): (6, 0, "left"), ("siglip_so400m_384", "fine-tune"): (-7, 0, "right"),
             ("scratch_cnn", "fine-tune"): (-7, 0, "right"), ("mobilenetv3_large", "fine-tune"): (-7, 0, "right")}
    for (_, r), k in zip(both.iterrows(), keep):
        o2 = offs2.get((r.backbone, r.regime))
        if o2 is None:
            continue
        lbl = short[r.backbone] + (" probe" if r.regime == "linear_probe" else
                                   (" partial FT (3 folds)" if r.folds != 5 else " FT"))
        a2.annotate(lbl, (r.x, r.f1), xytext=o2[:2], textcoords="offset points", fontsize=FS_TICK,
                    color=GROUP_COLOR[r.group], va="center", ha=o2[2])
    a2.set_xlabel("Trainable parameters (log scale)")
    a2.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else (f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}")))
    a2.set_title("23k-parameter probe ≈ 76M-parameter fine-tune")
    a2.legend(handles=[Line2D([], [], ls="", marker="o", color="#333333", label="linear probe", markersize=5),
                       Line2D([], [], ls="", marker="s", color="#333333", label="fine-tune", markersize=5),
                       Line2D([], [], color="#BBBBBB", label="Pareto front")], loc="lower right")
    panel_letter(a2, "b")
    a1.text(0.02, 0.02, "higher = better ↑", transform=a1.transAxes, fontsize=FS_TICK, va="bottom", ha="left",
            color="#555555")
    save(fig, "fig_c_pareto")


# ----------------------------------------------------------------------------------------------- (d) probe vs FT
PAIRS = [("siglip_so400m_384", "probe_siglip_so400m_384", "siglip_so400m_384_partial_v2"),
         ("siglip2_b16", "probe_siglip2_b16", "siglip2_b16"), ("clip_vitb16", "probe_clip_vitb16", "clip_vitb16"),
         ("dinov2_vitb14", "probe_dinov2_vitb14", "dinov2_vitb14"),
         ("convnext_tiny", "probe_convnext_tiny", "convnext_tiny"), ("resnet50", "probe_resnet50", "resnet50")]


def fig_probe_vs_ft(d: dict) -> None:
    F = d["perfold"].pivot_table(index="name", columns="fold", values="macro_f1")
    rows = []
    for bb, p, f in PAIRS:
        dd = (F.loc[f] - F.loc[p]).dropna()
        rows.append(dict(backbone=bb, display=BACKBONE[bb][0], group=BACKBONE[bb][2], probe=F.loc[p, dd.index].mean(),
                         finetune=F.loc[f, dd.index].mean(), delta=dd.mean(), wins=int((dd > 0).sum()), n=len(dd),
                         folds=list(dd.index), deltas=dd.to_numpy(),
                         kind="partial FT (last 4 blocks)" if "partial" in f else "full FT"))
    df = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    for i, r in df.iterrows():
        col = GROUP_COLOR[r.group]
        ax.scatter(100 * r.deltas, np.full(r.n, i) + (np.arange(r.n) - (r.n - 1) / 2) * 0.08, s=12, color=col,
                   alpha=0.55, lw=0, zorder=2)
        ax.plot([100 * r.delta] * 2, [i - 0.3, i + 0.3], color=col, lw=2.0, zorder=3)
        ax.text(11.8, i, f"{r.wins}/{r.n}", va="center", ha="right", fontsize=FS_ANN, color=col)
    ax.axvline(0, color="#333333", lw=0.7)
    ax.set_yticks(range(len(df)), [f"{r.display} ({'partial FT' if 'partial' in r.kind else 'full FT'})"
                                   for _, r in df.iterrows()])
    for t, g in zip(ax.get_yticklabels(), df.group):
        t.set_color(GROUP_COLOR[g])
    ax.set_xlim(-6, 12)
    ax.set_xlabel("Macro-F1 gain of fine-tuning over the frozen linear probe (points, per fold)")
    ax.set_ylim(-1.0, len(df) - 0.4)
    ax.text(11.8, -0.8, "folds where FT wins", ha="right", va="center", fontsize=FS_TICK, color="#555555")
    ax.set_title("Fine-tuning helps ImageNet, DINOv2 and CLIP-B backbones, not SigLIP")
    save(fig, "fig_d_probe_vs_finetune")
    out = df.assign(probe=df.probe.round(4), finetune=df.finetune.round(4), delta_pts=(100 * df.delta).round(2),
                    wins=[f"{w}/{n}" for w, n in zip(df.wins, df.n)],
                    folds=df.folds.map(lambda v: ",".join(map(str, v))))[
        ["display", "kind", "folds", "probe", "finetune", "delta_pts", "wins"]]
    out.columns = ["Backbone", "Fine-tune type", "Folds", "Probe macro-F1", "Fine-tune macro-F1", "Δ (pts)",
                   "FT wins"]
    write_table(out, "probe_vs_finetune", "Paired probe vs fine-tune comparison on identical folds (macro-F1).",
                align="llrrrrr")
    for _, r in df.iterrows():
        key(f"probe_vs_ft_{r.backbone}", {"delta_pts": round(100 * r.delta, 2), "wins": f"{r.wins}/{r.n}"},
            "results/phase1/raw/runs.jsonl + results/phase2/runs_fixed.jsonl")


# ----------------------------------------------------------------------------------------------- (e) leakage
def fig_leakage(d: dict) -> None:
    lk = d["leak"].copy()
    lk["display"] = lk.model.map(lambda m: BACKBONE[m][0] + " probe")
    lk["group"] = lk.model.map(lambda m: BACKBONE[m][2])
    lk = lk.sort_values("grouped_cv_acc").reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True, gridspec_kw={"wspace": 0.12})
    for i, r in lk.iterrows():
        col = GROUP_COLOR[r.group]
        a1.annotate("", xy=(100 * r.legacy_style_acc, i), xytext=(100 * r.grouped_cv_acc, i),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, shrinkA=3, shrinkB=3))
        a1.scatter(100 * r.grouped_cv_acc, i, s=30, color=col, zorder=3)
        a1.scatter(100 * r.legacy_style_acc, i, s=30, facecolor="white", edgecolor=col, lw=1.0, zorder=3)
        a1.text(100 * r.legacy_style_acc + 1.2, i, f"+{r.inflation_pts:.1f} pt", va="center", fontsize=FS_ANN,
                color=col)
        a2.scatter(100 * r.acc_on_clean, i, s=30, marker="o", facecolor="white", edgecolor=col, lw=1.0, zorder=3)
        a2.scatter(100 * r.acc_on_leaked, i, s=30, marker="o", color=col, zorder=3)
        a2.plot([100 * r.acc_on_clean, 100 * r.acc_on_leaked], [i, i], color=col, lw=0.8, zorder=2)
    a1.set_yticks(range(len(lk)), lk.display)
    a1.set_ylim(-1.1, len(lk) - 0.4)
    a1.set_xlim(48, 96)
    a1.set_xlabel("Accuracy (%)")
    a1.set_title("Legacy split inflates weaker models most")
    a1.legend(handles=[Line2D([], [], ls="", marker="o", color="#333333", label="grouped CV (dev, honest)"),
                       Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor="#333333",
                              label="legacy-style split")], loc="lower right", fontsize=FS_TICK)
    a2.set_xlim(48, 102)
    a2.set_xlabel("Accuracy on legacy eval rows (%)")
    n_leak, n_eval = int(lk.leaked_rows.iloc[0]), int(lk.legacy_eval_rows.iloc[0])
    a2.set_title("Duplicated rows are near-memorised")
    a2.legend(handles=[Line2D([], [], ls="", marker="o", color="#333333", label=f"rows duplicated in train (n={n_leak})"),
                       Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor="#333333",
                              label=f"other rows (n={n_eval - n_leak})")], loc="lower left", fontsize=FS_TICK)
    panel_letter(a1, "a")
    panel_letter(a2, "b")
    save(fig, "fig_e_leakage")
    out = lk[["display", "grouped_cv_acc", "legacy_style_acc", "inflation_pts", "acc_on_leaked", "acc_on_clean"]].copy()
    out.columns = ["Model", "Grouped-CV acc.", "Legacy-style acc.", "Inflation (pts)", "Acc. on duplicated rows",
                   "Acc. on other rows"]
    write_table(out.round(4), "leakage_simulation",
                f"Leakage simulation (results/leakage_simulation.csv): legacy-style split, {n_eval} evaluation rows of "
                f"which {n_leak} duplicate a training image.")
    key("leakage", lk[["model", "legacy_style_acc", "grouped_cv_acc", "inflation_pts", "acc_on_leaked",
                       "acc_on_clean"]].to_dict("records"), "results/leakage_simulation.csv")
    key("leakage_rows", {"eval": n_eval, "leaked": n_leak, "share": round(n_leak / n_eval, 4)},
        "results/leakage_simulation.csv")


# ----------------------------------------------------------------------------------------------- (f) confusion
def fig_confusion(d: dict) -> None:
    P = d["preds"]
    classes = d["labels"]
    y, probs, ids = P["y"], P["probs"], P["ids"]
    pred = probs.argmax(1)
    C = len(classes)
    cm = np.zeros((C, C), int)
    np.add.at(cm, (y, pred), 1)
    jt = d["test"]
    assert np.array_equal(cm, np.array(jt["confusion"])), "confusion from preds.npz != final_test_v2.json"
    acc = float((pred == y).mean())
    mf1 = macro_f1(y, pred, C)
    assert abs(acc - jt["accuracy"]) < 1e-9 and abs(mf1 - jt["macro_f1"]) < 1e-9
    flags = {r["id"]: r["flags"] for r in d["manifest"]["rows"]}
    ins = np.array(["ood_suspect" not in flags[str(i)] for i in ids])
    src = "results/final/final_test_v2_preds.npz (+ final_test_v2.json)"
    se = math.sqrt(acc * (1 - acc) / len(y))
    key("test_n", len(y), src)
    key("test_accuracy", round(acc, 4), src)
    key("test_macro_f1", round(mf1, 4), src)
    key("test_accuracy_se_pts", round(100 * se, 2), "binomial SE sqrt(p(1-p)/n) from test accuracy")
    key("test_accuracy_wilson95", [round(v, 4) for v in wilson(int((pred == y).sum()), len(y))], src)
    key("test_inscope", {"n": int(ins.sum()), "acc": round(float((pred[ins] == y[ins]).mean()), 4),
                         "macro_f1": round(macro_f1(y[ins], pred[ins], C), 4)}, src)
    key("test_ood_rows", {"n": int((~ins).sum()), "acc": round(float((pred[~ins] == y[~ins]).mean()), 4)}, src)
    key("test_probe_only_accuracy", round(jt["probe_only_accuracy"], 4), "results/final/final_test_v2.json")
    sw = (y == classes.index("switch"))
    key("test_switch_ood", {"n": int(sw.sum()), "ood_flagged": int((sw & ~ins).sum()),
                            "ood_flagged_correct": int((sw & ~ins & (pred == y)).sum())}, src)

    rec = cm.diagonal() / cm.sum(1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1c = np.where(rec + prec > 0, 2 * rec * prec / np.maximum(rec + prec, 1e-12), 0)
    cmn = cm / cm.sum(1, keepdims=True)
    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 0.75], wspace=0.95)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    im = a1.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(C):
        for j in range(C):
            if cm[i, j]:
                a1.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=4.5,
                        color="white" if cmn[i, j] > 0.55 else "#222222")
    a1.set_xticks(range(C), classes, rotation=90)
    a1.set_yticks(range(C), classes)
    a1.tick_params(labelsize=5.5)
    a1.set_xlabel("Predicted class")
    a1.set_ylabel("True class")
    a1.set_title(f"Sealed test (n={len(y)}): errors are diffuse, not systematic")
    cb = fig.colorbar(im, ax=a1, fraction=0.04, pad=0.02)
    cb.set_label("Row-normalised rate (cell text = count)", fontsize=FS_TICK)
    cb.ax.tick_params(labelsize=FS_TICK)
    panel_letter(a1, "a")

    o = np.argsort(rec)
    n_c = cm.sum(1)
    lo_hi = np.array([wilson(int(cm[i, i]), int(n_c[i])) for i in range(C)])
    yy = np.arange(C)
    xerr = np.clip([rec[o] - lo_hi[o, 0], lo_hi[o, 1] - rec[o]], 0, None)  # float round-off at recall = 1
    a2.errorbar(rec[o], yy, xerr=xerr, fmt="o", color="#0072B2",
                ecolor="#9ECAE1", elinewidth=1.2, markersize=3.5, capsize=0)
    a2.set_yticks(yy, [f"{classes[i]} (n={n_c[i]})" for i in o])
    a2.tick_params(axis="y", labelsize=5.5)
    a2.axvline(acc, color="#333333", lw=0.7, ls="--")
    a2.text(acc - 0.015, C - 0.2, f"overall acc. {acc:.2f}", fontsize=FS_TICK, ha="right", va="bottom")
    a2.set_xlim(0.2, 1.02)
    a2.set_ylim(-0.7, C + 0.6)
    a2.set_xlabel("Recall (95% Wilson CI)")
    a2.set_title("Resistor recall is lowest")
    panel_letter(a2, "b")
    save(fig, "fig_f_confusion")

    # top confusions
    off = [(cm[i, j], classes[i], classes[j]) for i in range(C) for j in range(C) if i != j and cm[i, j] >= 2]
    off.sort(reverse=True)
    key("test_top_confusions", [f"{a} -> {b}: {k}" for k, a, b in off[:10]], src)
    per = pd.DataFrame({"Class": classes, "Support": n_c, "Precision": prec.round(3), "Recall": rec.round(3),
                        "F1": f1c.round(3),
                        "Recall 95% CI": [f"{a:.2f}–{b:.2f}" for a, b in lo_hi]}).sort_values("Recall")
    write_table(per, "sealed_test_per_class", "Sealed-test per-class metrics of the final model (418 images).",
                align="lrrrrr")
    worst = per.iloc[0]
    key("test_worst_class", {"class": worst.Class, "recall": float(worst.Recall), "precision": float(worst.Precision),
                             "support": int(worst.Support)}, src)
    r_row = cm[classes.index("resistor")]
    key("test_resistor_misrouted_to", {classes[j]: int(r_row[j]) for j in np.argsort(-r_row) if
                                       j != classes.index("resistor") and r_row[j] > 0}, src)


# ----------------------------------------------------------------------------------------------- (g) calibration
def fig_calibration(d: dict) -> None:
    P = d["preds"]
    y, probs = P["y"], P["probs"]
    conf, pred = probs.max(1), probs.argmax(1)
    corr = pred == y
    e = ece(probs, y)
    jt, fit = d["test"], d["fit"]
    assert abs(e - jt["ece"]) < 1e-9, (e, jt["ece"])
    thr = fit["abstain_threshold"]
    key("test_ece", round(e, 4), "results/final/final_test_v2_preds.npz (15-bin ECE as in compclass.final)")
    key("oof_ece_before_after", [round(fit["ece_before"], 4), round(fit["ece_after"], 4)],
        "exports/final/fit_report.json")
    key("temperature", round(fit["temperature"], 4), "exports/final/fit_report.json")
    key("abstain_threshold", thr, "exports/final/fit_report.json")
    key("test_coverage_accepted_acc", [round(jt["coverage"], 4), round(jt["accepted_accuracy"], 4)],
        "results/final/final_test_v2.json")
    edges = np.linspace(0, 1, 16)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"wspace": 0.32})
    mids, accs, cnts, confs = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            mids.append((lo + hi) / 2)
            accs.append(corr[m].mean())
            cnts.append(int(m.sum()))
            confs.append(conf[m].mean())
    w = edges[1] - edges[0]
    a1.bar(mids, accs, width=w * 0.92, color="#0072B2", alpha=0.85, label="accuracy in bin")
    a1.plot([0, 1], [0, 1], color="#333333", lw=0.7, ls="--", label="perfect calibration")
    a1.scatter(confs, accs, s=8, color="#08306B", zorder=3, label="mean confidence in bin")
    for mx, ac, n in zip(mids, accs, cnts):
        a1.text(mx, min(ac + 0.03, 1.02), str(n), ha="center", fontsize=4.8, color="#333333")
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1.08)
    a1.set_xlabel("Predicted confidence (max softmax, temperature-scaled)")
    a1.set_ylabel("Observed accuracy")
    a1.set_title(f"Sealed test calibration: ECE = {e:.3f} (bin text = n)")
    a1.legend(loc="upper left", fontsize=FS_TICK)
    panel_letter(a1, "a")

    tab = pd.DataFrame(fit["abstain_table"])
    tcov = [(conf >= t).mean() for t in tab.threshold]
    tacc = [corr[conf >= t].mean() if (conf >= t).any() else np.nan for t in tab.threshold]
    tab["test_coverage"], tab["test_accepted_acc"] = tcov, tacc
    a2.plot(100 * tab.coverage, 100 * tab.accepted_acc, "-o", color="#999999", markersize=2.5,
            label="dev out-of-fold (n=2,512)")
    a2.plot(100 * tab.test_coverage, 100 * tab.test_accepted_acc, "-o", color="#0072B2", markersize=2.5,
            label=f"sealed test (n={len(y)})")
    sel = tab[tab.threshold == thr].iloc[0]
    a2.scatter([100 * sel.coverage], [100 * sel.accepted_acc], s=40, facecolor="none", edgecolor="#333333", zorder=4)
    a2.scatter([100 * sel.test_coverage], [100 * sel.test_accepted_acc], s=40, facecolor="none", edgecolor="#0072B2",
               zorder=4)
    a2.annotate(f"threshold {thr:.2f}: test coverage {100 * sel.test_coverage:.0f}%,\naccepted acc. "
                f"{100 * sel.test_accepted_acc:.1f}%", (100 * sel.test_coverage, 100 * sel.test_accepted_acc),
                xytext=(50, 87.0), textcoords="data", fontsize=FS_TICK, ha="left",
                arrowprops=dict(arrowstyle="-", color="#0072B2", lw=0.6))
    a2.axhline(90, color="#333333", lw=0.6, ls=":")
    a2.text(48, 90.3, "90% target", fontsize=FS_TICK, va="bottom")
    a2.set_xlabel("Coverage: images answered (%)")
    a2.set_ylabel("Accuracy on answered images (%)")
    a2.set_xlim(46, 102)
    a2.set_title("Abstaining on low confidence trades coverage for accuracy")
    a2.legend(loc="lower left", fontsize=FS_TICK)
    panel_letter(a2, "b")
    save(fig, "fig_g_calibration_abstention")
    out = tab.rename(columns={"threshold": "Threshold", "coverage": "OOF coverage", "accepted_acc": "OOF accepted acc.",
                              "test_coverage": "Test coverage", "test_accepted_acc": "Test accepted acc."}).round(4)
    write_table(out, "abstention", "Coverage vs accepted accuracy (OOF from exports/final/fit_report.json; "
                                   "test from results/final/final_test_v2_preds.npz).", align="rrrrr")


# ----------------------------------------------------------------------------------------------- (h) schedule
def fig_schedule(d: dict) -> None:
    O = d["old"].pivot_table(index="name", columns="fold", values="macro_f1")
    FX = d["fx"].pivot_table(index="name", columns="fold", values="macro_f1")
    names = list(d["sched"].name)  # the 10 models with identical recipes under both schedules
    D = FX.loc[names, [0, 1, 2]] - O.loc[names, [0, 1, 2]]
    assert (D.mean(axis=1).round(4) - d["sched"].set_index("name").delta).abs().max() < 1e-4
    sd = float(np.std(D.to_numpy().ravel(), ddof=1))
    se3 = sd / math.sqrt(3)
    k, n = int((D.to_numpy() > 0).sum()), D.size
    p = sign_test_p(k, n)
    key("schedule_folds_improved", f"{k}/{n}", "results/phase2/runs_fixed.jsonl vs raw_oldschedule/runs_{A,B}.jsonl")
    key("schedule_mean_delta_pts", round(100 * float(D.to_numpy().mean()), 2), "same")
    key("schedule_perfold_delta_sd_pts", round(100 * sd, 2), "same")
    key("schedule_noise_floor_2se_pts", round(200 * se3, 2), "2 x SD(per-fold delta)/sqrt(3)")
    key("schedule_sign_test_p", round(p, 3), "exact two-sided binomial")
    order = D.mean(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axvspan(-200 * se3, 200 * se3, color="#EEEEEE", zorder=0)
    ax.axvline(0, color="#333333", lw=0.7)
    for i, nm in enumerate(order):
        bb = backbone_of(nm)
        col = GROUP_COLOR[BACKBONE[bb][2]]
        v = 100 * D.loc[nm].to_numpy()
        ax.scatter(v, np.full(3, i) + np.array([-0.12, 0, 0.12]), s=12, color=col, alpha=0.6, lw=0)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=col, lw=2)
        ax.text(5.4, i, f"{int((v > 0).sum())}/3", fontsize=FS_ANN, va="center", ha="right", color=col)
    ax.set_yticks(range(len(order)), [BACKBONE[backbone_of(nm)][0] for nm in order])
    for t, nm in zip(ax.get_yticklabels(), order):
        t.set_color(GROUP_COLOR[BACKBONE[backbone_of(nm)][2]])
    ax.set_xlim(-3, 5.5)
    ax.set_xlabel("Macro-F1 change, fixed − old schedule (points; folds 0–2)")
    ax.text(5.4, len(order) - 0.4, "folds improved", fontsize=FS_TICK, ha="right", va="bottom", color="#555555")
    ax.text(0, -1.05, f"grey band: ±2 SE noise floor (±{200 * se3:.1f} pt)", fontsize=FS_TICK, ha="center",
            va="center", color="#555555")
    ax.set_ylim(-1.5, len(order) - 0.2)
    ax.set_title(f"LR-schedule fix: {k}/{n} folds improved, no consistent effect")
    save(fig, "fig_h_schedule_ablation")
    out = pd.DataFrame({"Model": [BACKBONE[backbone_of(nm)][0] for nm in order],
                        "Old (bug)": O.loc[order, [0, 1, 2]].mean(axis=1).round(4).to_numpy(),
                        "Fixed": FX.loc[order, [0, 1, 2]].mean(axis=1).round(4).to_numpy(),
                        "Δ fold 0": (100 * D.loc[order, 0]).round(2).to_numpy(),
                        "Δ fold 1": (100 * D.loc[order, 1]).round(2).to_numpy(),
                        "Δ fold 2": (100 * D.loc[order, 2]).round(2).to_numpy(),
                        "Mean Δ (pts)": (100 * D.loc[order].mean(axis=1)).round(2).to_numpy()})
    write_table(out, "schedule_ablation", "LR-schedule ablation, macro-F1 on folds 0-2 (old = warm-up not restarted at "
                                          "unfreeze).")


# ----------------------------------------------------------------------------------------------- other ablations
def ablation_tables(d: dict, bm: pd.DataFrame) -> None:
    F = d["perfold"].pivot_table(index="name", columns="fold", values="macro_f1")
    rows = []
    for b in ("probe_siglip_so400m_384", "probe_siglip2_so400m_384"):
        dd = F.loc[b + "_tta"] - F.loc[b]
        rows.append(dict(Ablation="hflip TTA (probe features averaged over flip)", Model=BACKBONE[backbone_of(b)][0],
                         Baseline=round(F.loc[b].mean(), 4), Variant=round(F.loc[b + "_tta"].mean(), 4),
                         Delta=round(100 * dd.mean(), 2), Evidence=f"{int((dd > 0).sum())}/5 folds improved"))
        key(f"tta_{b}", {"delta_pts": round(100 * dd.mean(), 2), "folds_improved": f"{int((dd > 0).sum())}/5"},
            "results/phase1/raw/runs.jsonl + results/phase1b/runs.jsonl")
    cl = d["clean"].set_index("variant")
    for v in ("drop_issues", "drop_ood", "drop_ood+issues"):
        rows.append(dict(Ablation=f"training-data cleaning: {v} (all rows / in-scope acc.)",
                         Model="SigLIP So400m/14@384 probe",
                         Baseline=f"{cl.loc['keep_all', 'all_acc']:.4f} / {cl.loc['keep_all', 'in_acc']:.4f}",
                         Variant=f"{cl.loc[v, 'all_acc']:.4f} / {cl.loc[v, 'in_acc']:.4f}",
                         Delta=round(100 * (cl.loc[v, "in_acc"] - cl.loc["keep_all", "in_acc"]), 2),
                         Evidence="pooled OOF, accuracy; Δ = in-scope"))
    key("cleaning", cl.reset_index().to_dict("records"), "results/phase1/cleaning_ablation_siglip_so400m.csv")
    fit, jt = d["fit"], d["test"]
    rows.append(dict(Ablation="zero-shot fusion (w=0.25, T=50), deployable pipeline, OOF", Model="final model",
                     Baseline=round(fit["oof_probe_only_accuracy"], 4), Variant=round(fit["oof_accuracy"], 4),
                     Delta=round(100 * (fit["oof_accuracy"] - fit["oof_probe_only_accuracy"]), 2),
                     Evidence="accuracy, 2,512 dev rows"))
    rows.append(dict(Ablation="zero-shot fusion, sealed test", Model="final model",
                     Baseline=round(jt["probe_only_accuracy"], 4), Variant=round(jt["accuracy"], 4),
                     Delta=round(100 * (jt["accuracy"] - jt["probe_only_accuracy"]), 2),
                     Evidence="accuracy, 418 rows (SE ≈ 1.8 pt)"))
    key("fusion_oof", {"probe_only_acc": round(fit["oof_probe_only_accuracy"], 4), "fused_acc": round(fit["oof_accuracy"], 4),
                       "fused_macro_f1": round(fit["oof_macro_f1"], 4)}, "exports/final/fit_report.json")
    key("fusion_test", {"probe_only_acc": round(jt["probe_only_accuracy"], 4), "fused_acc": round(jt["accuracy"], 4)},
        "results/final/final_test_v2.json")
    # partial FT vs probe on the same folds is in probe_vs_finetune; old-schedule partial (137M trainable) for context
    o = bm  # noqa: F841
    old = d["bm"][(d["bm"].name == "siglip_so400m_384_partial")].iloc[0]
    pv2 = d["bm"][(d["bm"].name == "siglip_so400m_384_partial_v2")].iloc[0]
    probe012 = F.loc["probe_siglip_so400m_384", [0, 1, 2]].mean()
    rows.append(dict(Ablation="partial fine-tune, old schedule, 137M trainable", Model="SigLIP So400m/14@384",
                     Baseline=round(probe012, 4), Variant=round(old.f1, 4), Delta=round(100 * (old.f1 - probe012), 2),
                     Evidence="folds 0-2, macro-F1 vs probe"))
    rows.append(dict(Ablation="partial fine-tune (last 4 blocks), fixed schedule, 76M trainable",
                     Model="SigLIP So400m/14@384", Baseline=round(probe012, 4), Variant=round(pv2.f1, 4),
                     Delta=round(100 * (pv2.f1 - probe012), 2), Evidence="folds 0-2, macro-F1 vs probe"))
    key("partial_ft", {"probe_folds012_f1": round(probe012, 4), "old_137M_f1": round(old.f1, 4),
                       "v2_76M_f1": round(pv2.f1, 4), "old_min_per_fold": round(old.train_min_per_fold, 1),
                       "v2_min_per_fold": round(pv2.train_min_per_fold, 1)}, "results/benchmark.csv")
    out = pd.DataFrame(rows).rename(columns={"Delta": "Δ (pts)"})
    write_table(out, "ablations", "Ablations: TTA, training-data cleaning, zero-shot fusion and partial fine-tuning.",
                align="llllrl")


def model_card(d: dict) -> None:
    fit, jt, ex, pre = d["fit"], d["test"], d["export"], d["pre"]
    rows = [
        ("Backbone", f"{pre['model']} (frozen SigLIP So400m/14 image tower, {pre['input_size']} px)"),
        ("Heads", f"logistic-regression probe (C = {fit['final_C']}) + zero-shot text head, fused: "
                  f"log p_probe + {fit['w_zs']} log softmax({fit['t_zs']:g} cos)"),
        ("Preprocessing", f"resize short side {pre['resize_short_side']} ({pre['interpolation']}), centre crop "
                          f"{pre['input_size']}, mean/std {pre['mean'][0]}/{pre['std'][0]}, RGB, NCHW"),
        ("Temperature T", f"{fit['temperature']:.4f} (fit on dev OOF scores)"),
        ("OOF ECE before → after T", f"{fit['ece_before']:.4f} → {fit['ece_after']:.4f}"),
        ("Abstain threshold", f"{fit['abstain_threshold']} (smallest grid value with OOF accepted acc. ≥ 0.90)"),
        ("Dev OOF accuracy / macro-F1", f"{fit['oof_accuracy']:.4f} / {fit['oof_macro_f1']:.4f} (n = {fit['n_dev']})"),
        ("Sealed test accuracy / macro-F1", f"{jt['accuracy']:.4f} / {jt['macro_f1']:.4f} (n = {jt['n']})"),
        ("Sealed test in-scope accuracy / macro-F1",
         f"{jt['inscope_accuracy']:.4f} / {jt['inscope_macro_f1']:.4f} (n = {jt['inscope_n']})"),
        ("Sealed test probe-only accuracy", f"{jt['probe_only_accuracy']:.4f}"),
        ("Sealed test coverage / accepted accuracy", f"{jt['coverage']:.4f} / {jt['accepted_accuracy']:.4f}"),
        ("Sealed test ECE", f"{jt['ece']:.4f}"),
        ("ONNX size", f"{ex['onnx_mb']} MB (external data: {ex['external_data']})"),
        ("onnxruntime parity", f"max |Δp| = {ex['ort_max_abs_prob_diff']:.1e}, argmax agreement {ex['ort_argmax_agree']}"),
        ("onnxruntime CPU latency", f"{ex['ort_cpu_ms_per_image']} ms / image"),
        ("OpenCV DNN", f"fails: child process {ex['cv2_error'].strip()} (segmentation fault)"),
        ("Data", f"data/v2 manifest fingerprint {pre['manifest_fingerprint']}"),
    ]
    write_table(pd.DataFrame(rows, columns=["Field", "Value"]), "model_card",
                "Final model card (exports/final/*.json, results/final/final_test_v2.json).", align="lp{10cm}")
    key("export", ex, "exports/final/export_report.json")
    key("fit_report_core", {k: v for k, v in fit.items() if k != "abstain_table"}, "exports/final/fit_report.json")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    apply_figure_style(SIZES)
    d = load_all()
    check_benchmark(d)
    bm = main_table(d)
    for f in (fig_data_audit,):
        f(d)
    fig_benchmark(d, bm)
    fig_pareto(d, bm)
    fig_probe_vs_ft(d)
    fig_leakage(d)
    fig_confusion(d)
    fig_calibration(d)
    fig_schedule(d)
    ablation_tables(d, bm)
    model_card(d)
    (TAB / "key_numbers.json").write_text(json.dumps(KEY, indent=1, default=float))
    print("wrote", (TAB / "key_numbers.json").relative_to(ROOT), f"({len(KEY)} entries)")


if __name__ == "__main__":
    main()
