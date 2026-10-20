"""Helpers for notebooks/ITRI626_three_models.ipynb: loading saved results, tables and report figures.

Kept out of the notebook so its cells stay short. Nothing here trains a model; training lives in
scripts/itri626_experiments.py (the notebook calls it when RETRAIN=True).
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def find_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for q in [p, *p.parents]:
        if (q / "pyproject.toml").exists() and (q / "src" / "compclass").is_dir():
            return q
    raise FileNotFoundError("run the notebook from inside the 626-Project repository")


ROOT = find_root()
RES = ROOT / "results" / "itri626"
FIG = RES / "figures"
SPLITS = ROOT / "data" / "v2" / "splits"
V2 = ROOT / "data" / "v2"

MAIN = ["A_scratch_cnn", "B_convnext_tiny", "C_siglip_so400m_linear"]
REFS = ["R1_sklearn_probe", "R2_deployed_fused"]
SHORT = {"A_scratch_cnn": "A: scratch CNN", "B_convnext_tiny": "B: ConvNeXt-T (fine-tuned)",
         "C_siglip_so400m_linear": "C: SigLIP ViT (frozen + linear)", "R1_sklearn_probe": "R1: SigLIP + sklearn probe",
         "R2_deployed_fused": "R2: deployed fused model"}
# Okabe-Ito colours for the three main models (orange, blue, bluish green); neutral greys for the reference rows
COLOR = {"A_scratch_cnn": "#E69F00", "B_convnext_tiny": "#0072B2", "C_siglip_so400m_linear": "#009E73",
         "R1_sklearn_probe": "#999999", "R2_deployed_fused": "#555555"}
WRONG = "#B2182B"   # dark red: error marking, kept distinct from model A's orange
OK = "#1A7F37"      # dark green tick for correct predictions in the disagreement panel
TEXTW = 6.30        # inches = \\textwidth of the report (16 cm); report figures are drawn at their printed width
# compact class names for the thumbnail titles only (the captions spell them out)
ABBR = {"microcontroller board": "MCU board", "power supply module": "PSU module", "crystal oscillator": "crystal osc.",
        "battery holder": "batt. holder"}


def short_cls(c: str) -> str:
    return ABBR.get(c, c)


# ------------------------------------------------------------------------------------------------ style
def apply_figure_style(sizes=(8.5, 7.5, 7)) -> None:
    """Publication rcParams (open frame, 3-size font ladder, frameless legends, 300 dpi, embedded fonts)."""
    base, secondary, tick = sizes
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "legend.fontsize": secondary, "xtick.labelsize": tick, "ytick.labelsize": tick, "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.titleweight": "normal", "axes.titlelocation": "left",
        "lines.linewidth": 1.2, "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, name: str) -> list[Path]:
    """Save a report figure as PDF + PNG under results/itri626/figures/."""
    FIG.mkdir(parents=True, exist_ok=True)
    out = [FIG / f"{name}.pdf", FIG / f"{name}.png"]
    for p in out:
        fig.savefig(p)
    return out


# ------------------------------------------------------------------------------------------------ environment
def environment_report() -> dict:
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return "not installed"

    rep = {"python": platform.python_version(), "os": platform.platform()}
    for p in ["torch", "torchvision", "timm", "open_clip_torch", "scikit-learn", "numpy", "pandas", "onnxruntime",
              "matplotlib", "pillow"]:
        rep[p] = ver(p)
    try:
        import torch

        rep["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            rep["gpu"] = f"{pr.name}, {pr.total_memory / 2**30:.1f} GiB"
            rep["cuda (torch build)"] = torch.version.cuda
    except ImportError:
        pass
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                    if ln.startswith("model name")), cpu)
    rep["cpu"] = f"{cpu} ({os.cpu_count()} logical cores)"
    if Path("/proc/meminfo").exists():
        kb = next(int(ln.split()[1]) for ln in Path("/proc/meminfo").read_text().splitlines()
                  if ln.startswith("MemTotal"))
        rep["ram"] = f"{kb / 2**20:.1f} GiB"
    return rep


# ------------------------------------------------------------------------------------------------ loading
def load_json(p: Path):
    return json.loads(Path(p).read_text())


def load_splits() -> dict[str, pd.DataFrame]:
    return {s: pd.read_csv(SPLITS / f"{s}.csv", dtype={"id": str}) for s in ("train", "val", "test")}


def load_preds(name: str, split: str = "test") -> dict:
    z = np.load(RES / "preds" / f"{name}_{split}.npz", allow_pickle=False)
    return {"ids": z["ids"].astype(str), "y": z["y"], "probs": z["probs"], "classes": list(z["classes"].astype(str))}


def img_path(rel: str) -> Path:
    return V2 / rel


def thumb(path: Path, size: int = 160) -> Image.Image:
    """Centre square crop, resized (display only)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return im.resize((size, size), Image.BICUBIC)


# ------------------------------------------------------------------------------------------------ figures
# Every report figure is drawn at the width it is printed at (see TEXTW), so the font sizes set here are the
# printed sizes (>= 6.5 pt). main.tex includes each figure at that width.

def _frame(ax, color="#BBBBBB", lw=0.5):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)


def fig_class_counts(summary: pd.DataFrame):
    s = summary[summary["class"] != "TOTAL"].sort_values("final_rows")
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 3.6), sharey=True, layout="constrained")
    y = np.arange(len(s))
    ax = axes[0]
    ax.barh(y, s["raw_files"], color="#D9D9D9", label="raw files (Kaggle download)")
    ax.barh(y, s["final_rows"], color="#4D4D4D", label="final (after cleaning)")
    ax.set_yticks(y, s["class"])
    ax.set_xlabel("images per class")
    ax.margins(x=0.02, y=0.01)
    ax = axes[1]
    left = np.zeros(len(s))
    for col, c, lab in (("train", "#4D4D4D", "train"), ("val", "#8C8C8C", "validation"), ("test", "#C8C8C8", "test")):
        ax.barh(y, s[col], left=left, color=c, label=lab)
        left += s[col].to_numpy()
    ax.set_xlabel("images per class (final)")
    ax.margins(x=0.02, y=0.01)
    ax.tick_params(axis="y", length=0)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, handlelength=1.2, borderaxespad=0.2)
    axes[1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.2, columnspacing=1.0,
                   borderaxespad=0.2)
    return fig


def _two_line(c: str, n: int, width: int = 15) -> str:
    """Class name and training count (in brackets) on exactly two lines; long names wrap at the last space."""
    if len(c) <= width:
        return f"{c}\n({n})"
    head, tail = c.rsplit(" ", 1)
    return f"{head}\n{tail} ({n})"


def fig_examples(train: pd.DataFrame, classes: list[str], per_class: int = 4, seed: int = 0, size: int = 110):
    """One 2x2 mosaic of random training images per class (rows without the OOD flag preferred)."""
    rng = np.random.default_rng(seed)
    ncol = 5
    nrow = int(np.ceil(len(classes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.05, 1.06 * nrow))
    for ax, c in zip(axes.flat, classes):
        rows = train[train["label"] == c]
        pool = rows[rows["ood_suspect"] == 0]
        pool = pool if len(pool) >= per_class else rows
        pick = pool.iloc[rng.choice(len(pool), per_class, replace=False)]
        tiles = [thumb(img_path(p), size) for p in pick["path"]]
        k = int(np.sqrt(per_class))
        mosaic = Image.new("RGB", (k * size + (k - 1) * 3, k * size + (k - 1) * 3), "white")
        for i, t in enumerate(tiles):
            mosaic.paste(t, ((i % k) * (size + 3), (i // k) * (size + 3)))
        ax.imshow(mosaic)
        ax.set_title(_two_line(c, len(rows)), fontsize=6.5, loc="center", pad=2, linespacing=1.05)
        ax.axis("off")
    for ax in list(axes.flat)[len(classes):]:
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.05, hspace=0.30)
    return fig


ROWLABEL = {"A_scratch_cnn": "A: scratch CNN", "B_convnext_tiny": "B: ConvNeXt-T\n(fine-tuned)",
            "C_siglip_so400m_linear": "C: SigLIP ViT\n(frozen + linear)"}


def fig_augmentation(path: Path, specs: dict, augs: dict, n: int = 5, seed: int = 0):
    """Original image, the eval crop and n random training augmentations for each model."""
    import torch
    from compclass.data import build_transforms
    from compclass.models.registry import PreprocessSpec

    with Image.open(path) as im:
        im = im.convert("RGB")
    fig, axes = plt.subplots(len(specs), n + 2, figsize=(TEXTW, 0.74 * len(specs) + 0.2))
    axes = np.atleast_2d(axes)
    for r, (name, sd) in enumerate(specs.items()):
        spec = PreprocessSpec(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in sd.items()
                                 if k != "resize_short_side"})
        mean, std = np.array(spec.mean)[:, None, None], np.array(spec.std)[:, None, None]
        show = lambda x: np.clip((x.numpy() * std + mean).transpose(1, 2, 0), 0, 1)  # noqa: E731
        axes[r, 0].imshow(im)
        axes[r, 1].imshow(show(build_transforms(spec, None, train=False)(im)))
        tr = build_transforms(spec, augs[name], train=True) if augs.get(name) is not None else None
        torch.manual_seed(seed)
        for j in range(n):
            ax = axes[r, j + 2]
            if tr is not None:
                ax.imshow(show(tr(im)))
        axes[r, 0].set_ylabel(ROWLABEL.get(name, SHORT[name]), fontsize=7, rotation=0, ha="right", va="center",
                              labelpad=4)
        for ax in axes[r]:
            _frame(ax, color="white", lw=0)
        if tr is None:  # one label across the empty augmentation cells instead of five repeats
            for ax in axes[r, 2:]:
                ax.set_visible(False)
            fig.canvas.draw()
            b0, b1 = axes[r, 2].get_position(), axes[r, -1].get_position()
            fig.text((b0.x0 + b1.x1) / 2, (b0.y0 + b0.y1) / 2,
                     "no augmentation (frozen features of the evaluation crop)",
                     ha="center", va="center", fontsize=7, color="#444444")
    axes[0, 0].set_title("original", fontsize=7, loc="center")
    axes[0, 1].set_title("eval crop", fontsize=7, loc="center")
    for j in range(n):
        axes[0, j + 2].set_title(f"train aug. {j + 1}", fontsize=7, loc="center")
    return fig


def fig_learning_curves(histories: dict, best: dict):
    """2 x 3 grid: losses (top) and accuracy / macro-F1 (bottom); one column per model; one shared legend."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, len(MAIN), figsize=(TEXTW, 3.25), sharex="col", sharey="row",
                             layout="constrained")
    for c_, name in enumerate(MAIN):
        h = pd.DataFrame(histories[name])
        e = h["epoch"] + 1
        c = COLOR[name]
        ax = axes[0, c_]
        ax.plot(e, h["train_ce"], color=c, lw=1.2)
        if name != "C_siglip_so400m_linear":
            ax.plot(e, h["train_loss"], color=c, lw=1.0, ls=":")
        ax.plot(e, h["val_loss"], color="black", lw=1.1, ls="--")
        ax.set_title(SHORT[name], loc="left", fontsize=8)
        ax = axes[1, c_]
        ax.plot(e, h["train_acc"], color=c, lw=1.2)
        ax.plot(e, h["val_acc"], color="black", lw=1.1, ls="--")
        ax.plot(e, h["val_macro_f1"], color=c, lw=1.0, ls="-.")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("epoch")
        for ax in axes[:, c_]:
            ax.axvline(best[name] + 1, color="#888888", lw=0.7)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.margins(x=0.02)
    axes[0, 0].set_ylabel("cross-entropy")
    axes[1, 0].set_ylabel("accuracy / macro-F1")
    g = "#555555"
    handles = [Line2D([], [], color=g, lw=1.2), Line2D([], [], color=g, lw=1.0, ls=":"),
               Line2D([], [], color="black", lw=1.1, ls="--"), Line2D([], [], color=g, lw=1.0, ls="-."),
               Line2D([], [], color="#888888", lw=0.7)]
    labels = ["training (model colour)", "training objective, label-smoothed", "validation loss / accuracy",
              "validation macro-F1", "selected epoch"]
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=7, handlelength=2.0, columnspacing=1.5)
    return fig


def fig_test_metrics(table: pd.DataFrame):
    """Grouped bars: rows of ``table`` = models, columns = metrics (notebook figure)."""
    metrics = list(table.columns)
    fig, ax = plt.subplots(figsize=(TEXTW, 2.6), layout="constrained")
    x = np.arange(len(metrics))
    w = 0.8 / len(table)
    for i, (name, row) in enumerate(table.iterrows()):
        off = (i - (len(table) - 1) / 2) * w
        ax.bar(x + off, row.values, w * 0.92, color=COLOR[name], label=SHORT[name])
        for xi, v in zip(x, row.values):
            ax.text(xi + off, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, [m.replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("test score (n = 418)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    return fig


def _draw_cm(ax, mat, classes, vmax, counts: bool, cell_fs=6.0, percent=False):
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = mat[i, j]
            if (counts and v > 0) or (not counts and v >= 0.05):
                if counts:
                    txt = f"{int(v)}"
                elif percent:
                    txt = f"{100 * v:.0f}"
                else:
                    txt = f"{v:.2f}"[1:] if v < 1 else "1"
                ax.text(j, i, txt, ha="center", va="center", fontsize=cell_fs,
                        color="white" if v > 0.6 * vmax else "black")
    return im


def fig_confusion(cm: np.ndarray, classes: list[str], name: str, vmax: float | None = None):
    """Counts and row-normalised matrix of one model (notebook figure). Pass ``vmax`` to share the count scale."""
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW * 1.25, 4.3), layout="constrained")
    for ax, mat, title, counts in ((axes[0], cm, "counts", True), (axes[1], norm, "row-normalised (recall)", False)):
        vm = (vmax or mat.max()) if counts else 1.0
        im = _draw_cm(ax, mat, classes, vm, counts, cell_fs=5.5)
        ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(classes)), classes if ax is axes[0] else [], fontsize=6.5)
        ax.set_xlabel("predicted class")
        ax.set_title(f"{SHORT[name]}: {title}", fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label("test images" if counts else "share of the true class", fontsize=7)
    axes[0].set_ylabel("true class")
    return fig


def fig_confusion_grid(cms: dict, classes: list[str]):
    """Report figure: (a)-(c) counts of A, B, C on one shared colour scale, (d) C row-normalised."""
    names = list(cms)
    mats = {n: np.asarray(cms[n]) for n in names}
    vmax = max(m.max() for m in mats.values())
    fig, axes = plt.subplots(2, 2, figsize=(TEXTW, 5.55), layout="constrained")
    panels = [(names[0], "counts", True), (names[1], "counts", True), (names[2], "counts", True),
              (names[2], "row-normalised, %", False)]
    ims = []
    for k, (ax, (n, what, counts)) in enumerate(zip(axes.flat, panels)):
        m = mats[n] if counts else mats[n] / np.maximum(mats[n].sum(1, keepdims=True), 1)
        ims.append(_draw_cm(ax, m, classes, vmax if counts else 1.0, counts, cell_fs=6.0, percent=True))
        ax.set_title(f"({'abcd'[k]}) {SHORT[n].split(' (')[0]}: {what}", fontsize=7.5, loc="left")
        ax.set_xticks(range(len(classes)), classes if k >= 2 else [], rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(classes)), classes if k % 2 == 0 else [], fontsize=6.5)
        ax.tick_params(length=2)
        if k >= 2:
            ax.set_xlabel("predicted class", fontsize=7.5)
        if k % 2 == 0:
            ax.set_ylabel("true class", fontsize=7.5)
    cb = fig.colorbar(ims[0], ax=axes[0, :].tolist(), shrink=0.85, pad=0.01, aspect=30)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label("test images, panels (a)-(c)", fontsize=7)
    cb2 = fig.colorbar(ims[3], ax=axes[1, :].tolist(), shrink=0.85, pad=0.01, aspect=30)
    cb2.ax.tick_params(labelsize=6.5)
    cb2.set_label("share of the true class, panel (d)", fontsize=7)
    cb2.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0", "20%", "40%", "60%", "80%", "100%"])
    return fig


def fig_per_class_f1(pc: pd.DataFrame, train_counts: pd.Series):
    piv = pc.pivot(index="class", columns="model", values="f1")[MAIN]
    piv = piv.sort_values(MAIN[-1])
    fig, ax = plt.subplots(figsize=(5.0, 2.8), layout="constrained")
    y = np.arange(len(piv))
    for i, (_, row) in enumerate(piv.iterrows()):
        ax.plot([row.min(), row.max()], [i, i], color="#CCCCCC", lw=0.8, zorder=1)
    for name in MAIN:
        ax.scatter(piv[name], y, s=22, color=COLOR[name], label=SHORT[name], zorder=3, linewidths=0)
    ax.set_yticks(y, [f"{c} ({train_counts[c]})" for c in piv.index])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, len(piv) - 0.4)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.xaxis.grid(True, color="#E5E5E5", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("test F1-score (higher = better)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, handletextpad=0.3,
              columnspacing=1.0, borderaxespad=0.2)
    return fig


def fig_gallery(df: pd.DataFrame, title: str = "", ncol: int = 6, size: int = 150):
    """``df`` rows: path, true, pred, conf, correct (bool). Errors get a red frame and a 'true -> pred' title.

    ``title`` is kept for compatibility with the notebook but no longer drawn (the report caption names the model).
    """
    nrow = int(np.ceil(len(df) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(TEXTW, 1.33 * nrow))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, df.iterrows()):
        ax.axis("on")
        ax.imshow(thumb(img_path(r["path"]), size))
        if r["correct"]:
            t = f"{short_cls(r['true'])}\n{r['conf']:.2f}"
            _frame(ax)
        else:
            t = f"{short_cls(r['true'])}\n\u2192 {short_cls(r['pred'])}\n{r['conf']:.2f}"
            _frame(ax, WRONG, 1.5)
        ax.set_title(t, fontsize=6.5, loc="center", pad=2, linespacing=1.05)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.86, wspace=0.08, hspace=0.42)
    return fig


DISAGREE_ROWS = ("C right,\nB wrong", "B right,\nC wrong", "all three\nwrong")


def fig_disagreement(df: pd.DataFrame, title: str = "", ncol: int = 4, size: int = 150, row_labels=DISAGREE_ROWS):
    """``df`` rows: path, true, and one prediction column per main model; one group of ``ncol`` per row.

    The title of each thumbnail is the true class, then each model's prediction (a tick if it is right).
    ``title`` is kept for compatibility but not drawn.
    """
    nrow = int(np.ceil(len(df) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.85, 1.27 * nrow))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, df.iterrows()):
        ax.axis("on")
        ax.imshow(thumb(img_path(r["path"]), size))
        _frame(ax)
        step = 8.0  # points per title line
        ax.annotate(short_cls(r["true"]), (0.5, 1), xycoords="axes fraction", xytext=(0, 2 + 3 * step),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
        for i, k in enumerate(MAIN):
            ok = r[k] == r["true"]
            txt = f"{k[0]}: \u2713" if ok else f"{k[0]}: {short_cls(r[k])} \u2717"
            ax.annotate(txt, (0.5, 1), xycoords="axes fraction", xytext=(0, 2 + step * (2 - i)),
                        textcoords="offset points", ha="center", va="bottom", fontsize=6.5,
                        color=OK if ok else WRONG)
    for rr, lab in enumerate(row_labels[:nrow]):
        axes[rr, 0].set_ylabel(lab, fontsize=6.5, rotation=90, labelpad=2)
        axes[rr, 0].yaxis.label.set_visible(True)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.01, top=0.87, wspace=0.12, hspace=0.98)
    return fig


AGREE_LABEL = {"111": "all three", "011": "B and C", "001": "C only", "101": "A and C", "110": "A and B",
               "010": "B only", "100": "A only", "000": "none"}


def fig_agreement(correct: pd.DataFrame):
    """Horizontal bars: test images by the set of models (A, B, C) that classify them correctly."""
    pat = correct[MAIN].astype(int).astype(str).agg("".join, axis=1)
    order = ["111", "011", "001", "101", "110", "010", "100", "000"]
    counts = pat.value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(2.25, 2.2), layout="constrained")
    y = np.arange(len(order))[::-1]
    ax.barh(y, counts.values, color="#4D4D4D", height=0.7)
    for yi, v in zip(y, counts.values):
        ax.text(v + 3, yi, str(v), va="center", fontsize=6.5)
    ax.set_yticks(y, [AGREE_LABEL[p] for p in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("test images")
    ax.set_ylabel("models that are right")
    ax.set_xlim(0, counts.max() * 1.18)
    return fig, counts


def fig_efficiency(eff: pd.DataFrame, acc: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.6), layout="constrained")
    cols = (("gpu_ms_per_image_batch1_fp16", "GPU latency, batch 1 (ms per image, log scale)"),
            ("params", "total parameters (log scale)"))
    for ax, (col, xl) in zip(axes, cols):
        for name, r in eff.iterrows():
            ax.scatter(r[col], acc[name], s=30, color=COLOR[name], zorder=3)
            ax.annotate(SHORT[name].split(": ")[1], (r[col], acc[name]), textcoords="offset points",
                        xytext=(5, -3), fontsize=6.5)
        ax.set_xscale("log")
        ax.set_ylabel("test accuracy")
        ax.set_xlabel(xl)
        ax.set_ylim(0, 1)
        ax.margins(x=0.35)
    axes[0].set_title("Accuracy vs inference time")
    axes[1].set_title("Accuracy vs model size")
    return fig


def fig_grouped(df: pd.DataFrame, title: str, ylabel: str):
    """Grouped bars: index = model, columns = groups (notebook figure)."""
    fig, ax = plt.subplots(figsize=(5.2, 2.6), layout="constrained")
    x = np.arange(len(df.columns))
    w = 0.8 / len(df)
    for i, (name, row) in enumerate(df.iterrows()):
        off = (i - (len(df) - 1) / 2) * w
        ax.bar(x + off, row.values, w * 0.92, color=COLOR[name], label=SHORT[name])
        for xi, v in zip(x, row.values):
            ax.text(xi + off, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, df.columns)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


def most_confused(cm: np.ndarray, classes: list[str], k: int = 8) -> pd.DataFrame:
    cm = np.asarray(cm)
    rows = [{"true": classes[i], "predicted": classes[j], "count": int(cm[i, j]),
             "share_of_true_class": round(cm[i, j] / max(cm[i].sum(), 1), 3)}
            for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
    return (pd.DataFrame(rows).sort_values(["count", "share_of_true_class"], ascending=False)
            .head(k).reset_index(drop=True))
