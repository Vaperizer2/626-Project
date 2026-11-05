"""The efficiency study's one figure: development-fold accuracy against measured batch-1 latency on the RTX 4070.

    uv run python tools/views/efficiency_figure.py        # -> docs/figures/efficiency_frontier.{png,svg}

x: onnxruntime CUDA EP fp16 median batch-1 latency (results/efficiency/latency_ort_cuda_fp16_with308.json, plus the
PE-Core L session); y: out-of-fold macro-F1 of the fused probe + zero-shot recipe on the five development folds
(EXPERIMENT_LOG.md 2026-11-05 entries). Two series: the team lead's SigLIP So400m trunks and the study's candidates.
Every point is labelled directly; the legend names the two series. Static image for docs/EFFICIENCY_STUDY.md.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/figures")
BLUE, ORANGE, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#1a1a19", "#5f5e59", "#e6e5e0", "#fcfcfb"

lat = {r["graph"].replace("\\", "/").split("/")[-2]: r["median_ms"]
       for f in ("latency_ort_cuda_fp16.json", "latency_ort_cuda_fp16_with308.json")  # the later session wins where both have a graph
       for r in json.load(open(Path("results/efficiency") / f))["rows"]}
POINTS = [  # name, latency key, fused OOF macro-F1, series, label offset (points)
    ("SigLIP So400m @384\n(Deployed trunk, crop 1.0)", "siglip_so400m_384", 0.8403, "ref", (-8, 10)),
    ("SigLIP So400m @308\n(Fast trunk, fp16)", "siglip_so400m_308", 0.8222, "ref", (8, -16)),
    ("PE-Core L/14 @336", "pe_core_l14_336", 0.8351, "cand", (8, -14)),
    ("MobileCLIP2-S3 @256\n(Small)", "mobileclip2_s3", 0.8013, "cand", (8, -6)),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    done = set()
    for name, key, f1, series, (dx, dy) in POINTS:
        x = lat[key]
        c = ORANGE if series == "ref" else BLUE
        lbl = "SigLIP So400m trunks (team lead)" if series == "ref" else "study candidates"
        ax.scatter(x, f1, s=64, color=c, edgecolor=SURFACE, linewidth=2, zorder=3, label=None if lbl in done else lbl)
        done.add(lbl)
        ax.annotate(name, (x, f1), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK,
                    ha="left" if dx > 0 else "right", va="center", linespacing=1.15)
    ax.set_xlim(0, 36)
    ax.set_ylim(0.78, 0.86)
    ax.set_xlabel("batch-1 latency, onnxruntime CUDA fp16, RTX 4070 Laptop (ms)")
    ax.set_ylabel("OOF macro-F1, fused recipe, 5 dev folds")
    ax.set_title("Accuracy against measured latency, 20 component classes", loc="left", fontsize=11, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"efficiency_frontier.{ext}", facecolor=SURFACE)
    print("wrote", OUT / "efficiency_frontier.png", {k: lat[k] for _, k, _, _, _ in POINTS})


if __name__ == "__main__":
    main()
