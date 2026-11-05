"""Efficiency study, step 1: the small-backbone probe sweep against SigLIP So400m at the same (crop 1.0) view.

    uv run python tools/views/small_sweep_summary.py [--results results/probes_win.jsonl]

For every key in registry section 6 with five fold records: CV macro-F1 / accuracy, paired Δ vs the SigLIP So400m
probe fitted on the crop-1.0 cache (and vs PE-Core L/14@336), parameters and GFLOPs per image
(torch FlopCounterMode; relative to SigLIP So400m at 384 px). Prints a markdown table sorted by macro-F1.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
from compclass.data import load_manifest, get_dev, get_fold
from compclass.features import fit_probe
from compclass.evaluate import compute_metrics
from compclass.models import build_model

KEYS = ["pe_core_t16_384", "pe_core_s16_384", "pe_core_b16_224", "pe_core_b16_336", "siglip2_b16_384", "mobileclip2_s2",
        "mobileclip2_b", "mobileclip2_s3", "dfn2b_b16_224", "metaclip_b16_224", "eva02_b16_clip_224",
        "convnext_base_clip_320", "pe_core_l14_336"]
LABEL = {"pe_core_t16_384": "PE-Core T/16@384", "pe_core_s16_384": "PE-Core S/16@384", "pe_core_b16_224": "PE-Core B/16@224",
         "pe_core_b16_336": "PE-Core B/16@336", "siglip2_b16_384": "SigLIP 2 B/16@384", "mobileclip2_s2": "MobileCLIP2-S2@256",
         "mobileclip2_b": "MobileCLIP2-B@224", "mobileclip2_s3": "MobileCLIP2-S3@256", "dfn2b_b16_224": "DFN-2B B/16@224",
         "metaclip_b16_224": "MetaCLIP B/16@224", "eva02_b16_clip_224": "EVA-02 B/16 CLIP@224",
         "convnext_base_clip_320": "ConvNeXt-B CLIP@320", "pe_core_l14_336": "PE-Core L/14@336"}
FP = "431315e87d2612cb-1fdfce4c916e"


def size(key):
    """Parameters, input size and GFLOPs per image (torch's FlopCounterMode, fp32, batch 1, random weights on CPU).
    Hybrid and convolutional encoders have no token count, so FLOPs are the compute axis for every backbone."""
    from torch.utils.flop_counter import FlopCounterMode

    m, spec = build_model(key, 0, pretrained=False)
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    with FlopCounterMode(display=False) as fc:  # grad left on: the module tracker needs grad_fns under attention pooling
        m(torch.zeros(1, 3, spec.input_size, spec.input_size))
    return n, spec.input_size, fc.get_total_flops() / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/probes_win.jsonl")
    a = ap.parse_args()
    m = load_manifest("data/v2/manifest.json")
    dev = get_dev(m)
    pos = {r["id"]: i for i, r in enumerate(dev)}
    y = np.array([m.class_to_idx[r["label"]] for r in dev])
    cl = np.array([r["cluster"] for r in dev])
    folds = [([pos[r["id"]] for r in tr], [pos[r["id"]] for r in va]) for tr, va in (get_fold(m, k) for k in range(5))]
    # reference: SigLIP So400m probe on the crop-1.0 cache, fitted here (same recipe as the sweep records)
    F = np.load(Path("results/cache/features") / f"siglip_so400m_384-{FP}-crop100.npy").astype(np.float32)
    ref = []
    for itr, iva in folds:
        s = fit_probe(F[itr], y[itr], cl[itr])
        ref.append(compute_metrics(y[iva], s.predict(F[iva]), m.classes)["macro_f1"])
    ref = np.array(ref)
    recs = [json.loads(l) for l in open(a.results) if l.strip()]
    per = {}
    for r in recs:
        if r.get("method") == "linear_probe" and r["model"] in KEYS:
            per.setdefault(r["model"], {})[r["fold"]] = r  # the latest record per fold wins
    n_sig, _, gf_sig = size("siglip_so400m_384")
    rows = []
    for k in KEYS:
        d = per.get(k, {})
        if len(d) < 5:
            print(f"(skipping {k}: {len(d)} folds)"); continue
        f1 = np.array([d[i]["macro_f1"] for i in range(5)]); acc = np.array([d[i]["accuracy"] for i in range(5)])
        n, px, gf = size(k)
        rows.append((k, f1, acc, n, px, gf))
    pe = next((r for r in rows if r[0] == "pe_core_l14_336"), None)
    rows.sort(key=lambda r: -r[1].mean())
    print(f"\nReference: SigLIP So400m/14@384 probe at crop 1.0: macro-F1 {ref.mean():.4f} folds {ref.round(3)} | {n_sig/1e6:.0f} M, {gf_sig:.0f} GFLOPs/image\n")
    print("| Backbone (probe, crop 1.0) | params | px | GFLOPs/img | compute vs SigLIP | CV macro-F1 (± SD) | CV acc | Δ vs SigLIP probe (± SE) | folds | Δ vs PE-Core L |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, f1, acc, n, px, gf in rows:
        dd = f1 - ref
        dpe = (f1 - pe[1]) if pe is not None else None
        print(f"| {LABEL[k]} | {n/1e6:.0f} M | {px} | {gf:.0f} | {gf/gf_sig:.2f}× | {f1.mean():.4f} ± {f1.std(ddof=1):.4f} | {acc.mean():.4f} | "
              f"{dd.mean()*100:+.2f} ± {dd.std(ddof=1)/5**.5*100:.2f} pt | {(dd>0).sum()}/5 | "
              + (f"{dpe.mean()*100:+.2f} ± {dpe.std(ddof=1)/5**.5*100:.2f} pt |" if dpe is not None else "|"))


if __name__ == "__main__":
    main()
