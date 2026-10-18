#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "pillow", "pyyaml"]
# ///
"""Build the self-contained interactive project explorer: docs/explorer/index.html.

    uv run --no-config scripts/build_explorer.py            # from the repo root (CPU, ~10-30 s)
    uv run --no-config scripts/build_explorer.py --check    # build + run all assertions, then report sizes

Reads ONLY files already in the repository (results/, exports/final/, data/v2 + data/web_holdout_v1 manifests,
review CSVs and images, docs/models/*.md, configs/*.yaml, git log) and writes ONE HTML file with inline CSS, inline
vanilla JS (docs/explorer/src/app.js), embedded JSON data and small base64 JPEG thumbnails. No CDN, no external
fonts; works offline from file:// and inside a sandboxed iframe.

Hand-written prose (family explainers, glossary, timeline) lives in docs/explorer/src/content.py. Every number
shown in the page is read from a source file here; `check_headline_numbers` asserts the headline values against
their sources (sealed test, web holdout, original-Kaggle evaluation, benchmark, leakage, schedule, probe-vs-FT)
and that the numbers typed in the timeline prose match the files.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import re
import runpy
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "explorer"
SRC_DIR = OUT_DIR / "src"
THUMB_PX = 112          # longest side of every embedded thumbnail
THUMB_Q = 70            # JPEG quality
PROB_DECIMALS = 5       # rounding of embedded probabilities (assertions re-run on the rounded values)


def rd(path: str):
    p = ROOT / path
    if p.suffix == ".json":
        return json.loads(p.read_text())
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return p.read_text()


def r4(x, n=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), n)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ----------------------------------------------------------------------------------------------------------------
# Metrics (mirrors compclass.evaluate / compclass.final so the embedded probabilities can be re-checked)
# ----------------------------------------------------------------------------------------------------------------
def macro_f1(y, pred, k=20):
    f = []
    for c in range(k):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f))


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


# ----------------------------------------------------------------------------------------------------------------
# Minimal Markdown -> HTML (headings, fenced code, tables, lists, block quotes, paragraphs; inline code/bold/
# italic/links). Enough for docs/models/*.md; everything is HTML-escaped first.
# ----------------------------------------------------------------------------------------------------------------
def md_inline(s: str) -> str:
    codes: list[str] = []

    def keep(m):
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"

    s = re.sub(r"`([^`]+)`", keep, s)
    s = html.escape(s, quote=False)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
               lambda m: f'<a href="{html.escape(m.group(2))}" target="_blank" rel="noopener">{m.group(1)}</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*\\])\*([^*\s][^*]*?)\*(?![\w*])", r"<em>\1</em>", s)
    s = s.replace("\\*", "*")
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], s)


def md_to_html(text: str, h_offset: int = 0) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j])
                j += 1
            out.append(f"<pre class='code'><code>{html.escape(chr(10).join(buf))}</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lvl = min(6, len(m.group(1)) + h_offset)
            out.append(f"<h{lvl}>{md_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            head, body = cells[0], [r for r in cells[1:] if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
            t = "<div class='tablewrap'><table class='md'><thead><tr>" + "".join(
                f"<th>{md_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>" for r in body)
            out.append(t + "</tbody></table></div>")
            continue
        if re.match(r"^\s*([*-]|\d+\.)\s+", ln):
            ordered = bool(re.match(r"^\s*\d+\.", ln))
            items: list[str] = []
            while i < len(lines) and (re.match(r"^\s*([*-]|\d+\.)\s+", lines[i]) or
                                      (lines[i].startswith("  ") and lines[i].strip() and items)):
                if re.match(r"^\s*([*-]|\d+\.)\s+", lines[i]):
                    items.append(re.sub(r"^\s*([*-]|\d+\.)\s+", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{md_inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        if ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> "))
                i += 1
            out.append(f"<blockquote>{md_inline(' '.join(buf))}</blockquote>")
            continue
        if ln.strip() == "":
            i += 1
            continue
        if re.match(r"^\s{4,}\S", ln):     # indented formula / code line
            buf = []
            while i < len(lines) and re.match(r"^\s{4,}\S", lines[i]):
                buf.append(lines[i].strip())
                i += 1
            out.append(f"<pre class='code'><code>{html.escape(chr(10).join(buf))}</code></pre>")
            continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6}\s|```|\||>|\s*([*-]|\d+\.)\s+)",
                                                                   lines[i]):
            buf.append(lines[i].strip())
            i += 1
        out.append(f"<p>{md_inline(' '.join(buf))}</p>")
    return "\n".join(out)


def split_doc(md: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (meta paragraph, [(section title, section markdown)]) for a docs/models file."""
    parts = re.split(r"\n(?=## )", md)
    head = parts[0]
    meta = "\n".join(head.splitlines()[1:]).strip()
    secs = []
    for p in parts[1:]:
        title, _, body = p.partition("\n")
        secs.append((title[3:].strip(), body))
    return meta, secs


# ----------------------------------------------------------------------------------------------------------------
# Thumbnails
# ----------------------------------------------------------------------------------------------------------------
class Thumbs:
    def __init__(self):
        self.data: dict[str, str] = {}
        self.missing: list[str] = []

    def add(self, key: str, path: Path) -> str | None:
        if key in self.data:
            return key
        if not path.exists():
            self.missing.append(str(path.relative_to(ROOT)))
            return None
        im = Image.open(path)
        try:
            im.seek(0)
        except EOFError:
            pass
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        im.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=THUMB_Q, optimize=True, progressive=False)
        self.data[key] = base64.b64encode(buf.getvalue()).decode()
        return key


# ----------------------------------------------------------------------------------------------------------------
# Configs (base: inheritance, like compclass.config.load_config) flattened to dotted keys
# ----------------------------------------------------------------------------------------------------------------
def load_cfg(name: str) -> dict:
    def merge(a, b):
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out

    raw = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    base = raw.pop("base", None)
    cfg = merge(load_cfg(Path(base).stem), raw) if base else raw
    return cfg


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict) and k not in ("kwargs", "randaugment"):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


# ----------------------------------------------------------------------------------------------------------------
def git_log() -> tuple[list[dict], str | None, str | None]:
    try:
        lg = subprocess.run(["git", "log", "--format=%h|%ad|%s", "--date=short"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return [], None, None
    commits = []
    for line in lg.splitlines():
        h, d, s = line.split("|", 2)
        commits.append({"h": h, "d": d, "s": s})
    return commits, head, br


GROUP_OF_FAMILY = {}
FAMILY_OF_NAME = {}


def regime_label(row) -> str:
    r = row["regime"]
    if r == "fine-tune" and "partial" in row["name"]:
        return "partial fine-tune"
    if r == "fine-tune" and row["name"] == "scratch_cnn":
        return "from scratch"
    return {"linear_probe": "linear probe", "zeroshot": "zero-shot"}.get(r, r)


PRETTY = {
    "siglip_so400m_384": "SigLIP So400m/14@384", "siglip2_so400m_384": "SigLIP 2 So400m/14@384",
    "siglip_b16": "SigLIP B/16", "siglip2_b16": "SigLIP 2 B/16", "clip_vitl14_336": "CLIP ViT-L/14@336",
    "clip_vitl14": "CLIP ViT-L/14", "clip_vitb16": "CLIP ViT-B/16", "dinov2_vitl14": "DINOv2 ViT-L/14",
    "dinov2_vitb14": "DINOv2 ViT-B/14", "dinov2_vits14": "DINOv2 ViT-S/14", "convnext_tiny": "ConvNeXt-T",
    "resnet50": "ResNet-50", "vit_b16": "ViT-B/16", "swin_t": "Swin-T", "efficientnet_v2_s": "EfficientNetV2-S",
    "mobilenetv3_large": "MobileNetV3-L", "scratch_cnn": "Scratch CNN", "hog_hsv_svm": "HOG + HSV + SVM",
}


def pretty_name(name: str) -> str:
    n = name
    suffix = ""
    for pre in ("probe_", "zeroshot_"):
        if n.startswith(pre):
            n = n[len(pre):]
    if n.endswith("_tta"):
        n, suffix = n[:-4], " + flip TTA"
    if n.endswith("_partial_v2"):
        n, suffix = n[: -len("_partial_v2")], " (partial, last 4 blocks)"
    elif n.endswith("_partial"):
        n, suffix = n[: -len("_partial")], " (partial, 137M trainable)"
    return PRETTY.get(n, n) + suffix


def backbone_of(name: str) -> str:
    n = name
    for pre in ("probe_", "zeroshot_"):
        if n.startswith(pre):
            n = n[len(pre):]
    return re.sub(r"(_tta|_partial_v2|_partial)$", "", n)


# ----------------------------------------------------------------------------------------------------------------
def build(check_only: bool = False) -> dict:
    content = runpy.run_path(str(SRC_DIR / "content.py"))
    families = content["FAMILIES"]
    for f in families:
        for n in f["bench"]:
            FAMILY_OF_NAME[n] = f["id"]
        GROUP_OF_FAMILY[f["id"]] = f["group"]

    classes = rd("exports/final/labels.json")
    K = len(classes)
    thumbs = Thumbs()
    manifest = rd("data/v2/manifest.json")
    rows_by_id = {r["id"]: r for r in manifest["rows"]}
    lr = rd("data/v2/leakage_report.json")
    kn = rd("docs/report/tables/key_numbers.json")

    def v2_thumb(rid: str):
        r = rows_by_id.get(rid)
        return thumbs.add(rid, ROOT / "data/v2" / r["path"]) if r else None

    # ---------------- benchmark + per-fold records ------------------------------------------------------------
    bench = rd("results/benchmark.csv")
    rec_files = {"phase1": rd("results/phase1/raw/runs.jsonl"), "phase1b": rd("results/phase1b/runs.jsonl"),
                 "fixed": rd("results/phase2/runs_fixed.jsonl"),
                 "old": rd("results/phase2/raw_oldschedule/runs_A.jsonl")
                 + rd("results/phase2/raw_oldschedule/runs_B.jsonl")}
    per_fold: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for src, recs in rec_files.items():
        for r in recs:
            sched = {"fixed": "fixed", "old": "old (bug)"}.get(src, "-")
            per_fold[(r["name"], sched)].append(r)
    bench_rows = []
    max_diff = 0.0
    for _, b in bench.iterrows():
        recs = sorted(per_fold[(b["name"], b["schedule"])], key=lambda r: r["fold"])
        assert len(recs) == int(b["folds"]), (b["name"], b["schedule"], len(recs))
        accs = [r["accuracy"] for r in recs]
        f1s = [r["macro_f1"] for r in recs]
        max_diff = max(max_diff, abs(np.mean(accs) - b["acc"]), abs(np.mean(f1s) - b["f1"]))
        fam = FAMILY_OF_NAME.get(b["name"]) or FAMILY_OF_NAME.get(backbone_of(b["name"]))
        if fam is None:
            fam = {"siglip": "siglip"}.get(b["name"].split("_")[0], None)
        assert fam is not None, b["name"]
        feat_dim = recs[0].get("feature_dim")
        trainable = b["trainable_M"]
        if b["regime"] == "linear_probe":
            trainable = K * (feat_dim + 1) / 1e6
        pc = np.array([[r["per_class"][c]["f1"] for c in classes] for r in recs]).mean(0)
        hist = None
        if b["regime"] == "fine-tune":
            hist = [[[h["epoch"], r4(h["val_macro_f1"]), r4(h["train_loss"])] for h in r["history"]] for r in recs]
        bench_rows.append({
            "name": b["name"], "label": pretty_name(b["name"]), "family": fam, "group": GROUP_OF_FAMILY[fam],
            "backbone": backbone_of(b["name"]), "regime": regime_label(b), "schedule": b["schedule"],
            "folds": int(b["folds"]), "acc": r4(b["acc"]), "acc_sd": r4(b["acc_sd"]), "f1": r4(b["f1"]),
            "f1_sd": r4(b["f1_sd"]), "inscope_acc": r4(b["inscope_acc"]), "inscope_f1": r4(b["inscope_f1"]),
            "params_M": r4(b["params_M"], 3), "trainable_M": r4(trainable, 4), "feature_dim": feat_dim,
            "train_min": r4(b["train_min_per_fold"], 2), "vram_gb": r4(b["peak_vram_gb"], 2),
            "best_epoch": r4(b["best_epoch"], 1),
            "fold_ids": [r["fold"] for r in recs], "fold_acc": [r4(a) for a in accs], "fold_f1": [r4(f) for f in f1s],
            "per_class_f1": [r4(x, 3) for x in pc], "history": hist,
        })
    assert max_diff < 1e-4, f"benchmark.csv vs per-fold records differ by {max_diff}"

    # ---------------- probe vs fine-tune (paired per fold) ----------------------------------------------------
    pvf = rd("docs/report/tables/probe_vs_finetune.csv")
    pairs_def = [("siglip2_b16", "probe_siglip2_b16", "siglip2_b16", "fixed"),
                 ("siglip_so400m_384", "probe_siglip_so400m_384", "siglip_so400m_384_partial_v2", "fixed"),
                 ("dinov2_vitb14", "probe_dinov2_vitb14", "dinov2_vitb14", "fixed"),
                 ("clip_vitb16", "probe_clip_vitb16", "clip_vitb16", "fixed"),
                 ("convnext_tiny", "probe_convnext_tiny", "convnext_tiny", "fixed"),
                 ("resnet50", "probe_resnet50", "resnet50", "fixed")]
    paired = []
    for (bb, pn, fn, sch), (_, t) in zip(pairs_def, pvf.iterrows()):
        pr = {r["fold"]: r["macro_f1"] for r in per_fold[(pn, "-")]}
        ft = {r["fold"]: r["macro_f1"] for r in per_fold[(fn, sch)]}
        folds = sorted(set(pr) & set(ft))
        d = [100 * (ft[k] - pr[k]) for k in folds]
        assert abs(np.mean(d) - t["Δ (pts)"]) < 0.006, (bb, np.mean(d), t["Δ (pts)"])
        assert f"{sum(x > 0 for x in d)}/{len(d)}" == t["FT wins"], bb
        paired.append({"backbone": bb, "label": PRETTY[bb], "type": t["Fine-tune type"], "folds": folds,
                       "probe": [r4(pr[k]) for k in folds], "ft": [r4(ft[k]) for k in folds],
                       "delta": [round(x, 2) for x in d], "mean_delta": round(float(np.mean(d)), 2),
                       "wins": t["FT wins"], "probe_mean": r4(t["Probe macro-F1"]),
                       "ft_mean": r4(t["Fine-tune macro-F1"])})

    # ---------------- schedule ablation (paired folds 0-2) ---------------------------------------------------
    sab = rd("results/schedule_ablation_folds012.csv")
    sched = []
    all_d = []
    for _, s in sab.iterrows():
        old = {r["fold"]: r["macro_f1"] for r in per_fold[(s["name"], "old (bug)")]}
        new = {r["fold"]: r["macro_f1"] for r in per_fold[(s["name"], "fixed")]}
        folds = [0, 1, 2]
        d = [100 * (new[k] - old[k]) for k in folds]
        all_d += d
        assert abs(np.mean(d) / 100 - s["delta"]) < 6e-5, (s["name"], np.mean(d), s["delta"])
        assert sum(x > 0 for x in d) == s["folds_improved"]
        sched.append({"name": s["name"], "label": PRETTY[s["name"]], "old": r4(s["old"]), "fixed": r4(s["fixed"]),
                      "delta": [round(x, 2) for x in d], "mean": round(float(np.mean(d)), 2),
                      "improved": int(s["folds_improved"])})
    n_impr = sum(x > 0 for x in all_d)
    sd_pf = float(np.std(all_d, ddof=1))
    assert f"{n_impr}/{len(all_d)}" == kn["schedule_folds_improved"]["value"]
    assert abs(np.mean(all_d) - kn["schedule_mean_delta_pts"]["value"]) < 0.006
    assert abs(sd_pf - kn["schedule_perfold_delta_sd_pts"]["value"]) < 0.06
    schedule = {"rows": sched, "n_improved": n_impr, "n": len(all_d), "mean": round(float(np.mean(all_d)), 2),
                "sd": round(sd_pf, 2), "noise2se": kn["schedule_noise_floor_2se_pts"]["value"],
                "p": kn["schedule_sign_test_p"]["value"]}

    # ---------------- other ablations ------------------------------------------------------------------------
    abl = rd("docs/report/tables/ablations.csv")
    clean = rd("results/phase1/cleaning_ablation_siglip_so400m.csv")
    tta = []
    for base, var in [("probe_siglip_so400m_384", "probe_siglip_so400m_384_tta"),
                      ("probe_siglip2_so400m_384", "probe_siglip2_so400m_384_tta")]:
        a = {r["fold"]: r["macro_f1"] for r in per_fold[(base, "-")]}
        b = {r["fold"]: r["macro_f1"] for r in per_fold[(var, "-")]}
        d = [round(100 * (b[k] - a[k]), 2) for k in sorted(a)]
        tta.append({"label": pretty_name(base), "base": r4(np.mean(list(a.values()))),
                    "tta": r4(np.mean(list(b.values()))), "delta": d})
    assert abs(np.mean(tta[0]["delta"]) - kn["tta_probe_siglip_so400m_384"]["value"]["delta_pts"]) < 0.01
    assert abs(np.mean(tta[1]["delta"]) - kn["tta_probe_siglip2_so400m_384"]["value"]["delta_pts"]) < 0.01

    # ---------------- final model: sealed test + web holdout --------------------------------------------------
    fit = rd("exports/final/fit_report.json")
    pre = rd("exports/final/preprocess.json")
    exp = rd("exports/final/export_report.json")
    finals = {}
    web_manifest = rd("data/web_holdout_v1/manifest.json")
    web_rows = {r["id"]: r for r in web_manifest["rows"]}
    for tag, jpath, npath in [("test", "results/final/final_test_v2.json", "results/final/final_test_v2_preds.npz"),
                              ("web", "results/final/final_test_web_holdout_v1.json",
                               "results/final/final_test_web_holdout_v1_preds.npz")]:
        js = rd(jpath)
        z = np.load(ROOT / npath)
        probs = np.round(z["probs"].astype(np.float64), PROB_DECIMALS)
        y = z["y"].astype(int)
        pred, conf = probs.argmax(1), probs.max(1)
        thr = js["abstain_threshold"]
        acc = float((pred == y).mean())
        cm = np.zeros((K, K), int)
        np.add.at(cm, (y, pred), 1)
        assert abs(acc - js["accuracy"]) < 1e-9, tag
        assert abs(macro_f1(y, pred, K) - js["macro_f1"]) < 1e-9, tag
        assert cm.tolist() == js["confusion"], tag
        assert abs((conf >= thr).mean() - js["coverage"]) < 1e-9, tag
        assert abs((pred[conf >= thr] == y[conf >= thr]).mean() - js["accepted_accuracy"]) < 1e-9, tag
        assert abs(ece(probs, y) - js["ece"]) < 5e-4, (tag, ece(probs, y), js["ece"])
        ids = [str(i) for i in z["ids"]]
        if tag == "test":
            ood = [int("ood_suspect" in rows_by_id[i]["flags"]) for i in ids]
            for i in ids:
                v2_thumb(i)
            credit = None
        else:
            ood = [0] * len(ids)
            credit = []
            for i in ids:
                w = web_rows[i]
                thumbs.add("w" + i, ROOT / "data/web_holdout_v1" / w["path"])
                credit.append([w.get("author", ""), w.get("license", ""), w.get("source_url", "")])
        lo, hi = wilson(int((pred == y).sum()), len(y))
        finals[tag] = {
            "n": len(y), "ids": ids, "y": y.tolist(), "probs": probs.tolist(), "ood": ood, "credit": credit,
            "accuracy": r4(js["accuracy"]), "macro_f1": r4(js["macro_f1"]), "ci": [r4(lo), r4(hi)],
            "probe_only_accuracy": r4(js["probe_only_accuracy"]), "coverage": r4(js["coverage"]),
            "accepted_accuracy": r4(js["accepted_accuracy"]), "ece": r4(js["ece"]),
            "inscope_n": js["inscope_n"], "inscope_accuracy": r4(js["inscope_accuracy"]),
            "inscope_macro_f1": r4(js["inscope_macro_f1"]), "fingerprint": js["fingerprint"],
            "per_class": {c: {k: r4(v) for k, v in js["per_class"][c].items()} for c in classes},
            "source": jpath,
        }
    assert [r4(x) for x in kn["test_accuracy_wilson95"]["value"]] == finals["test"]["ci"]

    # ---------------- original Kaggle evaluation -------------------------------------------------------------
    ks = rd("results/kaggle_original/summary.json")
    kaggle = {
        "n_files": ks["n_files"], "n_rows": ks["n_rows"], "all": ks["all_files"], "by_split": ks["by_v2_split"],
        "test_inscope": ks["test_inscope"], "by_ood": ks["by_ood_flag"],
        "legacy_as_deployed": ks["legacy_test_as_deployed"],
        "legacy_unseen": ks["legacy_test_as_deployed_unseen_rows_only"], "replay": ks["legacy_test_replay"],
        "test_per_class": ks["test_per_class_file_accuracy"],
    }

    # ---------------- data audit ----------------------------------------------------------------------------
    cnt = lr["counts"]
    per_class = []
    for c in classes:
        rs = [r for r in manifest["rows"] if r["label"] == c]
        per_class.append({"cls": c, "n": len(rs), "ood": sum("ood_suspect" in r["flags"] for r in rs),
                          "test": sum(r["split"] == "test" for r in rs),
                          "near": sum("near_dup" in r["flags"] for r in rs)})
    opc = rd("docs/report/tables/ood_per_class.csv").set_index("Class")
    for p in per_class:
        assert opc.loc[p["cls"], "Rows"] == p["n"] and opc.loc[p["cls"], "OOD-suspect rows"] == p["ood"], p
    oods = rd("data/v2/review/ood_suspects.csv")
    distr = oods["top_distractor"].value_counts().to_dict()
    assert {k: int(v) for k, v in distr.items()} == kn["ood_top_distractors"]["value"]
    fold_sizes = [sum(v.values()) for k, v in sorted(lr["per_class_split_counts"].items()) if k.startswith("fold")]
    assert sum(fold_sizes) == cnt["dev_rows"]

    gal: dict[str, list] = {}
    # out-of-scope: top examples per distractor concept
    g = []
    for dname in kn["ood_top_distractors"]["value"]:
        sub = oods[oods.top_distractor == dname].sort_values(["top_distractor_p", "id"], ascending=[False, True])
        for _, r in sub.head(8).iterrows():
            if v2_thumb(r["id"]):
                g.append({"id": r["id"], "label": r["label"], "distractor": dname,
                          "p": r4(r["top_distractor_p"], 3), "score": r4(r["ood_score"], 3), "split": r["split"]})
    gal["ood"] = g
    # exact duplicates whose copies were in different legacy splits
    pf = rd("results/kaggle_original/per_file.csv")
    pfv = pf[pf.v2_split != "excluded"]
    grp = pfv.groupby("row")
    ex = []
    for rid, sub in grp:
        sp = sorted(set(sub.legacy_split.dropna()) - {"not_in_legacy"})
        if len(sub) > 1 and "train" in sp and ("test" in sp or "val" in sp):
            ex.append((-len(sub), rid, sub))
    ex.sort(key=lambda t: (t[0], t[1]))
    g = []
    for negn, rid, sub in ex[:24]:
        if v2_thumb(rid):
            g.append({"id": rid, "label": rows_by_id[rid]["label"], "copies": int(-negn),
                      "legacy": sorted(sub.legacy_split.tolist()), "files": sub.file.tolist()[:6]})
    gal["exact"] = g
    n_multi = sum(1 for r in manifest["rows"] if "legacy_multi_split" in r["flags"])
    # near-duplicate pairs (both rules) + the largest cluster
    edges = rd("data/v2/review/near_dup_edges.csv")
    g = []
    for rule in ["phash+clip", "clip_only"]:
        sub = edges[edges.rule == rule] if rule in set(edges.rule) else edges[edges.rule != "clip_only"]
        sub = sub.sort_values(["clip_cos", "id_a"], ascending=[False, True]).iloc[::max(1, len(sub) // 9)].head(9)
        for _, e in sub.iterrows():
            if v2_thumb(e["id_a"]) and v2_thumb(e["id_b"]):
                g.append({"a": e["id_a"], "b": e["id_b"], "la": e["label_a"], "lb": e["label_b"],
                          "phash": int(e["phash_hamming"]), "cos": r4(e["clip_cos"], 3), "rule": e["rule"]})
    gal["near"] = g
    ndc = rd("data/v2/review/near_dup_clusters.csv")
    big = ndc[ndc["size"] == ndc["size"].max()].sort_values("id")
    gal["near_big"] = {"cluster": big.cluster.iloc[0], "size": int(big["size"].iloc[0]),
                       "labels": big.labels.iloc[0],
                       "ids": [i for i in big.id.tolist()[:16] if v2_thumb(i)]}
    # conflicting labels
    conf = rd("data/v2/review/conflicts.csv")
    g = []
    for _, c in conf.iterrows():
        if c["kind"] == "exact":
            key = thumbs.add("x" + c["id"], ROOT / "data/kaggle_eval_extra/images" / f"{c['id']}.jpg")
        else:
            key = v2_thumb(c["id"])
        if key:
            g.append({"key": key, "kind": c["kind"], "group": c["group"], "labels": c["labels"].split("|"),
                      "decision": c["decision"], "cos": r4(c["max_cross_label_cos"], 3)})
    gal["conflict"] = g
    # label issues (all)
    li = rd("results/phase1/label_issues_v2.csv")
    g = []
    for _, r in li.iterrows():
        if v2_thumb(r["id"]):
            g.append({"id": r["id"], "label": r["label"], "pred": r["pred"], "self": r4(r["self_conf"], 4),
                      "alt": r4(r["alt_conf"], 4), "ood": bool(r["ood"])})
    gal["issues"] = g
    patt = Counter(f"{r['label']} -> {r['pred']}" for r in g)
    # typical in-scope examples per class
    zs = rd("data/v2/review/zeroshot_scores.csv")
    dev_ids = {r["id"] for r in manifest["rows"] if r["split"] == "dev"}
    g = {}
    for c in classes:
        sub = zs[(zs.label == c) & (zs.ood_score < 0.05) & (zs.id.isin(dev_ids))]
        sub = sub.sort_values(["label_p", "id"], ascending=[False, True]).head(6)
        g[c] = [i for i in sub.id.tolist() if v2_thumb(i)]
    gal["classes"] = g

    audit = {
        "counts": cnt, "formats": cnt["orig_format_unique_decoded"], "flags": cnt["flags"],
        "near_hist": cnt["near_dup_cluster_size_hist"], "near_rule": lr["near_dup_rule"], "ood_rule": lr["ood_rule"],
        "per_class": per_class, "distractors": kn["ood_top_distractors"]["value"], "fold_sizes": fold_sizes,
        "outer_tvd_max": max(lr["split"]["outer_tvd"]), "n_legacy_multi": n_multi,
        "issue_patterns": patt.most_common(8), "n_issues": len(li), "n_issues_ood": int(li.ood.sum()),
        "ood_dev": kn["ood_dev_rows"]["value"], "ood_test": kn["ood_test_rows"]["value"],
    }

    # ---------------- leakage --------------------------------------------------------------------------------
    leak = rd("results/leakage_simulation.csv")
    leakage = {"rows": [{k: (r4(v) if isinstance(v, float) else v) for k, v in r.items()} for r in
                        leak.to_dict("records")], "eval": int(leak.legacy_eval_rows.iloc[0]),
               "leaked": int(leak.leaked_rows.iloc[0])}

    # ---------------- families: docs, configs, results -------------------------------------------------------
    fam_out = []
    for f in families:
        md = (ROOT / "docs/models" / f"{f['doc']}.md").read_text()
        meta, secs = split_doc(md)
        sec_html = []
        for title, body in secs:
            if title.lower().startswith("results on data v2"):
                continue          # placeholder in the doc; the explorer shows the measured results instead
            sec_html.append({"title": md_inline(title), "html": md_to_html(body, h_offset=1)})
        cfgs = []
        for cname in [f.get("config"), f.get("extra_config")]:
            if not cname:
                continue
            fl = flat(load_cfg(cname))
            keys = list(dict.fromkeys(f.get("hp_keys", []) + ["train.lr", "train.batch_size", "train.accum_steps",
                                                             "train.layer_decay", "train.freeze_epochs",
                                                             "train.warmup_epochs", "train.weight_decay",
                                                             "train.epochs", "train.patience",
                                                             "train.label_smoothing", "train.trainable_blocks",
                                                             "train.grad_checkpointing", "model.kwargs",
                                                             "aug.randaugment", "aug.rrc_scale"]))
            own = flat(yaml.safe_load((ROOT / "configs" / f"{cname}.yaml").read_text()))
            hp = [[k, json.dumps(fl[k]) if not isinstance(fl.get(k), str) else fl[k], k in own,
                   content["HP_EXPLAIN"].get(k, "")] for k in keys if k in fl]
            cfgs.append({"file": f"configs/{cname}.yaml", "hp": hp,
                         "raw": (ROOT / "configs" / f"{cname}.yaml").read_text()})
        fam_out.append({**{k: v for k, v in f.items() if k not in ("hp_keys",)},
                        "meta": md_inline(meta.replace("\n", " ")), "sections": sec_html, "configs": cfgs,
                        "idea": [md_inline(p) for p in f["idea"]], "lesson": md_inline(f["lesson"]),
                        "tagline": md_inline(f["tagline"]), "papers": [md_inline(p) for p in f["papers"]],
                        "regimes": md_inline(f["regimes"]),
                        "chars": {k: md_inline(v) for k, v in f["chars"].items()},
                        "hp": [[k, md_inline(v)] for k, v in f.get("hp", [])]})
    for b in bench_rows:
        assert any(b["name"] in f["bench"] or b["backbone"] in [backbone_of(x) for x in f["bench"]]
                   for f in families), b["name"]

    # ---------------- timeline + commits ---------------------------------------------------------------------
    commits, head, branch = git_log()
    known = {c["h"]: c for c in commits}
    timeline = []
    for t in content["TIMELINE"]:
        cs = []
        for h in t["commits"]:
            if commits:
                assert h in known, f"timeline commit {h} not in git log"
                cs.append(known[h])
            else:
                cs.append({"h": h, "d": "", "s": ""})
        timeline.append({**t, "commits": cs, "what": md_inline(t["what"]), "why": md_inline(t["why"]),
                         "result": md_inline(t["result"])})
    used = {h for t in content["TIMELINE"] for h in t["commits"]}
    other_commits = [c for c in commits if c["h"] not in used]

    data = {
        "meta": {"head": head, "branch": branch, "manifest": manifest["fingerprint"],
                 "web_fingerprint": web_manifest["fingerprint"], "n_commits": len(commits)},
        "classes": classes, "kn": {k: v["value"] for k, v in kn.items()},
        "timeline": timeline, "other_commits": other_commits, "audit": audit, "gal": gal,
        "explain": {k: md_inline(v) for k, v in content["AUDIT_EXPLAIN"].items()},
        "bench": bench_rows, "paired": paired, "schedule": schedule, "tta": tta,
        "clean": clean.to_dict("records"), "ablations": abl.to_dict("records"), "leakage": leakage,
        "fit": {k: v for k, v in fit.items()}, "pre": pre, "export": exp, "final": finals, "kaggle": kaggle,
        "families": fam_out, "glossary": [[a, md_inline(b)] for a, b in content["GLOSSARY"]],
        "reproduce": content["REPRODUCE"],
        "demo_parity": rd("results/demo/demo_parity.json"),
        "model_card": rd("docs/report/tables/model_card.csv").to_dict("records"),
        "fusion_grid": rd("results/phase1/fusion_ablation.csv").to_dict("records"),
        "abstention_table": rd("docs/report/tables/abstention.csv").to_dict("records"),
        "web_meta": {"n": len(web_manifest["rows"]), "counts": web_manifest["counts"],
                     "source": web_manifest["source"]},
    }
    check_headline_numbers(data, content)
    data["thumbs_missing"] = thumbs.missing
    return data, thumbs


# ----------------------------------------------------------------------------------------------------------------
def check_headline_numbers(d: dict, content: dict) -> None:
    """Headline values against their source files, and numbers typed into the timeline prose."""
    t, w, k = d["final"]["test"], d["final"]["web"], d["kaggle"]
    kn = d["kn"]
    assert t["n"] == 418 and t["accuracy"] == kn["test_accuracy"] and t["macro_f1"] == kn["test_macro_f1"]
    assert t["ece"] == kn["test_ece"] and t["probe_only_accuracy"] == kn["test_probe_only_accuracy"]
    assert [t["coverage"], t["accepted_accuracy"]] == kn["test_coverage_accepted_acc"]
    assert t["inscope_accuracy"] == kn["test_inscope"]["acc"] and t["inscope_n"] == kn["test_inscope"]["n"]
    se = math.sqrt(t["accuracy"] * (1 - t["accuracy"]) / t["n"]) * 100
    assert round(se, 2) == kn["test_accuracy_se_pts"]
    assert w["n"] == 407 and abs(w["accuracy"] - 0.8206) < 1e-9 and abs(w["macro_f1"] - 0.8159) < 1e-9
    assert round(100 * (w["accuracy"] - w["probe_only_accuracy"]), 1) == 4.2
    a = k["all"]
    assert k["n_files"] == 3530 and a["n"] == 3530 and round(a["accuracy"], 3) == 0.955
    assert k["by_split"]["test"]["n"] == 493 and round(k["by_split"]["test"]["accuracy"], 3) == 0.830
    assert k["by_split"]["dev"]["n"] == 3010 and round(k["by_split"]["dev"]["accuracy"], 3) == 0.981
    assert k["by_split"]["excluded"]["n"] == 27 and round(k["by_split"]["excluded"]["accuracy"], 3) == 0.370
    rp = k["replay"]
    assert rp["all"]["n"] == 521 and round(rp["all"]["accuracy"], 3) == 0.877
    assert rp["leaked_copy_of_train"]["n"] == 113 and rp["leaked_copy_of_train"]["accuracy"] == 1.0
    assert rp["clean_not_in_train"]["n"] == 408 and round(rp["clean_not_in_train"]["accuracy"], 3) == 0.843
    assert rp["old_best_for_reference"]["accuracy"] == 0.7132
    best = max((b for b in d["bench"]), key=lambda b: b["f1"])
    assert best["name"] == "probe_siglip_so400m_384" and best["f1"] == 0.8238 and best["f1_sd"] == 0.0119
    c = d["audit"]["counts"]
    assert (c["raw_files"], c["unique_sha256"], c["rows"], c["clusters"], c["ood_suspect"], c["test_rows"],
            c["dev_rows"]) == (3530, 2943, 2930, 2787, 816, 418, 2512)
    assert d["leakage"]["eval"] == 853 and d["leakage"]["leaked"] == 198
    assert d["audit"]["n_issues"] == 69
    # numbers typed in the timeline prose (4-decimal source values rounded half-up, as in the report)
    def f3(x):
        return f"{math.floor(x * 1000 + 0.5 + 1e-9) / 1000:.3f}"

    bn = {b["name"] + "|" + b["schedule"]: b for b in d["bench"]}
    tl = {x["id"]: x["result"] + " " + x["what"] for x in content["TIMELINE"]}
    must = {
        "data": ["3,530", "2,943", "2,930", "2,787", "816", "418", "2,512"],
        "phase1": [f"{bn['probe_siglip_so400m_384|-']['f1']:.3f} +/- {bn['probe_siglip_so400m_384|-']['f1_sd']:.3f}",
                   str(d["audit"]["n_issues"])],
        "phase2": [f3(bn['siglip2_b16|fixed']['f1']), f3(bn['probe_siglip2_b16|-']['f1']),
                   f3(min(b['f1'] for b in d['bench'] if b['group'].startswith('ImageNet') and b['schedule'] == 'fixed')),
                   f3(max(b['f1'] for b in d['bench'] if b['group'].startswith('ImageNet') and b['schedule'] == 'fixed'))],
        "final": [f3(t['accuracy']), f"{t['ci'][0]:.3f}-{t['ci'][1]:.3f}", f3(t['macro_f1']),
                  f3(t['ece']), f"{100 * t['coverage']:.1f} %", f"{100 * t['accepted_accuracy']:.1f} %"],
        "web": [f3(w['accuracy']), f"{w['ci'][0]:.3f}-{w['ci'][1]:.3f}", f3(w['macro_f1']), "+4.2", "407"],
        "kaggle": [f"{100 * a['accuracy']:.1f} %", f"{100 * k['by_split']['test']['accuracy']:.1f} %",
                   f"{k['by_split']['test']['n']}", f"{100 * rp['all']['accuracy']:.1f} %",
                   f"{100 * rp['old_best_for_reference']['accuracy']:.1f} %"],
        "legacy": [f"{rp['old_best_for_reference']['accuracy']:.4f}", f"{rp['old_best_for_reference']['macro_f1']:.4f}"],
        "report": [f"{d['demo_parity']['n_images']}/{d['demo_parity']['n_images']}"],
    }
    for key, frags in must.items():
        for fr in frags:
            assert fr in tl[key], f"timeline '{key}' should contain '{fr}' (from source files)"
    assert d["demo_parity"]["argmax_agreement"] == 1.0


# ----------------------------------------------------------------------------------------------------------------
def clean_nan(o):
    """NaN/inf -> None (JSON null), numpy scalars -> Python."""
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_nan(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def assemble(data: dict, thumbs: Thumbs) -> str:
    css = (SRC_DIR / "style.css").read_text()
    js = (SRC_DIR / "app.js").read_text()

    def safe(obj) -> str:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>compclass explorer — electronic-component classification across model families</title>
<style>
{css}
</style>
</head>
<body>
<div id="errbar" role="alert" hidden></div>
<header class="top">
  <div class="brand"><span class="logo" aria-hidden="true">◆</span><div><div class="t1">compclass explorer</div>
  <div class="t2">20-class electronic-component recognition · 11 model families · leakage-free evaluation</div></div></div>
  <div class="topctl"><span id="buildinfo" class="muted small"></span>
  <button id="themebtn" class="btn" type="button" aria-label="Toggle light or dark theme">◐ Theme</button></div>
</header>
<nav id="tabs" role="tablist" aria-label="Sections"></nav>
<main id="main"></main>
<div id="tip" role="tooltip" hidden></div>
<script type="application/json" id="data">{safe(clean_nan(data))}</script>
<script type="application/json" id="thumbs">{safe(thumbs.data)}</script>
<script>
{js}
</script>
</body>
</html>
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DIR / "index.html"))
    ap.add_argument("--check", action="store_true", help="print size breakdown after building")
    a = ap.parse_args(argv)
    data, thumbs = build()
    page = assemble(data, thumbs)
    Path(a.out).write_text(page, encoding="utf-8")
    tb = sum(len(v) for v in thumbs.data.values())
    print(f"wrote {a.out}: {len(page.encode()) / 1e6:.2f} MB  ({len(thumbs.data)} thumbnails, {tb / 1e6:.2f} MB base64; "
          f"missing images: {len(thumbs.missing)})")
    if a.check:
        dj = json.dumps(clean_nan(data))
        print(f"data JSON {len(dj) / 1e6:.2f} MB; probs test/web embedded at {PROB_DECIMALS} decimals; "
              f"all assertions passed")
    if thumbs.missing:
        print("missing:", thumbs.missing[:10], file=sys.stderr)


if __name__ == "__main__":
    main()
