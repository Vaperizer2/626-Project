#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "numpy==2.5.3",
#   "pillow==12.3.0",
#   "pillow-avif-plugin==1.6.0",
#   "imagehash==4.3.2",
#   "scikit-learn==1.9.1",
#   "torch==2.14.1",
#   "torchvision==0.29.1",
#   "open_clip_torch==3.3.0",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# ///
"""Build the content-addressed, leakage-gated dataset v2.

Run from anywhere (paths resolve from this file's location):

    uv run --no-config tools/build_dataset_v2.py              # build / verify
    uv run --no-config tools/build_dataset_v2.py --overwrite  # allow a new split fingerprint

Pipeline
--------
1. Enumerate every non-hidden file under data/raw/electronic-component-recognizer/<class>/<class>/,
   hash raw bytes (SHA-256), decode with Pillow regardless of suffix (JPEG/WebP/PNG/AVIF/MPO/GIF...),
   convert to RGB JPEG q95 (EXIF-transposed, alpha composited on white, first frame of animations,
   longest side <= 512 px, never upscaled) under data/v2/images/<class>/<id>.jpg, id = sha256[:16].
   Exact duplicates collapse to one row; byte-identical files filed under different class folders are
   excluded ("conflicting_labels") -- no label is guessed.
2. Near-duplicates: 64-bit pHash + dHash on the converted image and OpenCLIP ViT-B-32
   (laion2b_s34b_b79k) image embeddings.  An edge joins two images when
       (pHash Hamming <= PHASH_RADIUS and CLIP cosine >= COS_WITH_PHASH)   [resized/re-encoded copies]
    or (CLIP cosine >= COS_ONLY)                                          [crops, re-framings, recolours]
   Defaults (PHASH_RADIUS=10, COS_WITH_PHASH=0.85, COS_ONLY=0.92) were chosen by visual inspection of
   sampled pairs per band on this data: pHash alone is unreliable here (of pairs with Hamming <= 10 only
   ~40% share a label; plain white-background shots collide), pHash<=10 pairs at cosine 0.80-0.90 were
   re-renders/recolours of one template or the same item, while 0.70-0.80 began pairing distinct
   objects; cosine-only pairs >= 0.92 were the same object/photo shoot from another angle, 0.90-0.92
   began to include distinct look-alike products.  Grouping errs toward over-merging (leakage-safe).
   Union-find over edges gives clusters (the split group).  Mixed-label clusters are flagged
   "near_dup_label_conflict"; rows joined to a differently-labelled row at cosine >= EXCLUDE_COS
   (clearly the same picture) are excluded ("near_dup_conflicting_labels").
3. Out-of-scope suspects: CLIP zero-shot over 20 component concepts + 9 distractor concepts.
   ood_score = softmax mass on distractors; rows with ood_score >= OOD_THRESHOLD are flagged
   "ood_suspect" (never excluded automatically).  Default 0.25: flags are only a review queue, and a
   visual spot check found most images even in the 0.2-0.5 band to be out of scope (homonym scrapes such
   as Nintendo Switch for "switch", relay races for "relay", brand logos for "fuse").
   review/zeroshot_scores.csv carries the score for every row so the cut can be moved.
4. Splits: StratifiedGroupKFold(7, shuffle, 42) on clusters -> the fold whose class distribution is
   closest (total-variation distance) to the overall one becomes the sealed test set; the remaining dev
   rows get StratifiedGroupKFold(5, shuffle, 42) folds 0..4.
5. Leakage gate (exit 2 on violation, manifest is NOT written): a cluster in more than one of
   {test, fold0..4}; duplicated sha256/id; missing image; class list != the 20 raw folder names;
   a raw file unaccounted for or accounted twice.
6. Writes data/v2/manifest.json, leakage_report.json, clip_b32_embeddings.npy (+ clip_b32_ids.json),
   review/*.csv + contact sheets, and a deterministic images.tar.

Fingerprint definition: lines "id\\tlabel\\tsplit\\tfold" (fold "" for test), sorted, joined with "\\n",
UTF-8, SHA-256, first 16 hex chars.  If data/v2/manifest.json already exists with a different
fingerprint the script exits 3 unless --overwrite is given (splits are frozen; fail closed).

Nothing under data/raw is modified.  Files are never deleted: stale images in data/v2/images that are no
longer rows are moved to data/v2/stale/.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps

try:  # Pillow >= 11.3 decodes AVIF natively; the plugin is a fallback for older builds.
    import pillow_avif  # noqa: F401
except ImportError:  # pragma: no cover
    pass

Image.MAX_IMAGE_PIXELS = 250_000_000

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "electronic-component-recognizer"
OUT = REPO / "data" / "v2"
LEGACY_SPLIT = REPO / "data" / "splits" / "split_manifest.json"
LEGACY_ANCHOR = "electronic-component-recognizer/"

VERSION = "v2.0"
MAX_SIDE = 512
JPEG_QUALITY = 95
SEED = 42
N_TEST_SPLITS = 7
N_FOLDS = 5
CLIP_ARCH, CLIP_PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"

# ---- prompts ---------------------------------------------------------------------------------------
CLASS_TEXT = {
    "battery holder": "battery holder",
    "breadboard": "solderless breadboard",
    "capacitor": "capacitor",
    "connector": "electrical connector",
    "crystal oscillator": "crystal oscillator",
    "diode": "diode",
    "fuse": "electrical fuse",
    "heat sink": "heat sink",
    "ic chip": "integrated circuit chip",
    "inductor coil": "inductor coil",
    "jumper wire": "jumper wire",
    "led": "LED light emitting diode",
    "microcontroller board": "microcontroller board",
    "potentiometer": "potentiometer",
    "power supply module": "power supply module",
    "relay": "relay",
    "resistor": "resistor",
    "switch": "electrical switch",
    "transformer": "transformer",
    "transistor": "transistor",
}
CLASS_TEMPLATES = [
    "a photo of a {}.",
    "a close-up photo of a {}, an electronic component.",
    "a product photo of a {} on a white background.",
]
DISTRACTORS = {
    "person": ["a photo of a person.", "a photo of a person's face.", "a photo of people."],
    "car": ["a photo of a car.", "a photo of a vehicle on a road."],
    "food": ["a photo of food.", "a photo of a meal on a plate."],
    "logo": ["a company logo.", "a brand logo on a plain background.", "an icon."],
    "text_screenshot": ["a screenshot of text.", "a page of a document full of text."],
    "cartoon_game_art": ["a cartoon illustration.", "video game art.", "a clip-art drawing."],
    "circuit_schematic": ["a circuit schematic diagram.", "a line drawing of an electronic circuit diagram."],
    "website_screenshot": ["a screenshot of a website.", "a screenshot of an online shop web page."],
    "collage": ["a collage of many different items.", "a grid of many different product photos."],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- step 1: enumerate / hash / convert -----------------------------------------------------------
def list_classes() -> list[str]:
    return sorted(p.name for p in RAW.iterdir() if p.is_dir() and not p.name.startswith("."))


def enumerate_raw() -> list[str]:
    rels = []
    for root, dirs, files in os.walk(RAW):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for f in sorted(files):
            if f.startswith("."):
                continue
            rels.append(Path(root, f).relative_to(RAW).as_posix())
    return sorted(rels)


def to_rgb_jpeg(data: bytes) -> dict:
    """Decode raw bytes -> converted JPEG bytes + metadata.  Never raises."""
    flags = []
    try:
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except OSError:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                im = Image.open(io.BytesIO(data))
                im.load()
                flags.append("truncated_decode")
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = False
        fmt = im.format or "UNKNOWN"
        w, h = im.size  # stored pixel size, before EXIF transpose
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
            flags.append("multi_frame_first_used")
        has_alpha = im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
            im.mode == "P" and "transparency" in im.info)
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
            arr = np.asarray(im, dtype=np.float64)
            hi = 65535.0 if arr.max() > 255 else 255.0
            im = Image.fromarray(np.clip(arr / hi * 255.0, 0, 255).astype(np.uint8), "L")
        if has_alpha:
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            im = bg.convert("RGB")
            flags.append("alpha_composited")
        else:
            im = im.convert("RGB")
        cw, ch = im.size
        if max(cw, ch) > MAX_SIDE:
            s = MAX_SIDE / max(cw, ch)
            im = im.resize((max(1, round(cw * s)), max(1, round(ch * s))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY)
        return {"ok": True, "format": fmt, "w": w, "h": h, "jpeg": buf.getvalue(), "flags": flags}
    except Exception as e:  # undecodable
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300], "flags": flags}


def _convert_job(path: str) -> dict:
    return to_rgb_jpeg(Path(path).read_bytes())


# ---- union-find ------------------------------------------------------------------------------------
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# ---- features ---------------------------------------------------------------------------------------
def perceptual_hashes(jpegs: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    import imagehash

    def h2int(h):
        return int("".join("1" if b else "0" for b in h.hash.flatten()), 2)

    ph, dh = np.zeros(len(jpegs), np.uint64), np.zeros(len(jpegs), np.uint64)
    for i, b in enumerate(jpegs):
        im = Image.open(io.BytesIO(b))
        ph[i] = h2int(imagehash.phash(im, hash_size=8))
        dh[i] = h2int(imagehash.dhash(im, hash_size=8))
    return ph, dh


def hamming_matrix(h: np.ndarray) -> np.ndarray:
    out = np.empty((len(h), len(h)), np.uint8)
    for i in range(len(h)):
        out[i] = np.bitwise_count(np.bitwise_xor(h, h[i]))
    return out


def clip_features(jpegs: list[bytes], threads: int, batch: int):
    import open_clip
    import torch

    torch.set_num_threads(threads)
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_PRETRAINED)
    model.eval()
    tok = open_clip.get_tokenizer(CLIP_ARCH)
    embs = []
    with torch.no_grad():
        for s in range(0, len(jpegs), batch):
            x = torch.stack([preprocess(Image.open(io.BytesIO(b)).convert("RGB")) for b in jpegs[s:s + batch]])
            e = model.encode_image(x)
            embs.append(torch.nn.functional.normalize(e, dim=-1).numpy().astype(np.float32))
            if (s // batch) % 20 == 0:
                log(f"  clip {s + len(x)}/{len(jpegs)}")

        def concept(prompts):
            t = model.encode_text(tok(prompts))
            t = torch.nn.functional.normalize(t, dim=-1).mean(0)
            return torch.nn.functional.normalize(t, dim=0).numpy()

        classes = sorted(CLASS_TEXT)
        T = [concept([tp.format(CLASS_TEXT[c]) for tp in CLASS_TEMPLATES]) for c in classes]
        T += [concept(p) for p in DISTRACTORS.values()]
        scale = float(model.logit_scale.exp())
    return np.concatenate(embs), np.stack(T).astype(np.float32), scale


# ---- contact sheets ---------------------------------------------------------------------------------
def contact_sheet(tiles: list[tuple[bytes, list[str]]], out: Path, cols: int, tile: int = 160,
                  title: str = "", row_breaks: list[int] | None = None) -> None:
    """tiles: (jpeg bytes, caption lines).  row_breaks: tile indices that must start a new row."""
    font = ImageFont.load_default(size=11)
    cap_h = 14 * max((len(c) for _, c in tiles), default=1) + 4
    positions, r, c = [], 0, 0
    breaks = set(row_breaks or [])
    for i in range(len(tiles)):
        if i and (c >= cols or i in breaks):
            r, c = r + 1, 0
        positions.append((r, c))
        c += 1
    nrows = (positions[-1][0] + 1) if positions else 1
    top = 22 if title else 0
    sheet = Image.new("RGB", (cols * tile, top + nrows * (tile + cap_h)), "white")
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((4, 4), title, fill="black", font=font)
    for (b, caps), (r, c) in zip(tiles, positions):
        im = Image.open(io.BytesIO(b)).convert("RGB")
        im.thumbnail((tile - 6, tile - 6))
        x0, y0 = c * tile, top + r * (tile + cap_h)
        sheet.paste(im, (x0 + (tile - im.width) // 2, y0 + 3))
        for k, line in enumerate(caps):
            d.text((x0 + 3, y0 + tile + 14 * k), line[:28], fill="black", font=font)
        d.rectangle([x0, y0, x0 + tile - 1, y0 + tile + cap_h - 1], outline=(200, 200, 200))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=88)


# ---- helpers -----------------------------------------------------------------------------------------
def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == len(data) and path.read_bytes() == data:
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fingerprint(rows: list[dict]) -> str:
    lines = sorted(f"{r['id']}\t{r['label']}\t{r['split']}\t{'' if r['fold'] is None else r['fold']}"
                   for r in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- main ------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phash-radius", type=int, default=10)
    ap.add_argument("--cos-with-phash", type=float, default=0.85)
    ap.add_argument("--cos-only", type=float, default=0.92)
    ap.add_argument("--exclude-cos", type=float, default=0.97)
    ap.add_argument("--ood-threshold", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cache", type=Path, default=None,
                    help="optional .npz feature cache (phash/dhash/clip); reused when the sha256 list matches")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing a manifest with a new fingerprint")
    ap.add_argument("--no-tar", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    classes = list_classes()
    rels = enumerate_raw()
    log(f"{len(rels)} raw files, {len(classes)} class folders")
    layout_warnings = [r for r in rels if len(r.split("/")) != 3 or r.split("/")[0] != r.split("/")[1]]

    # 1. hash raw bytes, group exact duplicates
    by_sha: dict[str, list[str]] = collections.defaultdict(list)
    for r in rels:
        by_sha[hashlib.sha256((RAW / r).read_bytes()).hexdigest()].append(r)
    shas = sorted(by_sha)
    log(f"{len(shas)} unique sha256; converting")
    with ProcessPoolExecutor(args.workers) as ex:
        conv = dict(zip(shas, ex.map(_convert_job, [str(RAW / by_sha[s][0]) for s in shas], chunksize=16)))

    units, excluded = [], []  # unit = unique-content candidate row
    for s in shas:
        srcs = sorted(by_sha[s])
        labels = sorted({p.split("/")[0] for p in srcs})
        c = conv[s]
        if not c["ok"]:
            excluded.append({"sha256": s, "src_paths": srcs, "labels": labels, "reason": "undecodable",
                             "error": c["error"]})
            continue
        if len(labels) > 1:
            excluded.append({"sha256": s, "src_paths": srcs, "labels": labels, "reason": "conflicting_labels",
                             "orig_format": c["format"]})
            continue
        units.append({"id": s[:16], "sha256": s, "label": labels[0], "src_paths": srcs,
                      "orig_format": c["format"], "orig_width": c["w"], "orig_height": c["h"],
                      "flags": list(c["flags"]) + (["exact_dup_collapsed"] if len(srcs) > 1 else []),
                      "jpeg": c["jpeg"]})
    n_exact_groups = sum(1 for s in shas if len(by_sha[s]) > 1)
    n_exact_conflict = sum(1 for e in excluded if e["reason"] == "conflicting_labels")
    fmt_counts = collections.Counter(conv[s]["format"] for s in shas if conv[s]["ok"])
    log(f"units={len(units)} exact_dup_groups={n_exact_groups} conflicting={n_exact_conflict} "
        f"undecodable={sum(1 for e in excluded if e['reason'] == 'undecodable')} formats={dict(fmt_counts)}")
    assert len({u["id"] for u in units}) == len(units), "sha256[:16] id collision"

    # 2. features
    N = len(units)
    u_shas = [u["sha256"] for u in units]
    cached = None
    if args.cache and args.cache.exists():
        z = np.load(args.cache, allow_pickle=False)
        if list(z["shas"]) == u_shas:
            cached = z
            log("feature cache hit")
    if cached is not None:
        ph, dh, E, T, scale = cached["ph"], cached["dh"], cached["E"], cached["T"], float(cached["scale"])
    else:
        jpegs = [u["jpeg"] for u in units]
        ph, dh = perceptual_hashes(jpegs)
        log("perceptual hashes done; CLIP embeddings")
        E, T, scale = clip_features(jpegs, args.threads, args.batch)
        if args.cache:
            np.savez(args.cache, shas=np.array(u_shas), ph=ph, dh=dh, E=E, T=T, scale=scale)
    P, D = hamming_matrix(ph), hamming_matrix(dh)
    C = E @ E.T
    iu = np.triu_indices(N, 1)
    rule_ph = (P[iu] <= args.phash_radius) & (C[iu] >= args.cos_with_phash)
    rule_cos = C[iu] >= args.cos_only
    sel = np.nonzero(rule_ph | rule_cos)[0]
    edges = [(int(iu[0][k]), int(iu[1][k])) for k in sel]
    n_phash_cand = int((P[iu] <= args.phash_radius).sum())
    n_phash_rejected = int(((P[iu] <= args.phash_radius) & (C[iu] < args.cos_with_phash)).sum())
    log(f"pHash<={args.phash_radius} candidates={n_phash_cand} (rejected by CLIP: {n_phash_rejected}); "
        f"edges={len(edges)} (phash-rule {int(rule_ph.sum())}, cos-only-rule {int(rule_cos.sum())}, "
        f"cos-only not caught by phash rule {int((rule_cos & ~rule_ph).sum())})")
    dsu = DSU(N)
    for a, b in edges:
        dsu.union(a, b)
    comp = collections.defaultdict(list)
    for i in range(N):
        comp[dsu.find(i)].append(i)
    order = sorted(comp.values(), key=lambda m: min(units[i]["id"] for i in m))
    for k, members in enumerate(order):
        for i in members:
            units[i]["cluster"] = f"c{k:05d}"

    # 3. zero-shot OOD
    logits = scale * (E @ T.T)
    logits -= logits.max(1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(1, keepdims=True)
    nC = len(CLASS_TEXT)
    dnames = list(DISTRACTORS)
    ood_score = prob[:, nC:].sum(1)
    cls_sorted = sorted(CLASS_TEXT)
    for i, u in enumerate(units):
        j = int(prob[i, nC:].argmax())
        k = int(prob[i, :nC].argmax())
        u.update(_ood=float(ood_score[i]), _top_distractor=dnames[j], _top_distractor_p=float(prob[i, nC + j]),
                 _zs_class=cls_sorted[k], _zs_class_p=float(prob[i, k]),
                 _label_p=float(prob[i, cls_sorted.index(u["label"])]))
        if ood_score[i] >= args.ood_threshold:
            u["flags"].append("ood_suspect")

    # near-dup clusters, label conflicts
    nbrs = collections.defaultdict(list)
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    nd_clusters = [m for m in order if len(m) > 1]
    conflict_clusters = [m for m in nd_clusters if len({units[i]["label"] for i in m}) > 1]
    drop = set()
    for m in conflict_clusters:
        for i in m:
            units[i]["flags"].append("near_dup_label_conflict")
            for j in nbrs[i]:
                if units[j]["label"] != units[i]["label"] and C[i, j] >= args.exclude_cos:
                    drop.add(i)
    for m in nd_clusters:
        for i in m:
            units[i]["flags"].append("near_dup")
    for i in sorted(drop):
        u = units[i]
        excluded.append({"sha256": u["sha256"], "src_paths": u["src_paths"], "labels": [u["label"]],
                         "reason": "near_dup_conflicting_labels", "id": u["id"], "cluster": u["cluster"],
                         "conflicts_with": sorted(units[j]["id"] for j in nbrs[i]
                                                  if units[j]["label"] != u["label"]
                                                  and C[i, j] >= args.exclude_cos),
                         "orig_format": u["orig_format"]})
    keep = [i for i in range(N) if i not in drop]
    log(f"clusters={len(order)} near_dup_clusters={len(nd_clusters)} (max size "
        f"{max((len(m) for m in nd_clusters), default=0)}) mixed-label={len(conflict_clusters)} "
        f"excluded_near_dup={len(drop)} ood_suspects={sum('ood_suspect' in units[i]['flags'] for i in keep)}")

    # 4. splits
    from sklearn.model_selection import StratifiedGroupKFold
    import sklearn

    y = np.array([classes.index(units[i]["label"]) for i in keep])
    g = np.array([units[i]["cluster"] for i in keep])
    overall = np.bincount(y, minlength=len(classes)) / len(y)
    outer = list(StratifiedGroupKFold(N_TEST_SPLITS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, g))
    tvd = [0.5 * np.abs(np.bincount(y[te], minlength=len(classes)) / len(te) - overall).sum() for _, te in outer]
    best = int(np.argmin(tvd))
    dev_idx, test_idx = outer[best]
    split_of = {}
    for k in test_idx:
        split_of[keep[k]] = ("test", None)
    inner = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for f, (_, va) in enumerate(inner.split(np.zeros(len(dev_idx)), y[dev_idx], g[dev_idx])):
        for k in va:
            split_of[keep[dev_idx[k]]] = ("dev", f)
    log(f"test = outer fold {best} (TVD {tvd[best]:.4f}; all {[round(float(t), 4) for t in tvd]}) -> "
        f"{len(test_idx)} test / {len(dev_idx)} dev")

    # legacy split
    legacy = json.loads(LEGACY_SPLIT.read_text())
    legacy_of_rel, legacy_unmatched = {}, 0
    rel_set = set(rels)
    for sp in ("train", "val", "test"):
        for e in legacy.get(sp, []):
            p = e["path"].replace("\\", "/")
            rel = p.split(LEGACY_ANCHOR, 1)[1] if LEGACY_ANCHOR in p else p
            if rel in rel_set:
                legacy_of_rel[rel] = sp
            else:
                legacy_unmatched += 1

    rows = []
    for i in keep:
        u = units[i]
        sps = sorted({legacy_of_rel[p] for p in u["src_paths"] if p in legacy_of_rel},
                     key=["train", "val", "test"].index)
        if len(sps) > 1:
            u["flags"].append("legacy_multi_split")
        split, fold = split_of[i]
        rows.append({"id": u["id"], "path": f"images/{u['label']}/{u['id']}.jpg", "label": u["label"],
                     "cluster": u["cluster"], "split": split, "fold": fold, "sha256": u["sha256"],
                     "src_paths": u["src_paths"], "orig_format": u["orig_format"],
                     "orig_width": u["orig_width"], "orig_height": u["orig_height"],
                     "legacy_split": sps[0] if sps else None, "flags": sorted(set(u["flags"]))})
    rows.sort(key=lambda r: (r["label"], r["id"]))
    excluded.sort(key=lambda e: (e["reason"], e["sha256"]))
    fp = fingerprint(rows)

    # write images (rows only); move stale files aside (never delete)
    by_id = {u["id"]: u for u in units}
    for r in rows:
        atomic_write_bytes(OUT / r["path"], by_id[r["id"]]["jpeg"])
    wanted = {r["path"] for r in rows}
    stale = []
    for p in sorted((OUT / "images").rglob("*")):
        if p.is_file() and p.relative_to(OUT).as_posix() not in wanted:
            dst = OUT / "stale" / p.relative_to(OUT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(p, dst)
            stale.append(p.relative_to(OUT).as_posix())
    if stale:
        log(f"moved {len(stale)} stale images to data/v2/stale/")

    # 5. leakage gate
    violations = []
    part_of = collections.defaultdict(set)
    for r in rows:
        part_of[r["cluster"]].add("test" if r["split"] == "test" else f"fold{r['fold']}")
    crossing = [c for c, s in part_of.items() if len(s) > 1]
    if crossing:
        violations.append(f"{len(crossing)} clusters cross partitions: {crossing[:10]}")
    all_shas = [r["sha256"] for r in rows] + [e["sha256"] for e in excluded]
    if len(all_shas) != len(set(all_shas)):
        violations.append("a sha256 appears more than once across rows/excluded")
    if len({r["id"] for r in rows}) != len(rows):
        violations.append("duplicate row id")
    missing = [r["path"] for r in rows if not (OUT / r["path"]).is_file()]
    if missing:
        violations.append(f"{len(missing)} row images missing: {missing[:5]}")
    if classes != sorted(CLASS_TEXT) or len(classes) != 20:
        violations.append(f"class list mismatch: {classes}")
    if any(r["label"] not in classes for r in rows):
        violations.append("row label not in class list")
    accounted = collections.Counter(p for x in rows + excluded for p in x["src_paths"])
    if set(accounted) != rel_set or any(v != 1 for v in accounted.values()):
        violations.append("raw files not accounted for exactly once")
    if any(r["split"] == "dev" and r["fold"] not in range(N_FOLDS) for r in rows) or \
            any(r["split"] == "test" and r["fold"] is not None for r in rows):
        violations.append("bad split/fold values")

    # ---- report numbers
    def counts(sel):
        c = collections.Counter(r["label"] for r in sel)
        return {k: c.get(k, 0) for k in classes}

    test_rows = [r for r in rows if r["split"] == "test"]
    per_split = {"test": counts(test_rows)}
    for f in range(N_FOLDS):
        per_split[f"fold{f}"] = counts([r for r in rows if r["fold"] == f])
    per_split["total"] = counts(rows)
    low_test = {k: v for k, v in per_split["test"].items() if v < 15}

    # legacy leakage, measured on legacy entries (files)
    rel2cluster = {p: r["cluster"] for r in rows for p in r["src_paths"]}
    rel2sha = {p: s for s, ps in by_sha.items() for p in ps}
    excl_rel = {p for e in excluded for p in e["src_paths"]}
    train_clusters = {rel2cluster[p] for p, sp in legacy_of_rel.items() if sp == "train" and p in rel2cluster}
    train_shas = {rel2sha[p] for p, sp in legacy_of_rel.items() if sp == "train"}
    legacy_leak = {}
    for sp in ("val", "test"):
        ents = [p for p, s in legacy_of_rel.items() if s == sp]
        in_rows = [p for p in ents if p in rel2cluster]
        legacy_leak[sp] = {
            "entries": len(ents),
            "excluded_in_v2": sum(p in excl_rel for p in ents),
            "exact_byte_copy_of_legacy_train": sum(rel2sha[p] in train_shas for p in ents),
            "share_cluster_with_legacy_train": sum(rel2cluster[p] in train_clusters for p in in_rows),
            "near_dup_only_leak": sum(rel2cluster[p] in train_clusters and rel2sha[p] not in train_shas
                                      for p in in_rows),
        }

    excl_counts = collections.Counter(e["reason"] for e in excluded)
    nd_sizes = collections.Counter(len(m) for m in nd_clusters)
    created = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

    # 6. review material
    rev = OUT / "review"
    uidx = {u["id"]: i for i, u in enumerate(units)}
    row_by_id = {r["id"]: r for r in rows}
    conf_rows, conf_tiles, breaks = [], [], []
    for e in excluded:
        if e["reason"] == "conflicting_labels":
            conf_rows.append(["exact", e["sha256"][:16], e["sha256"][:16], "|".join(e["labels"]),
                              ";".join(e["src_paths"]), "excluded:conflicting_labels", ""])
            conf_tiles.append((conv[e["sha256"]]["jpeg"], [f"EXACT {e['sha256'][:12]}", "|".join(e["labels"])]))
    for m in conflict_clusters:
        breaks.append(len(conf_tiles))
        for i in sorted(m, key=lambda i: units[i]["label"]):
            u = units[i]
            xmax = max((float(C[i, j]) for j in m if units[j]["label"] != u["label"]), default=0.0)
            dec = "excluded:near_dup_conflicting_labels" if i in drop else "kept:flag near_dup_label_conflict"
            conf_rows.append(["near_dup", u["cluster"], u["id"], u["label"], ";".join(u["src_paths"]), dec,
                              f"{xmax:.4f}"])
            conf_tiles.append((u["jpeg"], [f"{u['cluster']} {u['id'][:10]}",
                                           f"{u['label']}{' [EXCL]' if i in drop else ''}"]))
    write_csv(rev / "conflicts.csv", ["kind", "group", "id", "labels", "src_paths", "decision",
                                      "max_cross_label_cos"], conf_rows)
    if conf_tiles:
        contact_sheet(conf_tiles, rev / "conflicts.jpg", cols=8,
                      title="Label conflicts: EXACT = byte-identical file in several class folders (excluded); "
                            "cNNNNN rows = near-dup cluster with mixed labels", row_breaks=breaks)

    nd_rows = []
    for m in nd_clusters:
        labs = sorted({units[i]["label"] for i in m})
        for i in m:
            u = units[i]
            others = [j for j in m if j != i]
            r = row_by_id.get(u["id"])
            nd_rows.append([u["cluster"], len(m), "|".join(labs), int(len(labs) > 1), u["id"], u["label"],
                            r["path"] if r else "", r["split"] if r else "excluded",
                            "" if not r or r["fold"] is None else r["fold"],
                            f"{max(float(C[i, j]) for j in others):.4f}", int(min(P[i, j] for j in others)),
                            ";".join(u["src_paths"])])
    write_csv(rev / "near_dup_clusters.csv", ["cluster", "size", "labels", "mixed_labels", "id", "label", "path",
                                              "split", "fold", "max_cos_to_mate", "min_phash_to_mate", "src_paths"],
              nd_rows)
    write_csv(rev / "near_dup_edges.csv", ["id_a", "id_b", "label_a", "label_b", "phash_hamming", "dhash_hamming",
                                           "clip_cos", "rule"],
              [[units[a]["id"], units[b]["id"], units[a]["label"], units[b]["label"], int(P[a, b]), int(D[a, b]),
                f"{C[a, b]:.4f}",
                "phash+clip" if (P[a, b] <= args.phash_radius and C[a, b] >= args.cos_with_phash) else "clip_only"]
               for a, b in edges])
    rng = np.random.default_rng(SEED)
    by_size = sorted(nd_clusters, key=lambda m: (-len(m), min(m)))
    pick, rest = by_size[:6], by_size[6:]
    pick += [rest[k] for k in sorted(rng.choice(len(rest), size=min(24, len(rest)), replace=False))] if rest else []
    nd_tiles, nd_breaks = [], []
    for m in pick:
        nd_breaks.append(len(nd_tiles))
        for i in m[:8]:
            u = units[i]
            others = [j for j in m if j != i]
            nd_tiles.append((u["jpeg"], [f"{u['cluster']} {u['id'][:10]}", u["label"],
                                         f"cos{max(float(C[i, j]) for j in others):.3f} "
                                         f"ph{min(int(P[i, j]) for j in others)}"]))
    if nd_tiles:
        contact_sheet(nd_tiles, rev / "near_dup_examples.jpg", cols=8, tile=150,
                      title="Near-duplicate clusters, one per row (6 largest, then 24 random, seed 42; <=8 shown)",
                      row_breaks=nd_breaks)

    ood_list = sorted((units[uidx[r["id"]]] for r in rows if "ood_suspect" in r["flags"]), key=lambda u: -u["_ood"])
    write_csv(rev / "ood_suspects.csv", ["id", "label", "path", "split", "fold", "ood_score", "top_distractor",
                                         "top_distractor_p", "zeroshot_class", "zeroshot_class_p", "label_p"],
              [[u["id"], u["label"], row_by_id[u["id"]]["path"], row_by_id[u["id"]]["split"],
                "" if row_by_id[u["id"]]["fold"] is None else row_by_id[u["id"]]["fold"], f"{u['_ood']:.4f}",
                u["_top_distractor"], f"{u['_top_distractor_p']:.4f}", u["_zs_class"], f"{u['_zs_class_p']:.4f}",
                f"{u['_label_p']:.4f}"] for u in ood_list])
    write_csv(rev / "zeroshot_scores.csv", ["id", "label", "ood_score", "top_distractor", "top_distractor_p",
                                            "zeroshot_class", "zeroshot_class_p", "label_p"],
              [[r["id"], r["label"], f"{units[uidx[r['id']]]['_ood']:.4f}", units[uidx[r["id"]]]["_top_distractor"],
                f"{units[uidx[r['id']]]['_top_distractor_p']:.4f}", units[uidx[r["id"]]]["_zs_class"],
                f"{units[uidx[r['id']]]['_zs_class_p']:.4f}", f"{units[uidx[r['id']]]['_label_p']:.4f}"]
               for r in rows])
    if ood_list:
        contact_sheet([(u["jpeg"], [u["id"][:12], u["label"], f"{u['_top_distractor'][:16]} {u['_ood']:.2f}"])
                       for u in ood_list[:80]], rev / "ood_suspects_top80.jpg", cols=10, tile=150,
                      title=f"Top {min(80, len(ood_list))} of {len(ood_list)} OOD suspects "
                            f"(CLIP zero-shot distractor mass)")

    # embeddings for rows (manifest order)
    emb_ids = [r["id"] for r in rows]
    np.save(OUT / "clip_b32_embeddings.npy", E[[uidx[i] for i in emb_ids]].astype(np.float32))
    atomic_write_text(OUT / "clip_b32_ids.json", json.dumps(
        {"model": f"open_clip {CLIP_ARCH} {CLIP_PRETRAINED}", "normalized": True, "input": "converted v2 JPEG",
         "order": "manifest rows order", "ids": emb_ids}))

    report = {
        "version": VERSION, "created_utc": created, "fingerprint": fp,
        "gate": {"passed": not violations, "violations": violations},
        "counts": {
            "raw_files": len(rels), "unique_sha256": len(shas),
            "decoded_unique": sum(1 for s in shas if conv[s]["ok"]),
            "exact_dup_groups": n_exact_groups,
            "exact_dup_redundant_files": len(rels) - len(shas),
            "exact_dup_conflicting_label_groups": n_exact_conflict,
            "rows": len(rows), "clusters": len({r["cluster"] for r in rows}),
            "near_dup_clusters": len(nd_clusters), "rows_in_near_dup_clusters": sum(len(m) for m in nd_clusters),
            "near_dup_cluster_size_hist": {str(k): v for k, v in sorted(nd_sizes.items())},
            "near_dup_mixed_label_clusters": len(conflict_clusters),
            "near_dup_label_conflict_rows_kept": sum("near_dup_label_conflict" in r["flags"] for r in rows),
            "near_dup_edges": len(edges),
            "excluded_total": len(excluded), "excluded_by_reason": dict(sorted(excl_counts.items())),
            "ood_suspect": len(ood_list),
            "test_rows": len(test_rows), "dev_rows": len(rows) - len(test_rows),
            "orig_format_unique_decoded": dict(fmt_counts.most_common()),
            "flags": dict(collections.Counter(f for r in rows for f in r["flags"]).most_common()),
            "raw_layout_warnings": len(layout_warnings),
        },
        "near_dup_rule": {
            "phash_radius": args.phash_radius, "cos_with_phash": args.cos_with_phash,
            "cos_only": args.cos_only, "exclude_cos": args.exclude_cos,
            "phash_candidates": n_phash_cand, "phash_candidates_rejected_by_clip": n_phash_rejected,
            "edges_phash_rule": int(rule_ph.sum()), "edges_cos_only_rule": int(rule_cos.sum()),
            "edges_cos_only_not_in_phash_rule": int((rule_cos & ~rule_ph).sum()),
        },
        "ood_rule": {"threshold": args.ood_threshold, "distractors": list(DISTRACTORS),
                     "score": "softmax(logit_scale * cos) mass on distractor concepts vs 20 class concepts"},
        "split": {"test_outer_fold": best, "outer_tvd": [round(float(t), 5) for t in tvd],
                  "sklearn": sklearn.__version__, "low_test_classes_lt15": low_test},
        "per_class_split_counts": per_split,
        "legacy_split": {"matched_entries": len(legacy_of_rel), "unmatched_entries": legacy_unmatched,
                         "rows_with_legacy_split": {str(k): v for k, v in
                                                    collections.Counter(r["legacy_split"] for r in rows).items()},
                         "val_test_vs_train": legacy_leak},
        "fingerprint_definition": "sha256 over sorted 'id\\tlabel\\tsplit\\tfold' lines (fold '' for test) "
                                  "joined by '\\n', first 16 hex",
        "stale_images_moved": stale,
    }

    if violations:
        atomic_write_text(OUT / "leakage_report.json", json.dumps(report, indent=2))
        log("LEAKAGE GATE FAILED:\n  " + "\n  ".join(violations))
        return 2

    mpath = OUT / "manifest.json"
    if mpath.exists() and not args.overwrite:
        old_fp = json.loads(mpath.read_text()).get("fingerprint")
        if old_fp != fp:
            log(f"existing manifest fingerprint {old_fp} != new {fp}; refusing (use --overwrite)")
            return 3

    if not args.no_tar:
        tpath = OUT / "images.tar"
        tmp = tpath.with_name("images.tar.tmp")
        with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tf:
            for r in sorted(rows, key=lambda r: r["path"]):
                data = (OUT / r["path"]).read_bytes()
                ti = tarfile.TarInfo(r["path"])
                ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
                tf.addfile(ti, io.BytesIO(data))
        os.replace(tmp, tpath)
        report["images_tar"] = {"path": "images.tar", "bytes": tpath.stat().st_size, "sha256": sha256_file(tpath),
                                "members": len(rows)}
        log(f"images.tar {tpath.stat().st_size / 1e6:.1f} MB sha256 {report['images_tar']['sha256']}")
    elif (OUT / "images.tar").exists():
        tpath = OUT / "images.tar"
        with tarfile.open(tpath) as tf:
            same = sorted(tf.getnames()) == sorted(r["path"] for r in rows)
        report["images_tar"] = {"path": "images.tar", "bytes": tpath.stat().st_size, "sha256": sha256_file(tpath),
                                "members": len(rows), "rebuilt": False, "member_names_match_rows": same}

    manifest = {
        "version": VERSION, "created_utc": created, "classes": classes, "images_root": "images",
        "split_policy": {"seed": SEED, "test_fraction": round(len(test_rows) / len(rows), 4), "n_folds": N_FOLDS,
                         "group_key": "cluster"},
        "fingerprint": fp, "rows": rows, "excluded": excluded,
    }
    atomic_write_text(mpath, json.dumps(manifest, indent=1))
    atomic_write_text(OUT / "leakage_report.json", json.dumps(report, indent=2))
    log(f"OK rows={len(rows)} test={len(test_rows)} fingerprint={fp} ({time.time() - t0:.0f}s)")
    if low_test:
        log(f"WARNING classes with <15 test rows: {low_test}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
