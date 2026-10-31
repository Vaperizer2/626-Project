"""Build the unlabeled distillation pool: images that are disjoint from every data/v2 image (dev and test) and
from data/web_holdout_v1, so a student distilled on it cannot have seen an evaluation fold.

    uv run python tools/distill/pool.py --component RAW_DIR --generic GENERIC_DIR --out POOL \
        --v2 data/v2 --v2-clip DIR_WITH_clip_b32_embeddings.npy --holdout data/web_holdout_v1

Every immediate subdirectory of --component / --generic is one source. Each file is converted exactly like
build_dataset_v2.py (RGB, EXIF-transposed, alpha on white, longest side <= 512, JPEG q95) into
POOL/images/<sha256(raw)[:16]>.jpg. A pool image is dropped when

* its raw sha256 equals a v2 or holdout image (exact duplicate),
* the max OpenCLIP ViT-B-32 (laion2b) cosine between any of its variants (identity, h-flip, rot90, rot270) and any
  v2 / holdout image reaches --cos-gate (crops, flips, rotations and re-encodes of one photo count as that photo),
* the 64-bit pHash of the image or its mirror is within --phash-gate bits of a v2 / holdout pHash.

Survivors are deduplicated among themselves (cosine >= --internal-cos, largest image kept) and capped per source.
Outputs: POOL/pool.json (kept rows + counts), POOL/dropped.csv, POOL/clip_b32.npy (kept rows), and review sheets
POOL/review/{nearest_kept,dropped_boundary}.jpg pairing pool images with their nearest v2 image.
Holdout images are only embedded and hashed for this gate; they are never displayed or used otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from build_dataset_v2 import (CLIP_ARCH, CLIP_PRETRAINED, DSU, atomic_write_text, contact_sheet,  # noqa: E402
                              to_rgb_jpeg, write_csv)

EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".avif", ".tif", ".tiff"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def phash64(im: Image.Image) -> int:
    import imagehash

    return int("".join("1" if b else "0" for b in imagehash.phash(im, hash_size=8).hash.flatten()), 2)


def convert_job(args: tuple[str, str]) -> dict:
    """Convert one raw file into the pool (idempotent: an existing output is reused). Never raises."""
    src, out_dir = args
    data = Path(src).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    out = Path(out_dir) / f"{sha[:16]}.jpg"
    if out.exists():
        jpeg = out.read_bytes()
    else:
        r = to_rgb_jpeg(data)
        if not r["ok"]:
            return {"src": src, "sha": sha, "ok": False, "error": r["error"]}
        jpeg = r["jpeg"]
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_bytes(jpeg)
        tmp.replace(out)
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    return {"src": src, "sha": sha, "ok": True, "w": im.width, "h": im.height,
            "phash": phash64(im), "phash_flip": phash64(im.transpose(Image.Transpose.FLIP_LEFT_RIGHT))}


def ref_job(path: str) -> dict:
    data = Path(path).read_bytes()
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return {"sha": hashlib.sha256(data).hexdigest(), "phash": phash64(im)}


VARIANTS = (None, Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270)


class _Files:
    def __init__(self, paths: list[Path], preprocess, variants=VARIANTS):
        self.paths, self.pre, self.variants = paths, preprocess, variants

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        import torch

        im = Image.open(self.paths[i]).convert("RGB")
        return torch.stack([self.pre(im if v is None else im.transpose(v)) for v in self.variants])


def clip_embed(paths: list[Path], variants=VARIANTS, batch: int = 128) -> np.ndarray:
    """L2-normalised OpenCLIP ViT-B-32 embeddings [N, len(variants), 512] (same model and preprocessing as v2)."""
    import open_clip
    import torch
    from torch.utils.data import DataLoader

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, pre = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_PRETRAINED, device=dev)
    model.eval()
    out = []
    dl = DataLoader(_Files(paths, pre, variants), batch_size=batch, num_workers=8)
    with torch.no_grad():
        for k, x in enumerate(dl):
            b, v = x.shape[:2]
            e = model.encode_image(x.flatten(0, 1).to(dev))
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).view(b, v, -1).cpu().numpy())
            if k % 50 == 0:
                log(f"  clip {k * batch + b}/{len(paths)}")
    return np.concatenate(out).astype(np.float32)


def max_cos(q: np.ndarray, refs: np.ndarray, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """For q [N, V, D] and refs [M, D]: max over variants and refs of the cosine, and the arg-max ref."""
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = torch.from_numpy(refs).to(dev)
    best, arg = [], []
    for s in range(0, len(q), chunk):
        S = torch.from_numpy(q[s:s + chunk]).to(dev) @ R.T  # [n, V, M]
        m, a = S.max(1).values.max(1)
        best.append(m.cpu().numpy())
        arg.append(a.cpu().numpy())
    return np.concatenate(best), np.concatenate(arg)


def min_hamming(q: np.ndarray, refs: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """q [N, K] uint64 hashes, refs [M] uint64 -> min Hamming distance over K and M, per row."""
    out = np.empty(len(q), np.int64)
    for s in range(0, len(q), chunk):
        x = np.bitwise_xor(q[s:s + chunk, :, None], refs[None, None, :])
        out[s:s + chunk] = np.bitwise_count(x).min(axis=(1, 2))
    return out


def internal_pairs(E: np.ndarray, thr: float, chunk: int = 4096) -> list[tuple[int, int]]:
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(E).to(dev)
    pairs = []
    for s in range(0, len(E), chunk):
        S = X[s:s + chunk] @ X.T
        i, j = torch.nonzero(S >= thr, as_tuple=True)
        i = i + s
        keep = i < j
        pairs += list(zip(i[keep].tolist(), j[keep].tolist()))
    return pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--component", nargs="+", default=[], help="dirs whose subdirs are in-domain sources")
    ap.add_argument("--generic", nargs="+", default=[], help="dirs whose subdirs are generic-image sources")
    ap.add_argument("--out", required=True)
    ap.add_argument("--v2", required=True, help="data/v2 (manifest.json + images/)")
    ap.add_argument("--v2-clip", required=True, help="dir holding v2 clip_b32_embeddings.npy + clip_b32_ids.json")
    ap.add_argument("--holdout", required=True, help="data/web_holdout_v1 (manifest.json + images/)")
    ap.add_argument("--cos-gate", type=float, default=0.85)
    ap.add_argument("--phash-gate", type=int, default=10)
    ap.add_argument("--internal-cos", type=float, default=0.95)
    ap.add_argument("--min-side", type=int, default=64)
    ap.add_argument("--cap", nargs="*", default=[], help="SOURCE=N caps applied after deduplication")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)

    out = Path(a.out)
    img_dir, cache = out / "images", out / "cache"
    img_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=True)

    files: list[tuple[str, str, Path]] = []  # (source, domain, path)
    for domain, roots in (("component", a.component), ("generic", a.generic)):
        for root in roots:
            for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
                files += [(d.name, domain, f) for f in sorted(d.rglob("*")) if f.is_file() and f.suffix.lower() in EXT]
    log(f"{len(files)} candidate files from {len({s for s, _, _ in files})} sources")

    with ProcessPoolExecutor(a.workers) as ex:
        conv = list(ex.map(convert_job, [(str(p), str(img_dir)) for _, _, p in files], chunksize=64))
    log(f"converted: {sum(c['ok'] for c in conv)} ok, {sum(not c['ok'] for c in conv)} undecodable")

    # ---- references: every v2 image (dev + test) and every holdout image
    v2m = json.loads((Path(a.v2) / "manifest.json").read_text())
    hom = json.loads((Path(a.holdout) / "manifest.json").read_text())
    ref_paths = [Path(a.v2) / r["path"] for r in v2m["rows"]] + [Path(a.holdout) / r["path"] for r in hom["rows"]]
    ref_kind = np.array(["v2"] * len(v2m["rows"]) + ["holdout"] * len(hom["rows"]))
    ref_ids = [r["id"] for r in v2m["rows"]] + [r["id"] for r in hom["rows"]]
    ref_sha = {r["sha256"] for r in v2m["rows"]} | {r["sha256"] for r in hom["rows"]}
    ref_sha |= {r["sha256"] for r in v2m.get("excluded", [])}
    with ProcessPoolExecutor(a.workers) as ex:
        refh = list(ex.map(ref_job, map(str, ref_paths), chunksize=32))
    ref_sha |= {r["sha"] for r in refh}
    ref_phash = np.array([r["phash"] for r in refh], np.uint64)

    v2E = np.load(Path(a.v2_clip) / "clip_b32_embeddings.npy")
    v2ids = json.loads((Path(a.v2_clip) / "clip_b32_ids.json").read_text())
    v2ids = v2ids["ids"] if isinstance(v2ids, dict) else v2ids
    pos = {i: k for k, i in enumerate(v2ids)}
    v2E = v2E[[pos[r["id"]] for r in v2m["rows"]]]
    hoE_path = cache / "holdout_clip.npy"
    if not hoE_path.exists():
        np.save(hoE_path, clip_embed([Path(a.holdout) / r["path"] for r in hom["rows"]], variants=(None,))[:, 0])
    refE = np.concatenate([v2E, np.load(hoE_path)])
    probe = clip_embed(ref_paths[:64], variants=(None,))[:, 0]
    agree = float((probe * v2E[:64]).sum(1).min())
    log(f"recomputed vs stored v2 CLIP embeddings: min cosine {agree:.5f} over 64 images")
    assert agree > 0.99, "pool CLIP embeddings are not comparable with data/v2/clip_b32_embeddings.npy"

    # ---- pool rows: one per unique raw sha
    rows: dict[str, dict] = {}
    dropped: list[list] = []
    for (src, domain, path), c in zip(files, conv):
        if not c["ok"]:
            dropped.append([c["sha"][:16], src, str(path), "undecodable", "", ""])
            continue
        r = rows.get(c["sha"][:16])
        if r is not None:
            r["src_paths"].append(str(path))
            continue
        rows[c["sha"][:16]] = {"id": c["sha"][:16], "path": f"images/{c['sha'][:16]}.jpg", "source": src,
                               "domain": domain, "src_paths": [str(path)], "w": c["w"], "h": c["h"],
                               "sha256": c["sha"], "phash": (c["phash"], c["phash_flip"])}
    pool = list(rows.values())
    log(f"{len(pool)} unique decoded pool images")

    E_path = cache / f"pool_clip_{len(pool)}.npy"
    if not E_path.exists():
        np.save(E_path, clip_embed([out / r["path"] for r in pool]))
    E = np.load(E_path)
    mc, arg = max_cos(E, refE)
    ham = min_hamming(np.array([r["phash"] for r in pool], np.uint64), ref_phash)

    keep = []
    for k, r in enumerate(pool):
        r["max_cos_ref"], r["nearest_ref"] = round(float(mc[k]), 4), f"{ref_kind[arg[k]]}:{ref_ids[arg[k]]}"
        r["min_phash_ref"] = int(ham[k])
        reason = ("too_small" if min(r["w"], r["h"]) < a.min_side else
                  "exact_dup_ref" if r["sha256"] in ref_sha else
                  f"near_dup_{ref_kind[arg[k]]}" if mc[k] >= a.cos_gate else
                  "phash_near_ref" if ham[k] <= a.phash_gate else None)
        if reason:
            dropped.append([r["id"], r["source"], r["src_paths"][0], reason, r["max_cos_ref"], r["nearest_ref"]])
        else:
            keep.append(k)
    log(f"leakage gate kept {len(keep)} / {len(pool)}")

    dsu = DSU(len(keep))
    for i, j in internal_pairs(E[keep, 0], a.internal_cos):
        dsu.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(keep)):
        groups.setdefault(dsu.find(i), []).append(i)
    survivors = []
    for g in groups.values():
        g.sort(key=lambda i: (-pool[keep[i]]["w"] * pool[keep[i]]["h"], pool[keep[i]]["id"]))
        survivors.append(keep[g[0]])
        for i in g[1:]:
            r = pool[keep[i]]
            dropped.append([r["id"], r["source"], r["src_paths"][0], "internal_dup", r["max_cos_ref"],
                            f"pool:{pool[keep[g[0]]]['id']}"])
    caps = {s: int(n) for s, n in (c.split("=") for c in a.cap)}
    survivors.sort(key=lambda k: pool[k]["id"])
    final, per_src = [], {}
    for k in survivors:
        r = pool[k]
        per_src[r["source"]] = per_src.get(r["source"], 0) + 1
        if per_src[r["source"]] > caps.get(r["source"], 1 << 30):
            dropped.append([r["id"], r["source"], r["src_paths"][0], "over_cap", r["max_cos_ref"], ""])
        else:
            final.append(k)
    log(f"after internal dedupe and caps: {len(final)}")

    kept_rows = [{k: v for k, v in pool[i].items() if k != "phash"} for i in final]
    counts: dict[str, dict] = {}
    for s, _, _ in files:
        counts.setdefault(s, {"files": 0})["files"] += 1
    for d in dropped:
        counts[d[1]][d[3]] = counts[d[1]].get(d[3], 0) + 1
    for r in kept_rows:
        counts[r["source"]]["kept"] = counts[r["source"]].get("kept", 0) + 1
    cos_kept = np.array([r["max_cos_ref"] for r in kept_rows])
    report = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "params": {k: v for k, v in vars(a).items()}, "clip": f"{CLIP_ARCH}/{CLIP_PRETRAINED}",
              "clip_v2_recompute_min_cos": agree, "n_refs": {"v2": int((ref_kind == "v2").sum()),
                                                             "holdout": int((ref_kind == "holdout").sum())},
              "n_kept": len(kept_rows), "n_kept_by_domain": {d: sum(r["domain"] == d for r in kept_rows)
                                                             for d in ("component", "generic")},
              "kept_max_cos_ref_quantiles": {q: round(float(np.quantile(cos_kept, q)), 4)
                                             for q in (0.5, 0.9, 0.99, 1.0)},
              "kept_min_phash_ref_min": int(min(r["min_phash_ref"] for r in kept_rows)),
              "sources": counts}
    atomic_write_text(out / "pool.json", json.dumps({**report, "rows": kept_rows}, indent=1))
    write_csv(out / "dropped.csv", ["id", "source", "src_path", "reason", "max_cos_ref", "nearest"], dropped)
    np.save(out / "clip_b32.npy", E[final, 0])
    print(json.dumps(report, indent=1))

    v2_by_id = {r["id"]: Path(a.v2) / r["path"] for r in v2m["rows"]}

    def pair_sheet(items: list[tuple[Path, float, str]], name: str, title: str) -> None:
        tiles = []
        for p, c, ref in items:
            tiles += [(p.read_bytes(), [f"pool cos {c:.3f}"]), (v2_by_id[ref].read_bytes(), [f"v2 {ref}"])]
        contact_sheet(tiles, out / "review" / name, cols=8, title=title)

    near = sorted((r for r in kept_rows if r["nearest_ref"].startswith("v2:")), key=lambda r: -r["max_cos_ref"])[:48]
    pair_sheet([(out / r["path"], r["max_cos_ref"], r["nearest_ref"][3:]) for r in near], "nearest_kept.jpg",
               "KEPT pool images with the highest CLIP cosine to any v2 image (pool | nearest v2)")
    bd = sorted((d for d in dropped if d[3] == "near_dup_v2"), key=lambda d: d[4])[:48]
    pair_sheet([(out / "images" / f"{d[0]}.jpg", d[4], d[5][3:]) for d in bd], "dropped_boundary.jpg",
               f"DROPPED pool images just above the cosine gate {a.cos_gate} (pool | nearest v2)")


if __name__ == "__main__":
    main()
