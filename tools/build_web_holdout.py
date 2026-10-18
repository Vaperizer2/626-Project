#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "numpy>=2.0",
#   "pillow>=10.4",
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
"""Build the independent *web holdout* test set (data/web_holdout_v1) from Wikimedia Commons.

This set is for FINAL evaluation / demo only.  No classifier is ever run on it here; the only model
used is OpenCLIP ViT-B-32 (laion2b_s34b_b79k) as a leakage detector against data/v2.

    uv run --no-config tools/build_web_holdout.py collect   # query Commons + download 640-px thumbnails
    uv run --no-config tools/build_web_holdout.py build     # QC, leakage gate, review drops, write outputs

Wikimedia throttles file downloads (~50 per 10 min per client); ``collect`` is resumable and paces itself
(DL_GAP_S) and honours Retry-After, so a full run takes a few hours.  Candidates that could not be downloaded
are listed in dropped.csv as ``not_downloaded``.

Pipeline
--------
1. ``collect``: for every class, list files in a hand-picked set of Commons categories (CATEGORIES below;
   one level of subcategories is named explicitly, never recursed blindly) with the MediaWiki API
   (generator=categorymembers + prop=imageinfo, iiurlwidth=640).  Keep only image/jpeg|png, CC or
   public-domain licences, originals >= 240 px on both sides, and titles/categories free of diagram /
   schematic / symbol / logo / map / drawing keywords.  Candidates are taken round-robin over the class'
   categories with at most MAX_PER_UPLOADER per uploader, up to N_CANDIDATES per class.  A Commons file
   that qualifies for two classes is dropped from both (``cross_class_ambiguous``).  Thumbnails are
   downloaded sequentially (<= 1 req/s, descriptive User-Agent, Retry-After honoured) into --cache.
2. ``build``: files listed in results/demo/wikimedia_credits.json (already shown to the final model) are
   dropped as ``used_in_demo`` (title, sha256 prefix, or CLIP cosine >= LEAK_COS to a demo thumbnail);
   sha256 of the downloaded bytes -> id = sha256[:16]; exact duplicates collapse (first kept);
   convert to RGB JPEG q95, EXIF-transposed, longest side <= 512 (same rules as build_dataset_v2.py);
   CLIP embeddings of the converted JPEG; drop any image with cosine >= LEAK_COS to ANY data/v2 image
   (``leakage_v2``) and near-duplicates inside the holdout (cosine >= INTERNAL_DUP_COS, ``near_duplicate_holdout``);
   apply the human visual-review decisions in data/web_holdout_v1/review_decisions.json
   ({"<id>": "<reason>"}, or "relabel:<class>" when the photo clearly shows another class); then cap each class at TARGET_PER_CLASS (``over_target_cap``, deterministic
   order: lowest uploader count first, then id).
3. Outputs in data/web_holdout_v1/: images/<class>/<id>.jpg, manifest.json (v2 row schema + license,
   author, source_url), dropped.csv, ATTRIBUTION.md, contact_sheet.jpg; review sheets go to --cache/review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "data" / "v2"
OUT = REPO / "data" / "web_holdout_v1"
DEMO_CREDITS = REPO / "results" / "demo" / "wikimedia_credits.json"  # files the final model has already seen
VERSION = "web_holdout_v1"
API = "https://commons.wikimedia.org/w/api.php"
UA = "compclass-holdout/0.1 (educational research project)"
MIN_GAP_S = 1.0           # seconds between API requests (sequential, polite)
DL_GAP_S = 12.0           # seconds between file downloads (upload+thumb hosts allow ~50 req / 10 min)
THUMB_W = 500             # a standard Wikimedia thumbnail bucket (https://w.wiki/GHai); others are throttled
MAX_SIDE = 512
JPEG_QUALITY = 95
N_CANDIDATES = 48         # downloaded per class (before QC / review)
N_CANDIDATES_CLASS = {"power supply module": 64, "led": 64, "connector": 60, "transformer": 60,
                      "crystal oscillator": 60}  # classes whose Commons pool is review-heavy
TARGET_PER_CLASS = 25
MIN_PER_CLASS = 15
MAX_PER_UPLOADER = 4
MAX_PER_UPLOADER_CLASS = {"jumper wire": 8}  # tiny Commons pool; diversity cap relaxed
MIN_ORIG_SIDE = 240
LEAK_COS = 0.92
INTERNAL_DUP_COS = 0.95
CLIP_ARCH, CLIP_PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"

# class -> Commons categories (verified to exist 2026-10-17).  Order matters for round-robin.
CATEGORIES: dict[str, list[str]] = {
    "battery holder": ["Battery holders"],
    "breadboard": ["Breadboards", "Electronic circuits on breadboards"],
    "capacitor": ["Aluminum electrolytic capacitors", "Ceramic capacitors", "Film capacitors",
                  "Tantalum capacitors", "Electrolytic capacitors"],
    "connector": ["Header connectors", "JST connector", "D-subminiature", "Electrical connection terminals",
                  "DC connectors", "Crimp connectors", "Pin headers", "Screw terminals", "Ribbon cable connectors"],
    "crystal oscillator": ["Crystal oscillators", "Crystal resonator", "Soviet crystal resonators"],
    "diode": ["Diodes", "Rectifiers", "Zener diodes", "DO-41 (package)", "DO-35 (package)", "Diodes (SMD)"],
    "fuse": ["Electrical fuses", "Fuses", "Blade type fuses", "Thermal fuses", "Automobile fuses"],
    "heat sink": ["Heat sinks", "Computer heatsinks"],
    "ic chip": ["Integrated circuits (SMD)", "Unidentified integrated circuits", "DIP integrated circuit packages",
                "SOIC integrated circuit packages", "QFP integrated circuit packages", "PLCC integrated circuit packages"],
    "inductor coil": ["Inductors", "Ferrite inductors", "Air coils"],
    "jumper wire": ["Jumper wires", "Jumper wires female–female", "Jumper wires male–male"],
    "led": ["Light-emitting diodes (Through Hole)", "Light-emitting diodes (SMD)", "Light-emitting diodes",
            "Red light-emitting diodes", "Blue light-emitting diodes", "Green light-emitting diodes",
            "White light-emitting diodes", "RGB light-emitting diodes"],
    "microcontroller board": ["Arduino UNO", "Arduino Nano", "Arduino Mega", "ESP32", "NodeMCU", "Raspberry Pi Pico"],
    "potentiometer": ["Potentiometers", "Trimmer resistors"],
    "power supply module": ["Voltage regulator modules", "Buck converters", "Switched-mode power supplies",
                            "Power supplies for LED lamps"],
    "relay": ["PCB-mounted relays", "Relays", "Automobile relays"],
    "resistor": ["Resistors (Through Hole)", "Resistors (SMD)", "Power resistors", "Electronic color code"],
    "switch": ["Toggle switches", "Tactile switches", "Rocker switches", "Slide switches", "DIP switches",
               "Microswitches"],
    "transformer": ["Toroidal transformers", "Transformers", "Ferrite transformers", "Tube amp transformers",
                    "Signal transformers", "SELV transformers", "Isolating transformers",
                    "Control circuit transformers"],
    "transistor": ["Transistors", "Transistors (SMD)", "Power transistors", "Silicon bipolar power transistors",
                   "Transistor packages"],
}
# individual files found by full-text search for classes whose categories are too small
EXTRA_FILES: dict[str, list[str]] = {
    "jumper wire": ["File:A few Jumper Wires.jpg", "File:Jumper wire.jpg",
                    "File:Jumper Wires with Crocodile Clips.jpg",
                    "File:Jumper Wires with Crocodile Clips (x10 closeup) - JUMP-03 02 (4150694002).jpg",
                    "File:FF Jumper Wires.jpg", "File:Female-Female Jumper Wire.jpg", "File:F-F jumper wires.jpg",
                    "File:Jumper male-to-male.jpg", "File:Jumper male-to-female.jpg", "File:Cables puentes.jpg",
                    "File:Jumper cables.jpg", "File:Cjam-jumper-wires.png"],
}
BAD_WORDS = re.compile(
    r"diagram|schemat|symbol|\blogos?\b|\bmaps?\b|drawing|pinout|pin-out|\bcharts?\b|\bgraphs?\b|\bplots?\b|"
    r"\bicons?\b|spectr|sketch|\brender|\.svg|\bcad\b|\bstl\b|blueprint|patent|datasheet|screenshot|fritzing|"
    r"x-ray|xray|micrograph|\bdies?\b|decap|animation|clip ?art|equivalent circuit|waveform", re.I)
OK_LICENSE = re.compile(r"^(cc|public domain|pd|cc0)", re.I)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---- polite HTTP -------------------------------------------------------------------------------------
_last = [0.0]


def http_get(url: str, tries: int = 8, gap: float = MIN_GAP_S) -> bytes:
    for attempt in range(tries):
        dt = time.time() - _last[0]
        if dt < gap:
            time.sleep(gap - dt)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            _last[0] = time.time()
            return data
        except urllib.error.HTTPError as e:
            _last[0] = time.time()
            if e.code in (429, 500, 502, 503, 504):
                ra = e.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else 10.0 * (attempt + 1)
                log(f"  HTTP {e.code}; backing off {wait:.0f}s")
                time.sleep(min(wait, 900) + 5)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            _last[0] = time.time()
            log(f"  network error {e}; retry")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def api(**params) -> dict:
    params.update(format="json", formatversion="2")
    return json.loads(http_get(API + "?" + urllib.parse.urlencode(params)))


II = dict(prop="imageinfo", iiprop="url|mime|size|sha1|user|extmetadata", iiurlwidth=THUMB_W,
          iiextmetadatafilter="LicenseShortName|LicenseUrl|Artist|Credit|Categories|ImageDescription|UsageTerms")


def category_files(cat: str, cap: int = 150) -> list[dict]:
    pages, cont = [], {}
    while len(pages) < cap:
        r = api(action="query", generator="categorymembers", gcmtitle=f"Category:{cat}", gcmtype="file",
                gcmlimit=50, **II, **cont)
        pages += r.get("query", {}).get("pages", [])
        if "continue" not in r:
            break
        cont = r["continue"]
    return pages


def titles_info(titles: list[str]) -> list[dict]:
    if not titles:
        return []
    r = api(action="query", titles="|".join(titles), **II)
    return [p for p in r["query"]["pages"] if not p.get("missing")]


def strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def candidate(page: dict, cat: str) -> tuple[dict | None, str]:
    """-> (record, '') or (None, reject reason)"""
    if not page.get("imageinfo"):
        return None, "no_imageinfo"
    ii = page["imageinfo"][0]
    em = ii.get("extmetadata", {})
    g = lambda k: (em.get(k) or {}).get("value", "")
    title = page["title"]
    if ii.get("mime") not in ("image/jpeg", "image/png"):
        return None, "mime"
    lic = strip_html(g("LicenseShortName"))
    if not OK_LICENSE.match(lic):
        return None, "license"
    if min(ii.get("width", 0), ii.get("height", 0)) < MIN_ORIG_SIDE:
        return None, "too_small"
    cats = g("Categories")
    if BAD_WORDS.search(title) or BAD_WORDS.search(cats.replace("|", " ")):
        return None, "keyword_non_photo"
    return {
        "title": title, "category": cat, "mime": ii["mime"], "width": ii["width"], "height": ii["height"],
        "sha1": ii.get("sha1"), "user": ii.get("user", ""), "thumburl": ii.get("thumburl") or ii["url"],
        "url": ii["url"], "descriptionurl": ii["descriptionurl"], "license": lic,
        "license_url": strip_html(g("LicenseUrl")), "author": strip_html(g("Artist"))[:300] or ii.get("user", ""),
        "credit": strip_html(g("Credit"))[:300],
    }, ""


def cmd_collect(cache: Path, refresh: bool = False) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "candidates.json"
    if meta_path.exists() and not refresh:
        log(f"reusing {meta_path} (pass --refresh to re-query Commons)")
        meta = json.loads(meta_path.read_text())
        have = {r["title"] for r in meta["selected"]} | {r["title"] for r in meta["dropped"]}
        added = 0
        for cls, titles in EXTRA_FILES.items():  # merge extras added to the list after the metadata pass
            for p in titles_info([t for t in titles if t not in have]):
                rec, _ = candidate(p, "full-text search")
                if rec:
                    ext = ".png" if urllib.parse.urlsplit(rec["thumburl"]).path.lower().endswith(".png") else ".jpg"
                    meta["selected"].append({**rec, "label": cls, "file": f"raw/{rec['sha1']}{ext}"})
                    added += 1
        added += extend_with_new_categories(meta)
        if added:
            atomic_write_text(meta_path, json.dumps(meta, indent=1, ensure_ascii=False))
            log(f"merged {added} extra files")
        download(cache, meta["selected"])
        fetch_demo(cache)
        return
    pools: dict[str, dict[str, list[dict]]] = {}
    rejects: dict[str, int] = {}
    for cls, cats in CATEGORIES.items():
        pools[cls] = {}
        for cat in cats:
            recs = []
            for p in category_files(cat):
                rec, why = candidate(p, cat)
                if rec:
                    recs.append(rec)
                else:
                    rejects[why] = rejects.get(why, 0) + 1
            pools[cls][cat] = sorted(recs, key=lambda r: r["title"])
            log(f"{cls:22s} {cat:40s} {len(recs)} usable")
        for t in EXTRA_FILES.get(cls, []):
            for p in titles_info([t]):
                rec, why = candidate(p, "search:" + t)
                if rec:
                    pools[cls].setdefault("full-text search", []).append(rec)
    # files usable by >1 class are ambiguous -> excluded everywhere
    owner: dict[str, set] = {}
    for cls, by in pools.items():
        for recs in by.values():
            for r in recs:
                owner.setdefault(r["title"], set()).add(cls)
    ambiguous = {t for t, s in owner.items() if len(s) > 1}
    selected, dropped = [], []
    for cls, by in pools.items():
        seen, per_user, picked = set(), {}, []
        queues = [list(v) for v in by.values()]
        while len(picked) < N_CANDIDATES and any(queues):
            for q in queues:
                while q:
                    r = q.pop(0)
                    if r["title"] in seen:
                        continue
                    seen.add(r["title"])
                    if r["title"] in ambiguous:
                        dropped.append({**r, "label": cls, "reason": "cross_class_ambiguous",
                                        "detail": "+".join(sorted(owner[r["title"]]))})
                        continue
                    if per_user.get(r["user"], 0) >= MAX_PER_UPLOADER_CLASS.get(cls, MAX_PER_UPLOADER):
                        continue
                    per_user[r["user"]] = per_user.get(r["user"], 0) + 1
                    picked.append({**r, "label": cls})
                    break
                if len(picked) >= N_CANDIDATES:
                    break
        selected += picked
        log(f"{cls:22s} selected {len(picked)} candidates from {len(seen)} usable")
    for r in selected:
        ext = ".png" if urllib.parse.urlsplit(r["thumburl"]).path.lower().endswith(".png") else ".jpg"
        r["file"] = f"raw/{r['sha1']}{ext}"
    atomic_write_text(meta_path, json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "api_rejects": rejects,
        "pool_sizes": {c: {k: len(v) for k, v in by.items()} for c, by in pools.items()},
        "selected": selected, "dropped": dropped}, indent=1, ensure_ascii=False))
    log(f"metadata done: {len(selected)} candidates, {len(dropped)} cross-class drops, rejects={rejects}")
    download(cache, selected)
    fetch_demo(cache)


def extend_with_new_categories(meta: dict) -> int:
    """Categories added to CATEGORIES after the first metadata pass: query them and top classes up to
    N_CANDIDATES with the same filters, uploader cap and cross-class exclusion."""
    sizes = meta.setdefault("pool_sizes", {})
    done = meta.setdefault("extended", {})
    taken = {r["title"]: r["label"] for r in meta["selected"]} | {r["title"]: r["label"] for r in meta["dropped"]}
    added = 0
    for cls, cats in CATEGORIES.items():
        target = N_CANDIDATES_CLASS.get(cls, N_CANDIDATES)
        have_n = sum(r["label"] == cls for r in meta["selected"])
        new = [c for c in cats if c not in sizes.get(cls, {})]
        cap_key = f"{target}/{MAX_PER_UPLOADER_CLASS.get(cls, MAX_PER_UPLOADER)}"
        if not new and (have_n >= target or done.get(cls) in (target, cap_key)):
            continue
        new = new or list(cats)  # below target: re-query every category, already-taken titles are skipped
        pools = []
        for cat in new:
            recs = [rec for rec, _ in (candidate(p, cat) for p in category_files(cat)) if rec]
            sizes.setdefault(cls, {})[cat] = len(recs)
            pools.append(sorted(recs, key=lambda r: r["title"]))
            log(f"{cls:22s} {cat:40s} {len(recs)} usable (extension)")
        mine = [r for r in meta["selected"] if r["label"] == cls]
        per_user: dict[str, int] = {}
        for r in mine:
            per_user[r["user"]] = per_user.get(r["user"], 0) + 1
        n = len(mine)
        while n < target and any(pools):
            for q in pools:
                while q:
                    r = q.pop(0)
                    if r["title"] in taken:
                        if taken[r["title"]] != cls:
                            meta["dropped"].append({**r, "label": cls, "reason": "cross_class_ambiguous",
                                                    "detail": f"{taken[r['title']]}+{cls}"})
                        continue
                    if per_user.get(r["user"], 0) >= MAX_PER_UPLOADER_CLASS.get(cls, MAX_PER_UPLOADER):
                        continue
                    per_user[r["user"]] = per_user.get(r["user"], 0) + 1
                    ext = ".png" if urllib.parse.urlsplit(r["thumburl"]).path.lower().endswith(".png") else ".jpg"
                    meta["selected"].append({**r, "label": cls, "file": f"raw/{r['sha1']}{ext}"})
                    taken[r["title"]] = cls
                    n += 1
                    added += 1
                    break
                if n >= target:
                    break
        done[cls] = cap_key
        log(f"{cls:22s} now {n} candidates")
    return added


def demo_files() -> list[dict]:
    if not DEMO_CREDITS.exists():
        return []
    d = json.loads(DEMO_CREDITS.read_text())
    return [e for k in ("in_scope", "out_of_scope") for e in d.get(k, [])]


def fetch_demo(cache: Path) -> None:
    """Thumbnails of the demo images, used only for a CLIP near-duplicate exclusion."""
    out = cache / "demo"
    out.mkdir(parents=True, exist_ok=True)
    todo = [e for e in demo_files() if not (out / (hashlib.sha1(e["title"].encode()).hexdigest() + ".img")).exists()]
    for i in range(0, len(todo), 20):
        for p in titles_info([e["title"] for e in todo[i:i + 20]]):
            ii = (p.get("imageinfo") or [{}])[0]
            url = ii.get("thumburl") or ii.get("url")
            if url:
                url = re.sub(r"/\d+px-", f"/{THUMB_W}px-", url) if "/thumb/" in url else url
                atomic_write_bytes(out / (hashlib.sha1(p["title"].encode()).hexdigest() + ".img"),
                                   http_get(url, gap=DL_GAP_S))


def download(cache: Path, selected: list[dict]) -> None:
    """Sequential, resumable thumbnail download (files already on disk are skipped)."""
    (cache / "raw").mkdir(exist_ok=True)
    by_cls: dict[str, list[dict]] = {}
    for r in selected:
        by_cls.setdefault(r["label"], []).append(r)
    order = [q[i] for i in range(max(map(len, by_cls.values()))) for q in by_cls.values() if i < len(q)]
    for i, r in enumerate(order):  # interleaved over classes so a partial download stays balanced
        f = cache / r["file"]
        if not f.exists():
            # force the standard bucket even if the API answered with another width
            url = re.sub(r"/\d+px-", f"/{THUMB_W}px-", r["thumburl"]) if "/thumb/" in r["thumburl"] else r["thumburl"]
            atomic_write_bytes(f, http_get(url, gap=DL_GAP_S))
        if i % 50 == 0:
            log(f"  downloaded {i + 1}/{len(order)}")
    log("download done")


# ---- image conversion (same rules as build_dataset_v2.py) --------------------------------------------
def to_rgb_jpeg(data: bytes) -> tuple[bytes, str, list[str]]:
    flags = []
    im = Image.open(io.BytesIO(data))
    im.load()
    fmt = im.format or "UNKNOWN"
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    im = ImageOps.exif_transpose(im)
    if has_alpha:
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        im = bg.convert("RGB")
        flags.append("alpha_composited")
    else:
        im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue(), fmt, flags


def clip_embed(jpegs: list[bytes], batch: int = 16, threads: int = 8) -> np.ndarray:
    import open_clip
    import torch
    torch.set_num_threads(threads)
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_PRETRAINED)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(jpegs), batch):
            x = torch.stack([preprocess(Image.open(io.BytesIO(b)).convert("RGB")) for b in jpegs[s:s + batch]])
            out.append(torch.nn.functional.normalize(model.encode_image(x), dim=-1).numpy().astype(np.float32))
    return np.concatenate(out)


def contact_sheet(tiles: list[tuple[bytes, list[str]]], out: Path, cols: int, tile: int = 170,
                  title: str = "", breaks: list[int] | None = None) -> None:
    font = ImageFont.load_default(size=11)
    cap_h = 13 * max((len(c) for _, c in tiles), default=1) + 4
    pos, r, c, brk = [], 0, 0, set(breaks or [])
    for i in range(len(tiles)):
        if i and (c >= cols or i in brk):
            r, c = r + 1, 0
        pos.append((r, c))
        c += 1
    rows = (pos[-1][0] + 1) if pos else 1
    top = 22 if title else 0
    sheet = Image.new("RGB", (cols * tile, top + rows * (tile + cap_h)), "white")
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((4, 4), title, fill="black", font=ImageFont.load_default(size=14))
    for (b, cap), (r, c) in zip(tiles, pos):
        im = Image.open(io.BytesIO(b)).convert("RGB")
        im.thumbnail((tile - 6, tile - 6))
        x0, y0 = c * tile, top + r * (tile + cap_h)
        sheet.paste(im, (x0 + (tile - im.width) // 2, y0 + 3))
        for k, line in enumerate(cap):
            d.text((x0 + 3, y0 + tile + 13 * k), line[:28], fill="black", font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=88)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def fingerprint(rows: list[dict]) -> str:  # identical definition to data/v2
    lines = sorted(f"{r['id']}\t{r['label']}\t{r['split']}\t{'' if r['fold'] is None else r['fold']}" for r in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def cmd_build(cache: Path, threads: int) -> None:
    meta = json.loads((cache / "candidates.json").read_text())
    classes = json.loads((V2 / "manifest.json").read_text())["classes"]
    assert sorted(classes) == sorted(CATEGORIES), "class list drifted from data/v2"
    drops: list[dict] = []

    def drop(rec, reason, detail="", rid=""):
        drops.append({"id": rid, "label": rec["label"], "reason": reason, "detail": detail,
                      "title": rec["title"], "source_url": rec["descriptionurl"], "category": rec["category"]})

    for d in meta["dropped"]:
        drop(d, d["reason"], d.get("detail", ""))
    # 0. images already shown to the final model in the demo are not unseen any more
    demo = demo_files()
    demo_titles = {e["title"] for e in demo} | {urllib.parse.unquote(e["page"]).rsplit("/", 1)[-1].replace("_", " ")
                                                 for e in demo if e.get("page")}
    demo_sha = {e["sha256"][:16] for e in demo if e.get("sha256")}
    # 1. hash + exact dedup + conversion
    items, by_sha = [], {}
    for rec in meta["selected"]:
        f = cache / rec["file"]
        if not f.exists():
            drop(rec, "not_downloaded", "thumbnail download incomplete")
            continue
        data = f.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if rec["title"] in demo_titles or rec["title"].removeprefix("File:") in demo_titles or sha[:16] in demo_sha:
            drop(rec, "used_in_demo", f"listed in {DEMO_CREDITS.relative_to(REPO)}", sha[:16])
            continue
        if sha in by_sha:
            drop(rec, "exact_duplicate", f"same sha256 as {by_sha[sha]['title']}", sha[:16])
            continue
        if any(it["rec"]["sha1"] == rec["sha1"] for it in items):
            drop(rec, "exact_duplicate", "same Commons original (sha1)", sha[:16])
            continue
        try:
            jpg, fmt, flags = to_rgb_jpeg(data)
        except Exception as e:  # noqa: BLE001
            drop(rec, "undecodable", f"{type(e).__name__}: {e}"[:200], sha[:16])
            continue
        it = {"rec": rec, "sha": sha, "id": sha[:16], "jpg": jpg, "flags": flags}
        by_sha[sha] = rec
        items.append(it)
    log(f"{len(items)} unique images after exact dedup")
    # 2. CLIP leakage vs data/v2 (+ internal near-dups)
    emb_cache = cache / "clip_holdout.npz"
    ids = [it["id"] for it in items]
    if emb_cache.exists() and list(np.load(emb_cache)["ids"]) == ids:
        E = np.load(emb_cache)["E"]
    else:
        log("computing CLIP embeddings ...")
        E = clip_embed([it["jpg"] for it in items], threads=threads)
        np.savez(emb_cache, E=E, ids=np.array(ids))
    demo_jpgs = [to_rgb_jpeg(p.read_bytes())[0] for p in sorted((cache / "demo").glob("*.img"))]
    D = clip_embed(demo_jpgs, threads=threads) if demo_jpgs else np.zeros((0, E.shape[1]), np.float32)
    if len(demo_jpgs) < len(demo):
        log(f"WARNING: only {len(demo_jpgs)}/{len(demo)} demo thumbnails available for the CLIP demo check")
    V = np.load(V2 / "clip_b32_embeddings.npy")
    vids = json.loads((V2 / "clip_b32_ids.json").read_text())["ids"]
    S = E @ V.T
    best = S.argmax(1)
    leak_report = []
    keep = []
    for k, it in enumerate(items):
        it["max_cos_v2"] = float(S[k, best[k]])
        it["nn_v2"] = vids[best[k]]
        dcos = float((E[k] @ D.T).max()) if len(D) else 0.0
        if dcos >= LEAK_COS:
            drop(it["rec"], "used_in_demo", f"CLIP cos={dcos:.3f} to a demo image", it["id"])
        elif it["max_cos_v2"] >= LEAK_COS:
            drop(it["rec"], "leakage_v2", f"cos={it['max_cos_v2']:.3f} to v2 {it['nn_v2']}", it["id"])
            leak_report.append({"id": it["id"], "cos": round(it["max_cos_v2"], 4), "v2_id": it["nn_v2"]})
        else:
            keep.append(k)
    log(f"leakage drops: {len(leak_report)} (cos >= {LEAK_COS})")
    kept_idx = []
    for k in keep:
        dup = next((j for j in kept_idx if float(E[k] @ E[j]) >= INTERNAL_DUP_COS), None)
        if dup is not None:
            drop(items[k]["rec"], "near_duplicate_holdout",
                 f"cos={float(E[k] @ E[dup]):.3f} to {items[dup]['id']}", items[k]["id"])
        else:
            kept_idx.append(k)
    # review sheets (pre-review, everything still in play) for the human visual pass
    rev = cache / "review"
    for cls in classes:
        tiles = [(items[k]["jpg"], [items[k]["id"], items[k]["rec"]["category"][:28]])
                 for k in kept_idx if items[k]["rec"]["label"] == cls]
        if tiles:
            contact_sheet(tiles, rev / f"{cls}.jpg", cols=8, title=f"REVIEW {cls}  n={len(tiles)}")
    # 3. visual-review decisions
    rpath = OUT / "review_decisions.json"
    review = json.loads(rpath.read_text()) if rpath.exists() else {}
    if not review:
        log(f"no review decisions at {rpath}; wrote review sheets to {rev} and stopped")
        return
    final = []
    for k in kept_idx:
        it = items[k]
        verdict = review.get(it["id"])
        if verdict and verdict.startswith("relabel:"):  # reviewer moved it to the class it visibly shows
            new = verdict.split(":", 1)[1]
            assert new in classes, verdict
            it["flags"] = it["flags"] + [f"relabeled_from:{it['rec']['label']}"]
            it["rec"] = {**it["rec"], "label": new}
            final.append(it)
        elif verdict:
            drop(it["rec"], "visual_review", verdict, it["id"])
        else:
            final.append(it)
    # 4. cap per class: prefer uploader diversity, then id order (deterministic)
    rows, kept_items = [], []
    for cls in classes:
        cl = [it for it in final if it["rec"]["label"] == cls]
        counts: dict[str, int] = {}
        order = []
        for it in sorted(cl, key=lambda it: it["id"]):
            u = it["rec"]["user"]
            order.append((counts.get(u, 0), it["id"], it))
            counts[u] = counts.get(u, 0) + 1
        order.sort(key=lambda t: (t[0], t[1]))
        for n, (_, _, it) in enumerate(order):
            if n >= TARGET_PER_CLASS:
                drop(it["rec"], "over_target_cap", f"class already has {TARGET_PER_CLASS}", it["id"])
            else:
                kept_items.append(it)
    for it in kept_items:
        rec = it["rec"]
        rel = f"images/{rec['label']}/{it['id']}.jpg"
        atomic_write_bytes(OUT / rel, it["jpg"])
        rows.append({
            "id": it["id"], "path": rel, "label": rec["label"], "cluster": it["id"], "split": "test", "fold": None,
            "sha256": it["sha"], "src_paths": [rec["descriptionurl"]],
            "orig_format": "JPEG" if rec["mime"] == "image/jpeg" else "PNG",
            "orig_width": rec["width"], "orig_height": rec["height"], "legacy_split": None, "flags": it["flags"],
            "license": rec["license"], "license_url": rec["license_url"], "author": rec["author"],
            "source_url": rec["descriptionurl"], "commons_category": rec["category"],
            "max_clip_cos_v2": round(it["max_cos_v2"], 4),
        })
    rows.sort(key=lambda r: (classes.index(r["label"]), r["id"]))
    counts = {c: sum(r["label"] == c for r in rows) for c in classes}
    short = [c for c, n in counts.items() if n < MIN_PER_CLASS]
    manifest = {
        "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": classes, "images_root": "images", "fingerprint": fingerprint(rows),
        "split_policy": {"n_folds": 5, "test": "all rows (independent web holdout)", "dev": "none",
                         "cluster": "one per row (near-duplicates removed)"},
        "source": "Wikimedia Commons (MediaWiki API, 640-px thumbnails)", "commons_categories": CATEGORIES,
        "qc": {"exact_dedup": "sha256 of downloaded bytes + Commons sha1",
               "leakage": f"open_clip {CLIP_ARCH} {CLIP_PRETRAINED} cosine >= {LEAK_COS} to any data/v2 image",
               "internal_near_dup": f"cosine >= {INTERNAL_DUP_COS}", "visual_review": "manual, see dropped.csv",
               "leakage_drops": leak_report},
        "counts": counts, "classes_below_min": short,
        "usage": "FINAL evaluation / demo only. Never train, tune, calibrate or select thresholds on this set.",
        "rows": rows,
    }
    atomic_write_text(OUT / "manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["id", "label", "reason", "detail", "title", "source_url", "category"])
    w.writeheader()
    w.writerows(drops)
    atomic_write_text(OUT / "dropped.csv", buf.getvalue())
    md = [f"# Attribution — {VERSION}", "",
          "All images come from Wikimedia Commons and are redistributed under their original licences "
          "(public domain or Creative Commons). Images were downscaled (longest side <= 512 px) and re-encoded "
          "as JPEG; no other changes were made. Author / licence / source for every image:", "",
          "| id | class | author | licence | source |", "|---|---|---|---|---|"]
    for r in rows:
        lic = f"[{r['license']}]({r['license_url']})" if r["license_url"] else r["license"]
        author = r["author"].replace("|", "/") or "unknown"
        title = urllib.parse.unquote(r["source_url"].rsplit("/", 1)[-1]).replace("|", "/")
        md.append(f"| {r['id']} | {r['label']} | {author} | {lic} | [{title}]({r['source_url']}) |")
    atomic_write_text(OUT / "ATTRIBUTION.md", "\n".join(md) + "\n")
    tiles, breaks = [], []
    jmap = {it["id"]: it["jpg"] for it in kept_items}
    for cls in classes:
        breaks.append(len(tiles))
        tiles += [(jmap[r["id"]], [r["label"], r["id"]]) for r in rows if r["label"] == cls]
    contact_sheet(tiles, OUT / "contact_sheet.jpg", cols=TARGET_PER_CLASS, tile=120,
                  title=f"{VERSION}: {len(rows)} images, 20 classes (one class per row)", breaks=breaks[1:])
    log(f"OK {len(rows)} rows, fingerprint {manifest['fingerprint']}, counts {counts}, below-min {short}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["collect", "build"])
    ap.add_argument("--cache", type=Path,
                    default=Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "compclass" / VERSION)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--refresh", action="store_true", help="collect: re-query Commons even if metadata is cached")
    a = ap.parse_args()
    if a.cmd == "collect":
        cmd_collect(a.cache, a.refresh)
    else:
        cmd_build(a.cache, a.threads)


if __name__ == "__main__":
    sys.exit(main())
