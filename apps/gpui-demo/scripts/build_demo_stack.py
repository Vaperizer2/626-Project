# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""Build apps/gpui-demo/demo_stack/: the curated images the GUI's Demo and Benchmark tabs click through.

Every image is NOT used for training or model selection: all but two come from the independent Wikimedia
Commons web holdout (data/web_holdout_v1, evaluated once), the two out-of-scope photos are the credited demo
images in results/demo/wikimedia_credits.json (downloaded from Commons, sha256-checked).

    uv run --no-config apps/gpui-demo/scripts/build_demo_stack.py [--bin apps/gpui-demo/target/release/gpui-demo]

Writes demo_stack/*.jpg (long side <= 1024 px; holdout files are copied byte-for-byte, they are <= 512 px),
demo_stack/demo_stack.json and demo_stack/ATTRIBUTION.md. With --bin, every image is classified by the built
binary (`classify --json`) and model_pred / model_conf / model_unsure record what it said. The script never
deletes anything; it overwrites only the files it writes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from PIL import Image

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
HOLDOUT = REPO / "data" / "web_holdout_v1"
CREDITS = REPO / "results" / "demo" / "wikimedia_credits.json"
OUT = APP / "demo_stack"
UA = "compclass-gpui-demo/0.1.0 (educational image-classification demo; build_demo_stack.py)"
MAX_SIDE = 1024

ONE = "One per class"
HARD = "Hard cases"
ABSTAIN = "Not a component / abstains"

# (section, holdout id, note). Chosen by looking at the photos (clean, recognisable, varied) among the deployed
# model's confidently-correct holdout predictions (results/final/final_test_web_holdout_v1_preds.npz).
PICKS: list[tuple[str, str, str]] = [
    (ONE, "f704209d40002ca7", ""),
    (ONE, "f83b9a9730c1d29e", ""),
    (ONE, "b4066c951b115f31", ""),
    (ONE, "e65ca8ca34e12511", "D-sub connectors"),
    (ONE, "4feb0638c1377367", "HC-49 quartz crystal"),
    (ONE, "44a1e8650ff7798a", ""),
    (ONE, "b9d27819984f367d", "automotive blade fuse"),
    (ONE, "dcd25c2687b463db", ""),
    (ONE, "8dc4b1db7d8229da", "DIP and SOIC packages"),
    (ONE, "15ceadb5840c0d21", "toroidal inductor"),
    (ONE, "b73c8691546e4bc4", ""),
    (ONE, "8f6d90957f152afb", "LED is the model's weakest class; 76% is typical"),
    (ONE, "a94c9d1f4dae72fb", "Arduino Mega"),
    (ONE, "40b83912b761ac25", ""),
    (ONE, "3b0ca7d7555deebb", "AC-DC power supply boards"),
    (ONE, "0dc5b8351c8265a7", ""),
    (ONE, "56beaae9d2b431dc", ""),
    (ONE, "a91ad29ca4d7781a", "slide switch"),
    (ONE, "cdea86d2efbc5977", ""),
    (ONE, "218e4e950d31a4ad", "TO-220 transistor"),
    (HARD, "837b0dee2c063da0", "look-alikes: coloured LEDs read as diodes (same leads, similar body)"),
    (HARD, "1b17145a43f1660c", "look-alikes: a small toroidal transformer is wound exactly like a toroidal inductor"),
    (HARD, "0acee82d84501217", "DIP switches share the dual-in-line package of IC chips"),
    (HARD, "ac5ab40b383482fe", "unusual view: SMD power inductors are black blocks like SMD diodes"),
    (HARD, "931547d1cca4664b", "cluttered scene: Arduino wired to a breadboard (two classes in one picture)"),
    (ABSTAIN, "5c15a03b8b4a2dc4", "in scope but ambiguous: a coin-cell holder on a laptop motherboard"),
]
# (section, file in wikimedia_credits.json "out_of_scope", note)
OUT_OF_SCOPE = [
    (ABSTAIN, "oos_multimeter.jpg", "not a component: multimeter (none of the 20 classes)"),
    (ABSTAIN, "oos_soldering_iron.jpg", "not a component: soldering iron. Borderline: at 1024 px the top class "
                                        "(jumper wire) lands just above the 50% threshold, so it is not flagged; "
                                        "a confidence threshold is not a full out-of-distribution detector"),
]
# Full author of the derivative work (credits.json kept only the first line of the Commons "Artist" field).
AUTHOR_FIX = {"oos_soldering_iron.jpg": "oomlout (Soldering_iron_and_accessories.jpg); derivative work: Bomazi"}
LICENCE_URLS = {
    "CC BY-SA 2.5": "https://creativecommons.org/licenses/by-sa/2.5",
    "CC BY-SA 2.0": "https://creativecommons.org/licenses/by-sa/2.0",
}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def title_from_url(url: str) -> str:
    name = url.rsplit("File:", 1)[-1]
    name = urllib.request.unquote(name)
    return re.sub(r"\.(jpe?g|png|gif|webp|tiff?)$", "", name, flags=re.I).replace("_", " ")


def save_jpeg(data: bytes, dst: Path) -> None:
    im = Image.open(io.BytesIO(data))
    if max(im.size) <= MAX_SIDE and im.format == "JPEG":
        dst.write_bytes(data)  # byte-for-byte: no second JPEG generation
        return
    im = im.convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    im.save(dst, "JPEG", quality=90, optimize=True)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, help="gpui-demo binary used to verify the predictions")
    ap.add_argument("--cache", type=Path, default=None, help="folder with already downloaded oos_*.jpg")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((HOLDOUT / "manifest.json").read_text())
    rows = {r["id"]: r for r in manifest["rows"]}
    credits = {c["file"]: c for c in json.loads(CREDITS.read_text())["out_of_scope"]}
    entries = []
    counters: dict[str, int] = {}
    prefix = {ONE: "", HARD: "hard", ABSTAIN: "x"}
    for section, rid, note in PICKS:
        r = rows[rid]
        counters[section] = counters.get(section, 0) + 1
        n = counters[section]
        name = f"{n:02d}_{slug(r['label'])}.jpg" if section == ONE else f"{prefix[section]}{n}_{slug(r['label'])}.jpg"
        save_jpeg((HOLDOUT / r["path"]).read_bytes(), OUT / name)
        entries.append({
            "file": name, "section": section, "true_label": r["label"], "title": title_from_url(r["source_url"]),
            "author": r["author"], "licence": r["license"], "licence_url": r["license_url"],
            "source_url": r["source_url"], "origin": f"web_holdout_v1:{rid}", "note": note,
        })
    for section, file, note in OUT_OF_SCOPE:
        c = credits[file]
        counters[section] = counters.get(section, 0) + 1
        name = f"x{counters[section]}_{slug(file.removeprefix('oos_').removesuffix('.jpg'))}.jpg"
        cached = args.cache / file if args.cache else None
        data = cached.read_bytes() if cached and cached.is_file() else fetch(c["url"])
        digest = hashlib.sha256(data).hexdigest()[:16]
        if digest != c["sha256"]:
            sys.exit(f"{file}: sha256 {digest} != {c['sha256']} in {CREDITS}")
        save_jpeg(data, OUT / name)
        entries.append({
            "file": name, "section": section, "true_label": None, "title": c["title"].removeprefix("File:").rsplit(".", 1)[0],
            "author": AUTHOR_FIX.get(file, c["author"]), "licence": c["license"],
            "licence_url": LICENCE_URLS.get(c["license"], ""), "source_url": c["page"],
            "origin": f"results/demo/wikimedia_credits.json:{file}", "note": note,
        })

    if args.bin:
        files = [str(OUT / e["file"]) for e in entries]
        res = subprocess.run([str(args.bin), "classify", "--json", *files], capture_output=True, text=True, check=True)
        out = {Path(r["source"]).name: r for r in json.loads(res.stdout)}
        for e in entries:
            r = out[e["file"]]
            e["model_pred"] = r["top3"][0][0]
            e["model_conf"] = round(r["top3"][0][1], 4)
            e["model_unsure"] = r["unsure"]
    for e in entries:
        im = Image.open(OUT / e["file"])
        e["width"], e["height"] = im.size

    doc = {
        "version": 1,
        "description": "Curated demo images for apps/gpui-demo (Demo and Benchmark tabs). None was used for training "
                       "or model selection. model_pred/model_conf: deployed model (exports/final) as verified by "
                       "`gpui-demo classify --json`; true_label null = not one of the 20 classes (expected: UNSURE).",
        "sections": [ONE, HARD, ABSTAIN],
        "images": entries,
    }
    (OUT / "demo_stack.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")

    lines = [
        "# Attribution — demo_stack", "",
        "Photos from Wikimedia Commons, redistributed under their original licences. Holdout photos are the",
        "640-px Commons thumbnails downscaled to <= 512 px (data/web_holdout_v1); the two out-of-scope photos are",
        "downscaled to <= 1024 px. No other changes were made.", "",
        "| file | section | true label | title | author | licence | source |", "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        lic = f"[{e['licence']}]({e['licence_url']})" if e["licence_url"] else e["licence"]
        lines.append(f"| {e['file']} | {e['section']} | {e['true_label'] or '(not a component)'} | {e['title']} | "
                     f"{e['author']} | {lic} | [Commons]({e['source_url']}) |")
    (OUT / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")
    size = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"{len(entries)} images, {size / 1e6:.2f} MB in {OUT}")
    if args.bin:
        for e in entries:
            ok = (e["model_unsure"] if e["true_label"] is None else (not e["model_unsure"] and e["model_pred"] == e["true_label"]))
            print(f"{e['file']:32s} {str(e['true_label']):22s} {e['model_pred']:22s} {e['model_conf']:.3f} {'OK' if ok else '--'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
