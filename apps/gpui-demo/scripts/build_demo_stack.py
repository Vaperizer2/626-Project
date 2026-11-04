# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""Build apps/gpui-demo/demo_stack/: the images the GUI's Demo and Benchmark tabs click through.

No image was used for training or model selection: all but two come from the independent Wikimedia
Commons web holdout (data/web_holdout_v1, evaluated once), the two out-of-scope photos are the credited demo
images in results/demo/wikimedia_credits.json (downloaded from Commons, sha256-checked).

Each class has three pictures: one hand-picked (PICKS) and two drawn at random with RANDOM_SEED from the class's
holdout pictures not already in the stack, so the Random section is an unselected sample.

    uv run --no-config apps/gpui-demo/scripts/build_demo_stack.py [--bin target/release/gpui-demo]

Writes demo_stack/*.jpg (long side <= 1024 px; holdout files are copied byte-for-byte, they are <= 512 px),
demo_stack/demo_stack.json and demo_stack/ATTRIBUTION.md. With --bin, every image is classified by the built
binary (`classify --json --model deployed --ep cpu`) and model_pred / model_conf / model_unsure record what it said.
Without --bin, a photo keeps the answer recorded for it when its origin and sha256 are unchanged; rerun with --bin
after a model change.
The out-of-scope photos are downloaded only when neither --cache nor an earlier build's file has them. The script
never deletes anything; it overwrites only the files it writes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
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

ONE = "Hand-picked, one per class"
RANDOM = "Random, two per class"
HARD = "Hard cases"
ABSTAIN = "Not a component / abstains"

RANDOM_SEED = 20261103
RANDOM_PER_CLASS = 2

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


def page_url(url: str) -> str:
    """The Commons file page. Some holdout rows recorded an archived revision's name, File:<14 digits>!Name."""
    return re.sub(r"File:\d{14}(?:!|%21)", "File:", url)


def changes_of(row: dict) -> str:
    """What was done to a holdout photo besides the thumbnail conversion; CC BY-SA asks for changes to be indicated."""
    out = []
    if m := re.search(r"File:(\d{4})(\d\d)(\d\d)(\d\d)(\d\d)(\d\d)(?:!|%21)", row["source_url"]):
        y, mo, d, h, mi, s = m.groups()
        out.append(f"the version replaced on {y}-{mo}-{d} {h}:{mi}:{s} UTC (see the file history)")
    if "alpha_composited" in row["flags"]:
        out.append("transparent background filled with white")
    if row["orig_format"].upper() != "JPEG":
        out.append(f"converted from {row['orig_format'].upper()} to JPEG")
    return "; ".join(out)


def title_from_url(url: str) -> str:
    name = url.rsplit("File:", 1)[-1]
    name = urllib.request.unquote(name)
    return re.sub(r"\.(jpe?g|png|gif|webp|tiff?)$", "", name, flags=re.I).replace("_", " ")


def random_picks(manifest: dict) -> list[str]:
    """RANDOM_PER_CLASS holdout ids per class, in class order, drawn with RANDOM_SEED from those not in PICKS."""
    taken = {rid for _, rid, _ in PICKS}
    rng = random.Random(RANDOM_SEED)
    out = []
    for label in manifest["classes"]:
        pool = sorted(r["id"] for r in manifest["rows"] if r["label"] == label and r["id"] not in taken)
        out += rng.sample(pool, RANDOM_PER_CLASS)
    return out


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
    # without --bin, each photo keeps the answer recorded for it (matched by origin, not by file name)
    previous = {e["origin"]: e for e in json.loads((OUT / "demo_stack.json").read_text())["images"]} \
        if (OUT / "demo_stack.json").is_file() else {}
    rows = {r["id"]: r for r in manifest["rows"]}
    credits = {c["file"]: c for c in json.loads(CREDITS.read_text())["out_of_scope"]}
    entries = []
    counters: dict[str, int] = {}
    prefix = {ONE: "", RANDOM: "r", HARD: "hard", ABSTAIN: "x"}
    one = [p for p in PICKS if p[0] == ONE]
    picks = one + [(RANDOM, rid, "") for rid in random_picks(manifest)] + PICKS[len(one):]
    for section, rid, note in picks:
        r = rows[rid]
        counters[section] = counters.get(section, 0) + 1
        n = counters[section]
        num = f"{n:02d}" if section in (ONE, RANDOM) else str(n)
        name = f"{prefix[section]}{num}_{slug(r['label'])}.jpg"
        save_jpeg((HOLDOUT / r["path"]).read_bytes(), OUT / name)
        entries.append({
            "file": name, "section": section, "true_label": r["label"],
            "title": title_from_url(page_url(r["source_url"])), "author": r["author"], "licence": r["license"],
            "licence_url": r["license_url"], "source_url": page_url(r["source_url"]),
            "origin": f"web_holdout_v1:{rid}", "note": note, "changes": changes_of(r),
        })
    for section, file, note in OUT_OF_SCOPE:
        c = credits[file]
        counters[section] = counters.get(section, 0) + 1
        name = f"x{counters[section]}_{slug(file.removeprefix('oos_').removesuffix('.jpg'))}.jpg"
        cached = args.cache / file if args.cache else None
        if cached and cached.is_file() or not (OUT / name).is_file():
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
            "changes": "; ".join(x for x in (
                f"the {c['size'][0]}-px Commons thumbnail" if "/thumb/" in c["url"] else "",
                f"downscaled to {MAX_SIDE} px on the long side" if max(c["size"]) > MAX_SIDE else "") if x),
        })

    if args.bin:
        files = [str(OUT / e["file"]) for e in entries]
        cmd = [str(args.bin), "classify", "--json", "--model", "deployed", "--ep", "cpu", *files]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = {Path(r["source"]).name: r for r in json.loads(res.stdout)}
        for e in entries:
            r = out[e["file"]]
            e["model_pred"] = r["top3"][0][0]
            e["model_conf"] = round(r["top3"][0][1], 4)
            e["model_unsure"] = r["unsure"]
    for e in entries:
        e["sha256"] = hashlib.sha256((OUT / e["file"]).read_bytes()).hexdigest()[:16]
    if not args.bin:
        for e in entries:
            old = previous.get(e["origin"], {})
            if old.get("sha256") == e["sha256"]:
                e.update({k: old[k] for k in ("model_pred", "model_conf", "model_unsure") if k in old})
            else:
                print(f"{e['file']}: no recorded answer for this photo; rerun with --bin", file=sys.stderr)
    for e in entries:
        im = Image.open(OUT / e["file"])
        e["width"], e["height"] = im.size

    doc = {
        "version": 1,
        "description": "Demo images for apps/gpui-demo (Demo and Benchmark tabs). None was used for training or model "
                       "selection. The Random section is drawn from the web holdout with random_seed by "
                       "scripts/build_demo_stack.py. model_pred/model_conf: deployed model (exports/final) as verified "
                       "by `gpui-demo classify --json --model deployed --ep cpu`; true_label null = not one of the 20 "
                       "classes (expected: UNSURE).",
        "random_seed": RANDOM_SEED,
        "sections": [ONE, RANDOM, HARD, ABSTAIN],
        "images": entries,
    }
    (OUT / "demo_stack.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")

    lines = [
        "# Attribution — demo_stack", "",
        "Photos from Wikimedia Commons, redistributed under their original licences. Holdout photos are Commons",
        "downloads (a thumbnail up to 640 px wide, or the original when it is smaller), saved as JPEG (quality 95)",
        "with the long side at most 512 px (data/web_holdout_v1). The out-of-scope photos are Commons files or",
        "thumbnails. Any other change to a photo is in its changes column.", "",
        "| file | section | true label | title | author | licence | source | changes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        lic = f"[{e['licence']}]({e['licence_url']})" if e["licence_url"] else e["licence"]
        lines.append(f"| {e['file']} | {e['section']} | {e['true_label'] or '(not a component)'} | {e['title']} | "
                     f"{e['author']} | {lic} | [Commons]({e['source_url']}) | {e['changes']} |")
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
