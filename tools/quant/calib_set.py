"""Pick calibration and sensitivity images from the distillation pool and copy them from carlpc2.

The pool (artifacts/distill/pool/pool.json) was gated against every data/v2 image and the web holdout by raw hash,
CLIP-B/32 cosine >= 0.85 under flips/rotations, and pHash, so no development-fold image or near-duplicate of one
reaches calibration. Only component-domain photos with a short side >= 224 px are used, balanced across sources.

    calib_set.py <out_dir> [--calib 512] [--sens 256] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

POOL = Path("/home/vaper/Work/Prototypes/626-artifacts/distill/pool/pool.json")
REMOTE = "carlpc2:/home/carl/626-runs/distill/pool/"


def balanced(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    for v in by_src.values():
        rng.shuffle(v)
    out: list[dict] = []
    while len(out) < n and any(by_src.values()):
        for src in sorted(by_src):
            if by_src[src] and len(out) < n:
                out.append(by_src[src].pop())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--calib", type=int, default=512)
    ap.add_argument("--sens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [r for r in json.loads(POOL.read_text())["rows"]
            if r["domain"] == "component" and min(r["w"], r["h"]) >= 224]
    rows.sort(key=lambda r: r["id"])
    rng = random.Random(a.seed)
    picked = balanced(rows, a.calib + a.sens, rng)
    rng.shuffle(picked)
    split = {"calib": picked[:a.calib], "sens": picked[a.calib:]}
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(r["path"] for r in picked)
    (a.out / "files.txt").write_text("\n".join(files) + "\n")
    subprocess.run(["rsync", "-a", "--files-from", str(a.out / "files.txt"), REMOTE, str(a.out)], check=True)
    meta = {"pool": str(POOL), "filter": "domain=component, min(w,h)>=224, balanced by source", "seed": a.seed,
            **{k: [{"id": r["id"], "path": r["path"], "source": r["source"], "max_cos_ref": r["max_cos_ref"]}
                   for r in v] for k, v in split.items()}}
    (a.out / "split.json").write_text(json.dumps(meta, indent=1))
    for k, v in split.items():
        src = defaultdict(int)
        for r in v:
            src[r["source"]] += 1
        print(k, len(v), dict(src), "max_cos_ref max", max(r["max_cos_ref"] for r in v))


if __name__ == "__main__":
    main()
