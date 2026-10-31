"""Write a pool listing of the development images outside fold K (the images fold K's head trains on).

    uv run python tools/distill/fold_pool.py --manifest data/v2/manifest.json --fold 0 --out DIR

The per-fold diagnostic distils on these images label-free, so its student never sees an image of fold K, and only
fold K's accuracy is reported for it. Test-split rows are never listed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from compclass.data import get_dev, load_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = load_manifest(a.manifest)
    rows = [{"id": r["id"], "path": str(Path(m.root) / r["path"]), "source": f"v2_dev_not_fold{a.fold}",
             "domain": "component"} for r in get_dev(m) if r["fold"] != a.fold]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pool.json").write_text(json.dumps({"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                               "manifest": m.fingerprint, "excluded_fold": a.fold,
                                               "n_kept": len(rows), "rows": rows}, indent=1))
    print(f"{len(rows)} dev rows outside fold {a.fold} -> {out / 'pool.json'}")


if __name__ == "__main__":
    main()
