"""Cache the deployed teacher's pooled embedding (SigLIP So400m-384, exactly the deployed trunk) for every pool view.

    uv run python tools/distill/teacher.py --pool POOL --out TEACHER [--views-component 8 --views-generic 3]

Writes TEACHER/index.npz (pool row and view number per embedding) and TEACHER/shard_XXX.npy (float16 [n, 1152]),
one shard per --shard views, so an interrupted run resumes at the first missing shard. Views are ordered view-major,
so any prefix of shards is usable for training. Pool images only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TEACHER_KEY, load_pool, render_view, to_tensor  # noqa: E402


class Views(Dataset):
    def __init__(self, pool_dir: Path, rows: list[dict], items: np.ndarray, size: int, mean, std):
        self.dir, self.rows, self.items, self.size, self.mean, self.std = pool_dir, rows, items, size, mean, std

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        r, v = self.items[i]
        row = self.rows[r]
        im = Image.open(self.dir / row["path"]).convert("RGB")
        return to_tensor(render_view(im, row["id"], int(v), self.size), self.mean, self.std)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views-component", type=int, default=8)
    ap.add_argument("--views-generic", type=int, default=3)
    ap.add_argument("--shard", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)

    from compclass.models import build_model

    pool_dir, out = Path(a.pool), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, _ = load_pool(pool_dir)
    nv = {"component": a.views_component, "generic": a.views_generic}
    # view-major order: a prefix of the shards holds whole views (0, 1, ...) of every image
    items = np.array([(k, v) for v in range(max(nv.values())) for k, r in enumerate(rows) if v < nv[r["domain"]]],
                     np.int64)
    ids = np.array([r["id"] for r in rows])
    idx_path = out / "index.npz"
    if idx_path.exists():
        old = np.load(idx_path)
        if not (np.array_equal(old["items"], items) and np.array_equal(old["ids"], ids)):
            raise SystemExit(f"{idx_path} was built for a different pool or view plan; use a new --out")
    else:
        np.savez(idx_path, items=items, ids=ids)

    dev = torch.device("cuda")
    model, spec = build_model(TEACHER_KEY, num_classes=0, pretrained=True)
    model.eval().to(dev)
    n_shards = (len(items) + a.shard - 1) // a.shard
    print(f"{len(rows)} pool images, {len(items)} views, {n_shards} shards", flush=True)
    t0, done = time.time(), 0
    for s in range(n_shards):
        path = out / f"shard_{s:03d}.npy"
        if path.exists():
            continue
        part = items[s * a.shard:(s + 1) * a.shard]
        dl = DataLoader(Views(pool_dir, rows, part, spec.input_size, spec.mean, spec.std), batch_size=a.batch,
                        num_workers=a.workers, pin_memory=True)
        feats = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for x in dl:
                feats.append(model(x.to(dev, non_blocking=True)).float().cpu())
        F = torch.cat(feats).numpy()
        assert np.isfinite(F).all(), f"non-finite teacher embeddings in shard {s}"
        np.save(path.with_suffix(".tmp.npy"), F.astype(np.float16))
        path.with_suffix(".tmp.npy").rename(path)
        done += len(part)
        rate = done / (time.time() - t0)
        print(f"shard {s + 1}/{n_shards}: {rate:.1f} views/s, eta {(len(items) - (s + 1) * a.shard) / rate / 60:.0f} min",
              flush=True)
    (out / "done.json").write_text(json.dumps({"n_views": int(len(items)), "n_images": len(rows),
                                               "teacher": TEACHER_KEY, "views": nv}))


if __name__ == "__main__":
    main()
