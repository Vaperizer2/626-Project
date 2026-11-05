"""Dev features of a backbone under another input view, cached like compclass.features.extract_features.

    uv run python tools/views/extract_views.py siglip_so400m_384 crop100 squash
    uv run python tools/views/extract_views.py pe_core_l14_336 squash
    uv run python tools/views/extract_views.py siglip_so400m_384@308 crop100   # input size override, cache tag crop100@308

crop100: resize the short side to input_size, centre crop input_size (crop fraction 1.0; PE-Core's default).
squash:  the whole picture resized to input_size x input_size (aug.squash).
Cache: results/cache/features/<key>-<fingerprint>-<idshash>-<view>.npy, same row order as the default-view cache.
"""
import hashlib, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "src")
from compclass.data import ImageDataset, build_transforms, get_dev, load_manifest
from compclass.models import build_model

key, _, px = sys.argv[1].partition("@")  # key@px overrides the input size (e.g. siglip_so400m_384@308, the Fast trunk)
views = sys.argv[2:]
kw = {"input_size": int(px)} if px else {}
m = load_manifest("data/v2/manifest.json")
rows = get_dev(m)
ids_hash = hashlib.sha256("".join(r["id"] for r in rows).encode()).hexdigest()[:12]
device = torch.device("cuda")
for view in views:
    cache = Path("results/cache/features") / f"{key}-{m.fingerprint}-{ids_hash}-{view}{'@' + px if px else ''}.npy"
    if cache.exists():
        print("cached:", cache); continue
    t0 = time.perf_counter()
    if view == "crop100":
        model, spec = build_model(key, num_classes=0, pretrained=True, crop_pct=1.0, **kw)
        tf = build_transforms(spec, None, train=False)
    elif view == "squash":
        model, spec = build_model(key, num_classes=0, pretrained=True, **kw)
        tf = build_transforms(spec, {"squash": True}, train=False)
    else:
        raise SystemExit(f"unknown view {view}")
    print(f"{key} {view}: input {spec.input_size}, crop_pct {spec.crop_pct}, transform: {tf}", flush=True)
    model.eval().to(device)
    dl = DataLoader(ImageDataset(rows, m.root, m.class_to_idx, tf), batch_size=32, num_workers=0)  # in-process decode: 2,512 images cost ~15 s of CPU; avoids Windows spawn
    feats = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i, (x, _) in enumerate(dl):
            feats.append(model(x.to(device)).float().cpu())
            if i % 20 == 0:
                print(f"  batch {i}/{len(dl)} {time.perf_counter()-t0:.0f}s", flush=True)
    out = torch.cat(feats).numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    print(f"saved {cache} {out.shape} in {time.perf_counter()-t0:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()
