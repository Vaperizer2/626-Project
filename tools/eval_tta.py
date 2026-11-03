"""Score saved scratch-CNN checkpoints on their own validation fold with and without flip test-time augmentation.

Test-time augmentation (TTA) changes nothing about training: the saved network is run on the image and on its
mirror image and the two probability vectors are averaged. It is therefore evaluated on existing checkpoints,
which keeps the comparison exact (same weights, same fold, same images).

    uv run python tools/eval_tta.py runs/win/scratch_cnn-b0afa2aa4d60/fold*/best.pt [--out results/tta_win.jsonl]

Development folds only: the checkpoint stores its fold and the sealed test split is never touched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from compclass.data import ImageDataset, build_transforms, get_fold, load_manifest
from compclass.evaluate import compute_metrics
from compclass.models import build_model
from compclass.models.registry import PreprocessSpec


@torch.no_grad()
def predict_views(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Return (probs_plain, probs_flip, y): softmax of the image and of its horizontal mirror."""
    model.eval()
    plain, flip, ys = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p0 = model(x).float().softmax(1)
            p1 = model(torch.flip(x, dims=[3])).float().softmax(1)
        plain.append(p0.cpu()), flip.append(p1.cpu()), ys.append(y)
    return torch.cat(plain).numpy(), torch.cat(flip).numpy(), torch.cat(ys).numpy()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="best.pt paths or globs")
    ap.add_argument("--out", help="append one JSON record per checkpoint to this .jsonl")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args(argv)
    paths = sorted(p for pat in args.checkpoints for p in glob.glob(pat))
    if not paths:
        raise SystemExit("no checkpoints matched")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'checkpoint':58s} {'fold':>4s} {'plain F1':>9s} {'flip-TTA F1':>12s} {'diff':>7s}  {'plain acc':>9s} {'TTA acc':>8s}")
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        m = load_manifest(cfg["data"]["manifest"])
        if m.fingerprint != ck["manifest_fingerprint"]:
            raise SystemExit(f"{path}: trained on manifest {ck['manifest_fingerprint']}, not {m.fingerprint}")
        rows = get_fold(m, ck["fold"])[1]
        spec = PreprocessSpec(**{k: v for k, v in ck["preprocess"].items() if k != "resize_short_side"})
        model, _ = build_model(cfg["model"]["key"], len(m.classes), pretrained=False, **cfg["model"].get("kwargs", {}))
        model.load_state_dict(ck["model"])
        loader = DataLoader(ImageDataset(rows, m.root, m.class_to_idx, build_transforms(spec, cfg.get("aug"), train=False)),
                            batch_size=args.batch_size, num_workers=cfg["data"].get("num_workers", 4))
        p0, p1, y = predict_views(model.to(device), loader, device)
        base = compute_metrics(y, p0.argmax(1), m.classes)
        tta = compute_metrics(y, (0.5 * (p0 + p1)).argmax(1), m.classes)
        print(f"{path:58s} {ck['fold']:4d} {base['macro_f1']:9.4f} {tta['macro_f1']:12.4f} "
              f"{tta['macro_f1'] - base['macro_f1']:+7.4f}  {base['accuracy']:9.4f} {tta['accuracy']:8.4f}", flush=True)
        if args.out:
            rec = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "method": "tta_eval",
                   "name": cfg["name"], "model": cfg["model"]["key"], "config_hash": ck["config_hash"],
                   "fold": ck["fold"], "checkpoint": str(path), "manifest": m.fingerprint, "views": "plain+hflip",
                   "plain": {"accuracy": base["accuracy"], "macro_f1": base["macro_f1"]},
                   "tta": {"accuracy": tta["accuracy"], "macro_f1": tta["macro_f1"]}}
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
