"""Per-class precision/recall/F1 and the most frequent confusions, pooled over saved checkpoints.

Each checkpoint is scored on its own development fold (the fold is stored in the checkpoint), so pooling the
checkpoints of one recipe over folds 0-4 gives out-of-fold predictions for the whole development set. The sealed
test split is never touched.

    uv run python tools/eval_per_class.py "runs/win/scratch_cnn-b0afa2aa4d60/fold*/best.pt" [--top 15] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from compclass.data import ImageDataset, build_transforms, get_fold, load_manifest
from compclass.models import build_model
from compclass.models.registry import PreprocessSpec


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            preds.append(model(x.to(device, non_blocking=True)).float().argmax(1).cpu())
        ys.append(y)
    return torch.cat(preds).numpy(), torch.cat(ys).numpy()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="best.pt paths or globs")
    ap.add_argument("--top", type=int, default=15, help="number of confusion pairs to print")
    ap.add_argument("--csv", help="write the per-class table here")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args(argv)
    paths = sorted(p for pat in args.checkpoints for p in glob.glob(pat))
    if not paths:
        raise SystemExit("no checkpoints matched")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_p, all_y, folds = [], [], []
    classes = None
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        m = load_manifest(cfg["data"]["manifest"])
        if m.fingerprint != ck["manifest_fingerprint"]:
            raise SystemExit(f"{path}: trained on manifest {ck['manifest_fingerprint']}, not {m.fingerprint}")
        classes = m.classes
        rows = get_fold(m, ck["fold"])[1]
        spec = PreprocessSpec(**{k: v for k, v in ck["preprocess"].items() if k != "resize_short_side"})
        model, _ = build_model(cfg["model"]["key"], len(m.classes), pretrained=False, **cfg["model"].get("kwargs", {}))
        model.load_state_dict(ck["model"])
        loader = DataLoader(ImageDataset(rows, m.root, m.class_to_idx, build_transforms(spec, cfg.get("aug"), train=False)),
                            batch_size=args.batch_size, num_workers=cfg["data"].get("num_workers", 4))
        p, y = predict(model.to(device), loader, device)
        all_p.append(p), all_y.append(y), folds.append(ck["fold"])
        print(f"{path}: fold {ck['fold']}, {len(y)} images, acc {np.mean(p == y):.4f}", flush=True)
    p, y = np.concatenate(all_p), np.concatenate(all_y)
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    rec = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    prec = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    print(f"\nPooled over folds {sorted(folds)}: {len(y)} images, accuracy {np.mean(p == y):.4f}, "
          f"macro-F1 {f1.mean():.4f}\n")
    print(f"{'class':24s} {'n':>4s} {'prec':>6s} {'rec':>6s} {'F1':>6s}  most confused with (count)")
    order = np.argsort(f1)
    table = []
    for i in order:
        off = [(cm[i, j], classes[j]) for j in range(k) if j != i and cm[i, j] > 0]
        off.sort(reverse=True)
        conf = ", ".join(f"{c} ({n})" for n, c in off[:3])
        print(f"{classes[i]:24s} {support[i]:4d} {prec[i]:6.3f} {rec[i]:6.3f} {f1[i]:6.3f}  {conf}")
        table.append({"class": classes[i], "support": int(support[i]), "precision": round(float(prec[i]), 4),
                      "recall": round(float(rec[i]), 4), "f1": round(float(f1[i]), 4), "confused_with": conf})
    pairs = Counter({(classes[i], classes[j]): int(cm[i, j]) for i in range(k) for j in range(k) if i != j})
    print(f"\nTop {args.top} confusions (true -> predicted): count, share of the true class")
    for (a, b), n in pairs.most_common(args.top):
        print(f"  {a:22s} -> {b:22s} {n:3d}  {n / support[classes.index(a)]:.0%}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader(), w.writerows(table)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
