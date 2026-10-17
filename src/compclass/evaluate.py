"""Metrics, prediction, and the append-only run log ``results/runs.jsonl``.

Every method (fine-tune, linear probe, zero-shot, classical) ends by calling ``log_run`` with the same
fields, so one ``pandas.read_json("results/runs.jsonl", lines=True)`` gives the whole comparison table.

CLI -- re-evaluate a saved checkpoint on its validation fold, or (once, at the very end) on the sealed test:

    uv run python -m compclass.evaluate --checkpoint runs/<run>/best.pt
    uv run python -m compclass.evaluate --checkpoint runs/<run>/best.pt --sealed-test
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader


def compute_metrics(y_true, y_pred, classes: list[str]) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class precision/recall/F1/support and the confusion matrix (rows = true)."""
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(classes)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device,
            amp_dtype: torch.dtype | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (softmax probabilities [N, C], true labels [N])."""
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x.to(device, non_blocking=True))
        probs.append(logits.float().softmax(1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def log_run(record: dict[str, Any], path: str | Path = "results/runs.jsonl") -> None:
    """Append one JSON line (adds timestamp + host info)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "host": platform.node(), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None, **record}
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=float) + "\n")


def main(argv: list[str] | None = None) -> None:
    from .data import ImageDataset, build_transforms, get_fold, get_test, load_manifest
    from .models import build_model
    from .models.registry import PreprocessSpec

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", help="override the manifest path stored in the checkpoint")
    ap.add_argument("--sealed-test", action="store_true", help="evaluate on the sealed test split (use once)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = load_manifest(args.manifest or cfg["data"]["manifest"])
    if m.fingerprint != ck["manifest_fingerprint"]:
        raise SystemExit(f"checkpoint was trained on manifest {ck['manifest_fingerprint']}, not {m.fingerprint}")
    rows = get_test(m, confirm=True) if args.sealed_test else get_fold(m, ck["fold"])[1]
    spec = PreprocessSpec(**{k: v for k, v in ck["preprocess"].items() if k != "resize_short_side"})
    model, _ = build_model(cfg["model"]["key"], len(m.classes), pretrained=False, **cfg["model"].get("kwargs", {}))
    model.load_state_dict(ck["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(ImageDataset(rows, m.root, m.class_to_idx, build_transforms(spec, None, train=False)),
                        batch_size=args.batch_size, num_workers=cfg["data"].get("num_workers", 4))
    probs, y = predict(model.to(device), loader, device)
    metrics = compute_metrics(y, probs.argmax(1), m.classes)
    split = "test" if args.sealed_test else f"val_fold{ck['fold']}"
    print(json.dumps({"split": split, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}))
    log_run({"method": "evaluate", "name": cfg["name"], "model": cfg["model"]["key"], "split": split,
             "config_hash": ck["config_hash"], "fold": ck["fold"], "manifest": m.fingerprint,
             "checkpoint": str(args.checkpoint), **metrics}, cfg["output"]["results"])


if __name__ == "__main__":
    main()
