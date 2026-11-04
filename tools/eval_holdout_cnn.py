"""Score frozen scratch-CNN checkpoints once on a sealed manifest (sealed test split or the web holdout).

    uv run python tools/eval_holdout_cnn.py runs/win/scratch_cnn-b0afa2aa4d60/fold0/best.pt \
        --manifest data/web_holdout_v1/manifest.json --tag plain_200ep --confirm

Every row with ``split == "test"`` in the manifest is scored (``get_test(..., confirm=True)``: this is the one
look, record it in EXPERIMENT_LOG.md). The checkpoint's own eval preprocessing is used; its training manifest only
has to share the class list. Writes ``<out>/<prefix>_<tag>.json`` (accuracy, macro-F1, per-class, confusion,
in-scope subset) and ``<out>/<prefix>_<tag>_preds.npz`` (ids, y, probs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from compclass.data import ImageDataset, build_transforms, get_test, load_manifest
from compclass.evaluate import compute_metrics
from compclass.models import build_model
from compclass.models.registry import PreprocessSpec


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="best.pt of the frozen model")
    ap.add_argument("--manifest", default="data/web_holdout_v1/manifest.json")
    ap.add_argument("--tag", required=True, help="output name, e.g. plain_200ep_b0afa2aa4d60")
    ap.add_argument("--out", default="results/scratch_cnn_improvement")
    ap.add_argument("--prefix", default="web_holdout")
    ap.add_argument("--confirm", action="store_true", help="acknowledge spending this model's one look")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args(argv)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = load_manifest(args.manifest)
    train_m = load_manifest(cfg["data"]["manifest"])
    if train_m.classes != m.classes:
        raise SystemExit("class lists differ between the training manifest and the evaluation manifest")
    if train_m.fingerprint != ck["manifest_fingerprint"]:
        raise SystemExit(f"checkpoint trained on manifest {ck['manifest_fingerprint']}, not {train_m.fingerprint}")
    rows = get_test(m, confirm=args.confirm)
    spec = PreprocessSpec(**{k: v for k, v in ck["preprocess"].items() if k != "resize_short_side"})
    model, _ = build_model(cfg["model"]["key"], len(m.classes), pretrained=False, **cfg["model"].get("kwargs", {}))
    model.load_state_dict(ck["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(ImageDataset(rows, m.root, m.class_to_idx, build_transforms(spec, cfg.get("aug"), train=False)),
                        batch_size=args.batch_size, num_workers=0)
    probs, y = predict(model.to(device), loader, device)
    pred = probs.argmax(1)
    met = compute_metrics(y, pred, m.classes)
    ood = np.array(["ood_suspect" in (r.get("flags") or []) for r in rows])
    rep = {"manifest": args.manifest, "fingerprint": m.fingerprint, "n": int(len(rows)),
           "checkpoint": args.checkpoint, "config_hash": ck.get("config_hash"), "fold": ck.get("fold"),
           "best_epoch": ck.get("epoch"), "training_manifest": train_m.fingerprint, **met,
           "inscope_n": int((~ood).sum()),
           "inscope_accuracy": float((pred[~ood] == y[~ood]).mean()) if (~ood).any() else None,
           "inscope_macro_f1": (compute_metrics(y[~ood], pred[~ood], m.classes)["macro_f1"] if (~ood).any() else None)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.prefix}_{args.tag}.json").write_text(json.dumps(rep, indent=1, default=float))
    np.savez_compressed(out / f"{args.prefix}_{args.tag}_preds.npz", ids=np.array([r["id"] for r in rows]), y=y,
                        probs=probs.astype(np.float32), classes=np.array(m.classes))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_class", "confusion")}, indent=1, default=float))


if __name__ == "__main__":
    main()
