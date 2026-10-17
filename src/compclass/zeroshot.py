"""Zero-shot classification with CLIP / SigLIP: no training images at all.

Each class name is inserted into several prompt templates ("a photo of a {}."); the text tower embeds them,
the embeddings are averaged and L2-normalised into one "text classifier" vector per class. An image is
assigned to the class whose vector has the highest cosine similarity with the image embedding. It is the
zero-data baseline every trained model should beat -- and it shows how much a web-scale model already knows.

    uv run python -m compclass.zeroshot --model clip_vitb16 --folds 0 1 2 3 4

Scored on the same validation folds as everything else (nothing is fitted, so all folds are fair).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImageDataset, get_fold, load_manifest
from .evaluate import compute_metrics, log_run
from .models.registry import REGISTRY

TEMPLATES = [
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a photo of a {}, an electronic component.",
    "a product photo of a {}.",
    "a {} on a white background.",
    "a photo of a {} on a circuit board.",
]
# Folder names -> more natural phrases (anything not listed is used verbatim).
PROMPT_NAMES = {"ic chip": "integrated circuit chip", "led": "LED", "inductor coil": "inductor coil",
                "microcontroller board": "microcontroller development board", "heat sink": "heat sink",
                "jumper wire": "jumper wire", "power supply module": "power supply module"}


@torch.no_grad()
def text_classifier(model, tokenizer, classes: list[str], device: torch.device) -> torch.Tensor:
    """[C, D] matrix of normalised, template-averaged class text embeddings."""
    weights = []
    for c in classes:
        tokens = tokenizer([t.format(PROMPT_NAMES.get(c, c)) for t in TEMPLATES]).to(device)
        e = model.encode_text(tokens).float()
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.mean(0)
        weights.append(e / e.norm())
    return torch.stack(weights)


def run_zeroshot(manifest_path: str, key: str, folds: list[int], results: str = "results/runs.jsonl",
                 batch_size: int = 64, num_workers: int = 4) -> list[dict]:
    import open_clip

    entry = REGISTRY[key]
    if entry.zeroshot is None:
        raise ValueError(f"{key} has no text tower; zero-shot models: "
                         f"{[k for k, e in REGISTRY.items() if e.zeroshot]}")
    arch, tag = entry.zeroshot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(arch)
    m = load_manifest(manifest_path)
    W = text_classifier(model, tokenizer, m.classes, device)
    records = []
    for k in folds:
        _, va = get_fold(m, k)
        dl = DataLoader(ImageDataset(va, m.root, m.class_to_idx, preprocess), batch_size=batch_size,
                        num_workers=num_workers)
        preds, ys = [], []
        with torch.no_grad():
            for x, y in dl:
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    f = model.encode_image(x.to(device)).float()
                preds.append(((f / f.norm(dim=-1, keepdim=True)) @ W.T).argmax(1).cpu())
                ys.append(y)
        metrics = compute_metrics(torch.cat(ys).numpy(), torch.cat(preds).numpy(), m.classes)
        rec = {"method": "zeroshot", "name": f"zeroshot_{key}", "model": key, "fold": k, "manifest": m.fingerprint,
               "open_clip": [arch, tag], "n_templates": len(TEMPLATES), **metrics}
        log_run(rec, results)
        print(f"[zeroshot {key} f{k}] acc {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}")
        records.append(rec)
    return records


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/v2/manifest.json")
    ap.add_argument("--model", required=True, choices=[k for k, e in REGISTRY.items() if e.zeroshot])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--results", default="results/runs.jsonl")
    args = ap.parse_args(argv)
    run_zeroshot(args.manifest, args.model, args.folds, args.results)


if __name__ == "__main__":
    main()
