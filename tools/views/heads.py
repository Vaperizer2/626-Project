"""The two fixed matrices of a backbone's zero-shot head, written to results/cache/zeroshot/ (git-ignored).

    uv run python tools/views/heads.py --model mobileclip2_s3 [--model ...]

    <key>_W.npy      [20, D] L2-normalised, template-averaged class text embeddings (compclass.zeroshot prompts,
                     the open_clip text tower named in the registry's ``zeroshot`` pair)
    <key>_proj.npz   the backbone's own projection into that text space, read from its timm checkpoint
                     (``head.weight`` / ``head.bias``). timm's num_classes=0 features, the probe's input and our
                     cache, are taken BEFORE this projection; the text space is after it. SigLIP So400m has no
                     projection (its text weights are in exports/final/head.npz), every other candidate has one.

Generalises tools/views/pe_core_heads.py (PE-Core L) to any registered key with a zeroshot pair.
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
OUT = Path("results/cache/zeroshot")


def write_heads(key: str) -> None:
    import open_clip
    from safetensors import safe_open
    from compclass.data import load_manifest
    from compclass.models.registry import REGISTRY
    from compclass.zeroshot import text_classifier

    entry = REGISTRY[key]
    if entry.zeroshot is None:
        raise SystemExit(f"{key} has no text tower registered")
    arch, tag = entry.zeroshot
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_manifest("data/v2/manifest.json")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag, device=dev,
                                                        force_quick_gelu=tag == "openai")
    model.eval()
    with torch.no_grad():
        W = text_classifier(model, open_clip.get_tokenizer(arch), m.classes, dev).cpu().numpy()
    np.save(OUT / f"{key}_W.npy", W.astype(np.float32))
    import timm
    timm_name = entry.timm_name
    pat = str(Path.home() / f".cache/huggingface/hub/models--timm--{timm_name}/snapshots/*/model.safetensors")
    ck = glob.glob(pat)
    if not ck:
        timm.create_model(timm_name, pretrained=True, num_classes=0)  # fills the hub cache
        ck = glob.glob(pat)
    with safe_open(ck[0], "pt") as f:
        keys = list(f.keys())
        wk = "head.weight" if "head.weight" in keys else next(k for k in keys if k.endswith("head.fc.weight") or k.endswith("proj.weight"))
        bk = wk.replace("weight", "bias")
        np.savez(OUT / f"{key}_proj.npz", weight=f.get_tensor(wk).float().numpy(),
                 bias=f.get_tensor(bk).float().numpy() if bk in keys else np.zeros(f.get_tensor(wk).shape[0], np.float32))
    print(f"{key}: W {W.shape} from open_clip {arch}/{tag}; projection {wk} from {Path(ck[0]).parent.parent.parent.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True)
    for k in ap.parse_args().model:
        write_heads(k)


if __name__ == "__main__":
    main()
