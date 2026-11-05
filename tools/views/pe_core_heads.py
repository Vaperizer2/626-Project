"""PE-Core L's two fixed matrices for the zero-shot head, written to results/cache/zeroshot/ (git-ignored):

    pe_core_l14_336_W.npy     [20, 1024] L2-normalised, template-averaged class text embeddings (compclass.zeroshot
                              prompts, open_clip PE-Core-L-14-336 / meta)
    pe_core_l14_336_proj.npz  the 1024 x 1024 projection ("head") from the timm checkpoint. timm's num_classes=0
                              features (our cache) are taken BEFORE this projection; the text space is after it.
                              SigLIP has no projection, so compclass.final never needed this step.

    uv run python tools/views/pe_core_heads.py
"""
import glob, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
OUT = Path("results/cache/zeroshot")


def main() -> None:
    import open_clip
    from safetensors import safe_open
    from compclass.data import load_manifest
    from compclass.zeroshot import text_classifier

    OUT.mkdir(parents=True, exist_ok=True)
    m = load_manifest("data/v2/manifest.json")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms("PE-Core-L-14-336", pretrained="meta", device=dev)
    model.eval()
    with torch.no_grad():
        W = text_classifier(model, open_clip.get_tokenizer("PE-Core-L-14-336"), m.classes, dev).cpu().numpy()
    np.save(OUT / "pe_core_l14_336_W.npy", W.astype(np.float32))
    ck = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--timm--vit_pe_core_large_patch14_336.fb/snapshots/*/model.safetensors"))
    if not ck:  # not downloaded yet: build the timm model once so the hub cache fills
        import timm
        timm.create_model("vit_pe_core_large_patch14_336.fb", pretrained=True, num_classes=0)
        ck = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--timm--vit_pe_core_large_patch14_336.fb/snapshots/*/model.safetensors"))
    with safe_open(ck[0], "pt") as f:
        np.savez(OUT / "pe_core_l14_336_proj.npz", weight=f.get_tensor("head.weight").float().numpy(),
                 bias=f.get_tensor("head.bias").float().numpy())
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
