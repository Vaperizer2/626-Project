"""Torch So400m trunks built from harness graph directories, and the unlabeled image sources speed3 distils on.

A graph directory is any harness candidate holding the deployed op sequence (baseline_fp32, or the tokens track's
resized and the speed2 track's MLP-pruned edits of it). Its initializers become timm's
`vit_so400m_patch14_siglip_384` at the directory's own input size and MLP widths, so a student trained here writes
back into the same graph with `tools/prune/build.py load`, and its shapes stay exactly the graph's.

Image sources, never labels:
  dev        the 2,512 development-fold images of data/v2 (transductive: the harness scores these same images)
  devout:K   dev images outside fold K (the generalisation diagnostic: fold K stays unseen)
  pool       the distill track's pool minus pool-val, gated disjoint from every data/v2 and web-holdout image
The sealed test split and data/web_holdout_v1 are never listed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

R = Path(os.environ.get("SPEED3_R", "/home/carl/626-runs/speed3"))
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools/prune")]
import so400m  # noqa: E402
from views import is_pool_val, load_pool, render_view, to_tensor  # noqa: E402

MANIFEST = Path(os.environ.get("SPEED3_MANIFEST", "/home/carl/Projects/626-Project/data/v2/manifest.json"))
POOL = Path("/home/carl/626-runs/distill/pool")
TEXT = Path("/home/carl/626-runs/distill/v2meta/text_weights.npy")
FINGERPRINT = "431315e87d2612cb"
MEAN = STD = (0.5, 0.5, 0.5)
GRAPHS = {
    "t384": Path("/home/carl/626-runs/harness/candidates/baseline_fp32"),
    "t294": R / "base/t294",
    "s294": R / "base/s294_mlp3840",
}


def graph_size(d: Path) -> int:
    return int(json.loads((d / "preprocess.json").read_text())["input_size"])


def trunk(d: Path) -> nn.Module:
    """timm So400m with `d`'s weights, input size and per-block MLP widths (eval mode, fp32, CPU)."""
    import timm

    sd = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in so400m.Weights(so400m.load(d)).torch_state().items()}
    m = timm.create_model("vit_so400m_patch14_siglip_384", pretrained=False, num_classes=0, img_size=graph_size(d))
    for i, b in enumerate(m.blocks):
        h = sd[f"blocks.{i}.mlp.fc1.weight"].shape[0]
        if h != b.mlp.fc1.out_features:
            b.mlp.fc1, b.mlp.fc2 = nn.Linear(b.mlp.fc1.in_features, h), nn.Linear(h, b.mlp.fc2.out_features)
    m.load_state_dict(sd, strict=True)
    return m.eval()


def block_state(m: nn.Module) -> dict[str, np.ndarray]:
    """Every block Linear as `tools/prune/build.py load` reads it ([out, in] weights)."""
    return {f"blocks.{i}.{n}.{p}": getattr(b.get_submodule(n), p).detach().float().cpu().numpy()
            for i, b in enumerate(m.blocks) for n in so400m.BLOCK_LINEARS for p in ("weight", "bias")}


def load_block_state(m: nn.Module, path: Path) -> nn.Module:
    sd = np.load(path)
    with torch.no_grad():
        for i, b in enumerate(m.blocks):
            for n in so400m.BLOCK_LINEARS:
                for p in ("weight", "bias"):
                    getattr(b.get_submodule(n), p).copy_(torch.from_numpy(sd[f"blocks.{i}.{n}.{p}"]))
    return m


def dev_rows() -> tuple[object, list[dict]]:
    from compclass.data import get_dev, load_manifest

    m = load_manifest(MANIFEST)
    assert m.fingerprint == FINGERPRINT, f"manifest fingerprint {m.fingerprint}"
    return m, get_dev(m)


def source(name: str) -> list[tuple[str, Path]]:
    """(image id, file) pairs of an image source; see the module docstring."""
    if name == "pool":
        rows, _ = load_pool(POOL)
        return [(r["id"], POOL / r["path"]) for r in rows if not is_pool_val(r["id"])]
    m, rows = dev_rows()
    if name == "dev":
        return [(r["id"], m.root / r["path"]) for r in rows]
    if name.startswith("devout:"):
        k = int(name.split(":")[1])
        return [(r["id"], m.root / r["path"]) for r in rows if r["fold"] != k]
    raise SystemExit(f"unknown image source {name}")


class Views(torch.utils.data.Dataset):
    """Item i = view `views[i]` of image `items[i]`, rendered at every size in `sizes` (same crop, same jitter)."""

    def __init__(self, items: list[tuple[str, Path]], views: np.ndarray, sizes: tuple[int, ...]):
        self.items, self.views, self.sizes = items, views, sizes

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        image_id, path = self.items[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
        return tuple(to_tensor(render_view(im, image_id, int(self.views[i]), s), MEAN, STD) for s in self.sizes)


def loader(items, views, sizes, batch: int, workers: int = 8, shuffle: bool = False):
    return torch.utils.data.DataLoader(Views(items, np.asarray(views), sizes), batch_size=batch, shuffle=shuffle,
                                       num_workers=workers, pin_memory=True, persistent_workers=False)
