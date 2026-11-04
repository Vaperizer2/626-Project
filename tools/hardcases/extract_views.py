"""Dev embeddings of other input views through the frozen harness's baseline trunk (ORT CUDA EP, TF32, batch 1).

Same graph, provider and batch size as tools/speed eval, so ``center`` reproduces the harness's own embeddings
and any change in a view's score comes from the view, not from numerics. Run with the harness venv:

    $HV/bin/python tools/hardcases/extract_views.py hflip squash letterbox [--limit 64] [--manifest PATH --out DIR]

``--manifest``/``--out`` embed another manifest's rows (all splits) instead of dev; used once, for the final
demo-stack look, never for selection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ART, dev_rows

TRUNK = Path("/home/vaper/Work/Prototypes/626-artifacts/harness/candidates/baseline_fp32/model.onnx")
SIZE, SHORT, MEAN, STD = 384, 427, 0.5, 0.5


def _norm(im: Image.Image) -> np.ndarray:
    a = np.asarray(im, np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)


def center(im: Image.Image) -> Image.Image:
    """compclass.data.build_transforms eval: short side to 427 (bicubic), center crop 384."""
    from torchvision.transforms import functional as TF

    return TF.center_crop(TF.resize(im, SHORT, interpolation=TF.InterpolationMode.BICUBIC), SIZE)


def squash(im: Image.Image) -> Image.Image:
    """SigLIP's own preprocessing: the whole picture resized to 384 x 384, aspect not kept."""
    return im.resize((SIZE, SIZE), Image.BICUBIC)


def letterbox(im: Image.Image, fill=(128, 128, 128)) -> Image.Image:
    """Whole picture, long side to 384, padded to a square with mid-grey (0 after normalisation)."""
    w, h = im.size
    s = SIZE / max(w, h)
    r = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    out = Image.new("RGB", (SIZE, SIZE), fill)
    out.paste(r, ((SIZE - r.size[0]) // 2, (SIZE - r.size[1]) // 2))
    return out


def crop100(im: Image.Image) -> Image.Image:
    """Short side to 384, center crop 384 (crop_pct 1.0): less border lost than the deployed 0.9."""
    from torchvision.transforms import functional as TF

    return TF.center_crop(TF.resize(im, SIZE, interpolation=TF.InterpolationMode.BICUBIC), SIZE)


def aug(seed: int):
    """Training-time views: RandomResizedCrop(scale 0.35-1) + hflip + mild colour jitter, seeded per image."""
    from torchvision import transforms as T

    def view(im: Image.Image, i: int) -> Image.Image:
        import torch

        torch.manual_seed(seed * 1_000_003 + i)
        t = T.Compose([T.RandomResizedCrop(SIZE, scale=(0.35, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                       T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2, 0.02)])
        return t(im)

    return view


VIEWS = {
    "center": lambda im, i: center(im),
    "hflip": lambda im, i: center(im).transpose(Image.FLIP_LEFT_RIGHT),
    "squash": lambda im, i: squash(im),
    "squash_hflip": lambda im, i: squash(im).transpose(Image.FLIP_LEFT_RIGHT),
    "letterbox": lambda im, i: letterbox(im),
    "letterbox_white": lambda im, i: letterbox(im, (255, 255, 255)),
    "crop100": lambda im, i: crop100(im),
    **{f"aug{s}": aug(s) for s in range(8)},
}


def session():
    import onnxruntime as ort

    ort.preload_dlls()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(TRUNK), so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("views", nargs="+", choices=sorted(VIEWS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=str(ART / "emb"))
    a = ap.parse_args()
    if a.manifest:
        m = json.loads(Path(a.manifest).read_text())
        root, rows = Path(a.manifest).parent, m["rows"]
    else:
        m, rows = dev_rows()
        root = m.root
    rows = rows[:a.limit] if a.limit else rows
    sess = session()
    if sess.get_providers()[0] != "CUDAExecutionProvider":
        raise SystemExit(f"CUDA EP did not load: {sess.get_providers()}")
    out = {v: np.zeros((len(rows), 1152), np.float32) for v in a.views}
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        with Image.open(root / r["path"]) as im:
            im = im.convert("RGB")
        for v in a.views:
            x = _norm(VIEWS[v](im, i))[None]
            out[v][i] = sess.run(["embedding"], {"pixel_values": x})[0][0]
        if i % 250 == 0:
            print(f"{i}/{len(rows)} {time.perf_counter() - t0:.0f}s", flush=True)
    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    suffix = f"_first{a.limit}" if a.limit else ""
    for v, F in out.items():
        if not np.isfinite(F).all():
            raise SystemExit(f"{v}: non-finite embeddings")
        np.save(dst / f"{v}{suffix}.npy", F)
    if a.manifest:
        (dst / "ids.json").write_text(json.dumps([r["id"] if "id" in r else r["path"] for r in rows]))
    print(f"done {len(rows)} rows x {len(a.views)} views in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
