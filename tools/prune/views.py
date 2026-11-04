"""Pool views and the distillation loss, copied verbatim from wip/distill `tools/distill/common.py` (cda6955).

The teacher cache (`teacher_v8g2`) was rendered with exactly these ops, so a view rendered here at 384 px is the
input whose So400m embedding the cache holds. Do not edit; re-copy if the distill track changes its views.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

EVAL_CROP_PCT = 0.9


T_ZS = 50.0


def view_seed(image_id: str, view: int) -> int:
    return int(hashlib.sha256(f"{image_id}:{view}".encode()).hexdigest()[:12], 16)


def render_view(im: Image.Image, image_id: str, view: int, size: int, crop_pct: float = EVAL_CROP_PCT) -> Image.Image:
    """Square ``size`` x ``size`` RGB view number ``view`` of ``im`` (view 0 = evaluation view)."""
    w, h = im.size
    if view == 0:  # the exact ops of compclass.data.build_transforms(train=False)
        import torchvision.transforms as T

        im = T.Resize(int(round(size / crop_pct)), interpolation=T.InterpolationMode.BICUBIC)(im)
        return T.CenterCrop(size)(im)
    rng = random.Random(view_seed(image_id, view))
    if rng.random() < 0.2:
        im = im.rotate(rng.uniform(-25, 25), Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    for _ in range(10):  # RandomResizedCrop(scale=(0.2, 1), ratio=(3/4, 4/3)) on the source pixels
        area = w * h * rng.uniform(0.2, 1.0)
        ar = math.exp(rng.uniform(math.log(3 / 4), math.log(4 / 3)))
        cw, ch = math.sqrt(area * ar), math.sqrt(area / ar)
        if cw <= w and ch <= h:
            x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
            break
    else:
        cw = ch = min(w, h)
        x0, y0 = (w - cw) / 2, (h - ch) / 2
    im = im.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch)))
    if rng.random() < 0.5:
        im = ImageOps.mirror(im)
    if rng.random() < 0.1:
        im = ImageOps.flip(im)
    if rng.random() < 0.8:
        ops = [(ImageEnhance.Brightness, 0.6, 1.4), (ImageEnhance.Contrast, 0.6, 1.4), (ImageEnhance.Color, 0.5, 1.5)]
        rng.shuffle(ops)
        for op, lo, hi in ops:
            im = op(im).enhance(rng.uniform(lo, hi))
    if rng.random() < 0.1:
        im = ImageOps.grayscale(im).convert("RGB")
    if rng.random() < 0.2:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.5) * min(im.size) / 256))
    if rng.random() < 0.2:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=rng.randint(30, 90))
        im = Image.open(buf).convert("RGB")
    return im.resize((size, size), Image.Resampling.BICUBIC)


def to_tensor(im: Image.Image, mean, std):
    import torch

    x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
    return (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)


def distill_loss(s, t, mu, sd, W, w_mse: float, w_cos: float, w_zs: float) -> tuple:
    """Feature distillation loss and its terms.

    ``mse`` compares the per-dimension standardised embeddings, the space the deployed probe reads (StandardScaler
    then logistic regression). ``cos`` aligns directions, the space the zero-shot head reads. ``zs`` is the KL
    between the teacher's and the student's zero-shot class distributions (text embeddings only, no dev labels).
    """
    import torch
    import torch.nn.functional as F

    mse = ((s - mu) / sd - (t - mu) / sd).pow(2).mean()
    cos = 1 - F.cosine_similarity(s, t, dim=1).mean()
    loss = w_mse * mse + w_cos * cos
    zs = torch.zeros((), device=s.device)
    if w_zs:
        ls = F.log_softmax(T_ZS * F.normalize(s, dim=1) @ W.T, dim=1)
        lt = F.log_softmax(T_ZS * F.normalize(t, dim=1) @ W.T, dim=1)
        zs = F.kl_div(ls, lt, log_target=True, reduction="batchmean")
        loss = loss + w_zs * zs
    return loss, {"mse": mse.item(), "cos": cos.item(), "zs": zs.item()}


def load_pool(pool_dir: str | Path) -> tuple[list[dict], dict]:
    p = json.loads((Path(pool_dir) / "pool.json").read_text())
    return p["rows"], p


def is_pool_val(image_id: str) -> bool:
    """~2% of pool images held out from student training to measure teacher fidelity (never data/v2)."""
    return int(image_id[:4], 16) % 50 == 0
