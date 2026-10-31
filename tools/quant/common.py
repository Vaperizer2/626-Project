"""Shared pieces of the quantisation track: paths, preprocessing of pool images, the legion GPU lock."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools/speed"))

from harness import gpu_lock  # noqa: E402,F401

ART = Path(os.environ.get("QUANT_ART", "/home/vaper/Work/Prototypes/626-artifacts/quant"))
CALIB = ART / "calib"
BASELINE = Path(os.environ.get("QUANT_BASELINE",
                               "/home/vaper/Work/Prototypes/626-artifacts/harness/candidates/baseline_fp32"))
SPEC = SimpleNamespace(input_size=384, crop_pct=0.9, interpolation="bicubic", mean=[0.5] * 3, std=[0.5] * 3)


def pool_images(split: str, n: int | None = None) -> list[Path]:
    rows = json.loads((CALIB / "split.json").read_text())[split]
    return [CALIB / r["path"] for r in rows[:n]]


def spec_of(model_dir: Path) -> SimpleNamespace:
    """The eval transform a candidate's preprocess.json declares; the deployed 384 px one when it has none."""
    p = model_dir / "preprocess.json"
    if not p.exists():
        return SPEC
    pre = json.loads(p.read_text())
    return SimpleNamespace(**{k: pre[k] for k in vars(SPEC)})


def preprocess(paths: list[Path], spec: SimpleNamespace = SPEC) -> np.ndarray:
    """The harness's eval transform (compclass.data.build_transforms, train=False) on pool images."""
    from PIL import Image

    from compclass.data import build_transforms

    tf = build_transforms(spec, None, train=False)
    return np.stack([tf(Image.open(p).convert("RGB")).numpy() for p in paths]).astype(np.float32)
