"""Dataset manifest, fold selection, image loading and augmentation.

The dataset is described by one JSON manifest (``data/v2/manifest.json``, schema ``v2.0``). Each row is one
unique image with a label, a near-duplicate ``cluster``, a ``split`` (``dev`` or ``test``) and, for dev rows,
a ``fold`` in ``0..n_folds-1``. Folds were assigned per cluster, so near-duplicates never straddle a fold
boundary — this is what makes the cross-validation honest.

Rules this module enforces:

* ``get_fold(m, k)`` returns (train, val) rows drawn from ``dev`` only.
* the sealed ``test`` rows are only reachable via ``get_test(m, confirm=True)``.
* ``load_manifest`` recomputes the fingerprint, so silently edited manifests are rejected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

try:  # registers AVIF with Pillow (only needed for raw images; v2 images are JPEG)
    import pillow_avif  # noqa: F401
except ImportError:  # pragma: no cover
    pass

Row = dict[str, Any]
REQUIRED_TOP = ("version", "classes", "images_root", "split_policy", "fingerprint", "rows")
REQUIRED_ROW = ("id", "path", "label", "cluster", "split", "fold")


# --------------------------------------------------------------------------------------------- manifest
@dataclass(frozen=True)
class Manifest:
    classes: list[str]
    rows: list[Row]
    root: Path  # directory that row["path"] is relative to (data/v2/)
    fingerprint: str
    n_folds: int
    raw: dict[str, Any]

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}


def compute_fingerprint(rows: list[Row]) -> str:
    """Split fingerprint, identical to tools/build_dataset_v2.py: lines ``id\\tlabel\\tsplit\\tfold`` (fold ``""``
    for test), sorted, joined with ``\\n``, UTF-8, SHA-256, first 16 hex chars."""
    lines = sorted(f"{r['id']}\t{r['label']}\t{r['split']}\t{'' if r['fold'] is None else r['fold']}" for r in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def load_manifest(path: str | Path, verify: bool = True) -> Manifest:
    """Read and validate a v2 manifest. Raises ``ValueError`` describing the first problem found."""
    path = Path(path)
    m = json.loads(path.read_text())
    missing = [k for k in REQUIRED_TOP if k not in m]
    if missing:
        raise ValueError(f"{path}: missing top-level keys {missing}")
    classes, rows = list(m["classes"]), list(m["rows"])
    if classes != sorted(set(classes)):
        raise ValueError("classes must be unique and sorted alphabetically")
    n_folds = int(m["split_policy"]["n_folds"])
    seen: set[str] = set()
    for r in rows:
        if any(k not in r for k in REQUIRED_ROW):
            raise ValueError(f"row {r.get('id')}: missing keys {[k for k in REQUIRED_ROW if k not in r]}")
        if r["id"] in seen:
            raise ValueError(f"duplicate row id {r['id']}")
        seen.add(r["id"])
        if r["label"] not in classes:
            raise ValueError(f"row {r['id']}: unknown label {r['label']!r}")
        if r["split"] == "dev":
            if not (isinstance(r["fold"], int) and 0 <= r["fold"] < n_folds):
                raise ValueError(f"dev row {r['id']}: fold must be an int in [0, {n_folds})")
        elif r["split"] == "test":
            if r["fold"] is not None:
                raise ValueError(f"test row {r['id']}: fold must be null")
        else:
            raise ValueError(f"row {r['id']}: split must be 'dev' or 'test'")
    # A cluster must live in exactly one place (one fold, or test) -- otherwise CV leaks.
    home: dict[str, tuple[str, Any]] = {}
    for r in rows:
        where = (r["split"], r["fold"])
        if home.setdefault(r["cluster"], where) != where:
            raise ValueError(f"cluster {r['cluster']} appears in both {home[r['cluster']]} and {where}")
    if verify and compute_fingerprint(rows) != m["fingerprint"]:
        raise ValueError(f"fingerprint mismatch: manifest says {m['fingerprint']}, rows hash to "
                         f"{compute_fingerprint(rows)} (manifest edited by hand?)")
    root = path.parent
    return Manifest(classes, rows, root, m["fingerprint"], n_folds, m)


def get_fold(m: Manifest, k: int) -> tuple[list[Row], list[Row]]:
    """Train/validation rows for CV fold ``k``: val = dev rows with fold k, train = the other dev rows."""
    if not 0 <= k < m.n_folds:
        raise ValueError(f"fold {k} out of range 0..{m.n_folds - 1}")
    dev = [r for r in m.rows if r["split"] == "dev"]
    return [r for r in dev if r["fold"] != k], [r for r in dev if r["fold"] == k]


def get_dev(m: Manifest) -> list[Row]:
    """All dev rows (e.g. to refit the final model on all folds)."""
    return [r for r in m.rows if r["split"] == "dev"]


def get_test(m: Manifest, *, confirm: bool = False) -> list[Row]:
    """The sealed test rows. Pass ``confirm=True`` to acknowledge you are spending the one test look."""
    if not confirm:
        raise PermissionError("the test split is sealed: call get_test(manifest, confirm=True) "
                              "only for the final, frozen model and record it in EXPERIMENT_LOG.md")
    return [r for r in m.rows if r["split"] == "test"]


# --------------------------------------------------------------------------------------------- dataset
class ImageDataset(Dataset):
    """Rows -> (image tensor, class index). Images are decoded lazily with Pillow."""

    def __init__(self, rows: list[Row], root: Path, class_to_idx: dict[str, int], transform=None):
        self.rows, self.root, self.class_to_idx, self.transform = rows, Path(root), class_to_idx, transform

    def __len__(self) -> int:
        return len(self.rows)

    def load(self, i: int) -> Image.Image:
        with Image.open(self.root / self.rows[i]["path"]) as im:
            return im.convert("RGB")

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img = self.load(i)
        x = self.transform(img) if self.transform else T.functional.to_tensor(img)
        return x, self.class_to_idx[self.rows[i]["label"]]


# --------------------------------------------------------------------------------------------- transforms
_INTERP = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC,
           "nearest": T.InterpolationMode.NEAREST}


def build_transforms(spec, aug: dict[str, Any] | None, train: bool) -> T.Compose:
    """Torchvision pipeline for a model's ``PreprocessSpec`` (see models.registry).

    Eval: resize the short side to ``input_size / crop_pct`` then centre-crop -- the same recipe the
    pretrained weights were evaluated with, and the one the ONNX/OpenCV demo reproduces.
    Train: RandomResizedCrop (scale/ratio from ``aug``), flips, optional RandAugment / colour jitter,
    optional RandomErasing after normalisation. Mixup/CutMix act on batches: see ``build_mixup``.
    """
    aug = aug or {}
    size, interp = spec.input_size, _INTERP[spec.interpolation]
    norm = [T.ToTensor(), T.Normalize(spec.mean, spec.std)]
    if not train:
        return T.Compose([T.Resize(int(round(size / spec.crop_pct)), interpolation=interp),
                          T.CenterCrop(size), *norm])
    ops: list = [T.RandomResizedCrop(size, scale=tuple(aug.get("rrc_scale", (0.5, 1.0))),
                                     ratio=tuple(aug.get("rrc_ratio", (3 / 4, 4 / 3))), interpolation=interp)]
    if aug.get("hflip", 0.5):
        ops.append(T.RandomHorizontalFlip(aug.get("hflip", 0.5)))
    if aug.get("vflip", 0.0):
        ops.append(T.RandomVerticalFlip(aug["vflip"]))
    if ra := aug.get("randaugment"):
        ops.append(T.RandAugment(num_ops=ra.get("n", 2), magnitude=ra.get("m", 9), interpolation=interp))
    if cj := aug.get("color_jitter", 0.0):
        ops.append(T.ColorJitter(cj, cj, cj, min(cj, 0.5) / 4))
    ops += norm
    if re := aug.get("random_erasing", 0.0):
        ops.append(T.RandomErasing(p=re, value="random"))
    return T.Compose(ops)


def build_mixup(aug: dict[str, Any] | None, num_classes: int, label_smoothing: float):
    """timm ``Mixup`` (mixes images *and* labels within a batch) or None if both alphas are 0."""
    aug = aug or {}
    if not (aug.get("mixup", 0.0) or aug.get("cutmix", 0.0)):
        return None
    from timm.data import Mixup

    return Mixup(mixup_alpha=aug.get("mixup", 0.0), cutmix_alpha=aug.get("cutmix", 0.0),
                 prob=aug.get("mix_prob", 1.0), label_smoothing=label_smoothing, num_classes=num_classes)


# --------------------------------------------------------------------------------------------- synthetic
def make_synthetic_manifest(out_dir: str | Path, n_classes: int = 3, per_class: int = 15,
                            test_per_class: int = 3, n_folds: int = 5, size: int = 96, seed: int = 0) -> Path:
    """Write a tiny, learnable v2-format dataset (coloured shapes) for tests and smoke runs.

    Class c is a filled square of a class-specific colour at a random position on a noisy background, so a
    model can reach high accuracy within one epoch. Returns the manifest path.
    """
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    palette = [(220, 40, 40), (40, 180, 60), (40, 80, 220), (230, 200, 30), (160, 60, 200), (30, 200, 200)]
    classes = [f"class_{chr(97 + c)}" for c in range(n_classes)]
    rows: list[Row] = []
    for c, name in enumerate(classes):
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        for j in range(per_class + test_per_class):
            arr = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
            s = int(rng.integers(size // 4, size // 2))
            y0, x0 = rng.integers(0, size - s, 2)
            arr[y0:y0 + s, x0:x0 + s] = palette[c % len(palette)]
            raw = arr.tobytes()
            sha = hashlib.sha256(raw).hexdigest()
            rel = f"images/{name}/{sha[:16]}.jpg"
            Image.fromarray(arr).save(out / rel, quality=95)
            is_test = j >= per_class
            rows.append({"id": sha[:16], "path": rel, "label": name, "cluster": f"c{len(rows):05d}",
                         "split": "test" if is_test else "dev", "fold": None if is_test else j % n_folds,
                         "sha256": sha, "src_paths": [], "orig_format": "JPEG", "orig_width": size,
                         "orig_height": size, "legacy_split": None, "flags": []})
    manifest = {"version": "v2.0", "created_utc": "synthetic", "classes": classes, "images_root": "images",
                "split_policy": {"seed": seed, "test_fraction": test_per_class / (per_class + test_per_class),
                                 "n_folds": n_folds, "group_key": "cluster"},
                "fingerprint": compute_fingerprint(rows), "rows": rows, "excluded": []}
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path
