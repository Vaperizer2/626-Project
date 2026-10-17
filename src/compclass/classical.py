"""Classical computer-vision baseline: hand-crafted features + an RBF support-vector machine.

Features (fixed, not learned):

* **HOG** (histogram of oriented gradients, Dalal & Triggs 2005) on a 128x128 grey image: gradient
  directions pooled in 16x16 cells and contrast-normalised over 2x2-cell blocks -> shape / edge layout.
* **HSV colour histogram** (8 hue x 4 saturation x 4 value bins) -> colour, which HOG ignores
  (e.g. blue potentiometers, green PCBs, banded resistors).

The SVM finds a maximum-margin boundary in an RBF kernel space. This is what "computer vision" meant before
deep learning, and the gap to the CNNs is the measure of what learned features buy us.

    uv run python -m compclass.classical --folds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import Manifest, Row, get_dev, get_fold, load_manifest
from .evaluate import compute_metrics, log_run

SIZE = 128


def image_features(img: Image.Image) -> np.ndarray:
    """HOG + HSV-histogram vector for one image (short side resized to 128, centre-cropped)."""
    w, h = img.size
    s = SIZE / min(w, h)
    img = img.convert("RGB").resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BILINEAR)
    left, top = (img.width - SIZE) // 2, (img.height - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    grey = np.asarray(img.convert("L"), dtype=np.float32) / 255
    f_hog = hog(grey, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    hsv = np.asarray(img.convert("HSV")).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=(8, 4, 4), range=((0, 256),) * 3)
    return np.concatenate([f_hog, hist.ravel() / hsv.shape[0]]).astype(np.float32)


def extract_all(rows: list[Row], m: Manifest) -> np.ndarray:
    def one(r: Row) -> np.ndarray:
        with Image.open(m.root / r["path"]) as im:
            return image_features(im)

    from concurrent.futures import ThreadPoolExecutor  # skimage/numpy release the GIL for most of the work

    with ThreadPoolExecutor(8) as pool:
        return np.stack(list(pool.map(one, rows)))


def run_classical(manifest_path: str, folds: list[int], C: float = 10.0, results: str = "results/runs.jsonl"):
    m = load_manifest(manifest_path)
    dev = get_dev(m)
    t0 = time.perf_counter()
    X = extract_all(dev, m)
    extract_s = time.perf_counter() - t0
    pos = {r["id"]: i for i, r in enumerate(dev)}
    records = []
    for k in folds:
        tr, va = get_fold(m, k)
        itr, iva = [pos[r["id"]] for r in tr], [pos[r["id"]] for r in va]
        ytr = np.array([m.class_to_idx[r["label"]] for r in tr])
        yva = np.array([m.class_to_idx[r["label"]] for r in va])
        clf = make_pipeline(StandardScaler(), SVC(C=C, kernel="rbf", gamma="scale")).fit(X[itr], ytr)
        metrics = compute_metrics(yva, clf.predict(X[iva]), m.classes)
        rec = {"method": "classical", "name": "hog_hsv_svm", "model": "hog_hsv_svm", "fold": k,
               "manifest": m.fingerprint, "feature_dim": int(X.shape[1]), "C": C,
               "extract_time_s": round(extract_s, 1), **metrics}
        log_run(rec, results)
        print(f"[classical f{k}] acc {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}")
        records.append(rec)
    return records


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/v2/manifest.json")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--results", default="results/runs.jsonl")
    args = ap.parse_args(argv)
    run_classical(args.manifest, args.folds, args.C, args.results)


if __name__ == "__main__":
    main()
