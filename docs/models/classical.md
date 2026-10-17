# Classical baseline: HOG + HSV colour histogram + RBF-SVM

**Code:** `src/compclass/classical.py` (not in the model registry — it has no neural network) ·
**CLI:** `uv run python -m compclass.classical [--folds 0 1 2 3 4] [--C 10.0]` ·
**Logged as:** `method="classical"`, `name="hog_hsv_svm"` in `results/runs.jsonl`

## 1. Core idea and why it was invented

Before 2012, image classification was a two-stage pipeline: a human designed a fixed feature extractor, and a
generic classifier learned a decision boundary on top of it. This module reproduces that pipeline with two
hand-crafted descriptors and a kernel SVM:

* **HOG — Histogram of Oriented Gradients.** N. Dalal and B. Triggs, 2005, *Histograms of Oriented Gradients for
  Human Detection*, CVPR 2005 (no arXiv). They showed that local gradient *orientation* histograms, contrast-
  normalised over overlapping blocks, describe object shape robustly to illumination changes and small
  deformations. HOG describes *where edges are and which way they point*, and ignores colour.
* **Colour histogram.** M. Swain and D. Ballard, 1991, *Color Indexing*, IJCV 7(1) (no arXiv). A global
  histogram of pixel colours is invariant to rotation and translation and captures cues that HOG discards
  (blue trimmer potentiometers, green PCBs, colour-banded resistors).
* **Support-vector machine.** C. Cortes and V. Vapnik, 1995, *Support-Vector Networks*, Machine Learning 20(3)
  (no arXiv). It finds the maximum-margin separating hyperplane; with an RBF kernel that hyperplane lives in an
  implicit infinite-dimensional feature space, so the boundary in input space is non-linear.

The pedagogical purpose here: **the gap between this baseline and the CNNs measures what learned features buy.**

## 2. Data flow

```
image (any size, RGB)
  │  resize short side → 128 px (bilinear), centre-crop 128×128           image_features()
  ├────────────────────────────┬───────────────────────────────────────────
  │ grey, float in [0,1]       │ HSV (Pillow, each channel 0..255)
  ▼                            ▼
HOG: 9 orientations,          3-D histogram 8 (H) × 4 (S) × 4 (V) bins
16×16-px cells (8×8 grid),    = 128 bins, divided by #pixels
2×2-cell blocks (7×7 blocks)
= 7·7·2·2·9 = 1764 dims
  └─────────────┬──────────────┘
                ▼  concatenate → 1892-dim float32 vector
        StandardScaler (per-feature z-score, fitted on the training fold)
                ▼
        SVC(kernel="rbf", C=10, gamma="scale")   one-vs-one multiclass (scikit-learn default)
                ▼
        predicted class
```

Feature extraction runs once for all dev rows (8 threads, `extract_all`), then each fold fits a fresh
scaler + SVM on its training rows and scores its validation rows.

## 3. "Pretraining"

None. The only learned parameters are the scaler statistics and the SVM's support vectors / dual
coefficients, fitted per fold on ~2,010 images.

## 4. Knobs that matter in this repo

This baseline does **not** read the YAML configs. Its knobs are:

| knob | where | default | effect of changing it |
|---|---|---|---|
| `C` | CLI `--C` | 10.0 | SVM soft-margin penalty. Larger C → fewer margin violations allowed → tighter fit (risk: memorising label noise). Smaller C → smoother boundary. Not tuned inside the fold — try a grid yourself. |
| `gamma` | code (`gamma="scale"`) | `1 / (n_features · Var(X))` | RBF width. With standardised features `Var(X)` ≈ 1, so γ ≈ 1/1892. |
| `SIZE` | code constant | 128 | Working resolution. Larger → finer HOG grid, more dimensions. |
| HOG `pixels_per_cell`, `cells_per_block`, `orientations` | code | 16, 2, 9 | Cell size trades spatial precision against robustness to misalignment. |
| HSV bins | code | 8×4×4 | More hue bins separate colours better but make the histogram sparser. |

Changing anything other than `C` means editing code; do that in a notebook or a copy (see "Try this"), not in
`src/`, so that the logged baseline stays reproducible.

## 5. Adaptation regimes

There is only one regime: fixed features + a trained classifier, the classical analogue of a *linear probe*
(see [`../concepts/transfer_learning.md`](../concepts/transfer_learning.md)). The kernel makes it non-linear, but the
features never adapt to the task.

## 6. Compute

CPU only. Feature extraction is one pass over ~2,500 dev images; the RBF-SVM fit scales roughly
quadratically in the number of training rows, which is trivial at this size. The GPU witness ran the CLI on a
3-class synthetic manifest only as a smoke test (`"classical"` step in `results/gpu_witness_carlpc2.json`);
that run is not a timing benchmark.

## 7. Typical failure modes on small, noisy web data

* **Background dominates.** HOG and colour are computed over the *whole* crop. A white product-shot background
  or a busy PCB contributes as much as the component, so the model learns "photo style" rather than component.
* **Scale and viewpoint.** A resistor filling the frame and a resistor in the corner give unrelated HOG vectors.
  CNNs learn some scale tolerance through pooling and augmentation; this pipeline has none (no augmentation).
* **Out-of-scope images.** Logos, game art (Nintendo Switch under `switch`) and text screenshots have strong,
  distinctive gradients and colours, so the SVM can *fit* them, but the result tells you nothing about
  components. See [`../concepts/data_quality.md`](../concepts/data_quality.md).
* **No tuned C.** The default `C=10` is a single guess; a different C may move the score by several points.

## 8. Try this

```bash
# 1. Baseline on all five dev folds
uv run python -m compclass.classical
# 2. Regularisation sweep (each run appends 5 lines to results/runs.jsonl)
for C in 0.1 1 10 100; do uv run python -m compclass.classical --C $C; done
# 3. Compare the sweep
uv run python -c "import pandas as pd; d=pd.read_json('results/runs.jsonl', lines=True); d=d[d.method=='classical']; print(d.groupby('C')[['accuracy','macro_f1']].agg(['mean','std']))"
# 4. Inspect one feature vector (shape should be (1892,))
uv run python -c "from PIL import Image; from compclass.classical import image_features; import glob; print(image_features(Image.open(sorted(glob.glob('data/v2/images/resistor/*'))[0])).shape)"
# 5. Ablation in a notebook: HOG only vs colour only — slice the vector at index 1764
```

## Results on data v2

> **Placeholder — no v2 results exist yet.** See `results/benchmark.csv` (generated from `results/runs.jsonl`) once the benchmark has run. Report mean ± std over the 5 dev folds; never quote the sealed test here.

| method | key | CV macro-F1 | CV accuracy | notes |
|---|---|---:|---:|---|
| — | — | *pending* | *pending* | |


## References

* N. Dalal, B. Triggs (2005). *Histograms of Oriented Gradients for Human Detection.* CVPR.
* M. J. Swain, D. H. Ballard (1991). *Color Indexing.* IJCV 7(1):11–32.
* C. Cortes, V. Vapnik (1995). *Support-Vector Networks.* Machine Learning 20(3):273–297.
