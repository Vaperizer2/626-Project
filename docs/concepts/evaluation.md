# Evaluation: grouped cross-validation, a sealed test, and honest uncertainty

The evaluation contract lives in [`../../EXPERIMENT_LOG.md`](../../EXPERIMENT_LOG.md); this page explains *why* it
looks the way it does.

## 1. The leakage that invalidated the legacy numbers

The pre-v2 notebooks reported 0.71–0.76 validation/test accuracy. The v2 audit
(`legacy/docs/AGENT_DATASET_V2_AUDIT.md`) found that **128 / 522 (24.5 %) legacy validation rows and 115 / 523
(22.0 %) legacy test rows were byte-identical copies of training images**, and 12 duplicate groups carried
conflicting labels. A model that memorises a training image is "correct" on its copy in validation, so roughly a
quarter of the evaluation measured memory, not generalisation. The first leakage-free estimate (5-fold, exact-hash
grouped CV, 14 fine-tuned ImageNet models) was ≈63–65 % macro-F1 (`legacy/README.md`,
`legacy/docs/CROSS_FOLD_AUGMENTATION_REPORT.md`). None of the legacy numbers are comparable to v2 results.

This failure is common in web-scraped datasets: B. Barz, J. Denzler (2019), *Do We Train on Test Data? Purging
CIFAR of Near-Duplicates*, arXiv:1902.00423; S. Kapoor, A. Narayanan (2022), *Leakage and the Reproducibility Crisis
in ML-based Science*, arXiv:2207.07048.

## 2. How v2 prevents it (`tools/build_dataset_v2.py`, `data/v2/leakage_report.json`)

| step | count |
|---|---:|
| raw files | 3,530 |
| unique SHA-256 (exact duplicates collapsed: 533 groups, 587 redundant files) | 2,943 |
| excluded: byte-identical files filed under different classes | 13 |
| rows in the manifest | 2,930 |
| near-duplicate clusters (size ≥ 2) / rows in them | 91 / 234 |
| clusters overall (the grouping unit) | 2,787 |
| sealed test rows / dev rows | 418 / 2,512 |

**Near-duplicates** (re-encodes, resizes, crops, recolours) are found with perceptual hashes plus OpenCLIP
ViT-B/32 (`laion2b_s34b_b79k`) image embeddings. Two images are linked when (pHash Hamming ≤ 10 **and** cosine ≥ 0.85)
**or** cosine ≥ 0.92; union-find over the links gives clusters. The thresholds were chosen by visual inspection of
sampled pairs and err towards over-merging (leakage-safe). The largest cluster has 29 images.

**Splits are made per cluster**: `StratifiedGroupKFold(7, shuffle=True, random_state=42)` on clusters; the outer fold
whose class distribution is closest to the overall one (total-variation distance 0.00558) becomes the
sealed test set; the remaining dev rows get `StratifiedGroupKFold(5)` folds 0–4 (sizes 502, 503, 503, 503, 501). A
build **gate** refuses to write the manifest if any cluster appears in more than one of {test, fold0..4}, and
`compclass.data.load_manifest` re-checks it at load time together with the split fingerprint (`431315e87d2612cb`).
Every result line in `results/runs.jsonl` records the fingerprint; results on different fingerprints are not comparable.

## 3. Why cross-validation *and* a sealed test

* **Cross-validation (model selection).** Every choice — family, LR, augmentation, epochs — is made on the 5 dev
  folds. Each dev row is scored exactly once by a model that never saw it (or its near-duplicates), giving
  out-of-fold (OOF) predictions for all 2,512 dev rows: `runs/<name>-<hash>/fold<k>/val_preds.npz` for
  fine-tunes, `results/oof/probe_<key>.npz` and `results/oof/zeroshot_<key>.npz` for probes and zero-shot.
* **Selection bias.** After comparing many configurations on the same folds, the best CV score is optimistically
  biased — you picked the luckiest one (G. C. Cawley, N. L. C. Talbot 2010, *On Over-fitting in Model Selection and
  Subsequent Selection Bias in Performance Evaluation*, JMLR 11). The **sealed test** (418 rows) is evaluated
  **once**, for the frozen final model, to give an unbiased estimate. The code enforces this socially, not
  cryptographically: `get_test(m, confirm=True)` / `compclass.evaluate --sealed-test`, and every look is logged in
  `EXPERIMENT_LOG.md`.
* **Early stopping uses the validation fold.** `train.py` keeps the best-macro-F1 epoch on the fold it reports, which
  makes per-fold fine-tune scores slightly optimistic relative to probe/zero-shot scores (which choose nothing on the
  validation fold; the probe chooses C by an inner grouped 3-fold CV on the training rows). Keep this asymmetry in mind
  when the gap between regimes is small.

## 4. Macro-F1 vs accuracy

Accuracy counts every image equally. **Macro-F1** computes F1 = 2PR/(P+R) per class and averages the 20 values
equally, so a class the model ignores costs 1/20 of the score regardless of its size. Dev classes range from
94 (`connector`) to 152 (`switch`) images — mild
imbalance, so the two metrics are close, but macro-F1 is the primary metric (and the early-stopping criterion)
because the demo should work for every component type. `compute_metrics` also stores per-class
precision/recall/F1/support and the confusion matrix (rows = true class) with every run.

## 5. How precise are the numbers? (~2,500 dev rows)

For a proportion p measured on n independent items, the 95 % normal-approximation half-width is
1.96·√(p(1−p)/n):

| evaluation set | n | half-width at 65 % (pp) | half-width at 80 % (pp) |
|---|---:|---:|---:|
| all dev rows, pooled out-of-fold | 2512 | ±1.9 | ±1.6 |
| one CV fold | 502 | ±4.2 | ±3.5 |
| sealed test | 418 | ±4.6 | ±3.8 |

Consequences:

1. **A difference of 1–2 points between two families on pooled OOF predictions is within noise** unless a *paired*
   test says otherwise. Both models are scored on the same images, so use a paired test: McNemar's test on the
   OOF correct/incorrect table (T. G. Dietterich 1998, *Approximate Statistical Tests for Comparing Supervised
   Classification Learning Algorithms*, Neural Computation 10(7)), or a paired bootstrap over rows/clusters.
2. **Report mean ± std over the 5 folds** (the contract), but do not treat the 5 fold scores as independent samples:
   training sets overlap, so the naive standard error is too small (Y. Bengio, Y. Grandvalet 2004, *No Unbiased
   Estimator of the Variance of K-Fold Cross-Validation*, JMLR 5).
3. **Macro-F1 has no closed-form interval**; bootstrap it by resampling clusters (the grouping unit), recomputing
   macro-F1 on each resample.
4. **The sealed test is small** (418 rows ≈ 20 per class): its interval is about ±4–5 pp; per-class test
   numbers are very noisy.

## 6. What the numbers measure

704 dev rows and 112 of 418 test rows carry the `ood_suspect` flag
([`data_quality.md`](data_quality.md)). Until they are reviewed, CV and test scores measure "accuracy on the
scraped web distribution, including off-topic images", not "accuracy on electronic components". Report metrics
on the flagged and unflagged subsets separately (the flags are in the manifest rows) so readers can see both.

## 7. Try this

```bash
# per-family CV summary (mean ± std over folds)
uv run python -c "import pandas as pd; d=pd.read_json('results/runs.jsonl', lines=True); print(d.groupby(['method','name'])['macro_f1'].agg(['mean','std','count']).sort_values('mean'))"
# re-evaluate a checkpoint on its validation fold (does NOT touch the test split)
uv run python -m compclass.evaluate --checkpoint runs/<name>-<hash>/fold0/best.pt
# the sealed test — once, for the final frozen model only; log it in EXPERIMENT_LOG.md
uv run python -m compclass.evaluate --checkpoint runs/<name>-<hash>/fold0/best.pt --sealed-test
# the leakage gate's own numbers
uv run python -c "import json; print(json.dumps(json.load(open('data/v2/leakage_report.json'))['counts'], indent=1))"
```
