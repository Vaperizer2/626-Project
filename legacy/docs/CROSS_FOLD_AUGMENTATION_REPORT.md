# Cross-fold augmentation and aggregation benchmark

**Status:** complete — held-out test evaluation intentionally deferred  
**Run date:** 2026-08-14  
**Compute:** `ai-pc-04` / NVIDIA GeForce RTX 5080  
**Dataset fingerprint:** `78637c8898aa10f8`

## Result at a glance

The full current set of 14 models completed 5-fold grouped cross-validation: 70 fold fits in total. The best individual model was **ConvNeXt-Base** with an out-of-fold (OOF) macro F1 of **62.88%**. The best aggregation was **majority vote across all 14 models**, reaching **64.72% OOF macro F1**—a **+1.84 percentage-point** improvement over the best single model.

The held-out legacy test split was not loaded, evaluated, or used for model selection. The numbers below are development-pool OOF results only.

## Model leaderboard

OOF metrics are the primary comparison: every development image is scored by a model whose fold training set did not contain that image. “Fold macro F1” is the mean ± standard deviation across the five validation folds. The train–validation gap is an accuracy gap in percentage points and is diagnostic only.

| Rank | Model | Family | OOF accuracy | OOF macro F1 | OOF weighted F1 | Fold macro F1 | Train–val gap |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `convnext_base` | ConvNeXt | 64.02% | **62.88%** | 64.04% | 62.79% ± 2.49% | 32.98% ± 4.26% |
| 2 | `convnext_small` | ConvNeXt | 62.91% | 61.81% | 62.91% | 61.75% ± 1.66% | 34.85% ± 1.04% |
| 3 | `swin_b` | Swin | 61.76% | 60.66% | 61.86% | 60.65% ± 1.26% | 33.02% ± 6.55% |
| 4 | `convnext_tiny` | ConvNeXt | 60.88% | 59.66% | 60.69% | 59.70% ± 1.66% | 34.03% ± 5.36% |
| 5 | `swin_t` | Swin | 60.61% | 59.55% | 60.68% | 59.48% ± 1.76% | 33.81% ± 2.35% |
| 6 | `resnet152` | ResNet | 60.14% | 59.12% | 60.26% | 59.09% ± 2.38% | 38.84% ± 2.25% |
| 7 | `swin_v2_t` | Swin V2 | 60.17% | 58.91% | 60.18% | 58.72% ± 1.92% | 34.33% ± 3.71% |
| 8 | `efficientnet_v2_s` | EfficientNet V2 | 58.55% | 57.36% | 58.42% | 57.33% ± 2.85% | 37.43% ± 3.93% |
| 9 | `resnet101` | ResNet | 58.45% | 57.24% | 58.40% | 57.24% ± 4.08% | 38.12% ± 2.77% |
| 10 | `efficientnet_v2_m` | EfficientNet V2 | 58.18% | 57.08% | 58.25% | 57.02% ± 1.34% | 37.69% ± 2.91% |
| 11 | `densenet121` | DenseNet | 57.33% | 56.04% | 57.12% | 55.96% ± 1.62% | 40.25% ± 2.85% |
| 12 | `efficientnet_b2` | EfficientNet | 55.88% | 54.68% | 55.77% | 54.63% ± 2.04% | 39.02% ± 5.35% |
| 13 | `resnet34` | ResNet | 55.37% | 54.22% | 55.24% | 54.12% ± 2.67% | 42.29% ± 1.39% |
| 14 | `mobilenet_v3_large` | MobileNet | 53.11% | 51.71% | 52.96% | 51.74% ± 1.57% | 41.71% ± 4.61% |

## Aggregation comparison

All aggregation rows use complete OOF predictions, so no model sees the sample it predicts. The full 14-model majority vote is the strongest OOF result in this run. The full probability average is a close second and has the smallest fold-to-fold macro-F1 spread among the aggregation choices.

| Aggregation | Models | OOF accuracy | OOF macro F1 | OOF weighted F1 | Fold macro F1 |
|---|---:|---:|---:|---:|---:|
| **All-model majority vote** | 14 | **65.74%** | **64.72%** | **65.55%** | 64.58% ± 1.38% |
| All-model mean probability | 14 | 65.41% | 64.38% | 65.24% | 64.17% ± 0.97% |
| All-model mean logits | 14 | 65.20% | 64.24% | 65.08% | 64.06% ± 1.23% |
| Top-3 ConvNeXt mean probability | 3 | 64.66% | 63.60% | 64.60% | 63.54% ± 1.44% |

### Practical reading

- If the selection metric is macro F1, use the **all-14 majority vote** for this experiment.
- If downstream code needs a soft score, retain the **all-14 mean-probability** output as the runner-up, but calibrate it separately before treating it as a probability.
- Restricting the ensemble to the top three ConvNeXt models did not beat the full ensemble; the broader model-family diversity helped the OOF result.

## Evaluation design

### Leakage-safe grouped folds

- Development pool: the frozen legacy train + validation split, **2,960 images** (`2,438 + 522`).
- Legacy test: **523 images**, deliberately deferred.
- Classes: 20.
- Splitter: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
- Group key: exact file-content SHA-256, so byte-identical images cannot cross a fold boundary.
- Exact-content groups: 2,527.
- Duplicate groups: 406.
- Groups with conflicting labels: 10; these remain grouped and should be reviewed as a data-quality follow-up.
- Each fold: 2,368 training rows and 592 validation rows.

This prevents the common failure mode where duplicate or near-identical copies inflate validation scores. Each sample receives exactly one OOF prediction.

### Augmentation and regularization

The run used the `strong` / `strong-regularized` recipe, applied to training images inside each fold only:

- resize to the existing model image size;
- random horizontal flip;
- random rotation up to 20°;
- color jitter `(brightness=.2, contrast=.2, saturation=.15)`;
- random affine translation up to 5% and scale `(0.95, 1.05)`;
- ImageNet normalization;
- random erasing with `p=.15`, scale `(0.02, .08)`, ratio `(0.5, 2.0)`;
- label smoothing `0.12`;
- class-weight power `0.5`;
- fivefold strong-recipe weight decay and at least two frozen-backbone epochs where the model setting required it.

OOF inference also averaged the original and horizontally flipped logits (two-view test-time augmentation). No augmented image was inserted into a validation or test fold.

### Training contract

- Pretrained versions of the registered backbones.
- Maximum 15 epochs per fold, early stopping patience 4.
- Batch size 16; 4 data-loader workers.
- Epoch selection used fold validation macro F1; final comparison used OOF macro F1.
- Aggregations: mean probability, mean logits, majority vote, and the top-three ConvNeXt probability mean.

## Caveats and next steps

1. This is a strong-augmentation benchmark, not a paired augmentation ablation. To quantify augmentation uplift, repeat the same folds with the standard recipe and compare paired per-image OOF predictions.
2. The 10 conflicting-label duplicate groups should be adjudicated before using the test split for a final claim.
3. The test split remains untouched by design. A final test evaluation should happen once the aggregation rule is frozen.
4. The train–validation gaps remain large for several models, so the ensemble improvement should be treated as an OOF selection result rather than evidence that the individual models are fully calibrated.

## Reproducibility artifacts

- [Benchmark runner](notebooks/16_cross_fold_augmentation_benchmark.py)
- [Methodology and source notes](CROSS_FOLD_AUGMENTATION_RESEARCH.md)
- [Machine-readable report](notebooks/outputs/metrics/cross_fold_augmentation.json)
- [Summary CSV](notebooks/outputs/metrics/cross_fold_augmentation.csv)
- [OOF logits](notebooks/outputs/metrics/cross_fold_augmentation_oof.pt)
- [Training log](notebooks/outputs/metrics/cross_fold_augmentation.log)

The completed remote run used:

```bash
COMPCLASS_WORK_DIR=/home/carlai04/SEB/notebooks \
/home/carlai04/SEB/.venv/bin/python \
/home/carlai04/SEB/notebooks/16_cross_fold_augmentation_benchmark.py \
  --folds 5 --epochs 15 --patience 4 --profile paper \
  --batch-size 16 --num-workers 4 --augmentation strong \
  --report /home/carlai04/SEB/notebooks/outputs/metrics/cross_fold_augmentation.json \
  --artifact-prefix cross_fold_strong --keep-checkpoints
```

## Method references

- [`StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [`torchvision.transforms`](https://docs.pytorch.org/vision/stable/transforms.html)
- [`RandomErasing`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomErasing.html)
- [Deep Ensembles](https://arxiv.org/abs/1612.01474)
