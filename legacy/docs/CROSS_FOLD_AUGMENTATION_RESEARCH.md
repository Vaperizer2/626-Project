# Cross-fold validation, augmentation, and aggregation

## Decision for this project

The current test split is a legacy audit set, not a new tuning target. The
benchmark therefore uses the existing raw train + validation pool only (2,960
images) and leaves the 523-image legacy test split untouched.

The fold splitter is `StratifiedGroupKFold(n_splits=5, shuffle=True,
random_state=42)`, with the SHA-256 of the complete image bytes as the group
key. This preserves class proportions as far as the group constraint allows,
and keeps every exact duplicate in one fold. That is the right minimum guard
for this dataset because the prior audit found duplicate content crossing the
old partitions. It does not catch resized, recompressed, cropped, or
source-family near duplicates; those require a future perceptual/provenance
manifest.

The local preflight over the synchronized dataset fingerprint (`78637c8898aa10f8`)
found 2,527 exact-content groups among the 2,960 development rows, including
406 duplicate groups and 10 groups with conflicting labels. The run records
these counts and the fold class counts in its JSON artifact.

## Augmentation policy

Augmentation is applied by the training dataset only. Fold validation uses
deterministic resize, tensor conversion, and ImageNet normalization. The
default run uses the repository's existing `strong-regularized` recipe:

- horizontal flip, 20-degree rotation, mild color jitter;
- small translation/scale affine perturbation;
- random erasing after normalization;
- the existing weight decay, label smoothing, class weighting, and two-epoch
  backbone warm-up associated with that recipe.

The transforms are deliberately modest: a component's class must remain true
after the transformation. Torchvision documents transforms as usable for
training or inference and provides the same classification pattern used here;
its `RandomErasing` API is a train-time occlusion transform. The runner exposes
the existing standard recipe as an explicit alternative for a later ablation.

## Aggregation policy

Each model/fold produces logits for its held-out fold. Those out-of-fold (OOF)
predictions are assembled in original development-row order. The report
contains:

1. each of the 14 models individually;
2. uniform mean of the 14 softmax probabilities;
3. uniform mean of the 14 logits;
4. 14-model majority vote; and
5. a pre-specified three-ConvNeXt probability average for comparison.

Aggregation is evaluated fold by fold and again over all OOF rows. No ensemble
weight is learned from the same OOF predictions, and no legacy-test result is
used to choose an ensemble. The aggregation is therefore a clean development
comparison, not a final deployment claim.

The scikit-learn `cross_val_predict` documentation describes the same OOF
property—each sample is predicted by a model fitted without that sample—but
warns that a metric over concatenated predictions can differ from the mean of
fold metrics. The report keeps both values, with macro-F1 as the primary metric,
plus accuracy, weighted-F1, fold standard deviation, and the train/validation
accuracy gap.

## Run contract

The executable is [`notebooks/16_cross_fold_augmentation_benchmark.py`](notebooks/16_cross_fold_augmentation_benchmark.py).
The compute-node command is:

```bash
cd /home/carlai04/SEB/notebooks
../.venv/bin/python 16_cross_fold_augmentation_benchmark.py \
  --folds 5 --epochs 15 --patience 4 --profile paper \
  --batch-size 16 --num-workers 4 --augmentation strong \
  --report cross_fold_augmentation.json \
  --artifact-prefix cross_fold
```

It resumes completed model/fold pairs, writes OOF logits after every fold,
and writes both JSON and CSV summaries under `outputs/metrics/`. By default it
removes the temporary fold checkpoint after its OOF predictions are captured;
pass `--keep-checkpoints` when persistent fold weights are wanted. The test
split is not passed to `make_loaders`.

## Sources

- scikit-learn, [`StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- scikit-learn, [`cross_val_predict`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html)
- Torchvision, [image transforms and augmentation](https://docs.pytorch.org/vision/stable/transforms.html)
- Torchvision, [`RandomErasing`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomErasing.html)
- Lakshminarayanan, Pritzel, and Blundell, [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474)
- Cubuk et al., [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)
