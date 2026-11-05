# Scratch CNN improvement log (model A)

Chronological, append-only record of the attempt to improve the from-scratch CNN. Each entry states what was
tried, why, the exact configuration, the measured result and the decision taken. The starting point is the
committed course model documented in `docs/SCRATCH_CNN_BASELINE.md`; the running summary is
`docs/SCRATCH_CNN_IMPROVEMENT_REPORT.md`.

Conventions:

- Hardware: Windows 11 laptop, RTX 4070 Laptop GPU (8 GB), torch 2.11.0+cu128, same `uv.lock` as the Linux PC.
- Data: v2 manifest, fingerprint `431315e87d2612cb`, rebuilt locally from the Kaggle download. Only the five
  development folds are scored. The sealed 418-image test split and the Wikimedia holdout are never touched.
- Metric: validation macro-F1 of the best epoch, per fold. Means are over the folds listed. "SE" is the standard
  error of the per-fold differences against the comparison recipe.
- Run records: `results/runs_win.jsonl` (one JSON line per fold); checkpoints: `runs/win/<name>-<hash>/fold<k>/`.
  The 12-hex `config_hash` identifies a recipe exactly; it is printed by `train.py` and stored in every record.
- Every run is `uv run python -m compclass.train --config configs/scratch_cnn.yaml --fold <k...> --set <overrides>
  output.runs_dir=runs/win output.results=results/runs_win.jsonl`. Only the `--set` overrides are listed below.

## 2026-11-03

### 0. Environment and baseline reproduction

- `pyproject.toml` torch/torchvision source markers widened from Linux-only to Linux or Windows so the CUDA
  wheels install on the laptop; `uv.lock` re-locked. `tools/build_dataset_v2.py` patched to tolerate the untracked
  legacy split file; the rebuild reproduced the committed manifest fingerprint.
- Baseline recipe (`4e4c6e8f6cf3`, 120 epochs, patience 25, 160 px) rerun on fold 0: val acc 0.3506, macro-F1
  0.3436 in 257 s at 925 img/s. Committed research run on the same fold: 0.3345; course run: 0.3287. The setup
  reproduces the baseline within normal seed variation.
- Committed per-fold baseline used throughout (from `results/phase2/runs_fixed.jsonl`):
  folds 0-4 macro-F1 0.3345, 0.3758, 0.3484, 0.3598, 0.3574 (mean 0.3552; folds 0-2 mean 0.3529).

### 1. Input resolution 224 px, fold 0 (`model.kwargs.input_size=224`, hash `ef722b62e00e`)

- Why: 160 px was the obvious suspect; small parts (diodes, LEDs, crystals) lose detail.
- Result: fold 0 macro-F1 0.3784 vs 0.3345 (+4.4 pt). Looked like a win.
- Decision: confirm on the remaining folds before adopting (entry 4).

### 2. Input resolution 256 px, fold 0

- Result: 0.342. No gain over 160 px, 2x the cost of 224 px.
- Decision: drop 256 px.

### 3. 224 px + mixup 0.2, 120 epochs, fold 0 (`aug.mixup=0.2`)

- Result: 0.349, with the lowest validation loss so far (2.308) and validation still rising at epoch 120.
- Decision: the schedule looked too short; retest with a longer one (entry 5).

### 4. 224 px on folds 1-4 (confirmation of entry 1)

| Fold | 160 px | 224 px | Diff |
|---:|---:|---:|---:|
| 0 | 0.3345 | 0.3784 | +0.044 |
| 1 | 0.3758 | 0.3626 | -0.013 |
| 2 | 0.3484 | 0.3444 | -0.004 |
| 3 | 0.3598 | 0.3333 | -0.027 |
| 4 | 0.3574 | 0.3679 | +0.011 |
| mean | 0.3552 | 0.3573 | +0.002 ± 0.012 SE, 2/5 folds |

- Verdict: no reliable gain; the fold-0 result was single-run noise. Cost would have been 1.3x time, 1.9x VRAM.
- Lesson adopted: a single fold-0 run varies by ±2-3 pt. Screen on folds 0-2; confirm anything promising on all
  five; treat single-fold effects under 5 pt as noise.

### 5. 224 px + mixup 0.2, 200 epochs, patience 40, fold 0

- Result: 0.404, best epoch 199 and still rising. +6 pt over the fold-0 baseline, above the noise band.
- Problem: schedule length, resolution and mixup confounded. Separated in entry 6.

### 6. Schedule length vs mixup at 160 px, folds 0-2 (`train.epochs=200 train.patience=40 [aug.mixup=0.2]`)

| Recipe | Fold 0 | Fold 1 | Fold 2 | Mean | vs baseline 0.3529 | Best epochs |
|---|---:|---:|---:|---:|---:|---|
| baseline 120 ep (committed) | 0.3345 | 0.3758 | 0.3484 | 0.3529 | | 110, 98, 79 |
| 200 ep, no mixup (`b0afa2aa4d60`) | 0.4021 | 0.3898 | 0.3866 | 0.3928 | +0.040 ± 0.016 SE | 172, 165, 147 |
| 200 ep, mixup 0.2 (`06ad9e7a1ed5`) | 0.3768 | 0.4097 | 0.3918 | 0.3928 | +0.040 ± 0.003 SE | 181, 185, 183 |

- Verdict: the longer schedule alone explains the gain. The committed 120-epoch cosine schedule stopped the model
  before it converged (best epochs 79-110 against a patience of 25). Mixup adds nothing at 200 epochs.
- Decision: **adopt `train.epochs: 200`, `train.patience: 40`** as the new reference recipe. All later
  comparisons are against `b0afa2aa4d60` (mean 0.3928 on folds 0-2).

### 7. 300 epochs, patience 60, with and without mixup, folds 0-2

| Recipe | Fold 0 | Fold 1 | Fold 2 | Mean | vs 200 ep | Best epochs |
|---|---:|---:|---:|---:|---:|---|
| 300 ep, mixup 0.2 (`dbc15047d48d`) | 0.4093 | 0.3824 | 0.4088 | 0.4002 | +0.007 | 258, 146, 232 |
| 300 ep, no mixup (`97da28d70f41`) | 0.4060 | 0.3829 | 0.4071 | 0.3987 | +0.006 | 197, 158, 246 |

- Verdict: at most +0.7 pt, inside noise, for about 45 % more training time. The schedule gain saturates at
  200 epochs. Mixup makes no difference at either length.
- Decision: keep 200 epochs, no mixup.

### 8. Capacity: width 48 and 64, 200 epochs, folds 0-2 (`model.kwargs.width=48|64`)

| Recipe | Params | Fold 0 | Fold 1 | Fold 2 | Mean | vs width 32 | Best epochs | VRAM |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| width 32 (`b0afa2aa4d60`) | 1.19 M | 0.4021 | 0.3898 | 0.3866 | 0.3928 | | 172, 165, 147 | 435 MB |
| width 48 (`bea1d524aef6`) | 2.67 M | 0.3828 | 0.3952 | 0.4123 | 0.3968 | +0.004 ± 0.013 SE | 160, 143, 164 | 648 MB |
| width 64 (`d967bc9bff50`) | 4.73 M | 0.3902 | 0.3828 | 0.3996 | 0.3909 | -0.002 ± 0.008 SE | 162, 175, 177 | 833 MB |

- Verdict: 2.2x and 4x the parameters change nothing; wider networks peak earlier, i.e. overfit sooner.
- Decision: keep width 32. The network is data-limited, not capacity-limited.

### 9. Structure: residual blocks, 200 epochs, folds 0-2 (`model.kwargs.residual=true [model.kwargs.blocks=2]`)

- Code: `src/compclass/models/scratch_cnn.py` gained an optional `residual` mode (each stage's conv pair becomes
  a ResNet basic block, `y = ReLU(x + F(x))`, 1x1 projection where channels change, last BN gamma zero-initialised
  so each block starts as the identity) and a `blocks` count. The plain network and its config hash are unchanged;
  smoke tests pass.

| Recipe | Params | Fold 0 | Fold 1 | Fold 2 | Mean | vs plain | Best epochs | VRAM |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| plain width 32 (`b0afa2aa4d60`) | 1.19 M | 0.4021 | 0.3898 | 0.3866 | 0.3928 | | 172, 165, 147 | 435 MB |
| residual, 1 block/stage (`85656a2a9540`) | 1.23 M | 0.3748 | 0.3851 | 0.3869 | 0.3823 | -0.011 ± 0.009 SE | 131, 154, 156 | 460 MB |
| residual, 2 blocks/stage (`e0d34a6f948c`) | 2.80 M | 0.3900 | 0.4022 | 0.3977 | 0.3967 | +0.004 ± 0.008 SE | 147, 168, 154 | 625 MB |

- Verdict: parameter-matched skips are slightly worse; doubling depth with skips lands on the plain number for
  1.5x the time. An 8-16 layer network has no optimisation problem for residuals to solve.
- Decision: keep the plain network. Architecture (resolution, width, depth, skips) is exhausted as a lever.

### 10. Literature check (web search)

- CIFAR-10 "airbench" (Jordan 2024, arXiv 2404.00498): 94 % with a plain VGG-style net of 1.97 M parameters
  and no residuals, almost the shape of model A. Its ablation credits the training recipe, not the structure:
  patch-whitening initialisation of the first layer (epochs to 94 %: 45 to 21), identity initialisation of later
  convolutions (to 18), BatchNorm-bias learning rate x64 (to 13.5), lookahead (to 12), flip and multi-crop
  test-time augmentation (to 10.8), label smoothing 0.2, Nesterov SGD, triangular LR.
- Small-data from-scratch comparisons (arXiv 2601.04727, 2601.04352): a small custom CNN beats ResNet-18 and
  VGG-16 trained from scratch on 4 of 5 datasets of 1k-30k images; VGG-16 last every time.
- ConvNeXt-style blocks train unstably from scratch on small data (arXiv 2204.05044); not a candidate.
- arXiv 2210.05657: auxiliary fully connected layers during training, distilled away at test time; up to 16 %
  relative gain in low-data regimes. Complex; last resort.
- Decision: test the recipe levers in order of cost: label smoothing 0.2 (flag), flip TTA (evaluation only,
  on existing checkpoints), whitening and identity initialisation (code).

### 11. Label smoothing 0.2, 200 epochs, folds 0-2 (`train.label_smoothing=0.2`, hash `52a716587386`)

- Why: the airbench recipe uses 0.2; the baseline uses 0.1. Smoothing is also a common defence against noisy
  web-scraped labels.

| Recipe | Fold 0 | Fold 1 | Fold 2 | Mean | vs reference | Best epochs |
|---|---:|---:|---:|---:|---:|---|
| label smoothing 0.1 (`b0afa2aa4d60`) | 0.4021 | 0.3898 | 0.3866 | 0.3928 | | 172, 165, 147 |
| label smoothing 0.2 (`52a716587386`) | 0.3736 | 0.3776 | 0.3851 | 0.3788 | -0.014 ± 0.008 SE | 146, 187, 148 |

- Verdict: worse on all three folds. With 20 classes and a model that is already under-confident (validation
  loss around 2.3), spreading more of the target mass over the wrong classes blunts the signal further.
- Decision: keep label smoothing 0.1.

### 12. Flip test-time augmentation on existing 200-epoch checkpoints (`tools/eval_tta.py`)

- Method: the saved best checkpoint is run on each validation image and on its mirror image; the two softmax
  vectors are averaged. No retraining, so the comparison is exact (same weights, fold and images). Records in
  `results/tta_win.jsonl`. The plain column reproduces the recorded macro-F1 of each run to four decimals.

| Checkpoint | Fold 0 | Fold 1 | Fold 2 | Mean diff (TTA - plain) |
|---|---:|---:|---:|---:|
| plain width 32 (`b0afa2aa4d60`) | 0.4021 -> 0.4144 | 0.3898 -> 0.3697 | 0.3866 -> 0.3793 | -0.005 |
| width 48 (`bea1d524aef6`) | 0.3828 -> 0.3837 | 0.3952 -> 0.3826 | 0.4123 -> 0.4122 | -0.004 |
| residual 2 blocks (`e0d34a6f948c`) | 0.3901 -> 0.3776 | 0.4022 -> 0.3913 | 0.3975 -> 0.3885 | -0.011 |

- Result: flip TTA lowered macro-F1 on 7 of 9 checkpoints, mean -0.008. Accuracy moved the same way.
- Caveat: each best checkpoint was selected by its plain validation score, which biases the plain column
  upward by roughly the fold-to-fold noise; TTA has no such advantage here. Even allowing for that, there is no
  sign of the gain the CIFAR literature reports. Training already uses horizontal flips, so the network is close
  to mirror-invariant and the second view adds little information; where it disagrees, it is as often wrong.
- Decision: drop flip TTA. Not worth the doubled inference cost.

### 13. Initialisation: identity init, whitened stem, both; 200 epochs, folds 0-2

- Code: `src/compclass/models/scratch_cnn.py` gained `identity_init` (Dirac initialisation of every 3x3 conv
  whose width does not change, so each stage starts as a pass-through) and `whiten_stem` (stem filters set to
  the PCA-whitening basis of 3x3 colour patches from eight training batches: top 16 eigenvectors and their
  negatives, scaled by 1/sqrt(eigenvalue); the stem's 864 weights are then frozen, as in airbench).
  `train.py` calls `model.init_from_data(train_dl, device)` once the loader exists and honours
  `model.frozen_param_ids`. Defaults unchanged; smoke tests pass; a 2-epoch smoke run confirmed the stem filters
  are antisymmetric pairs and 864 parameters are frozen.
- Overrides: `model.kwargs.identity_init=true`, `model.kwargs.whiten_stem=true`, and both together.
- Results (macro-F1 of best epoch per fold; reference is the 200-epoch plain run `b0afa2aa4d60`):

  | Recipe | Fold 0 | Fold 1 | Fold 2 | Mean | vs reference (0.3928) | Best epochs | VRAM |
  |---|---:|---:|---:|---:|---:|---|---:|
  | reference, default init (`b0afa2aa4d60`) | 0.4021 | 0.3898 | 0.3866 | 0.3928 | — | 172, 165, 147 | 435 MB |
  | identity init (`36c03cfccfd1`) | 0.3913 | 0.3893 | 0.4107 | 0.3971 | +0.004 ± 0.010 SE, better on 1/3 | 155, 150, 153 | 435 MB |
  | whitened frozen stem (`824733f758e3`) | 0.3815 | 0.3982 | 0.3850 | 0.3882 | -0.005 ± 0.009 SE, better on 1/3 | 165, 126, 117 | 424 MB |
  | both (`e2975dc659a5`) | 0.3945 | 0.4180 | 0.3892 | 0.4006 | +0.008 ± 0.011 SE, better on 2/3 | 194, 175, 145 | 424 MB |

- Reading: all three land inside the noise band. The combined recipe has the highest mean seen so far on these
  folds (0.4006) but the gain is 0.7 standard errors and rests on one fold (fold 1 +2.8 pt, the other two within
  ±0.8 pt). The whitened stem alone peaks 30-50 epochs earlier, consistent with airbench's claim that it speeds
  up convergence; on a 200-epoch schedule with early stopping that speed is not needed, and the final accuracy is
  unchanged. Train accuracy at the stop is 0.91-0.93 against 0.40 val in every run: the gap is overfitting, which
  no initialisation can fix.
- Decision: do not adopt. Keep `b0afa2aa4d60` as the reference. The initialisation code stays available behind
  its flags (defaults off, config hash of the plain recipe unchanged). The recipe-side list from the literature is
  now exhausted (label smoothing, TTA, whitening, identity); the next lever is the data side, starting with the
  train/val gap (stronger augmentation) and the per-class error structure.

### 14. Regularisation against the train/val gap; 200 epochs, folds 0-2

- Motivation: every converged run reaches 0.91-0.93 train accuracy against ~0.40 val, i.e. the network memorises
  the ~2,000 training photos. The effective augmentation of the reference recipe (from the run record) is
  RandomResizedCrop scale 0.35-1.0, horizontal flip 0.5, RandAugment n=2 m=9 (inherited from `base.yaml`),
  colour jitter 0.2, no vertical flip, no random erasing, no mixup. Dropout 0.3 before the classifier.
- Screens (each `train.epochs=200 train.patience=40`, otherwise the reference recipe):
  - 14a `aug.vflip=0.5 aug.random_erasing=0.25`: components have no canonical orientation, so vertical flips are
    label-preserving; random erasing forces the use of more than one cue per photo.
  - 14b `aug.randaugment.m=15`: stronger photometric/geometric distortion.
  - 14c `model.kwargs.dropout=0.5`: more regularisation at the classifier.
- Results (macro-F1 of best epoch per fold; "train acc" is the augmented-batch training accuracy at that epoch):

  | Recipe | Fold 0 | Fold 1 | Fold 2 | Mean | vs reference (0.3928) | Best epochs | Train acc at best |
  |---|---:|---:|---:|---:|---:|---|---|
  | reference (`b0afa2aa4d60`) | 0.4021 | 0.3898 | 0.3866 | 0.3928 | — | 172, 165, 147 | 0.84, 0.83, 0.80 |
  | 14a vflip 0.5 + erasing 0.25 (`f1bbc43a0f91`) | 0.3784 | 0.3671 | 0.3817 | 0.3757 | -0.017 ± 0.006 SE, worse on 3/3 | 145, 134, 142 | 0.56, 0.51, 0.57 |
  | 14b RandAugment m=15 (`7571d0e1baeb`) | 0.3737 | 0.4064 | 0.3807 | 0.3869 | -0.006 ± 0.013 SE, better on 1/3 | 139, 189, 197 | 0.66, 0.77, 0.78 |
  | 14c dropout 0.5 (`715eb49abefa`) | 0.3587 | 0.3918 | 0.3705 | 0.3737 | -0.019 ± 0.013 SE, better on 1/3 | 122, 140, 197 | 0.61, 0.70, 0.77 |

- Reading: every extra regulariser lowers validation macro-F1, and the strongest one (14a) is worse on all three
  folds by a margin outside noise. Training accuracy at the best epoch drops with it (0.51-0.57 against ~0.8 for
  the reference) while validation does not improve: making the task harder for the network does not make it
  generalise better, it just learns less. That is the behaviour of a network that is short of discriminative
  features, not of one that memorises; the 0.9 train / 0.4 val gap at the end of training is a symptom, not the
  lever. Combined with entries 8-9 (more width/depth also does nothing), the plain network has reached what it can
  extract from ~2,000 photos with supervised cross-entropy alone.
- Decision: keep the reference augmentation (RandomResizedCrop 0.35-1.0, hflip, RandAugment 2/9, colour jitter
  0.2) and dropout 0.3. Do not add vertical flips, random erasing or stronger RandAugment.

### 15. Analysis: where the reference model's errors are (`tools/eval_per_class.py`)

- Method: the three fold checkpoints of `b0afa2aa4d60` scored on their own validation folds and pooled
  (1,508 out-of-fold predictions, accuracy 0.4012, macro-F1 0.3934). Table in
  `results/per_class_b0afa2aa4d60_f012.csv`.
- Per-class F1 ranges from 0.22 (capacitor) to 0.59 (jumper wire). The weakest six are capacitor, connector,
  fuse, diode, relay and transistor (F1 0.22-0.29); the strongest are jumper wire, ic chip, microcontroller board,
  breadboard and switch (0.51-0.59). Class support is 56-91 images per class in the pooled set, so imbalance is
  mild and class-balanced sampling is not a lever.
- The errors are diffuse. The single most frequent confusion (jumper wire -> inductor coil, relay -> diode) is
  9 images, 11 % of its class; no pair exceeds 12 %. Every class's errors are spread over 10-15 other classes.
  This is not the picture of a few mislabeled or visually identical pairs that label cleaning would fix; it is a
  network whose features are not discriminative enough for any class. Cleaning the 18 conflict groups in
  `data/v2/review/conflicts.csv` (already excluded from the manifest) or other suspects cannot move macro-F1 by
  more than a fraction of a point.
- Consequence: data-side cleaning is deprioritised. What remains for a from-scratch network on ~2,000 photos is
  (a) stronger regularisation (entry 14, running), (b) weight averaging / sharpness-aware optimisation, and
  (c) changing what "from scratch" means, such as self-supervised pre-training on the same training photos.
  Option (c) alters what model A represents in the course comparison and needs the author's decision before it
  is tried.

### 16. Weight averaging: EMA 0.999; 200 epochs, folds 0-2

- Code: `train.py` gained an optional exponential moving average of the weights (`train.ema=<decay>`, timm
  `ModelEmaV3` with warm-up, updated after every optimizer step). When set, the averaged model is the one that is
  validated each epoch and saved to `best.pt`, so early stopping and the recorded score refer to it. Not in any
  config file, so the reference hash `b0afa2aa4d60` is unchanged; a 2-epoch smoke run passed.
- Why: the validation curve of every 200-epoch run is noisy from epoch to epoch (±1-2 pt), and the best epoch is
  a lucky draw from that noise. Averaging weights over the last ~30 epochs (decay 0.999 at 31 steps/epoch)
  smooths the curve and usually gives a slightly better and more reproducible endpoint.
- Override: `train.ema=0.999`. Hash `d0a915876c69`.
- Results: fold 0 0.3898 (best epoch 196), fold 1 0.3827 (114), fold 2 0.3957 (120); mean 0.3894,
  -0.003 ± 0.006 SE vs the reference, better on 1/3. The averaging does what it should to the curve: the
  epoch-to-epoch jitter of validation macro-F1 over the last 50 epochs falls from 0.008-0.013 (reference) to
  0.003-0.004, but the smoothed curve peaks at the same height. The reference's "best epoch" is therefore not a
  lucky spike; the plateau is real.
- Decision: not adopted. EMA stays available for the final confirmation runs, where a reproducible endpoint is
  worth more than a lucky one, but it is not an accuracy lever here.

### 17. Whole-picture "squash" input instead of resize-and-centre-crop; 200 epochs, folds 0-2

- Origin: the hard-cases study on branch `research/hard-cases` (EXPERIMENT_LOG.md, 2026-11-03) found that feeding
  the deployed SigLIP model the whole picture squashed to a square, instead of a short-side resize and centre crop,
  lifts its five-fold dev CV by +1.6 pt, entirely from non-square photos (62 % of dev). The scratch CNN evaluates on
  a centre crop of 87.5 % and trains on random-resized crops, so it discards the same borders. The author asked for
  the same test on the CNN.
- Code: `aug.squash=true` in `src/compclass/data.py`. Evaluation: `Resize((160, 160))`, aspect ratio ignored, no
  crop. Training: the same square resize first, then the usual RandomResizedCrop / flip / RandAugment / jitter on
  the squashed picture, so train and eval geometry agree. `tools/eval_tta.py` and `tools/eval_per_class.py` now
  build the evaluation transform from the checkpoint's own `aug`, so squash checkpoints are scored as validated.
  Default off; reference hash unchanged; smoke run and the 30 tests pass.
- Deployment note: `PreprocessSpec` (what the ONNX export and the Rust demo read) is not changed by this flag. If
  squash is adopted the export metadata must say so, otherwise the demo would feed the model a centre crop.
- Screen: squash with the reference recipe (clean A/B against `b0afa2aa4d60`).
- Results (macro-F1 of best epoch per fold):

  | Recipe | Fold 0 | Fold 1 | Fold 2 | Mean | vs its non-squash twin | Best epochs | Train acc at best |
  |---|---:|---:|---:|---:|---:|---|---|
  | squash, reference recipe (`34603637bab2`) | 0.3577 | 0.3596 | 0.3709 | 0.3627 | -0.030 ± 0.008 vs `b0afa2aa4d60` (0.3928), worse on 3/3 | 139, 101, 182 | 0.73, 0.58, 0.81 |

- Reading: a clear loss, well outside noise. The finding does not transfer from
  the 384 px SigLIP backbone to a 160 px network for two reasons that the numbers fit. First, resolution: the
  squash maps the whole photo onto 160x160, so the component itself ends up smaller in pixels than under the
  resize-and-crop, and entry 1-2 already showed this network gains nothing from more pixels but here it loses
  from fewer. Second, geometry: the squash changes the aspect ratio by up to the photo's own ratio, and a small
  network trained from scratch has to learn that invariance from ~2,000 photos, whereas SigLIP has already seen
  every distortion. The borders that squash preserves carry little for this model (its errors are diffuse, entry
  15), so there is nothing to offset the two costs.
- Decision: drop squash for the scratch CNN. Keep the resize-and-crop evaluation, which also keeps the export
  metadata and the demo preprocessing unchanged. The result is still useful for the course report as a case
  where an input-pipeline gain on the pretrained model does not carry over to the small one.

### 18. Five-fold confirmation of the 200-epoch reference (`b0afa2aa4d60`)

- Folds 3-4 run with the same recipe and seed as folds 0-2 (entry 6): fold 3 macro-F1 0.3834 (best epoch 161),
  fold 4 0.3897 (175).

  | Fold | 0 | 1 | 2 | 3 | 4 | Mean |
  |---|---:|---:|---:|---:|---:|---:|
  | committed 120-epoch recipe (`4e4c6e8f6cf3`) | 0.3345 | 0.3758 | 0.3484 | 0.3598 | 0.3574 | 0.3552 |
  | 200 epochs, patience 40 (`b0afa2aa4d60`) | 0.4021 | 0.3898 | 0.3866 | 0.3834 | 0.3897 | 0.3903 |

- Difference +0.035 ± 0.009 SE, better on 5/5 folds. The adoption of entry 6 is confirmed: the only change to
  the from-scratch recipe that holds up on every fold is training it to convergence.

### 19. Sealed test, the one look (2026-11-04, author's go-ahead)

- The final recipe trained once more on fold 0 (train 2,010 / val 502, seed 42: the same fixed split as the
  course table in `results/itri626/metrics.json`) and the best checkpoint scored once on the 418 sealed test
  images via `--test`:
  `uv run python -m compclass.train --config configs/scratch_cnn.yaml --fold 0 --test --set train.epochs=200
  train.patience=40 ...`. Outputs copied to `results/scratch_cnn_improvement/` (`test_*.json`,
  `test_preds_*.npz`, `summary.json`); `runs/win/` is git-ignored.

  | Model (fold-0 training) | Val macro-F1 | Val acc | **Test macro-F1** | **Test acc** | Best epoch |
  |---|---:|---:|---:|---:|---:|
  | A, committed course run (120 ep, `4e4c6e8f6cf3`) | 0.3287 | 0.3386 | 0.3361 | 0.3541 | 105 |
  | A, 200 ep (`b0afa2aa4d60`) | 0.3737 | 0.3785 | **0.3924** | **0.3995** | 183 |

- Reading. The schedule change carries over to the test split: +5.6 pt macro-F1 and +4.5 pt accuracy over the
  committed course model, in line with the +3.5 pt five-fold estimate. Two things limit what one look can say.
  (i) 418 images are about 21 per class, so a macro-F1 has a standard error near 2.5 pt. (ii) A single training
  run moves by 2-3 pt: this fold-0 rerun validated at 0.3737 where the earlier run of the same hash and seed gave
  0.4021 (fp16 AMP / cuDNN nondeterminism). The five-fold numbers remain the better estimate of expected
  performance; the test numbers are what the course table reports.
- Per class on test, the model is best on potentiometer (0.62), microcontroller board (0.68), switch (0.64) and
  battery holder (0.58), and fails on fuse, led and resistor.
- The sealed test split is now spent for this recipe. Any later change to it is a development-fold result only.

### 20. Wikimedia holdout, the one look (2026-11-04, author's go-ahead)

- The 407 photos of `data/web_holdout_v1` (20 classes, Wikimedia Commons, collected 2026-10-18, never used for
  training or selection; the deployed SigLIP model's recorded out-of-source result is accuracy 0.8206 / macro-F1
  0.8159) were restored to this laptop from the developer's archive
  `data/raw/626-non-kaggle-images-2026-11-04/` (all 407 files present under the manifest's names, same fingerprint
  `5458af44e043b3aa`, JPEG, long side <= 512, aspect ratios equal to the recorded originals). The frozen fold-0
  checkpoint of entry 19 was scored once with the new `tools/eval_holdout_cnn.py` (`--confirm`), eval
  preprocessing unchanged. Outputs in `results/scratch_cnn_improvement/web_holdout_*`.

  | Model (fold-0 training) | Test macro-F1 | Test acc | **Holdout macro-F1** | **Holdout acc** | Holdout macro-F1 95 % CI |
  |---|---:|---:|---:|---:|---|
  | A, 200 ep (`b0afa2aa4d60`) | 0.3924 | 0.3995 | **0.2570** | **0.2826** | 0.216-0.293 |
  | Deployed SigLIP probe + zero-shot (recorded 2026-10-18) | 0.8373 (acc) | | 0.8159 | 0.8206 | |

- Reading. (i) Source shift is where the from-scratch network really loses: -13.5 pt macro-F1 from test to
  holdout, against -2.1 pt for the deployed pretrained model (0.8373 to 0.8206 accuracy). A network that learned
  its features from one web scrape does not carry them to photos taken elsewhere; the pretrained features do.
  (ii) Per class the model collapses on diode (0.39 to 0.00), power supply module (0.39 to 0.05), relay, switch,
  battery holder and potentiometer (all 0.58-0.64 to 0.20-0.33), while breadboard (0.80), microcontroller board
  (0.55) and jumper wire (0.49) survive: the classes with a fixed, large-scale appearance. Top confusions are
  diode -> switch (7) and switch -> potentiometer (7).
- For the course report: the honest three-line summary of model A is now "0.39 on the Kaggle test split, 0.26 on
  photos from another source, 0.82 for the pretrained model on the same photos". The holdout is spent for this
  recipe; models B and C of the course comparison have not been scored on it (their checkpoints are not on this
  machine; the ONNX graphs are in the GitHub release bundle).
