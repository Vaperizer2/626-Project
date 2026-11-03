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
  (c) changing what "from scratch" means: self-supervised pre-training on the same training photos, or
  distilling model C's predictions into the CNN. Option (c) alters what model A represents in the course
  comparison and needs the author's decision before it is tried.

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

### 17. Knowledge distillation from model C into the scratch CNN; 200 epochs, folds 0-2

- Decision context: the author approved trying the options that change what model A learns from ("try it").
  Distillation is tried first because it is cheap (one feature pass with SigLIP, then the normal CNN training) and
  has the strongest record in low-data regimes. **Caveat for the report:** the resulting CNN still has no
  pretrained weights and no extra images, but its training signal comes from a pretrained model, so it is no longer
  "from scratch" in the strict sense. It must be presented as a separate variant (A + distillation), not as a
  drop-in replacement for A.
- Teacher (`tools/make_teacher.py`): for each fold k, SigLIP So400m/14@384 features of the dev images (eval
  preprocessing, cached under `results/cache/features/`) and a logistic-regression probe fitted on fold k's
  training rows only, exactly as `compclass.features.run_probe` does (C chosen by grouped 3-fold CV inside the
  training rows). Its logits for every dev row go to `results/cache/teacher/probe_siglip_so400m_384_fold<k>.npz`
  with an `is_train` mask. The trainer refuses a teacher whose mask does not cover the training rows, so the
  validation fold never leaks through the teacher. The teacher's own fold-k validation score is recorded as the
  ceiling the student is being pulled towards.
- Student (`train.py`, `train.distill.{teacher,alpha,temperature}`): offline distillation with fixed per-image
  teacher logits. Loss = (1 - alpha) * CE(student, label, label smoothing 0.1) + alpha * T^2 * KL(softmax(teacher/T)
  || softmax(student/T)). The teacher logits were computed on the clean image while the student sees the augmented
  view; that mismatch is accepted (running SigLIP on every augmented batch would make each epoch ~10x slower).
  Not combined with mixup. The key exists only when set on the command line, so the reference hash is unchanged.
- Teacher scores (its own validation fold; in-sample training accuracy in brackets, which is what the student
  actually sees): fold 0 macro-F1 0.8041 (0.990), fold 1 0.8330 (0.959), fold 2 0.8236 (0.958), fold 3 0.8309
  (0.992), fold 4 0.8283 (0.961). Chosen C 0.003-0.01. Summary in
  `results/cache/teacher/probe_siglip_so400m_384_summary.json`. Feature extraction + five probes took ~3 min.
- Screens, 200 epochs, patience 40, folds 0-2: alpha 0.7 / T 4 (common defaults) and alpha 0.9 / T 2 (teacher-
  dominated, sharper targets). A 2-epoch smoke run passed. Status: queued behind entry 16. Results to be appended.
- Known weakness of this first version: the teacher's logits on the training rows are in-sample (0.96-0.99
  accuracy), so they carry less "dark knowledge" than a cross-fitted teacher would. If the first screen helps, the
  next step is out-of-fold teacher logits inside the training split.
- Results (macro-F1 of best epoch per fold):

  | Recipe | Fold 0 | Fold 1 | Fold 2 | Mean | vs reference (0.3928) | Best epochs | Train acc at best | s/fold |
  |---|---:|---:|---:|---:|---:|---|---|---:|
  | reference (`b0afa2aa4d60`) | 0.4021 | 0.3898 | 0.3866 | 0.3928 | — | 172, 165, 147 | 0.84, 0.83, 0.80 | 430 |
  | KD alpha 0.7, T 4 (`0c8852bdad85`) | 0.4181 | 0.4055 | 0.4008 | 0.4081 | **+0.015 ± 0.001 SE, better on 3/3** | 187, 157, 161 | 0.79, 0.79, 0.80 | 461 |
  | KD alpha 0.9, T 2 (`74e41b7e867b`) | 0.4044 | 0.4213 | 0.4341 | 0.4199 | **+0.027 ± 0.013 SE, better on 3/3** | 156, 143, 195 | 0.76, 0.69, 0.76 | 452 |

- Reading: the first change since the schedule that moves every fold in the same direction. The gain grows as the
  loss leans further on the teacher (alpha 0.9) and as the targets get sharper (T 2), and it comes with *lower*
  training accuracy at the best epoch (0.69-0.76 vs 0.80-0.84): the soft targets stop the network from fitting
  the hard labels of hard photos and give it the teacher's class similarities instead, which transfers. Cost is
  +5-7 % training time (one extra KL term); inference is unchanged, the student is the same 1.19 M-parameter
  network.
- Next, along the same direction: alpha 1.0 / T 1 (pure imitation of the teacher's logits, no label term),
  alpha 0.9 / T 1, and alpha 0.9 / T 2 at 300 epochs (two of the three best epochs are in the last 15, so the
  schedule may again be too short). Queued as entry 18.
- Not yet a confirmed adoption: folds 3-4 and the five-fold mean are still to come, and the result has to be
  reported as the "A + distillation" variant with the caveat in the first bullet.

### 18. Distillation, second round: teacher weight, temperature, schedule; folds 0-2

- Screens: `train.distill.alpha=1.0 temperature=1` (200 ep), `alpha=0.9 temperature=1` (200 ep),
  `alpha=0.9 temperature=2` at 300 epochs / patience 60. Teacher unchanged (entry 17).
- Also queued, after the reference's folds 3-4 (needed for the five-fold confirmation whichever setting wins):
  a cross-fitted teacher. `tools/make_teacher.py --cross-fit` scores each training row with a probe fitted on the
  other four fifths of the training split (inner GroupKFold by cluster, C fixed to the full-split choice), so the
  logits the student sees are honest predictions (train-row accuracy 0.80-0.82 instead of 0.96-0.99) and carry the
  teacher's real uncertainty. Files `results/cache/teacher/probe_siglip_so400m_384_xfit_fold<k>.npz`. Screened
  with alpha 0.9 / T 2 for a direct comparison with `74e41b7e867b`.
- Results of the three settings (macro-F1 of best epoch per fold):

  | Recipe | Fold 0 | Fold 1 | Fold 2 | Mean | vs reference (0.3928) | Best epochs | Train acc at best | s/fold |
  |---|---:|---:|---:|---:|---:|---|---|---:|
  | KD alpha 0.9, T 2, 200 ep (`74e41b7e867b`, entry 17) | 0.4044 | 0.4213 | 0.4341 | 0.4199 | +0.027 ± 0.013, 3/3 | 156, 143, 195 | 0.76, 0.69, 0.76 | 452 |
  | KD alpha 1.0, T 1, 200 ep (`7259b0d2ed89`) | 0.3989 | 0.4224 | 0.4226 | 0.4146 | +0.022 ± 0.013, 2/3 | 173, 188, 165 | 0.83, 0.78, 0.78 | 464 |
  | KD alpha 0.9, T 1, 200 ep (`341efd07de7d`) | 0.4052 | 0.4141 | 0.4245 | 0.4146 | +0.022 ± 0.010, 3/3 | 171, 134, 157 | 0.84, 0.72, 0.80 | 424 |
  | **KD alpha 0.9, T 2, 300 ep, patience 60 (`18e1b819e52c`)** | 0.4195 | 0.4193 | 0.4474 | **0.4287** | **+0.036 ± 0.013, 3/3** | 240, 152, 204 | 0.86, 0.71, 0.79 | 494 |

- Reading: temperature 2 is the sweet spot (T 4: +1.5, T 1: +2.2, T 2: +2.7 at 200 epochs); dropping the label
  term entirely (alpha 1.0) gives nothing extra and loses a fold. Unlike the plain network, where 300 epochs was
  worth at most +0.6 pt (entry 7), the distilled student keeps improving: two of three best epochs lie beyond
  epoch 200, and the mean rises by +0.9 pt to 0.4287. Against the plain network's own 300-epoch run
  (`97da28d70f41`, 0.3987) the distillation gain is +3.0 pt. The soft targets are a richer signal, so the
  network has more to learn and takes longer to learn it.
- Cross-fitted teacher (18d, `d9f06ee849c6`, alpha 0.9 / T 2, 200 epochs): fold 0 0.4059, fold 1 0.4335,
  fold 2 0.4338; mean 0.4244, +0.032 ± 0.014 vs the reference (3/3) and +0.0045 ± 0.004 vs the in-sample teacher
  at the same settings (`74e41b7e867b`, 2/3). Best epochs 183, 160, 145; train accuracy at best 0.67-0.70, the
  lowest of any run, i.e. the student is pulled furthest from the hard labels. The difference to the in-sample
  teacher is inside noise, but it is in the expected direction and the cross-fitted teacher is the principled
  choice: the student never sees a logit from a probe that was fitted on that same photo. Queued at 300 epochs on
  all five folds so the final choice is between two complete five-fold candidates (in-sample vs cross-fitted,
  both alpha 0.9 / T 2 / 300 ep).
- Working best: alpha 0.9, T 2, 300 epochs, patience 60. Queued for folds 3-4 behind entry 19 so the five-fold
  comparison against the 200-epoch reference (folds 3-4 of which are also queued) can be made. The cross-fitted
  teacher (18d) and squash (19) remain to be read before the final choice.

### 19. Whole-picture "squash" input instead of resize-and-centre-crop; 200 epochs, folds 0-2

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
- Screens: squash with the reference recipe (clean A/B against `b0afa2aa4d60`), and squash combined with the best
  distillation setting so far (alpha 0.9 / T 2) to see whether the two gains stack. Teacher logits are unchanged
  (they come from the probe's own 384 px preprocessing, not from the student's input).
- Results (macro-F1 of best epoch per fold):

  | Recipe | Fold 0 | Fold 1 | Fold 2 | Mean | vs its non-squash twin | Best epochs | Train acc at best |
  |---|---:|---:|---:|---:|---:|---|---|
  | squash, reference recipe (`34603637bab2`) | 0.3577 | 0.3596 | 0.3709 | 0.3627 | -0.030 ± 0.008 vs `b0afa2aa4d60` (0.3928), worse on 3/3 | 139, 101, 182 | 0.73, 0.58, 0.81 |
  | squash + KD alpha 0.9 / T 2 (`20ae84129e37`) | 0.3717 | 0.3893 | 0.3866 | 0.3825 | -0.037 ± 0.005 vs `74e41b7e867b` (0.4199), worse on 3/3 | 146, 156, 156 | 0.73, 0.70, 0.71 |

- Reading: a clear loss, well outside noise, with and without distillation. The finding does not transfer from
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

### 20. Five-fold confirmation of the 200-epoch reference (`b0afa2aa4d60`)

- Folds 3-4 run with the same recipe and seed as folds 0-2 (entry 6): fold 3 macro-F1 0.3834 (best epoch 161),
  fold 4 0.3897 (175).

  | Fold | 0 | 1 | 2 | 3 | 4 | Mean |
  |---|---:|---:|---:|---:|---:|---:|
  | committed 120-epoch recipe (`4e4c6e8f6cf3`) | 0.3345 | 0.3758 | 0.3484 | 0.3598 | 0.3574 | 0.3552 |
  | 200 epochs, patience 40 (`b0afa2aa4d60`) | 0.4021 | 0.3898 | 0.3866 | 0.3834 | 0.3897 | 0.3903 |

- Difference +0.035 ± 0.009 SE, better on 5/5 folds. The adoption of entry 6 is confirmed: the only change to
  the from-scratch recipe that holds up on every fold is training it to convergence. This is the five-fold number
  the distilled variant (entries 17-18) is compared against.

### 21. Five-fold confirmation of distillation, in-sample teacher, alpha 0.9 / T 2 / 300 epochs (`18e1b819e52c`)

- Folds 3-4 added to the folds 0-2 of entry 18c, same recipe and seed.

  | Fold | 0 | 1 | 2 | 3 | 4 | Mean | Accuracy |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | reference, 200 ep (`b0afa2aa4d60`) | 0.4021 | 0.3898 | 0.3866 | 0.3834 | 0.3897 | 0.3903 | 0.3986 |
  | KD in-sample teacher, 300 ep (`18e1b819e52c`) | 0.4195 | 0.4193 | 0.4474 | 0.4171 | 0.4593 | **0.4325** | 0.4407 |
  | best epochs | 240 | 152 | 204 | 235 | 292 | | |

- Difference +0.042 ± 0.010 SE macro-F1, better on 5/5 folds. Against the committed 120-epoch model (0.3552) the
  distilled student is +7.7 pt. Fold 4 peaked at epoch 292 of 300, so a still longer schedule might add a little;
  not pursued, 300 epochs is already 1.15x the reference's cost.
- This is the first complete five-fold candidate. The cross-fitted-teacher twin (entry 18d at 300 epochs, all five
  folds) is running; the final choice is made between the two in entry 22.

### 22. Final choice: cross-fitted teacher, alpha 0.9 / T 2 / 300 epochs (`fbe64b2765cb`), `configs/scratch_cnn_distill.yaml`

- Five-fold results of the two complete distillation candidates against the confirmed reference:

  | Fold | 0 | 1 | 2 | 3 | 4 | Mean macro-F1 | Mean acc | vs reference | Best epochs |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---|
  | reference, 200 ep (`b0afa2aa4d60`) | 0.4021 | 0.3898 | 0.3866 | 0.3834 | 0.3897 | 0.3903 | 0.3986 | - | 172, 165, 147, 161, 175 |
  | KD, in-sample teacher, 300 ep (`18e1b819e52c`) | 0.4195 | 0.4193 | 0.4474 | 0.4171 | 0.4593 | 0.4325 | 0.4407 | +0.042 +/- 0.010, 5/5 | 240, 152, 204, 235, 292 |
  | **KD, cross-fitted teacher, 300 ep (`fbe64b2765cb`)** | 0.4250 | 0.4284 | 0.4341 | 0.4417 | 0.4514 | **0.4361** | **0.4459** | **+0.046 +/- 0.007, 5/5** | 244, 160, 244, 229, 240 |

  Cross-fitted vs in-sample: +0.004 +/- 0.007, better on 3/5, a tie on accuracy. The cross-fitted student has the
  smaller fold spread (0.425-0.451 vs 0.417-0.459) and the cleaner story (no training photo is ever scored by a
  probe that was fitted on it), so it is the one adopted. `configs/scratch_cnn_distill.yaml` (base
  `scratch_cnn.yaml` + 300 epochs, patience 60, `train.distill` with the cross-fitted teacher) reproduces the same
  config hash `fbe64b2765cb`, so `uv run python -m compclass.train --config configs/scratch_cnn_distill.yaml
  --fold 0 1 2 3 4` is the one-line reproduction after `uv run python tools/make_teacher.py --cross-fit`.
- Per-class, pooled out-of-fold over all 2,512 dev images (`tools/eval_per_class.py`,
  `results/per_class_fbe64b2765cb_f01234.csv` vs `results/per_class_b0afa2aa4d60_f01234.csv`): macro-F1 0.3904 to
  0.4381; 18 of 20 classes improve. Largest gains: capacitor 0.27 to 0.42, ic chip 0.59 to 0.70, potentiometer
  0.48 to 0.57, heat sink 0.40 to 0.50, resistor 0.36 to 0.44, crystal oscillator 0.36 to 0.43. Two classes do not:
  led 0.34 to 0.27 (now confused with diode, a pair the teacher also confuses) and inductor coil 0.33 to 0.33.
  The errors stay diffuse; distillation lifts the whole table rather than fixing particular pairs.
- Cost: same 1.19 M-parameter network and the same 160 px input, so inference is unchanged. Training is 300
  epochs instead of 200 plus one KL term: measured 360-390 s/fold for the reference against 410-940 s/fold for
  the two 300-epoch distillation runs. The spread is time of day, not recipe: folds run in the evening (in-sample
  folds 3-4, every cross-fitted fold) took 800-940 s while the laptop was in other use, the earlier ones 410-570 s.
  Expect about 1.3x the reference on an idle GPU. Teacher preparation is a one-off 3 min (SigLIP feature pass plus
  five probes); the teacher files live under `results/cache/` (ignored by Git) and are rebuilt by
  `tools/make_teacher.py --cross-fit`.
- What this result is and is not: the student still has no pretrained weights and sees no image outside its
  training fold. But its targets come from a model pretrained on web-scale image-text data, so in the course
  comparison it is a third thing between A and C, to be labelled "A + distillation from C". The strict
  from-scratch model A remains the 200-epoch plain recipe at 0.3903.
- The sealed test split was scored once afterwards, on the author's go-ahead (entry 24). The Wikimedia holdout
  is not scored.

### 23. Distilled variant, more signal per run: 500-epoch schedule and width 48; folds 0-2

- Both start from `configs/scratch_cnn_distill.yaml` (entry 22). 23a `train.epochs=500 train.patience=100`:
  the distilled student was still rising at 300 epochs (fold 4 best at 292). 23b `model.kwargs.width=48`: with
  hard labels wider nets learned nothing extra (entry 8) because the data ran out; soft targets carry more, so
  capacity may matter now. Screened on folds 0-2 against the adopted recipe's folds 0-2 (0.4250 / 0.4284 /
  0.4341, mean 0.4292).
- Run before the sealed test is scored, so that the dev-fold decision is final first.
- 23a result (`07be7ec77876`, 500 epochs, patience 100):

  | Recipe | Fold 0 | Fold 1 | Fold 2 | Mean | Mean acc | vs 300 ep | Best epochs | Train s/fold |
  |---|---:|---:|---:|---:|---:|---:|---|---:|
  | KD 300 ep (`fbe64b2765cb`, adopted) | 0.4250 | 0.4284 | 0.4341 | 0.4292 | 0.4390 | - | 244, 160, 244 | 551-827 |
  | KD 500 ep (`07be7ec77876`) | 0.4311 | 0.4450 | 0.4415 | 0.4392 | 0.4503 | +0.010 +/- 0.003 SE, 3/3 | 444, 368, 203 | 898-1288 |

  Every fold better, and the paired spread is tight, but the mean gain of +1.0 pt sits at the edge of what three
  folds can resolve. Folds 0 and 1 peak late (444, 368), fold 2 peaks at 203 and early-stops at 304, so the longer
  cosine helps on two folds of three. Cost is 1.6x the training time of the 300-epoch recipe.
- 23b (width 48) did not run: the queue was stopped by the agent harness during fold 0 because the laptop was
  critically low on memory while the session was idle. Nothing from it is recorded; it can be re-queued.
- 23a confirmation on folds 3-4 (same hash `07be7ec77876`):

  | Recipe | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Mean acc | vs KD 300 ep | Best epochs |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---|
  | KD 300 ep (`fbe64b2765cb`, adopted) | 0.4250 | 0.4284 | 0.4341 | 0.4417 | 0.4514 | 0.4361 | 0.4459 | - | 244, 160, 244, 229, 240 |
  | KD 500 ep (`07be7ec77876`) | 0.4311 | 0.4450 | 0.4415 | 0.4319 | 0.4504 | 0.4400 | 0.4486 | +0.004 +/- 0.004 SE, 3/5 | 444, 368, 203, 353, 265 |

  Folds 3 and 4 went the other way (-1.0 and -0.1 pt), so the five-fold gain is +0.4 pt with a standard error of
  the same size: inside noise. Both are +0.050 +/- 0.006 and +0.046 +/- 0.007 over the plain reference, i.e. the
  same result at 1.3x the training time (689-1288 s/fold vs 551-941 s/fold, both ranges inflated by other laptop
  use).
- Verdict: **not adopted.** The folds 0-2 screen was a 3/3 result of the size entry 4 warned about (224 px:
  +4.4 pt on fold 0, nothing on five). The distilled student's schedule is saturated at 300 epochs just as the
  plain one's was at 200 (entry 7).
  `configs/scratch_cnn_distill.yaml` stays at 300 epochs / patience 60 and entry 22 remains the final recipe. The
  width-48 screen (23b) is still an open, unrun option; given that the schedule lever is now flat on both students,
  the expected gain is small and it is not queued.

### 24. Sealed test, the one look (2026-11-04, author's go-ahead)

- Both final recipes trained once more on fold 0 (train 2,010 / val 502, seed 42: the same fixed split as the
  course table in `results/itri626/metrics.json`) and the best checkpoint scored once on the 418 sealed test
  images via `--test`:
  `uv run python -m compclass.train --config configs/scratch_cnn.yaml --fold 0 --test --set train.epochs=200
  train.patience=40 ...` and `... --config configs/scratch_cnn_distill.yaml --fold 0 --test ...`. Outputs copied
  to `results/scratch_cnn_improvement/` (`test_*.json`, `test_preds_*.npz`, `summary.json`); `runs/win/` is
  git-ignored.

  | Model (fold-0 training) | Val macro-F1 | Val acc | **Test macro-F1** | **Test acc** | Best epoch |
  |---|---:|---:|---:|---:|---:|
  | A, committed course run (120 ep, `4e4c6e8f6cf3`) | 0.3287 | 0.3386 | 0.3361 | 0.3541 | 105 |
  | A, 200 ep (`b0afa2aa4d60`) | 0.3737 | 0.3785 | **0.3924** | **0.3995** | 183 |
  | A + distillation, 300 ep (`fbe64b2765cb`) | 0.4234 | 0.4303 | **0.3933** | **0.4115** | 243 |

- Paired bootstrap over the 418 test images (2,000 resamples), distilled minus plain: macro-F1 +0.001
  [-0.044, +0.045], accuracy +0.012 [-0.031, +0.057]. The two models agree on 47.8 % of test images; 44 images
  are right only for the distilled model, 39 only for the plain one. Distilled is better on 9 of 20 test classes.
- Reading. The schedule change carries over to the test split: +5.6 pt macro-F1 and +4.5 pt accuracy over the
  committed course model, in line with the +3.5 pt five-fold estimate. The distillation gain does not show on this
  look: +0.1 pt macro-F1, +1.2 pt accuracy, both well inside the interval. Three things limit what one look can
  say. (i) 418 images are about 21 per class, so a macro-F1 has a standard error near 2.5 pt and a paired
  difference an interval of +/- 4.5 pt, wider than the +4.6 pt five-fold gain. (ii) A single training run moves
  by 2-3 pt: this plain fold-0 rerun validated at 0.3737 where the earlier run of the same hash and seed gave
  0.4021 (fp16 AMP / cuDNN nondeterminism; the distilled rerun, 0.4234 vs 0.4250, happened to land close). The
  plain run that was scored is therefore a weak draw on validation and a strong one on test, the distilled run
  the reverse. (iii) The five-fold estimate rests on 2,512 images and five trainings, the test look on 418 and
  one. The five-fold numbers remain the better estimate of expected performance; the test numbers are what the
  course table reports, and on them the two recipes are indistinguishable.
- Consequence for the write-up: the course report may claim the schedule gain on the test split. For
  distillation it must say "+4.6 pt on five development folds, not resolvable on the 418-image test split", not
  more. Per class on test, plain is best on potentiometer (0.62), microcontroller board (0.68), switch (0.64),
  battery holder (0.58); distilled on breadboard (0.57), heat sink (0.56), crystal oscillator (0.40),
  transformer (0.32); both fail on fuse (0.17-0.18), led (0.22-0.27) and resistor (0.16-0.29).
- The sealed test split is now spent for both recipes. No further scoring; any later change to either recipe is
  a development-fold result only.
