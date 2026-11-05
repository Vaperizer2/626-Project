# Scratch CNN improvement report (model A)

Living summary of the work to improve the from-scratch CNN. It answers four questions: what the model could do
before, how the research was conducted, what was tried with what result, and where it stands. The full
chronological detail is in `docs/SCRATCH_CNN_IMPROVEMENT_LOG.md`; the untouched record of the original model is
`docs/SCRATCH_CNN_BASELINE.md`; a diagram of the network is `docs/scratch_cnn_pipeline.html`.

Status (2026-11-04): **development-fold result final; sealed test and Wikimedia holdout each scored once
(sections 7 and 8).** One outcome, confirmed on all five folds with every fold improving:

| Model | Five-fold macro-F1 | Accuracy | vs committed model A (0.3552) |
|---|---:|---:|---:|
| A, committed recipe (120 epochs) | 0.3552 | 0.3639 | - |
| **A, trained to convergence (200 epochs)** | **0.3903** | 0.3986 | +0.035 +/- 0.009 |

Everything else that was tried (resolution, mixup, width, depth, residual blocks, label smoothing, TTA,
initialisation, regularisers, EMA, squash) made no difference or made it worse.

Sealed test, one look (section 7): the 200-epoch model scores 0.3924 macro-F1 / 0.3995 accuracy, against
0.3361 / 0.3541 for the committed course run. The schedule gain carries over to the test split. On the 407-photo
Wikimedia holdout (section 8) it drops hard, to 0.2570 macro-F1 against 0.8159 for the deployed pretrained model.

## 1. Starting point

Model A is a plain VGG-style CNN (stem conv3x3 stride 2, four stages of 2 x [conv3x3-BN-ReLU] at 32/64/128/256
channels, max-pool after the first three, global average pool, dropout 0.3, linear 256 -> 20; 1.19 M parameters,
160 px input) trained from random initialisation on the 2,512-image development set of the v2 manifest.

| Measure | Committed value |
|---|---:|
| Sealed test accuracy / macro-F1 (course run, single look) | 0.3541 / 0.3361 |
| Five-fold dev macro-F1, committed research run (`4e4c6e8f6cf3`) | 0.3345, 0.3758, 0.3484, 0.3598, 0.3574 (mean 0.3552) |
| Reference pretrained models on the same test split (`results/itri626/metrics.json`) | B ConvNeXt-Tiny 0.636, C SigLIP probe 0.823 accuracy |

The gap to the pretrained models is the point of model A in the course comparison; the aim here is to make the
from-scratch baseline as strong as the data allows without changing what it is (a CNN trained from scratch).

## 2. Method

- **Selection on development folds only.** The sealed test split and the Wikimedia holdout were each scored once
  in October and are spent. Nothing in this work scores them. Final confirmation of an adopted recipe on the
  test split is a separate, one-time decision.
- **One change at a time**, against a fixed comparison recipe identified by its config hash.
- **Screen on folds 0-2, confirm on all five.** Established after entry 4 of the log: a single fold-0 run varies
  by ±2-3 pt macro-F1, so single-fold differences under 5 pt are treated as noise. Three folds give a mean and a
  standard error of the paired differences; five folds decide adoption.
- **Same hardware and software for every run** (RTX 4070 laptop, locked `uv` environment), with run records in
  `results/runs_win.jsonl` and checkpoints in `runs/win/`, kept apart from the committed research log.
- **Cost is recorded** (training time, images/s, peak VRAM) so a gain can be weighed against it.

## 3. Experiments and results

All macro-F1, validation, best epoch. "Reference" is the 200-epoch plain recipe once adopted.

| # | Lever | Change | Folds | Result vs comparison | Verdict |
|---|---|---|---|---|---|
| 1-4 | input resolution | 224 px | 0-4 | +0.002 ± 0.012 vs 120-ep baseline | no gain (fold-0 +4.4 pt was noise) |
| 2 | input resolution | 256 px | 0 | 0.342 vs 0.335 | no gain |
| 3 | regularisation | mixup 0.2 at 224 px, 120 ep | 0 | 0.349 | no gain; schedule too short |
| 5-6 | **schedule** | **200 epochs, patience 40** | 0-2 | **0.393 vs 0.353, +0.040 ± 0.016** | **adopted** |
| 6 | regularisation | mixup 0.2 at 200 ep | 0-2 | 0.393, +0.000 vs reference | no gain |
| 7 | schedule | 300 epochs ± mixup | 0-2 | +0.006 / +0.007 vs reference | noise, 1.45x time |
| 8 | capacity | width 48 (2.67 M) | 0-2 | +0.004 ± 0.013 | no gain |
| 8 | capacity | width 64 (4.73 M) | 0-2 | -0.002 ± 0.008 | no gain, peaks earlier |
| 9 | structure | residual blocks, 1/stage (1.23 M) | 0-2 | -0.011 ± 0.009 | slightly worse |
| 9 | structure | residual blocks, 2/stage (2.80 M) | 0-2 | +0.004 ± 0.008 | no gain, 1.5x time |
| 11 | loss | label smoothing 0.2 | 0-2 | -0.014 ± 0.008, worse on 3/3 | dropped |
| 12 | inference | flip test-time augmentation (existing checkpoints) | 0-2, 3 recipes | -0.008 mean over 9 checkpoints, worse on 7/9 | dropped |
| 13a | initialisation | identity (Dirac) init of same-width convs | 0-2 | +0.004 ± 0.010 | no gain |
| 13b | initialisation | PCA-whitened frozen stem (airbench) | 0-2 | -0.005 ± 0.009 | no gain, peaks earlier |
| 13c | initialisation | both | 0-2 | +0.008 ± 0.011, better on 2/3 | inside noise, not adopted |
| 14a | regularisation | vertical flip 0.5 + random erasing 0.25 | 0-2 | -0.017 ± 0.006, worse on 3/3 | dropped |
| 14b | regularisation | RandAugment magnitude 9 -> 15 | 0-2 | -0.006 ± 0.013 | no gain |
| 14c | regularisation | dropout 0.3 -> 0.5 | 0-2 | -0.019 ± 0.013 | dropped |
| 15 | analysis | per-class errors of the reference (`tools/eval_per_class.py`) | 0-2 pooled | errors diffuse, no pair > 12 % | cleaning/balancing deprioritised |
| 16 | optimisation | EMA of weights, decay 0.999 (`train.ema`) | 0-2 | -0.003 ± 0.006; curve 3x smoother, same peak | no gain |
| 17 | input | whole picture squashed to 160x160 (`aug.squash`) | 0-2 | -0.030 ± 0.008, worse on 3/3 | dropped; SigLIP's +1.6 pt does not transfer to 160 px |
| 18 | confirmation | 200-epoch reference on folds 3-4 | 0-4 | five-fold 0.3903, +0.035 ± 0.009 vs committed, 5/5 | adoption confirmed |

Per-fold numbers, hashes, best epochs and VRAM for every row are in the log.

## 4. Current best recipe

`configs/scratch_cnn.yaml` with `train.epochs: 200`, `train.patience: 40`; everything else unchanged (160 px,
width 32, dropout 0.3, no mixup, label smoothing 0.1, AdamW lr 2e-3, wd 0.05, 5 warm-up epochs, cosine).
Config hash `b0afa2aa4d60`. Five-fold macro-F1 0.4021 / 0.3898 / 0.3866 / 0.3834 / 0.3897, mean **0.3903**,
against 0.3552 for the committed recipe: +0.035 ± 0.009 SE, better on 5/5 folds (entry 18). Cost about 430 s
per fold on the laptop GPU, 435 MB VRAM.

Confirmed on all five development folds. Sealed test, one look (section 7): accuracy 0.3995, macro-F1 0.3924.
Wikimedia holdout, one look (section 8): accuracy 0.2826, macro-F1 0.2570.

## 5. What was learned

1. The committed model was under-trained, not under-sized. Its best epochs (79-110) sat just inside a patience of
   25 on a 120-epoch cosine schedule; letting it run to 200 epochs is the whole gain.
2. Once converged, every variant lands in a 0.38-0.41 band on every fold regardless of resolution, width, depth,
   skip connections or mixup. That is the signature of a data-limited model: about 2,000 training photos for 20
   classes, several of which (diode/LED, resistor/inductor) overlap visually.
3. The literature on from-scratch CNNs at this scale agrees: the best small-scale results use a plain VGG-style
   network of this shape, and their gains come from initialisation and training details (whitening stem, identity
   init, label smoothing 0.2, test-time augmentation), not from architecture.
4. Methodologically, the single most useful rule was to distrust single-fold results. Two apparent wins of 4-6 pt
   evaporated on confirmation.
5. The reference model's errors are spread evenly over all 20 classes (per-class F1 0.22-0.59, no confusion pair
   above 12 % of a class, support 56-91 per class). Label cleaning and class balancing therefore cannot be large
   levers; the network simply lacks discriminative features.
6. Stronger regularisation (vertical flips, random erasing, RandAugment 15, dropout 0.5) makes it worse, not
   better, and lowers training accuracy at the same time. The large train/val gap is therefore not classical
   overfitting that a regulariser can trade away; supervised cross-entropy on ~2,000 photos has simply run out of
   signal for this network.

## 6. What was tried after the plateau, and what remains

In order of cost, each screened on folds 0-2 at 200 epochs:

1. Label smoothing 0.2: worse on all three folds; dropped.
2. Flip test-time augmentation: tested on nine existing checkpoints, no gain; dropped.
3. Whitening initialisation of the stem and identity initialisation of same-width convolutions: tested, all
   inside noise (best mean 0.4006 for both together, +0.8 ± 1.1 pt); not adopted.
4. Stronger regularisation (vertical flip + random erasing, RandAugment 15, dropout 0.5): all worse; dropped.
5. Weight averaging (EMA 0.999): smoother curve, same peak; no gain.
6. The regularisers and EMA stayed inside noise or worse, so ~0.39-0.40 macro-F1 is what this from-scratch
   network learns from ~2,000 photos.
7. Squash input (from the hard-cases study on the pretrained model): -3 pt here; dropped (entry 17).

Remaining, all the author's decisions rather than experiments:

- Done 2026-11-04: the sealed test split scored once for the 200-epoch recipe (section 7).
- Done 2026-11-04: the course report's model A is now the 200-epoch run (imported, not retrained, with
  `tools/itri626_import_run.py`), and the study, the test look and the holdout look are a subsection and table of
  `docs/itri626_report/main.tex` and section 8 of the notebook.
- Optional, not started: self-supervised pre-training on the same training photos.

When a recipe is adopted it is confirmed on all five folds, then documented here with the command to reproduce
it. Scoring the sealed test split is a one-time step taken only after the dev-fold decision is final.

## 7. Reproduce

```
uv sync --locked --group dev
uv run python tools/build_dataset_v2.py            # once; needs the Kaggle download in data/raw/
uv run python -m compclass.train --config configs/scratch_cnn.yaml --fold 0 1 2 \
    --set train.epochs=200 train.patience=40 output.runs_dir=runs/win output.results=results/runs_win.jsonl
uv run python tools/eval_tta.py "runs/win/scratch_cnn-b0afa2aa4d60/fold*/best.pt"
```

## 8. Files

| Path | Role |
|---|---|
| `docs/SCRATCH_CNN_BASELINE.md` | original model: architecture, config, committed results (unchanged by this work) |
| `docs/SCRATCH_CNN_IMPROVEMENT_LOG.md` | chronological experiment log with every number and hash |
| `docs/SCRATCH_CNN_IMPROVEMENT_REPORT.md` | this summary |
| `docs/scratch_cnn_pipeline.html` | diagram of the forward pass and the residual variant |
| `results/runs_win.jsonl`, `results/tta_win.jsonl` | run records from the laptop |
| `src/compclass/models/scratch_cnn.py` | model; `residual`/`blocks` options added, defaults unchanged |
| `tools/eval_tta.py` | test-time augmentation scorer for saved checkpoints |

## 7. Sealed test, the one look (2026-11-04)

Taken on the author's go-ahead after the development-fold decision was final (log entry 19). The final recipe was
trained once on fold 0 (train 2,010 / val 502, seed 42, the course table's fixed split) and its best checkpoint
scored once on the 418 sealed test images. Files: `results/scratch_cnn_improvement/`.

| Model | Dev five-fold macro-F1 | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|
| A, committed course run (120 epochs) | 0.3552 | 0.3361 | 0.3541 |
| A, 200 epochs | 0.3903 | **0.3924** | **0.3995** |
| Reference: B ConvNeXt-Tiny / C SigLIP probe (course run, `results/itri626/metrics.json`) | - | 0.628 / 0.817 | 0.636 / 0.823 |

What this means:

- The schedule gain is real on the test split: +5.6 pt macro-F1 over the committed course model, consistent with
  the five-fold estimate.
- One 418-image look (about 21 images per class) has an interval of about +/- 4.5 pt, and one training run varies
  by +/- 2-3 pt (the plain fold-0 rerun validated 2.8 pt below its earlier twin of the same seed). The five-fold
  numbers remain the better estimate of expected performance; the test numbers are what the course table reports.
- The model still fails on fuse, led and resistor.

The test split is now spent for this recipe.

## 8. Wikimedia holdout, the one look (2026-11-04)

The 407 Wikimedia Commons photos of `data/web_holdout_v1` are the project's only out-of-source test: a different
site, different photographers, backgrounds and lighting from the Kaggle scrape that supplies every training and
test image. They were scored once on 2026-10-18 with the deployed SigLIP model and never used for training or
selection. The developer's archive restored the files to this laptop (names, count and fingerprint match the
recorded manifest). The frozen fold-0 checkpoint of section 7 was scored once with `tools/eval_holdout_cnn.py`
(log entry 20; files in `results/scratch_cnn_improvement/web_holdout_*`).

| Model | Test macro-F1 (418 Kaggle) | Holdout macro-F1 (407 Wikimedia) | Holdout accuracy | Drop |
|---|---:|---:|---:|---:|
| A, 200 epochs | 0.3924 | **0.2570** [0.216, 0.293] | 0.2826 | -13.5 pt |
| Deployed SigLIP probe + zero-shot head (recorded) | 0.8373 acc | 0.8159 | 0.8206 | -1.7 pt acc |

What this means:

- **Source shift is the from-scratch network's real weakness.** It loses a third of its score when the photos come
  from somewhere else; the pretrained model loses almost nothing. Features learned from one web scrape describe
  that scrape. Classes with a fixed, large-scale look (breadboard, microcontroller board, jumper wire) survive;
  small parts photographed in varied ways (diode, power supply module, relay, fuse, transformer) do not.
- **For the course report** the from-scratch story is complete in three numbers: 0.39 on the Kaggle test split,
  0.26 on photos from another source, 0.82 for the pretrained model on the same photos. Models B and C have not
  been scored on the holdout (their checkpoints are not on this machine; a full column would need the ONNX graphs
  from the release bundle and would spend their one look).

Both holdouts are now spent for this recipe. Nothing in it will change in response to these numbers.
