# Scratch CNN (model A) — baseline record

Snapshot of the from-scratch CNN as recorded on 2026-10-20 for the ITRI626 course comparison, kept as the
reference point for any attempt to improve it. All numbers below are copied from committed result files;
nothing was re-run. Compare future variants against this file.

## Where the CNN is defined and configured

| What | File |
|---|---|
| Architecture | `src/compclass/models/scratch_cnn.py` (`ScratchCNN`, registered as `scratch_cnn`) |
| Training recipe | `configs/scratch_cnn.yaml`, inheriting every other knob from `configs/base.yaml` |
| Training loop | `src/compclass/train.py` (AdamW, warm-up + cosine, AMP, early stopping on val macro-F1) |
| Course run driver | `scripts/itri626_experiments.py` (entry `A_scratch_cnn`, fixed split, seed 42, val fold 0) |
| Recorded metrics | `results/itri626/metrics.json`, `per_class.csv`, `histories.json`, `hyperparameters.csv`, `efficiency.csv` |
| Checkpoint (not in Git) | `runs/itri626/scratch_cnn-4e4c6e8f6cf3/fold0/best.pt` |
| Figures | `results/itri626/figures/fig04_learning_curves.png`, `fig06_confusion_A_scratch_cnn.png`, `fig08_gallery_A_scratch_cnn.png` |

## Architecture (input 3 x 160 x 160, width w = 32)

Diagram of the forward pass, receptive fields and weight budget: open `docs/scratch_cnn_pipeline.html` in a
browser (standalone file, no build step). Improvement work is documented separately in
`docs/SCRATCH_CNN_IMPROVEMENT_LOG.md` (chronological) and `docs/SCRATCH_CNN_IMPROVEMENT_REPORT.md` (summary).

```
stem    conv3x3(3 -> 32), stride 2, BN, ReLU          160 -> 80
stage1  2 x [conv3x3 -> BN -> ReLU] at  32, maxpool    80 -> 40
stage2  2 x [conv3x3 -> BN -> ReLU] at  64, maxpool    40 -> 20
stage3  2 x [conv3x3 -> BN -> ReLU] at 128, maxpool    20 -> 10
stage4  2 x [conv3x3 -> BN -> ReLU] at 256             10 -> 10
head    global average pool -> dropout 0.3 -> linear(256 -> 20)
```

Plain VGG-style design: no residual connections, no attention, no pretrained weights. Convolutions use He
(Kaiming) initialisation. Constructor arguments `width` and `dropout` are the only architectural knobs exposed
through the YAML `model.kwargs`.

| Property | Value |
|---|---:|
| Parameters (all trainable) | 1,187,636 |
| Checkpoint size | 4.56 MB |
| fp32 weights | 4.5 MB |
| Input size | 160 px |

## Training configuration (course run)

Effective values after `configs/scratch_cnn.yaml` overrides `configs/base.yaml`.

| Setting | Value | Source |
|---|---|---|
| Pretrained | no (random init) | scratch_cnn.yaml |
| Epochs | 120 (ran all 120, best epoch 105) | scratch_cnn.yaml |
| Batch size | 64, no gradient accumulation | scratch_cnn.yaml |
| Optimiser | AdamW | train.py |
| Peak learning rate | 2.0e-3 | scratch_cnn.yaml |
| Schedule | linear warm-up 5 epochs, then cosine to 0 | scratch_cnn.yaml / base |
| Weight decay | 0.05 | scratch_cnn.yaml |
| Early stopping | val macro-F1, patience 25 | scratch_cnn.yaml |
| Loss | cross-entropy, label smoothing 0.1 | base.yaml |
| Mixed precision | fp16 with GradScaler | base.yaml |
| Gradient clipping | 1.0 | base.yaml |
| Dropout | 0.3 | scratch_cnn.yaml |
| Seed | 0 (training), 42 (split) | base.yaml / experiments script |
| Class weighting / sampling | none, uniform shuffle | — |

### Augmentation

| Transform | Value | Source |
|---|---|---|
| RandomResizedCrop scale | [0.35, 1.0], ratio [0.75, 1.33] | scratch_cnn.yaml |
| Horizontal flip | p = 0.5 | base.yaml |
| Vertical flip | off | base.yaml |
| RandAugment | n = 2, m = 9 | base.yaml |
| ColorJitter | 0.2 | scratch_cnn.yaml |
| Random erasing | off | base.yaml |
| Mixup / CutMix | off | base.yaml |

## Data used

Fixed ITRI626 split of the v2 manifest (fingerprint `431315e87d2612cb`): 2,010 train, 502 validation
(fold 0), 418 test. 20 classes, roughly 100 to 150 training images per class.

## Results

### Headline test metrics (418 images, single look)

| Metric | Value |
|---|---:|
| Accuracy | 0.3541 |
| Macro-F1 | 0.3361 |
| Macro precision | 0.3388 |
| Macro recall / balanced accuracy | 0.3426 |
| Weighted F1 | 0.3469 |
| Log loss | 2.334 |

Validation at the best epoch: accuracy 0.3386, macro-F1 0.3287, log loss 2.361.

### Comparison with the other course models (same test split)

| Model | Accuracy | Macro-F1 | Params | CPU ms/img |
|---|---:|---:|---:|---:|
| A: Scratch CNN | 0.354 | 0.336 | 1.19 M | 1.7 |
| B: ConvNeXt-Tiny fine-tuned | 0.636 | 0.628 | 27.8 M | 18.4 |
| C: SigLIP So400m frozen + linear | 0.823 | 0.817 | 428 M | 893.5 |

Research-track reference for the same recipe under five-fold grouped CV: accuracy 0.3654 ± 0.0155,
macro-F1 0.3552 ± 0.0152 (`results/benchmark.csv`). The classical HOG + HSV + SVM baseline sits at 0.2198.

### Per-class test scores

| Class | Precision | Recall | F1 | n |
|---|---:|---:|---:|---:|
| battery holder | 0.379 | 0.500 | 0.431 | 22 |
| breadboard | 0.529 | 0.474 | 0.500 | 19 |
| capacitor | 0.125 | 0.118 | 0.121 | 17 |
| connector | 0.176 | 0.188 | 0.182 | 16 |
| crystal oscillator | 0.292 | 0.333 | 0.311 | 21 |
| diode | 0.250 | 0.143 | 0.182 | 21 |
| fuse | 0.118 | 0.100 | 0.108 | 20 |
| heat sink | 0.667 | 0.435 | 0.526 | 23 |
| ic chip | 0.458 | 0.500 | 0.478 | 22 |
| inductor coil | 0.455 | 0.435 | 0.444 | 23 |
| jumper wire | 0.545 | 0.522 | 0.533 | 23 |
| led | 0.227 | 0.278 | 0.250 | 18 |
| microcontroller board | 0.536 | 0.682 | 0.600 | 22 |
| potentiometer | 0.321 | 0.429 | 0.367 | 21 |
| power supply module | 0.227 | 0.227 | 0.227 | 22 |
| relay | 0.267 | 0.182 | 0.216 | 22 |
| resistor | 0.304 | 0.318 | 0.311 | 22 |
| switch | 0.531 | 0.680 | 0.596 | 25 |
| transformer | 0.200 | 0.167 | 0.182 | 18 |
| transistor | 0.167 | 0.143 | 0.154 | 21 |

Strongest: microcontroller board, switch, jumper wire, heat sink, breadboard (large, distinctive shapes).
Weakest: fuse, capacitor, transistor, connector, diode, transformer (small parts, high intra-class variety).

### Most frequent confusions (true -> predicted, count)

| Count | True | Predicted |
|---:|---|---|
| 5 | fuse | connector |
| 4 | relay | transformer |
| 4 | inductor coil | battery holder |
| 4 | fuse | switch |
| 4 | crystal oscillator | potentiometer |
| 3 | transistor | ic chip |
| 3 | transformer | potentiometer |
| 3 | relay | power supply module |

### Training curve (from `histories.json`)

| Epoch | Train acc | Val acc | Val macro-F1 | Train loss | Val loss | LR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.071 | 0.100 | 0.059 | 2.998 | 2.969 | 4.1e-4 |
| 10 | 0.128 | 0.145 | 0.103 | 2.871 | 2.788 | 2.0e-3 |
| 20 | 0.173 | 0.177 | 0.130 | 2.763 | 2.717 | 1.9e-3 |
| 40 | 0.271 | 0.197 | 0.166 | 2.512 | 2.590 | 1.6e-3 |
| 60 | 0.384 | 0.267 | 0.254 | 2.260 | 2.487 | 1.1e-3 |
| 80 | 0.474 | 0.295 | 0.281 | 2.010 | 2.473 | 5.4e-4 |
| 100 | 0.555 | 0.335 | 0.325 | 1.817 | 2.390 | 1.5e-4 |
| 105 (best) | 0.586 | 0.339 | 0.329 | 1.747 | 2.361 | 7.2e-5 |
| 120 | 0.601 | 0.329 | 0.318 | 1.720 | 2.360 | 0 |

Reading: training accuracy keeps climbing while validation plateaus around 0.33 from epoch 100 onward, and
the train/val gap grows to about 27 points. The model is capacity- and data-limited rather than
under-trained: more epochs alone would not help, and the learning rate had already decayed to near zero when
validation peaked. Train loss includes label smoothing, so it stays above the plain cross-entropy value.

### Cost

| Measure | Value |
|---|---:|
| Training time (120 epochs, RTX 4060 Ti) | 146.9 s |
| Peak VRAM | 430 MB |
| GPU inference, batch 1, fp16 | 0.52 ms/img |
| GPU inference, batch 32, fp16 | 0.046 ms/img |
| CPU inference, batch 1, fp32, 8 threads | 1.7 ms/img |

Environment: Python 3.12.14, torch 2.11.0+cu128, timm 1.0.30, Linux, RTX 4060 Ti 8 GB, Ryzen 7 8700F.

## Rules for improvement experiments

- Select variants on development folds only (`uv run python -m compclass.train --config configs/scratch_cnn.yaml --fold 0 1 2 3 4`),
  or on the fixed train/validation split used above. The 418-image test split and the Wikimedia holdout are spent;
  do not score them during tuning.
- Change one thing at a time and record the config hash that `train.py` prints; runs append to `results/runs.jsonl`.
- Keep the comparison fair: same 160 px input unless input size is the variable, same split, same seed policy.
- Options are exposed without code changes via `--set`, e.g. `--set model.kwargs.width=48 train.lr=1.0e-3 aug.mixup=0.2`.
  Architectural changes (residual blocks, extra stages, SE blocks, different pooling) go in `scratch_cnn.py`
  and must keep `get_classifier()` so the training loop treats the model like the others.
- Requires `data/v2/images/` locally; rebuild it from the Kaggle download with `tools/build_dataset_v2.py` if absent.
