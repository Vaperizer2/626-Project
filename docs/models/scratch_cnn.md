# Scratch CNN — a VGG-style network trained from random initialisation

**Registry key:** `scratch_cnn` · **Code:** `src/compclass/models/scratch_cnn.py`, builder `_scratch` in
`src/compclass/models/registry.py` · **Config:** `configs/scratch_cnn.yaml`

## 1. Core idea and why it was invented

A convolutional network learns its own features: stacked 3×3 convolutions detect edges, then textures, then
parts, with pooling enlarging each unit's receptive field. The lineage:

* Y. LeCun, L. Bottou, Y. Bengio, P. Haffner (1998). *Gradient-Based Learning Applied to Document Recognition.*
  Proc. IEEE 86(11) — LeNet-5, convolution + subsampling trained by back-propagation.
* K. Simonyan, A. Zisserman (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.*
  arXiv:1409.1556 — VGG: only 3×3 convolutions, depth as the main lever. Two stacked 3×3 convs see a 5×5 region
  with fewer parameters and an extra non-linearity.
* S. Ioffe, C. Szegedy (2015). *Batch Normalization.* arXiv:1502.03167 — normalising activations per channel makes
  higher learning rates stable.
* K. He, X. Zhang, S. Ren, J. Sun (2015). *Delving Deep into Rectifiers.* arXiv:1502.01852 — "He/Kaiming"
  initialisation for ReLU networks (used here: `kaiming_normal_(mode="fan_out")`).
* M. Lin, Q. Chen, S. Yan (2013). *Network In Network.* arXiv:1312.4400 — global average pooling instead of large
  fully connected layers.

In this project it is the **lower bound**: everything must be learned from ~2,000 training images per fold,
so the gap to the ImageNet-pretrained families *is* the lesson about transfer learning.

## 2. Architecture (input 3×160×160, `width` w = 32)

```
x 3×160×160
 │ stem: conv3×3(3→w, stride 2) → BN → ReLU                         → w×80×80
 │ stage1: [conv3×3 → BN → ReLU] × 2 at w,    MaxPool 2             → 32×40×40
 │ stage2: [conv3×3 → BN → ReLU] × 2 at 2w,   MaxPool 2             → 64×20×20
 │ stage3: [conv3×3 → BN → ReLU] × 2 at 4w,   MaxPool 2             → 128×10×10
 │ stage4: [conv3×3 → BN → ReLU] × 2 at 8w    (no pool)             → 256×10×10
 │ AdaptiveAvgPool(1) → Flatten → Dropout(0.3) → Linear(8w → 20)
 ▼ logits [20]
```

`get_classifier()` returns the `head` Sequential (Flatten–Dropout–Linear), mirroring timm, so `train.py`
treats this model like every other one. It has **no** `group_matcher`, so `train.layer_decay` is ignored.
With `num_classes=0` (`features.py`) the Linear becomes `Identity` and the model returns the 256-d pooled vector.

## 3. Pretraining

None (`model.pretrained: false` in the config; the builder ignores the flag). Normalisation is a neutral
mean = std = 0.5 rather than ImageNet statistics, since there are no pretrained weights to match.

## 4. Hyperparameters that matter here

Effective recipe of `configs/scratch_cnn.yaml` after merging `base.yaml` (computed with `compclass.config.load_config`):

| key | value |
|---|---|
| `model.key` | `scratch_cnn` |
| `model.pretrained` | False |
| `model.kwargs` | `{'width': 32, 'dropout': 0.3}` |
| `train.epochs / train.patience` | 120 / 25 |
| `train.batch_size × train.accum_steps` | 64 × 1 = 64 |
| `train.lr (peak)` | 0.002 |
| `train.weight_decay` | 0.05 |
| `train.warmup_epochs` | 5 |
| `train.freeze_epochs` | 0 |
| `train.layer_decay` | None |
| `train.label_smoothing` | 0.1 |
| `train.amp` | fp16 |
| `train.grad_clip` | 1.0 |
| `train.grad_checkpointing` | False |
| `aug.rrc_scale` | [0.35, 1.0] |
| `aug.randaugment` | {'n': 2, 'm': 9} |
| `aug.color_jitter` | 0.2 |
| `aug.mixup / aug.cutmix` | 0.0 / 0.0 |
| `aug.random_erasing` | 0.0 |

Preprocessing (`PreprocessSpec`): input 160 px, mean (0.5, 0.5, 0.5), std (0.5, 0.5, 0.5), bilinear resize, `crop_pct` 0.875 → evaluation resizes the short side to 183 px, then centre-crops 160 px. Parameters with a 20-class head: 1,187,636.

Schedule arithmetic for fold 0 (2010 training rows, `drop_last=True`): 31 micro-batches → 31 optimiser steps per epoch; LR warm-up lasts 155 steps.


Why this recipe differs from the pretrained families:

| knob | value here vs `base.yaml` | reason |
|---|---|---|
| `train.epochs`, `train.patience` | 120 / 25 vs 30 / 8 | Random init needs many passes; validation macro-F1 is noisy early, so early stopping must be patient. |
| `train.lr` | 2.0e-3 vs 3.0e-4 | No pretrained features to protect; BatchNorm tolerates a high LR. |
| `train.warmup_epochs` | 5 | Limits early instability while the BN statistics settle. |
| `train.freeze_epochs` | 0 | Nothing to protect, so no head-only warm-up. |
| `aug.rrc_scale`, `aug.color_jitter` | [0.35, 1.0], 0.2 | Stronger augmentation is the main regulariser when every filter is learned from this data. |
| `model.kwargs.width` | 32 | Channel multiplier: params grow ≈ quadratically with width (conv weights are w_in·w_out·9). |
| `model.kwargs.dropout` | 0.3 | Dropout before the linear head. |
| `model.kwargs.input_size` | 160 (builder default) | Accepted by `_scratch`; resolution scales compute ≈ quadratically. |

Label smoothing 0.1 and RandAugment (n=2, m=9) are inherited from `base.yaml`; `amp: fp16` matters little at
this size.

## 5. Adaptation regimes

Only full training from scratch applies. A linear probe on a *random* network (`features.py` builds with
`pretrained=True`, which this builder ignores) would measure random features — an instructive but separate
baseline.

## 6. Compute on an 8 GB RTX 4060 Ti

| key | params (M) | input px | batch | grad ckpt | train it/s | train img/s | peak VRAM (MB) | infer img/s | ≈ s/epoch* |
|---|---:|---:|---:|:-:|---:|---:|---:|---:|---:|
| `scratch_cnn` | 1.19 | 160 | 16 | no | 282.52 | 4520.2 | 117 | 19870.3 | 0 |

Source: `results/gpu_witness_carlpc2.json` (RTX 4060 Ti 8 GB, torch 2.11.0+cu128, CUDA 12.8), produced by `scripts/gpu_witness.py`: fp16 AMP, plain AdamW over *all* parameters (i.e. full fine-tuning), random 20-class inputs at the native size, batch 16, 10 timed steps after 3 warm-up steps; inference = fp16 forward at batch 16. \*Derived, not measured: 2010 fold-0 training images ÷ train img/s — a compute-only lower bound per epoch that ignores JPEG decoding, augmentation, validation and the config's own batch size.


The config trains at batch 64, not the witness's 16, so expect higher VRAM than the 117 MB measured (still far
below 8 GB). At this throughput, data loading (`data.num_workers: 6`, JPEG decode + RandAugment) will be the
bottleneck, not the GPU.

## 7. Typical failure modes on small, noisy web data

* **Overfitting with little signal.** With ~100–150 training images per class, the network can memorise the
  training fold; training loss falls while validation macro-F1 plateaus. Watch the per-epoch print-out.
* **Shortcut learning.** Without pretrained shape features, the easiest cues are background colour, watermark
  text and photo style. Out-of-scope images (e.g. Nintendo Switch consoles labelled `switch`) are learnable
  shortcuts for exactly this reason.
* **Variance across seeds and folds.** Results move by several points between `train.seed` values; compare
  families only on the 5-fold mean ± std.
* **BatchNorm with small batches.** Reducing `train.batch_size` far below 32 makes BN statistics noisy.

## 8. Try this

```bash
# 1. Reference run on one fold
uv run python -m compclass.train --config configs/scratch_cnn.yaml --fold 0
# 2. Twice the width (≈4x parameters): does capacity help or overfit?
uv run python -m compclass.train --config configs/scratch_cnn.yaml --fold 0 --set model.kwargs.width=64
# 3. Remove the strong augmentation: how much of the score is augmentation?
uv run python -m compclass.train --config configs/scratch_cnn.yaml --fold 0 --set aug.randaugment=null aug.color_jitter=0.0 "aug.rrc_scale=[0.9,1.0]"
# 4. Mixup as an extra regulariser (soft targets; label smoothing is folded into the mixup targets)
uv run python -m compclass.train --config configs/scratch_cnn.yaml --fold 0 --set aug.mixup=0.2
# 5. Seed variance: same recipe, three seeds
for s in 0 1 2; do uv run python -m compclass.train --config configs/scratch_cnn.yaml --fold 0 --set train.seed=$s; done
```

## Results on data v2

> **Placeholder — no v2 results exist yet.** See `results/benchmark.csv` (generated from `results/runs.jsonl`) once the benchmark has run. Report mean ± std over the 5 dev folds; never quote the sealed test here.

| method | key | CV macro-F1 | CV accuracy | notes |
|---|---|---:|---:|---|
| — | — | *pending* | *pending* | |


## References

* Y. LeCun et al. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proc. IEEE 86(11).
* K. Simonyan, A. Zisserman (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* arXiv:1409.1556.
* S. Ioffe, C. Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* arXiv:1502.03167.
* K. He et al. (2015). *Delving Deep into Rectifiers.* arXiv:1502.01852.
* M. Lin, Q. Chen, S. Yan (2013). *Network In Network.* arXiv:1312.4400.
