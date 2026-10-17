# ResNet — residual convolutional networks

**Registry key:** `resnet50` (timm `resnet50.a1_in1k`) · **Builder:** `_resnet` in `src/compclass/models/registry.py` ·
**Config:** `configs/resnet50.yaml`

## 1. Core idea and why it was invented

K. He, X. Zhang, S. Ren, J. Sun (2015). *Deep Residual Learning for Image Recognition.* arXiv:1512.03385 (CVPR 2016).

Around 2015, simply stacking more layers made networks *worse even on the training set* (the "degradation"
problem): optimisation, not over-fitting, was the obstacle. ResNet re-parameterises each block to learn a
**residual** F(x) that is added to an identity shortcut, y = x + F(x). If the best thing a block can do is
nothing, it only has to drive F to zero; and gradients reach early layers through the identity path. This made
50–152-layer networks trainable and won ILSVRC 2015.

ResNet-50 uses **bottleneck** blocks — 1×1 (reduce channels) → 3×3 → 1×1 (expand ×4) — so depth is cheap.
The `a1_in1k` weights come from R. Wightman, H. Touvron, H. Jégou (2021), *ResNet strikes back: An improved
training procedure in timm*, arXiv:2110.00476: the same architecture, retrained with a modern recipe (long
schedule, LAMB optimiser, binary cross-entropy, Mixup/CutMix, RandAugment), which is markedly stronger than the
2015 torchvision weights.

## 2. Architecture (224×224 input; verified against timm 1.0.30)

```
x 3×224×224
 │ stem: conv7×7(3→64, stride 2) → BN → ReLU → maxpool 3×3 s2        → 64×56×56
 │ layer1: 3 bottlenecks  (out 256)                                    → 256×56×56
 │ layer2: 4 bottlenecks  (out 512,  first stride 2)                   → 512×28×28
 │ layer3: 6 bottlenecks  (out 1024, first stride 2)                   → 1024×14×14
 │ layer4: 3 bottlenecks  (out 2048, first stride 2)                   → 2048×7×7
 │ global average pool → fc(2048 → 20)                                 (head: 40,980 params)
 ▼ logits

bottleneck(x):  x ──► 1×1 conv (C→C/4) → BN → ReLU → 3×3 conv → BN → ReLU → 1×1 conv (→C) → BN ──(+)──► ReLU
                └────────────────────── identity (or 1×1 stride-2 projection) ───────────────────────┘
```

## 3. Pretraining data and objective

Supervised ImageNet-1k classification (1.28 M images, 1,000 classes), RSB "A1" recipe. Supervised
ImageNet features are object-centric and texture-sensitive; many of our 20 component classes have close
ImageNet relatives (e.g. there are no "resistor" classes, but there are many small manufactured objects),
which is why transfer works at all.

## 4. How it is wired in this repo

`_resnet` calls `_timm("resnet50.a1_in1k", num_classes, pretrained)`; timm replaces the 1000-way `fc` with a
fresh 20-way layer, and `PreprocessSpec` is read from the weights' pretrained config. `get_classifier()` is
`fc`, so `freeze_epochs` trains `fc` alone. With `num_classes=0` the model returns the 2048-d pooled vector
(linear probe input).

## 5. Hyperparameters that matter here

Effective recipe of `configs/resnet50.yaml` after merging `base.yaml` (computed with `compclass.config.load_config`):

| key | value |
|---|---|
| `model.key` | `resnet50` |
| `model.pretrained` | True |
| `model.kwargs` | `{}` |
| `train.epochs / train.patience` | 30 / 8 |
| `train.batch_size × train.accum_steps` | 32 × 1 = 32 |
| `train.lr (peak)` | 0.001 |
| `train.weight_decay` | 0.02 |
| `train.warmup_epochs` | 2 |
| `train.freeze_epochs` | 1 |
| `train.layer_decay` | None |
| `train.label_smoothing` | 0.1 |
| `train.amp` | fp16 |
| `train.grad_clip` | 1.0 |
| `train.grad_checkpointing` | False |
| `aug.rrc_scale` | [0.5, 1.0] |
| `aug.randaugment` | {'n': 2, 'm': 9} |
| `aug.color_jitter` | 0.0 |
| `aug.mixup / aug.cutmix` | 0.0 / 0.0 |
| `aug.random_erasing` | 0.0 |

Preprocessing (`PreprocessSpec`): input 224 px, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), bicubic resize, `crop_pct` 0.95 → evaluation resizes the short side to 236 px, then centre-crops 224 px. Parameters with a 20-class head: 23,549,012.

Schedule arithmetic for fold 0 (2010 training rows, `drop_last=True`): 62 micro-batches → 62 optimiser steps per epoch; LR warm-up lasts 124 steps; the backbone unfreezes after 62 steps, when the LR is at 0.51× its peak.


* **`train.lr: 1.0e-3`, no `layer_decay`.** CNNs with BatchNorm tolerate a higher fine-tuning LR than ViTs. timm
  *does* provide a `group_matcher` for ResNet: with `--set train.layer_decay=0.7` you get 18 LR scales (stem,
  16 bottlenecks, head) — try it as an exercise.
* **`train.weight_decay: 0.02`** (lower than the base 0.05): AdamW decoupled decay pulls weights toward zero, i.e.
  *away* from the pretrained solution; lighter decay keeps them closer.
* **`train.freeze_epochs: 1`.** One epoch of head-only training before full fine-tuning.
* **Input size.** CNNs are resolution-agnostic, and ResNet evaluates better at slightly larger sizes, but
  `--set model.kwargs.input_size=288` currently raises `TypeError` for this family (see *Known issue* below).

**Note — BatchNorm during the frozen warm-up.** `train.py` freezes backbone *weights* with `requires_grad_(False)` but still calls `model.train()`, so BatchNorm running means/variances keep updating from this dataset during `freeze_epochs`. That is usually harmless (it adapts the statistics to the new domain) but it means the backbone is not perfectly frozen. To freeze BN too you would need `module.eval()` on every BN layer.

## 6. Adaptation regimes

| regime | how in this repo | when it wins |
|---|---|---|
| linear probe | `uv run python -m compclass.features --model resnet50` | quick check of feature quality; supervised-ImageNet features are usually weaker than DINOv2/CLIP for probes |
| head-only | `--set train.freeze_epochs=30` (all epochs frozen) | tiny data, or as a diagnostic |
| full fine-tune | default config | the normal choice for supervised CNNs on a few thousand images |

## 7. Compute on an 8 GB RTX 4060 Ti

| key | params (M) | input px | batch | grad ckpt | train it/s | train img/s | peak VRAM (MB) | infer img/s | ≈ s/epoch* |
|---|---:|---:|---:|:-:|---:|---:|---:|---:|---:|
| `resnet50` | 23.55 | 224 | 16 | no | 25.91 | 414.6 | 1027 | 1679.4 | 5 |

Source: `results/gpu_witness_carlpc2.json` (RTX 4060 Ti 8 GB, torch 2.11.0+cu128, CUDA 12.8), produced by `scripts/gpu_witness.py`: fp16 AMP, plain AdamW over *all* parameters (i.e. full fine-tuning), random 20-class inputs at the native size, batch 16, 10 timed steps after 3 warm-up steps; inference = fp16 forward at batch 16. \*Derived, not measured: 2010 fold-0 training images ÷ train img/s — a compute-only lower bound per epoch that ignores JPEG decoding, augmentation, validation and the config's own batch size.


## 8. Typical failure modes on small, noisy web data

* **Texture bias.** ImageNet CNNs rely heavily on texture (R. Geirhos et al. 2018, *ImageNet-trained CNNs are
  biased towards texture*, arXiv:1811.12231). Components defined by *shape* (e.g. diode vs resistor body, both
  small cylinders with leads) get confused; check the confusion matrix stored with every run.
* **Over-confident wrong predictions** on out-of-scope images (consoles under `switch`, power-grid transformers
  under `transformer`) — the softmax must pick one of 20 classes. Use OOF probabilities for auditing
  ([`../concepts/data_quality.md`](../concepts/data_quality.md)).
* **LR too high for the backbone** wrecks the pretrained features in the first epoch (validation F1 drops after
  unfreezing). Lower `train.lr` or add `train.layer_decay`.

## 9. Try this

```bash
uv run python -m compclass.train --config configs/resnet50.yaml --fold 0
# layer-wise LR decay on a CNN (18 groups)
uv run python -m compclass.train --config configs/resnet50.yaml --fold 0 --set train.layer_decay=0.7
# no head warm-up: does the random head hurt the backbone?
uv run python -m compclass.train --config configs/resnet50.yaml --fold 0 --set train.freeze_epochs=0
# frozen backbone for the whole run (head-only ≈ linear probe with augmentation)
uv run python -m compclass.train --config configs/resnet50.yaml --fold 0 --set train.freeze_epochs=30
# linear probe for comparison
uv run python -m compclass.features --model resnet50 --folds 0
```

## Results on data v2

> **Placeholder — no v2 results exist yet.** See `results/benchmark.csv` (generated from `results/runs.jsonl`) once the benchmark has run. Report mean ± std over the 5 dev folds; never quote the sealed test here.

| method | key | CV macro-F1 | CV accuracy | notes |
|---|---|---:|---:|---|
| — | — | *pending* | *pending* | |


## Known issue

`_timm` always forwards `input_size` to `timm.create_model` as `img_size`. ResNet's constructor does not accept
`img_size`, so `--set model.kwargs.input_size=...` fails with `TypeError: ResNet.__init__() got an unexpected keyword
argument 'img_size'` (verified with timm 1.0.30, `pretrained=False`). The same applies to ConvNeXt, EfficientNet and
MobileNet. Do not change `src/` from this doc; this is reported to the maintainer.

## References

* K. He, X. Zhang, S. Ren, J. Sun (2015). *Deep Residual Learning for Image Recognition.* arXiv:1512.03385.
* R. Wightman, H. Touvron, H. Jégou (2021). *ResNet strikes back: An improved training procedure in timm.* arXiv:2110.00476.
* R. Geirhos et al. (2018). *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.* arXiv:1811.12231.
