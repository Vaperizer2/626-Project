# EfficientNet-V2 — architecture search plus compound scaling

**Registry key:** `efficientnet_v2_s` (timm `tf_efficientnetv2_s.in21k_ft_in1k`) · **Builder:** `_effnet` ·
**Config:** `configs/efficientnet_v2_s.yaml`

## 1. Core idea and why it was invented

* M. Tan, Q. V. Le (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*
  arXiv:1905.11946. A small base network (B0) is found by neural architecture search; larger models are made by
  **compound scaling** — growing depth, width and input resolution together by fixed ratios instead of one at a
  time. Resolution is part of the model: bigger EfficientNets expect bigger images.
* M. Tan, Q. V. Le (2021). *EfficientNetV2: Smaller Models and Faster Training.* arXiv:2104.00298. V1's
  depthwise convolutions are slow to *train* on accelerators, especially in early high-resolution stages. V2's
  training-aware search replaces them there with **Fused-MBConv** (one regular 3×3 conv instead of 1×1 expand +
  3×3 depthwise) and proposes progressive learning (image size and regularisation grow during training).

Building blocks: the **MBConv** inverted-residual block (M. Sandler et al. 2018, *MobileNetV2*, arXiv:1801.04381)
and **squeeze-and-excitation** channel attention (J. Hu, L. Shen, S. Albanie, G. Sun, E. Wu 2017, *Squeeze-and-Excitation Networks*,
arXiv:1709.01507).

## 2. Architecture (EfficientNetV2-S, 300×300 input; block counts verified against timm 1.0.30)

```
x 3×300×300
 │ stem conv3×3 (→24, stride 2, TF "same" padding)
 │ stage 0:  2 × Fused-MBConv1 (plain conv+BN+act, residual)     24 ch
 │ stage 1:  4 × Fused-MBConv  (EdgeResidual)                    48 ch   stride 2
 │ stage 2:  4 × Fused-MBConv                                    64 ch   stride 2
 │ stage 3:  6 × MBConv + SE   (InvertedResidual)               128 ch   stride 2
 │ stage 4:  9 × MBConv + SE                                    160 ch
 │ stage 5: 15 × MBConv + SE                                    256 ch   stride 2  → ≈10×10 map
 │ conv1×1 (256→1280) → BN → act → global avg pool → classifier (1280→20)
 ▼ logits

MBConv:        x → 1×1 expand (×k) → BN/SiLU → 3×3 depthwise → BN/SiLU → SE → 1×1 project → BN ─(+x)
Fused-MBConv:  x → 3×3 conv expand (×k) → BN/SiLU → 1×1 project → BN (no SE in V2-S) ────────────(+x)
SE:            global-pool → FC (reduce) → act → FC → sigmoid → rescale each channel
```

## 3. Pretraining data and objective

Supervised ImageNet-21k (≈14 M images, ≈21k classes, original TF release) then fine-tuned on ImageNet-1k. The
`tf_` prefix means the weights were ported from the official TensorFlow checkpoints, hence `Conv2dSame` padding and
mean = std = 0.5 normalisation (verified in the `PreprocessSpec` below).

## 4. Hyperparameters that matter here

Effective recipe of `configs/efficientnet_v2_s.yaml` after merging `base.yaml` (computed with `compclass.config.load_config`):

| key | value |
|---|---|
| `model.key` | `efficientnet_v2_s` |
| `model.pretrained` | True |
| `model.kwargs` | `{'drop_path_rate': 0.2}` |
| `train.epochs / train.patience` | 30 / 8 |
| `train.batch_size × train.accum_steps` | 24 × 1 = 24 |
| `train.lr (peak)` | 0.0005 |
| `train.weight_decay` | 0.05 |
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

Preprocessing (`PreprocessSpec`): input 300 px, mean (0.5, 0.5, 0.5), std (0.5, 0.5, 0.5), bicubic resize, `crop_pct` 1.0 → evaluation resizes the short side to 300 px, then centre-crops 300 px. Parameters with a 20-class head: 20,203,108.

Schedule arithmetic for fold 0 (2010 training rows, `drop_last=True`): 83 micro-batches → 83 optimiser steps per epoch; LR warm-up lasts 166 steps; the backbone unfreezes after 83 steps, when the LR is at 0.51× its peak.


* **`model.kwargs.drop_path_rate: 0.2` — stochastic depth** (G. Huang et al. 2016, *Deep Networks with Stochastic
  Depth*, arXiv:1603.09382): each residual branch is dropped at random during training (the rate grows with
  depth), a strong regulariser for deep, narrow networks.
* **Native 300 px with `crop_pct` 1.0.** Evaluation resizes the short side to 300 and centre-crops 300 — no border
  trimming. Lowering resolution is not currently possible via `model.kwargs.input_size` (see the known issue in
  [`resnet.md`](resnet.md)).
* **`train.batch_size: 24`, `train.lr: 5.0e-4`.** 300 px images cost ≈1.8× the activations of 224 px.
* **No `layer_decay`.** EfficientNet has a `group_matcher`, but the fine-grained grouping is per block: with
  `train.layer_decay=0.7` you get 40 LR scales and the stem's scale is ≈0.7^39 ≈ 1e-6 — effectively frozen. If you
  experiment with LLRD here, use a value close to 1 (e.g. 0.9).

**Note — BatchNorm during the frozen warm-up.** `train.py` freezes backbone *weights* with `requires_grad_(False)` but still calls `model.train()`, so BatchNorm running means/variances keep updating from this dataset during `freeze_epochs`. That is usually harmless (it adapts the statistics to the new domain) but it means the backbone is not perfectly frozen. To freeze BN too you would need `module.eval()` on every BN layer.

## 5. Adaptation regimes

Full fine-tuning (default) is the standard use. A linear probe on the 1280-d pooled features is a cheap check
(`compclass.features --model efficientnet_v2_s`). Head-only training via a long `freeze_epochs` is possible.

## 6. Compute on an 8 GB RTX 4060 Ti

| key | params (M) | input px | batch | grad ckpt | train it/s | train img/s | peak VRAM (MB) | infer img/s | ≈ s/epoch* |
|---|---:|---:|---:|:-:|---:|---:|---:|---:|---:|
| `efficientnet_v2_s` | 20.2 | 300 | 16 | no | 13.49 | 215.9 | 2527 | 997.2 | 9 |

Source: `results/gpu_witness_carlpc2.json` (RTX 4060 Ti 8 GB, torch 2.11.0+cu128, CUDA 12.8), produced by `scripts/gpu_witness.py`: fp16 AMP, plain AdamW over *all* parameters (i.e. full fine-tuning), random 20-class inputs at the native size, batch 16, 10 timed steps after 3 warm-up steps; inference = fp16 forward at batch 16. \*Derived, not measured: 2010 fold-0 training images ÷ train img/s — a compute-only lower bound per epoch that ignores JPEG decoding, augmentation, validation and the config's own batch size.


The witness ran at batch 16; the config uses 24, so peak VRAM will be roughly 1.5× the 2,527 MB measured —
still within 8 GB. Despite fewer parameters than ResNet-50, training throughput is lower because of the 300 px
input and the many depthwise layers.

## 7. Typical failure modes on small, noisy web data

* **Resolution mismatch.** Web images of tiny components are often small; upscaling to 300 px adds no detail.
* **Fragile BatchNorm + depthwise stack** at small batch sizes; if you reduce `train.batch_size` below ≈16, BN
  statistics get noisy.
* **Over-regularisation** when combining `drop_path_rate` 0.2, RandAugment, label smoothing and Mixup on only
  ~2k images; the model underfits (training loss stays high).

## 8. Try this

```bash
uv run python -m compclass.train --config configs/efficientnet_v2_s.yaml --fold 0
# less stochastic depth
uv run python -m compclass.train --config configs/efficientnet_v2_s.yaml --fold 0 --set model.kwargs.drop_path_rate=0.05
# gentle layer-wise decay over its 40 groups
uv run python -m compclass.train --config configs/efficientnet_v2_s.yaml --fold 0 --set train.layer_decay=0.9
# bf16 instead of fp16 (Ada GPUs support both; bf16 needs no GradScaler)
uv run python -m compclass.train --config configs/efficientnet_v2_s.yaml --fold 0 --set train.amp=bf16
```

## Results on data v2

> **Placeholder — no v2 results exist yet.** See `results/benchmark.csv` (generated from `results/runs.jsonl`) once the benchmark has run. Report mean ± std over the 5 dev folds; never quote the sealed test here.

| method | key | CV macro-F1 | CV accuracy | notes |
|---|---|---:|---:|---|
| — | — | *pending* | *pending* | |


## References

* M. Tan, Q. V. Le (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* arXiv:1905.11946.
* M. Tan, Q. V. Le (2021). *EfficientNetV2: Smaller Models and Faster Training.* arXiv:2104.00298.
* M. Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* arXiv:1801.04381.
* J. Hu, L. Shen, S. Albanie, G. Sun, E. Wu (2017). *Squeeze-and-Excitation Networks.* arXiv:1709.01507 (CVPR 2018 version: Hu, Shen, Sun).
* G. Huang et al. (2016). *Deep Networks with Stochastic Depth.* arXiv:1603.09382.
