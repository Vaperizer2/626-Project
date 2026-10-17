# MobileNet-V3 — a CNN designed for phones

**Registry key:** `mobilenetv3_large` (timm `mobilenetv3_large_100.ra_in1k`) · **Builder:** `_mobilenet` ·
**Config:** `configs/mobilenetv3_large.yaml`

## 1. Core idea and why it was invented

The MobileNet line optimises accuracy *per unit of latency on a phone CPU*:

* A. G. Howard et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
  Applications.* arXiv:1704.04861 — **depthwise-separable convolution**: a k×k conv over C_in→C_out channels
  (k²·C_in·C_out multiply-adds per pixel) is split into a per-channel k×k *depthwise* filter (k²·C_in) plus a 1×1
  *pointwise* mixer (C_in·C_out). For k = 3 this is ≈8–9× cheaper.
* M. Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* arXiv:1801.04381 — the
  **inverted residual**: expand channels with 1×1, filter depthwise, project back with a *linear* 1×1, skip
  connection between the thin ends.
* A. Howard et al. (2019). *Searching for MobileNetV3.* arXiv:1905.02244 — block choices found by
  platform-aware architecture search + NetAdapt, squeeze-and-excitation in some blocks, the cheap
  **hard-swish** activation x·ReLU6(x+3)/6 in the deeper layers, and a redesigned, cheaper final stage.

In this project it is the **latency end of the comparison** and the natural candidate for a CPU / OpenCV-DNN demo.

## 2. Architecture (verified against timm 1.0.30)

```
x 3×224×224
 │ stem conv3×3 (3→16) stride 2 → BN → hard-swish
 │ stage 0: 1 × depthwise-separable conv                     ReLU
 │ stage 1: 2 × inverted residual                            ReLU
 │ stage 2: 3 × inverted residual + SE                       ReLU
 │ stage 3: 4 × inverted residual                            hard-swish
 │ stage 4: 2 × inverted residual + SE                       hard-swish
 │ stage 5: 3 × inverted residual + SE                       hard-swish
 │ stage 6: 1 × conv1×1 → 960 ch                             hard-swish
 │ global avg pool → conv_head 1×1 (960→1280) → hard-swish → classifier (1280→20)
 ▼ logits                                          (classifier: 25,620 params)

inverted residual: x → 1×1 expand → BN/act → k×k depthwise → BN/act → [SE] → 1×1 linear project → BN ─(+x)
```

Note that `conv_head` (960→1280) sits *after* pooling and is part of the backbone, not of `get_classifier()`, so
it is frozen during `freeze_epochs`. The linear-probe features (`num_classes=0`) are the 1280-d output of
`conv_head`.

## 3. Pretraining data and objective

Supervised ImageNet-1k. The `ra_in1k` tag denotes timm's own retraining with a RandAugment-based recipe.

## 4. Hyperparameters that matter here

Effective recipe of `configs/mobilenetv3_large.yaml` after merging `base.yaml` (computed with `compclass.config.load_config`):

| key | value |
|---|---|
| `model.key` | `mobilenetv3_large` |
| `model.pretrained` | True |
| `model.kwargs` | `{}` |
| `train.epochs / train.patience` | 30 / 8 |
| `train.batch_size × train.accum_steps` | 64 × 1 = 64 |
| `train.lr (peak)` | 0.001 |
| `train.weight_decay` | 0.01 |
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

Preprocessing (`PreprocessSpec`): input 224 px, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), bicubic resize, `crop_pct` 0.875 → evaluation resizes the short side to 256 px, then centre-crops 224 px. Parameters with a 20-class head: 4,227,652.

Schedule arithmetic for fold 0 (2010 training rows, `drop_last=True`): 31 micro-batches → 31 optimiser steps per epoch; LR warm-up lasts 62 steps; the backbone unfreezes after 31 steps, when the LR is at 0.52× its peak.


* **`train.batch_size: 64`, `train.lr: 1.0e-3`.** Small model, small activations → large batches are affordable;
  BatchNorm benefits from them.
* **`train.weight_decay: 0.01`.** Small networks have little capacity to spare; heavy decay underfits.
* **No `drop_path_rate`.** Not set in the config; the model is already small.

**Note — BatchNorm during the frozen warm-up.** `train.py` freezes backbone *weights* with `requires_grad_(False)` but still calls `model.train()`, so BatchNorm running means/variances keep updating from this dataset during `freeze_epochs`. That is usually harmless (it adapts the statistics to the new domain) but it means the backbone is not perfectly frozen. To freeze BN too you would need `module.eval()` on every BN layer.

## 5. Adaptation regimes

Full fine-tuning is the normal choice; the model is small enough that freezing saves little. For deployment,
MobileNet's value is in the *export* step: `compclass.export` produces a static batch-1 ONNX graph that OpenCV DNN
can run on a CPU.

## 6. Compute on an 8 GB RTX 4060 Ti

| key | params (M) | input px | batch | grad ckpt | train it/s | train img/s | peak VRAM (MB) | infer img/s | ≈ s/epoch* |
|---|---:|---:|---:|:-:|---:|---:|---:|---:|---:|
| `mobilenetv3_large` | 4.23 | 224 | 16 | no | 72.1 | 1153.5 | 497 | 6841.0 | 2 |

Source: `results/gpu_witness_carlpc2.json` (RTX 4060 Ti 8 GB, torch 2.11.0+cu128, CUDA 12.8), produced by `scripts/gpu_witness.py`: fp16 AMP, plain AdamW over *all* parameters (i.e. full fine-tuning), random 20-class inputs at the native size, batch 16, 10 timed steps after 3 warm-up steps; inference = fp16 forward at batch 16. \*Derived, not measured: 2010 fold-0 training images ÷ train img/s — a compute-only lower bound per epoch that ignores JPEG decoding, augmentation, validation and the config's own batch size.


GPU throughput understates MobileNet's advantage: depthwise convolutions are memory-bound on GPUs, while the
design targets CPU latency. Measure CPU latency of the exported ONNX model for the deployment comparison.

## 7. Typical failure modes on small, noisy web data

* **Capacity limit.** Fine-grained confusions (diode vs resistor, crystal oscillator vs capacitor) need detailed
  features; a 4 M-parameter network has less room for them than a 28 M one.
* **Sensitivity to the LR.** Very high LRs destroy the pretrained depthwise filters quickly; if validation F1
  collapses after unfreezing, halve `train.lr`.
* **Quantisation / export drift.** Hard-swish and SE are exported fine to ONNX opset 18, but always run the
  parity check (`max_abs_logit_diff` is printed by `compclass.export`).

## 8. Try this

```bash
uv run python -m compclass.train --config configs/mobilenetv3_large.yaml --fold 0
uv run python -m compclass.train --config configs/mobilenetv3_large.yaml --fold 0 --set train.lr=3.0e-4
# export the fold-0 checkpoint and check PyTorch/onnxruntime parity
uv run python -m compclass.export --checkpoint runs/mobilenetv3_large-<hash>/fold0/best.pt --out exports/mobilenetv3_large
# linear probe on the 1280-d conv_head features
uv run python -m compclass.features --model mobilenetv3_large --folds 0
```

## Results on data v2

> **Placeholder — no v2 results exist yet.** See `results/benchmark.csv` (generated from `results/runs.jsonl`) once the benchmark has run. Report mean ± std over the 5 dev folds; never quote the sealed test here.

| method | key | CV macro-F1 | CV accuracy | notes |
|---|---|---:|---:|---|
| — | — | *pending* | *pending* | |


## References

* A. G. Howard et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861.
* M. Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* arXiv:1801.04381.
* A. Howard et al. (2019). *Searching for MobileNetV3.* arXiv:1905.02244.
