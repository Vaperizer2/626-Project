# ConvNeXt — a ResNet modernised with Transformer design choices

**Registry key:** `convnext_tiny` (timm `convnext_tiny.fb_in22k_ft_in1k`) · **Builder:** `_convnext` ·
**Config:** `configs/convnext_tiny.yaml`

## 1. Core idea and why it was invented

Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie (2022). *A ConvNet for the 2020s.*
arXiv:2201.03545 (CVPR 2022).

By 2021, Vision Transformers (ViT, Swin) had overtaken CNNs. ConvNeXt asks how much of that is *attention* and
how much is everything else. Starting from ResNet-50, the authors apply Transformer-era changes one at a time
and measure each: a modern training recipe; Swin-like stage ratios (3:3:9:3) and a non-overlapping 4×4
"patchify" stem; depthwise convolution (like per-channel spatial mixing in attention); an inverted bottleneck
MLP; large 7×7 kernels; GELU, fewer activations and normalisations, LayerNorm instead of BatchNorm; separate
downsampling layers. The result matches Swin at similar cost **with no attention at all** — evidence that macro
design and training recipe explain much of the ViT advantage.

## 2. Architecture (ConvNeXt-T; verified against timm 1.0.30)

```
x 3×224×224
 │ stem: conv4×4 stride 4 (3→96) → LayerNorm                         → 96×56×56
 │ stage 1: 3 × ConvNeXt block,  96 ch                                → 96×56×56
 │ down: LayerNorm → conv2×2 s2                                       → 192×28×28
 │ stage 2: 3 × block, 192 ch
 │ down                                                               → 384×14×14
 │ stage 3: 9 × block, 384 ch
 │ down                                                               → 768×7×7
 │ stage 4: 3 × block, 768 ch
 │ global avg pool → LayerNorm → fc(768→20)
 ▼ logits

ConvNeXt block(x):
  x ─► 7×7 depthwise conv ─► LayerNorm ─► Linear(C→4C) ─► GELU ─► Linear(4C→C) ─► ×γ (layer scale) ─► drop-path ─(+x)─►
```

## 3. Pretraining data and objective

Supervised ImageNet-22k (≈14 M images), then fine-tuned on ImageNet-1k (`fb_in22k_ft_in1k`, Meta's release).
The larger, more diverse 22k label space usually transfers better to unusual domains than 1k-only pretraining.

## 4. Hyperparameters that matter here

Effective recipe of `configs/convnext_tiny.yaml` after merging `base.yaml` (computed with `compclass.config.load_config`):

| key | value |
|---|---|
| `model.key` | `convnext_tiny` |
| `model.pretrained` | True |
| `model.kwargs` | `{'drop_path_rate': 0.1}` |
| `train.epochs / train.patience` | 30 / 8 |
| `train.batch_size × train.accum_steps` | 32 × 1 = 32 |
| `train.lr (peak)` | 0.0002 |
| `train.weight_decay` | 0.05 |
| `train.warmup_epochs` | 2 |
| `train.freeze_epochs` | 1 |
| `train.layer_decay` | 0.8 |
| `train.label_smoothing` | 0.1 |
| `train.amp` | fp16 |
| `train.grad_clip` | 1.0 |
| `train.grad_checkpointing` | False |
| `aug.rrc_scale` | [0.5, 1.0] |
| `aug.randaugment` | {'n': 2, 'm': 9} |
| `aug.color_jitter` | 0.0 |
| `aug.mixup / aug.cutmix` | 0.0 / 0.0 |
| `aug.random_erasing` | 0.0 |

Preprocessing (`PreprocessSpec`): input 224 px, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), bicubic resize, `crop_pct` 0.875 → evaluation resizes the short side to 256 px, then centre-crops 224 px. Parameters with a 20-class head: 27,835,508.

Schedule arithmetic for fold 0 (2010 training rows, `drop_last=True`): 62 micro-batches → 62 optimiser steps per epoch; LR warm-up lasts 124 steps; the backbone unfreezes after 62 steps, when the LR is at 0.51× its peak.

Layer-wise LR decay 0.8 yields 20 distinct LR scales over timm's `group_matcher` layer groups: the head trains at 2.0e-04; the bottom group (stem / patch embedding) gets scale 0.01441 = 0.8^19, i.e. LR 2.88e-06 (≈ 69× smaller than the head).


* **`train.layer_decay: 0.8` with 20 groups.** timm's fine-grained ConvNeXt `group_matcher` makes the stem one group, every one
  of the 18 blocks its own group (each downsampling layer joins the first block of its stage), and the head
  (pool-norm-fc) one group — 20 LR scales (verified with `timm.models._manipulate.group_parameters`). Because there are many
  groups, 0.8 is already strong: the stem learns ≈69× slower than the head. Compare ViT-B, where 0.75 over 14
  groups gives ≈42×.
* **`model.kwargs.drop_path_rate: 0.1`.** Stochastic depth on each residual block.
* **`train.lr: 2.0e-4`.** Lower than ResNet's 1e-3: LayerNorm networks trained with AdamW are typically fine-tuned
  with smaller LRs.
* **No BatchNorm.** Batch size and the frozen-warm-up BN caveat of the classic CNNs do not apply; LayerNorm is
  per-sample.

## 5. Adaptation regimes

Full fine-tuning with LLRD is the default and usually best for supervised CNNs. Linear probe:
`compclass.features --model convnext_tiny` (768-d pooled, normalised features). In the legacy cross-fold study (pre-v2 data,
exact-hash grouping only, 14 fine-tuned ImageNet models) ConvNeXt-*Base* was the best single model at 62.88 % OOF
macro-F1 (`legacy/docs/CROSS_FOLD_AUGMENTATION_REPORT.md`). That number was measured on a different dataset
version and is not comparable to v2 results (see [`../concepts/evaluation.md`](../concepts/evaluation.md)).

## 6. Compute on an 8 GB RTX 4060 Ti

| key | params (M) | input px | batch | grad ckpt | train it/s | train img/s | peak VRAM (MB) | infer img/s | ≈ s/epoch* |
|---|---:|---:|---:|:-:|---:|---:|---:|---:|---:|
| `convnext_tiny` | 27.84 | 224 | 16 | no | 22.1 | 353.6 | 1348 | 1449.0 | 6 |

Source: `results/gpu_witness_carlpc2.json` (RTX 4060 Ti 8 GB, torch 2.11.0+cu128, CUDA 12.8), produced by `scripts/gpu_witness.py`: fp16 AMP, plain AdamW over *all* parameters (i.e. full fine-tuning), random 20-class inputs at the native size, batch 16, 10 timed steps after 3 warm-up steps; inference = fp16 forward at batch 16. \*Derived, not measured: 2010 fold-0 training images ÷ train img/s — a compute-only lower bound per epoch that ignores JPEG decoding, augmentation, validation and the config's own batch size.


## 7. Typical failure modes on small, noisy web data

* **Too-strong LLRD.** With 20 groups, a decay of 0.65 would leave the stem at 0.65^19 ≈ 3e-4 of the head LR —
  effectively frozen; that may be fine, but know that you chose it.
* **Large kernels see context.** 7×7 depthwise kernels after a stride-4 stem see wide context; on cluttered PCB
  photos the model can learn the *board* rather than the part.
* **Over-confidence on out-of-scope images**, as for every softmax classifier.

## 8. Try this

```bash
uv run python -m compclass.train --config configs/convnext_tiny.yaml --fold 0
# LLRD ablation: off vs default vs strong
uv run python -m compclass.train --config configs/convnext_tiny.yaml --fold 0 --set train.layer_decay=null
uv run python -m compclass.train --config configs/convnext_tiny.yaml --fold 0 --set train.layer_decay=0.65
# Mixup + CutMix (soft-target loss is selected automatically)
uv run python -m compclass.train --config configs/convnext_tiny.yaml --fold 0 --set aug.mixup=0.2 aug.cutmix=1.0
# All five folds for a reportable mean ± std
uv run python -m compclass.train --config configs/convnext_tiny.yaml --fold 0 1 2 3 4
```

## Results on data v2

> **Placeholder — no v2 results exist yet.** See `results/benchmark.csv` (generated from `results/runs.jsonl`) once the benchmark has run. Report mean ± std over the 5 dev folds; never quote the sealed test here.

| method | key | CV macro-F1 | CV accuracy | notes |
|---|---|---:|---:|---|
| — | — | *pending* | *pending* | |


## References

* Z. Liu et al. (2022). *A ConvNet for the 2020s.* arXiv:2201.03545.
* G. Huang et al. (2016). *Deep Networks with Stochastic Depth.* arXiv:1603.09382.
