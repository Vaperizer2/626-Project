# Swin Transformer — hierarchical attention in shifted windows

**Registry key:** `swin_t` (timm `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`) · **Builder:** `_swin` ·
**Config:** `configs/swin_t.yaml`

## 1. Core idea and why it was invented

Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo (2021). *Swin Transformer: Hierarchical Vision
Transformer using Shifted Windows.* arXiv:2103.14030 (ICCV 2021).

ViT's global attention is quadratic in the number of tokens and works on one fixed, coarse resolution — awkward
for detection and segmentation, which need multi-scale feature maps. Swin restricts self-attention to
non-overlapping **local windows** (7×7 tokens), making cost linear in image area, and **shifts** the window
grid by half a window in every other block so information crosses window borders. **Patch merging** layers
halve the resolution and double the channels between stages, giving a CNN-like pyramid (strides 4, 8, 16, 32).
Locality and hierarchy — the inductive biases ViT dropped — come back, which helps on smaller datasets.

## 2. Architecture (Swin-T at 224 px; verified against timm 1.0.30)

```
x 3×224×224
 │ patch embed: 4×4 patches → linear → 56×56 tokens × 96 (C)
 │ stage 1: 2 blocks, C=96,  3 heads                                  56×56
 │ patch merging (concat 2×2 neighbours → 4C → linear → 2C)
 │ stage 2: 2 blocks, 192,  6 heads                                   28×28
 │ patch merging
 │ stage 3: 6 blocks, 384, 12 heads                                   14×14
 │ patch merging
 │ stage 4: 2 blocks, 768, 24 heads                                    7×7
 │ LayerNorm → global average pool → head fc(768 → 20)
 ▼ logits

block pair:  W-MSA   (attention inside each 7×7 window, + relative position bias) → MLP
             SW-MSA  (windows shifted by 3 tokens; cyclic shift + attention mask)  → MLP
```

timm keeps the feature map channels-last: `forward_features` returns 1×7×7×768 for a 224 px image.

## 3. Pretraining data and objective

Supervised ImageNet-22k, then fine-tuned on ImageNet-1k (`ms_in22k_ft_in1k`, Microsoft's release).

## 4. Hyperparameters that matter here

Effective recipe of `configs/swin_t.yaml` after merging `base.yaml` (computed with `compclass.config.load_config`):

| key | value |
|---|---|
| `model.key` | `swin_t` |
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

Preprocessing (`PreprocessSpec`): input 224 px, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), bicubic resize, `crop_pct` 0.9 → evaluation resizes the short side to 249 px, then centre-crops 224 px. Parameters with a 20-class head: 27,534,734.

Schedule arithmetic for fold 0 (2010 training rows, `drop_last=True`): 62 micro-batches → 62 optimiser steps per epoch; LR warm-up lasts 124 steps; the backbone unfreezes after 62 steps, when the LR is at 0.51× its peak.

Layer-wise LR decay 0.8 yields 14 distinct LR scales over timm's `group_matcher` layer groups: the head trains at 2.0e-04; the bottom group (stem / patch embedding) gets scale 0.05498 = 0.8^13, i.e. LR 1.10e-05 (≈ 18× smaller than the head).


* **`train.layer_decay: 0.8` over 14 groups.** timm's fine-grained Swin grouping: patch embedding, each of the
  12 blocks (each patch-merging layer joins the next stage's first block), head. The bottom group trains
  ≈18× slower than the head.
* **`train.lr: 2.0e-4`** — the same as ConvNeXt-T, a model of similar size (27.5 M vs 27.8 M parameters).
* **Input size.** `--set model.kwargs.input_size=256` builds and runs (verified with `pretrained=False`). The window
  stays 7×7 and the relative-position-bias tables keep their shape (13² = 169 entries per head), so pretrained
  weights load unchanged; but a 64×64 token map no longer tiles into 7×7 windows, so timm handles the remainder
  internally and the attention layout differs from pretraining. Stay with 224 unless you are testing exactly that.

**Design note — warm-up overlaps the frozen phase.** The LR warm-up (`warmup_epochs`) is counted from step 0, including the head-only epochs. Here `freeze_epochs` (1) < `warmup_epochs` (2), so the backbone is unfrozen half-way up the ramp; with larger `freeze_epochs` it is unfrozen at (or near) the *peak* LR with no ramp of its own; see the multiplier in the schedule line above. To give the backbone its own ramp, increase warm-up, e.g. `--set train.warmup_epochs=4`.

## 5. Adaptation regimes

Full fine-tuning with LLRD (default). The pooled 768-d feature is available for a linear probe
(`compclass.features --model swin_t`). Swin and ConvNeXt-T are the matched-size pair for the
"attention vs convolution" comparison in the report.

## 6. Compute on an 8 GB RTX 4060 Ti

| key | params (M) | input px | batch | grad ckpt | train it/s | train img/s | peak VRAM (MB) | infer img/s | ≈ s/epoch* |
|---|---:|---:|---:|:-:|---:|---:|---:|---:|---:|
| `swin_t` | 27.53 | 224 | 16 | no | 19.04 | 304.6 | 1466 | 1093.4 | 7 |

Source: `results/gpu_witness_carlpc2.json` (RTX 4060 Ti 8 GB, torch 2.11.0+cu128, CUDA 12.8), produced by `scripts/gpu_witness.py`: fp16 AMP, plain AdamW over *all* parameters (i.e. full fine-tuning), random 20-class inputs at the native size, batch 16, 10 timed steps after 3 warm-up steps; inference = fp16 forward at batch 16. \*Derived, not measured: 2010 fold-0 training images ÷ train img/s — a compute-only lower bound per epoch that ignores JPEG decoding, augmentation, validation and the config's own batch size.


## 7. Typical failure modes on small, noisy web data

* **Window boundaries.** A component that straddles windows is only fully "seen" after a shifted block; with
  only 2 blocks in the last stage there is limited global mixing at full semantic depth.
* **Same instability as ViT** at high LRs; LLRD helps.
* **Resolution changes** (see above) silently change the attention layout.

## 8. Try this

```bash
uv run python -m compclass.train --config configs/swin_t.yaml --fold 0
# matched comparison with ConvNeXt-T on the same fold
uv run python -m compclass.train --config configs/convnext_tiny.yaml --fold 0
# stronger stochastic depth
uv run python -m compclass.train --config configs/swin_t.yaml --fold 0 --set model.kwargs.drop_path_rate=0.2
# LLRD off
uv run python -m compclass.train --config configs/swin_t.yaml --fold 0 --set train.layer_decay=null
```

## Results on data v2

> **Placeholder — no v2 results exist yet.** See `results/benchmark.csv` (generated from `results/runs.jsonl`) once the benchmark has run. Report mean ± std over the 5 dev folds; never quote the sealed test here.

| method | key | CV macro-F1 | CV accuracy | notes |
|---|---|---:|---:|---|
| — | — | *pending* | *pending* | |


## References

* Z. Liu et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* arXiv:2103.14030.
* A. Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words.* arXiv:2010.11929.
