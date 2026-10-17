# DINOv2 — self-supervised Vision Transformers

**Registry keys:** `dinov2_vits14`, `dinov2_vitb14` (configs exist), `dinov2_vitl14` (no config) · timm
`vit_{small,base,large}_patch14_dinov2.lvd142m` · **Builder:** `_dinov2` · **Configs:** `configs/dinov2_vits14.yaml`,
`configs/dinov2_vitb14.yaml`

## 1. Core idea and why it was invented

M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa,
A. El-Nouby, et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193.

Supervised pretraining teaches a network only what the label set needs (ImageNet's 1k classes); text-supervised
pretraining (CLIP) teaches what captions mention. **Self-supervised** learning needs no labels at all: the
training signal comes from the images themselves. DINOv2 combines two objectives from earlier work:

* **DINO** (M. Caron et al. 2021, *Emerging Properties in Self-Supervised Vision Transformers*, arXiv:2104.14294):
  *self-distillation with no labels*. A student network sees several crops of an image (2 large "global" + several
  small "local" crops); a teacher — an exponential moving average (EMA) of the student — sees the global crops.
  The student is trained to match the teacher's softmax output over a large set of learned prototypes, with
  centring and sharpening of the teacher to avoid collapse. Matching local to global crops forces the features to
  encode *what the object is*, invariant to crop and augmentation.
* **iBOT** (J. Zhou et al. 2021, *iBOT: Image BERT Pre-Training with Online Tokenizer*, arXiv:2111.07832):
  masked-patch prediction — the student sees an image with patches masked out and must predict the teacher's
  patch-level outputs at those positions. This gives good *dense* (per-patch) features.

DINOv2's contribution is mostly *scale and curation*: the LVD-142M dataset (142 M images retrieved from a large
uncurated web pool by similarity to curated datasets, with de-duplication), a ViT-g/14 teacher trained with
these losses plus a KoLeo regulariser and Sinkhorn-Knopp centring, and smaller ViT-S/B/L models **distilled**
from it. The headline result: frozen DINOv2 features + a linear classifier rival task-specific fine-tuning on
many benchmarks.

## 2. Architecture (verified against timm 1.0.30, at the 224 px used here)

```
x 3×224×224
 │ patch embedding: conv14×14 stride 14 → 16×16 = 256 tokens   (dim 384 for ViT-S, 768 for ViT-B)
 │ prepend [CLS] → 257 tokens; + position embedding (interpolated from the 518 px / 37×37 grid)
 │ 12 × pre-norm Transformer block with LayerScale:
 │      t = t + γ1 · MHSA(LN(t))        ViT-S: 6 heads, ViT-B: 12 heads
 │      t = t + γ2 · MLP(LN(t))         4× expansion, GELU
 │ final LayerNorm → [CLS] token (global_pool="token") → head Linear(dim → 20)
 ▼ logits
```

**LayerScale** (H. Touvron et al. 2021, *Going deeper with Image Transformers*, arXiv:2103.17239) multiplies each
residual branch by a learned per-channel vector γ initialised small, stabilising deep ViTs.

## 3. How it is wired in this repo

`_dinov2` passes `input_size=224` (→ `img_size=224` in timm, which resamples the position embeddings) and
`crop_pct=1.0` (resize the short side to 224, centre-crop 224). The native 518 px would mean (518/14)² = 1,369
tokens instead of 256 — attention cost ≈ (1369/256)² ≈ 29× — unaffordable for fine-tuning on 8 GB. You can override
it per run, e.g. `--set model.kwargs.input_size=336` (24×24 = 576 tokens; builds and runs, verified with
`pretrained=False`).

The linear probe (`features.py`) uses the `[CLS]` token only. The DINOv2 paper's linear evaluation also uses the
average-pooled patch tokens alongside `[CLS]`. To compare pooling choices, build the backbone in a notebook with
`build_model("dinov2_vits14", 0, global_pool="avg")` (mean of patch tokens; extra kwargs go to `timm.create_model`,
verified to return a 384-d vector) and fit the same probe; concatenating both needs a few lines of your own code.

## 4. Pretraining data and objective

LVD-142M, no labels; DINO + iBOT losses; ViT-S/14 and ViT-B/14 are distilled from the ViT-g/14 teacher.

## 5. Hyperparameters that matter here

Effective recipe of `configs/dinov2_vitb14.yaml` after merging `base.yaml` (computed with `compclass.config.load_config`):

| key | value |
|---|---|
| `model.key` | `dinov2_vitb14` |
| `model.pretrained` | True |
| `model.kwargs` | `{'drop_path_rate': 0.1}` |
| `train.epochs / train.patience` | 30 / 8 |
| `train.batch_size × train.accum_steps` | 16 × 2 = 32 |
| `train.lr (peak)` | 5e-05 |
| `train.weight_decay` | 0.05 |
| `train.warmup_epochs` | 2 |
| `train.freeze_epochs` | 2 |
| `train.layer_decay` | 0.75 |
| `train.label_smoothing` | 0.1 |
| `train.amp` | fp16 |
| `train.grad_clip` | 1.0 |
| `train.grad_checkpointing` | False |
| `aug.rrc_scale` | [0.5, 1.0] |
| `aug.randaugment` | {'n': 2, 'm': 9} |
| `aug.color_jitter` | 0.0 |
| `aug.mixup / aug.cutmix` | 0.0 / 0.0 |
| `aug.random_erasing` | 0.0 |

Preprocessing (`PreprocessSpec`): input 224 px, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), bicubic resize, `crop_pct` 1.0 → evaluation resizes the short side to 224 px, then centre-crops 224 px. Parameters with a 20-class head: 85,740,308.

Schedule arithmetic for fold 0 (2010 training rows, `drop_last=True`): 125 micro-batches → 63 optimiser steps per epoch; LR warm-up lasts 126 steps; the backbone unfreezes after 126 steps, when the LR is at 1.00× its peak.

Layer-wise LR decay 0.75 yields 14 distinct LR scales over timm's `group_matcher` layer groups: the head trains at 5.0e-05; the bottom group (stem / patch embedding) gets scale 0.02376 = 0.75^13, i.e. LR 1.19e-06 (≈ 42× smaller than the head).


`configs/dinov2_vits14.yaml` differs only in `train.lr: 1.0e-4` and the default batch 32 without accumulation:

Preprocessing (`PreprocessSpec`): input 224 px, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), bicubic resize, `crop_pct` 1.0 → evaluation resizes the short side to 224 px, then centre-crops 224 px. Parameters with a 20-class head: 21,636,500.

Schedule arithmetic for fold 0 (2010 training rows, `drop_last=True`): 62 micro-batches → 62 optimiser steps per epoch; LR warm-up lasts 124 steps; the backbone unfreezes after 124 steps, when the LR is at 1.00× its peak.

Layer-wise LR decay 0.75 yields 14 distinct LR scales over timm's `group_matcher` layer groups: the head trains at 1.0e-04; the bottom group (stem / patch embedding) gets scale 0.02376 = 0.75^13, i.e. LR 2.38e-06 (≈ 42× smaller than the head).


**Design note — warm-up overlaps the frozen phase.** The LR warm-up (`warmup_epochs`) is counted from step 0, including the head-only epochs. When `freeze_epochs` ≥ `warmup_epochs`, the backbone is unfrozen at (or near) the *peak* LR with no ramp of its own; see the multiplier in the schedule line above. To give the backbone its own ramp, increase warm-up, e.g. `--set train.warmup_epochs=4`.

* **Low LR and LLRD 0.75.** DINOv2's features are the point; fine-tuning should adjust them gently.
* **`train.freeze_epochs: 2`.** A random head on strong frozen features learns quickly; two epochs of head-only
  training is a light form of LP-FT (see [`../concepts/transfer_learning.md`](../concepts/transfer_learning.md)).

## 6. Adaptation regimes — where DINOv2 is special

| regime | command | when it wins |
|---|---|---|
| linear probe | `uv run python -m compclass.features --model dinov2_vitb14` | **the expected sweet spot**: minutes of compute, no fine-tuning hyperparameters, robust to label noise because the backbone cannot memorise it |
| full fine-tune + LLRD | `uv run python -m compclass.train --config configs/dinov2_vitb14.yaml --fold 0` | when the probe saturates and you can afford 5-fold runs; gains over the probe are often small on ~2k images |
| larger backbone | `dinov2_vitl14` (probe only is realistic) | probe first; fine-tuning ViT-L on 8 GB needs gradient checkpointing |

## 7. Compute on an 8 GB RTX 4060 Ti

| key | params (M) | input px | batch | grad ckpt | train it/s | train img/s | peak VRAM (MB) | infer img/s | ≈ s/epoch* |
|---|---:|---:|---:|:-:|---:|---:|---:|---:|---:|
| `dinov2_vits14` | 21.64 | 224 | 16 | no | 25.91 | 414.6 | 1076 | 1936.1 | 5 |
| `dinov2_vitb14` | 85.74 | 224 | 16 | no | 8.95 | 143.1 | 2689 | 657.5 | 14 |

Source: `results/gpu_witness_carlpc2.json` (RTX 4060 Ti 8 GB, torch 2.11.0+cu128, CUDA 12.8), produced by `scripts/gpu_witness.py`: fp16 AMP, plain AdamW over *all* parameters (i.e. full fine-tuning), random 20-class inputs at the native size, batch 16, 10 timed steps after 3 warm-up steps; inference = fp16 forward at batch 16. \*Derived, not measured: 2010 fold-0 training images ÷ train img/s — a compute-only lower bound per epoch that ignores JPEG decoding, augmentation, validation and the config's own batch size.


Registered but **not** covered by a config file or the GPU witness: `dinov2_vitl14`. They build through `build_model`, but no VRAM/throughput numbers exist for them on the 8 GB card — measure with `uv run python scripts/gpu_witness.py` before planning runs.

## 8. Typical failure modes on small, noisy web data

* **Fine-tuning erases the advantage.** Too high an LR moves DINOv2 towards an ordinary supervised model trained
  on 2k noisy images. If fine-tuning loses to the probe, lower `train.lr` or raise `train.freeze_epochs`.
* **Features encode everything, including off-task content.** A self-supervised model represents the Nintendo
  Switch console under `switch` very well — as a games console. The probe will happily separate it, so OOD images
  inflate rather than hurt CV scores; audit with OOF probabilities.
* **Resolution.** At 224 px with 14 px patches, tiny components occupy few tokens.

## 9. Try this

```bash
# 1. linear probe on all five folds (features cached in results/cache/features/)
uv run python -m compclass.features --model dinov2_vits14
uv run python -m compclass.features --model dinov2_vitb14
# 2. fine-tune ViT-S/14 on one fold and compare with its probe on the same fold
uv run python -m compclass.train --config configs/dinov2_vits14.yaml --fold 0
# 3. backbone gets its own warm-up after the frozen phase
uv run python -m compclass.train --config configs/dinov2_vitb14.yaml --fold 0 --set train.warmup_epochs=4
# 4. higher resolution for small parts
uv run python -m compclass.train --config configs/dinov2_vits14.yaml --fold 0 --set model.kwargs.input_size=336
```

## Results on data v2

> **Placeholder — no v2 results exist yet.** See `results/benchmark.csv` (generated from `results/runs.jsonl`) once the benchmark has run. Report mean ± std over the 5 dev folds; never quote the sealed test here.

| method | key | CV macro-F1 | CV accuracy | notes |
|---|---|---:|---:|---|
| — | — | *pending* | *pending* | |


## References

* M. Oquab et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193.
* M. Caron et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers.* arXiv:2104.14294.
* J. Zhou et al. (2021). *iBOT: Image BERT Pre-Training with Online Tokenizer.* arXiv:2111.07832.
* H. Touvron et al. (2021). *Going deeper with Image Transformers.* arXiv:2103.17239.
