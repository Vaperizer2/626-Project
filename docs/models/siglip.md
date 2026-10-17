# SigLIP and SigLIP 2 — image–text pretraining with a sigmoid loss

**Registry keys:** `siglip_b16`, `siglip2_b16` (configs exist), `siglip_so400m_384`, `siglip2_so400m_384` (no configs) ·
timm `vit_base_patch16_siglip_224.{webli,v2_webli}` · open_clip `ViT-B-16-SigLIP` / `ViT-B-16-SigLIP2` (`webli`) for
zero-shot · **Builder:** `_siglip` · **Configs:** `configs/siglip_b16.yaml`, `configs/siglip2_b16.yaml` (inherits
from `siglip_b16.yaml` and changes only `model.key`)

## 1. Core idea and why it was invented

X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer (2023). *Sigmoid Loss for Language Image Pre-Training.*
arXiv:2303.15343 (ICCV 2023).

CLIP's softmax contrastive loss normalises over the whole batch: each image competes against every caption in
the batch, which needs very large batches and an all-gather of all embeddings across devices. SigLIP replaces it
with an independent **binary** decision per pair: for every (image i, text j) in the batch,

    loss_ij = −log σ( z_ij · (t · I_i·T_j + b) ),   z_ij = +1 if i = j else −1,

with learned temperature t and bias b (the bias offsets the heavy imbalance: N positives vs N² − N negatives).
No batch-wide normalisation is needed, so it works well at moderate batch sizes and scales memory-efficiently.

M. Tschannen, A. Gritsenko, X. Wang, et al. (2025). *SigLIP 2: Multilingual Vision-Language Encoders
with Improved Semantic Understanding, Localization, and Dense Features.* arXiv:2502.14786. SigLIP 2 keeps the
sigmoid loss and adds a captioning-based decoder objective, self-distillation and masked prediction (later in
training), and more multilingual, de-biased data — improving dense and localisation features. The B/16 image
tower has the same architecture and parameter count as SigLIP's (92.9 M with a 20-class head).

## 2. Architecture (B/16 at 224 px; verified against timm 1.0.30)

```
x 3×224×224
 │ patch embed 16×16 → 196 tokens × 768          (no [CLS] token)
 │ + learned position embedding (1×196×768)
 │ 12 × pre-norm Transformer block, 12 heads, MLP 3072, GELU
 │ final LayerNorm
 │ MAP head ("multihead attention pooling", timm AttentionPoolLatent):
 │     one learned query token attends over the 196 patch tokens → MLP → 768-d image feature
 │   ├─ fine-tune / probe: head Linear(768 → 20)
 │   └─ zero-shot (open_clip): image embedding vs prompt embeddings
 ▼ logits
```

The MAP head is part of the backbone in this repo: `get_classifier()` is only the final `Linear` (15,380
parameters), so the attention-pooling block is frozen during `freeze_epochs`. Under LLRD, timm puts `attn_pool.*` in the
*same group as the head* (verified with `group_parameters`), so after unfreezing it trains at the full head LR.

## 3. Pretraining data and objective

WebLI (Google's multilingual web image–text dataset; X. Chen et al. 2022, *PaLI*, arXiv:2209.06794), sigmoid
loss; SigLIP 2 adds the objectives listed above. Normalisation is mean = std = 0.5.

## 4. Hyperparameters that matter here

Effective recipe of `configs/siglip_b16.yaml` after merging `base.yaml` (computed with `compclass.config.load_config`):

| key | value |
|---|---|
| `model.key` | `siglip_b16` |
| `model.pretrained` | True |
| `model.kwargs` | `{'drop_path_rate': 0.1}` |
| `train.epochs / train.patience` | 30 / 8 |
| `train.batch_size × train.accum_steps` | 16 × 2 = 32 |
| `train.lr (peak)` | 3e-05 |
| `train.weight_decay` | 0.1 |
| `train.warmup_epochs` | 2 |
| `train.freeze_epochs` | 2 |
| `train.layer_decay` | 0.65 |
| `train.label_smoothing` | 0.1 |
| `train.amp` | fp16 |
| `train.grad_clip` | 1.0 |
| `train.grad_checkpointing` | False |
| `aug.rrc_scale` | [0.5, 1.0] |
| `aug.randaugment` | {'n': 2, 'm': 9} |
| `aug.color_jitter` | 0.0 |
| `aug.mixup / aug.cutmix` | 0.0 / 0.0 |
| `aug.random_erasing` | 0.0 |

Preprocessing (`PreprocessSpec`): input 224 px, mean (0.5, 0.5, 0.5), std (0.5, 0.5, 0.5), bicubic resize, `crop_pct` 0.9 → evaluation resizes the short side to 249 px, then centre-crops 224 px. Parameters with a 20-class head: 92,899,604.

Schedule arithmetic for fold 0 (2010 training rows, `drop_last=True`): 125 micro-batches → 63 optimiser steps per epoch; LR warm-up lasts 126 steps; the backbone unfreezes after 126 steps, when the LR is at 1.00× its peak.

Layer-wise LR decay 0.65 yields 14 distinct LR scales over timm's `group_matcher` layer groups: the head trains at 3.0e-05; the bottom group (stem / patch embedding) gets scale 0.003697 = 0.65^13, i.e. LR 1.11e-07 (≈ 270× smaller than the head).


The recipe mirrors `clip_vitb16.yaml` (low LR 3.0e-5, `layer_decay` 0.65, `freeze_epochs` 2, weight decay 0.1),
which makes CLIP-B/16 vs SigLIP-B/16 vs SigLIP2-B/16 a controlled comparison of *pretraining*, not of recipe.

**Design note — warm-up overlaps the frozen phase.** The LR warm-up (`warmup_epochs`) is counted from step 0, including the head-only epochs. When `freeze_epochs` ≥ `warmup_epochs`, the backbone is unfrozen at (or near) the *peak* LR with no ramp of its own; see the multiplier in the schedule line above. To give the backbone its own ramp, increase warm-up, e.g. `--set train.warmup_epochs=4`.

## 5. Adaptation regimes

| regime | command | notes |
|---|---|---|
| zero-shot | `uv run python -m compclass.zeroshot --model siglip2_b16` | `zeroshot.py` ranks classes by cosine similarity; SigLIP's calibrated sigmoid score is a monotone function of that cosine, so the argmax is identical |
| linear probe | `uv run python -m compclass.features --model siglip_b16` | 768-d MAP-pooled features |
| full fine-tune + LLRD | `uv run python -m compclass.train --config configs/siglip2_b16.yaml --fold 0` | same caveats as CLIP |

## 6. Compute on an 8 GB RTX 4060 Ti

| key | params (M) | input px | batch | grad ckpt | train it/s | train img/s | peak VRAM (MB) | infer img/s | ≈ s/epoch* |
|---|---:|---:|---:|:-:|---:|---:|---:|---:|---:|
| `siglip_b16` | 92.9 | 224 | 16 | no | 10.72 | 171.6 | 2334 | 806.7 | 12 |
| `siglip2_b16` | 92.9 | 224 | 16 | no | 10.4 | 166.4 | 2334 | 807.1 | 12 |

Source: `results/gpu_witness_carlpc2.json` (RTX 4060 Ti 8 GB, torch 2.11.0+cu128, CUDA 12.8), produced by `scripts/gpu_witness.py`: fp16 AMP, plain AdamW over *all* parameters (i.e. full fine-tuning), random 20-class inputs at the native size, batch 16, 10 timed steps after 3 warm-up steps; inference = fp16 forward at batch 16. \*Derived, not measured: 2010 fold-0 training images ÷ train img/s — a compute-only lower bound per epoch that ignores JPEG decoding, augmentation, validation and the config's own batch size.


Registered but **not** covered by a config file or the GPU witness: `siglip_so400m_384`, `siglip2_so400m_384`. They build through `build_model`, but no VRAM/throughput numbers exist for them on the 8 GB card — measure with `uv run python scripts/gpu_witness.py` before planning runs. The SO400M towers run at 384 px, so expect
far more VRAM than the B/16 models; a linear probe is the realistic use on this card.

## 7. Typical failure modes on small, noisy web data

* **Same homonym problem as CLIP** for zero-shot ("switch", "transformer", "relay", "fuse").
* **Prompt mismatch.** SigLIP's text tower was trained on WebLI alt-text; the 6 English templates in
  `zeroshot.py` are a reasonable but untuned choice.
* **Forgetting under fine-tuning**, as for CLIP; check that fine-tuning beats the probe before using it.
* **Mean/std 0.5.** Pre-processing must come from `PreprocessSpec` (it does in this repo, including ONNX
  `preprocess.json`); feeding ImageNet-normalised inputs degrades results silently.

## 8. Try this

```bash
# 1. zero-shot: CLIP vs SigLIP vs SigLIP 2 at matched size
uv run python -m compclass.zeroshot --model clip_vitb16
uv run python -m compclass.zeroshot --model siglip_b16
uv run python -m compclass.zeroshot --model siglip2_b16
# 2. probes at matched size
uv run python -m compclass.features --model siglip2_b16
# 3. fine-tune SigLIP 2 with weaker layer-wise decay
uv run python -m compclass.train --config configs/siglip2_b16.yaml --fold 0 --set train.layer_decay=0.8
```

## Results on data v2

> **Placeholder — no v2 results exist yet.** See `results/benchmark.csv` (generated from `results/runs.jsonl`) once the benchmark has run. Report mean ± std over the 5 dev folds; never quote the sealed test here.

| method | key | CV macro-F1 | CV accuracy | notes |
|---|---|---:|---:|---|
| — | — | *pending* | *pending* | |


## References

* X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer (2023). *Sigmoid Loss for Language Image Pre-Training.* arXiv:2303.15343.
* M. Tschannen et al. (2025). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.* arXiv:2502.14786.
* X. Chen et al. (2022). *PaLI: A Jointly-Scaled Multilingual Language-Image Model.* arXiv:2209.06794.
