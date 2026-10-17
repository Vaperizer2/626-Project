# CLIP — contrastive image–text pretraining

**Registry keys:** `clip_vitb16`, `clip_vitl14` (configs exist), `clip_vitl14_336` (no config) · timm image towers
`vit_{base_patch16,large_patch14}_clip_quickgelu_224.openai` · open_clip text towers for zero-shot ·
**Builder:** `_clip` · **Configs:** `configs/clip_vitb16.yaml`, `configs/clip_vitl14.yaml` · **Zero-shot:**
`src/compclass/zeroshot.py`

## 1. Core idea and why it was invented

A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark,
G. Krueger, I. Sutskever (2021). *Learning Transferable Visual Models From Natural Language Supervision.*
arXiv:2103.00020.

Fixed label sets limit what a model learns and require new labelled data for every new task. The web already
contains hundreds of millions of images with free-form captions. CLIP trains an **image encoder** and a **text
encoder** jointly so that an image and its own caption map to nearby unit vectors, and mismatched pairs map far
apart. Because the "classes" are arbitrary text, the trained model can classify **zero-shot**: embed the prompt
"a photo of a resistor." for each class and pick the class whose text embedding is closest to the image.

**Loss** — for a batch of N pairs with L2-normalised embeddings I_i, T_j and a learned temperature τ:
logits_ij = I_i · T_j / τ; loss = ½ [CE(rows → diagonal) + CE(columns → diagonal)]. Each image must pick its own
caption out of the N in the batch (and vice versa): a softmax over the batch, which is why CLIP was trained with
very large batches (32k).

## 2. Architecture

```
          image tower (timm, fine-tuned here)                    text tower (open_clip, zero-shot only)
x 3×224×224                                                      "a photo of a {class}." × 6 templates
 │ patch embed (16 px for B/16, 14 px for L/14)                    │ BPE tokens → 12-layer Transformer
 │ + [CLS] + pos-embed → LayerNorm ("ln_pre")                      │ → EOT token → projection
 │ 12 (B) / 24 (L) blocks, QuickGELU MLP                           ▼ text embedding, L2-normalised
 │ LayerNorm → [CLS] token                                         (averaged over templates → class vector)
 │   ├─ fine-tune / probe: head Linear(768 or 1024 → 20)   ◄── this repo's timm model (projection dropped)
 │   └─ zero-shot: projection → 512 (B) / 768 (L) ────────── cosine similarity ──► argmax over 20 classes
```

Verified in timm 1.0.30: `clip_vitb16` has a `LayerNorm` before the blocks (`norm_pre`), `[CLS]` pooling and
`QuickGELU` activations; with a 20-class head the new `Linear` sits on the 768-d pre-projection feature
(head = 15,380 parameters). The linear-probe features are therefore 768-d (B/16) / 1024-d (L/14), not CLIP's
512/768-d joint embedding.

**QuickGELU.** OpenAI trained with x·σ(1.702x); timm's `_quickgelu` variants and open_clip's
`force_quick_gelu=True` (set in `zeroshot.py` when the tag is `openai`) reproduce it. Plain GELU silently
degrades these weights.

## 3. Pretraining data and objective

400 M (image, text) pairs collected from the web (OpenAI's WIT dataset, not public); symmetric InfoNCE loss as
above. Normalisation: CLIP's own mean (0.4815, 0.4578, 0.4082) / std (0.2686, 0.2613, 0.2758).

## 4. Hyperparameters that matter here

Effective recipe of `configs/clip_vitb16.yaml` after merging `base.yaml` (computed with `compclass.config.load_config`):

| key | value |
|---|---|
| `model.key` | `clip_vitb16` |
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

Preprocessing (`PreprocessSpec`): input 224 px, mean (0.4815, 0.4578, 0.4082), std (0.2686, 0.2613, 0.2758), bicubic resize, `crop_pct` 0.9 → evaluation resizes the short side to 249 px, then centre-crops 224 px. Parameters with a 20-class head: 85,814,804.

Schedule arithmetic for fold 0 (2010 training rows, `drop_last=True`): 125 micro-batches → 63 optimiser steps per epoch; LR warm-up lasts 126 steps; the backbone unfreezes after 126 steps, when the LR is at 1.00× its peak.

Layer-wise LR decay 0.65 yields 14 distinct LR scales over timm's `group_matcher` layer groups: the head trains at 3.0e-05; the bottom group (stem / patch embedding) gets scale 0.003697 = 0.65^13, i.e. LR 1.11e-07 (≈ 270× smaller than the head).


`configs/clip_vitl14.yaml` (≈300 M parameters):

Effective recipe of `configs/clip_vitl14.yaml` after merging `base.yaml` (computed with `compclass.config.load_config`):

| key | value |
|---|---|
| `model.key` | `clip_vitl14` |
| `model.pretrained` | True |
| `model.kwargs` | `{'drop_path_rate': 0.1}` |
| `train.epochs / train.patience` | 30 / 8 |
| `train.batch_size × train.accum_steps` | 8 × 4 = 32 |
| `train.lr (peak)` | 2e-05 |
| `train.weight_decay` | 0.1 |
| `train.warmup_epochs` | 2 |
| `train.freeze_epochs` | 2 |
| `train.layer_decay` | 0.75 |
| `train.label_smoothing` | 0.1 |
| `train.amp` | fp16 |
| `train.grad_clip` | 1.0 |
| `train.grad_checkpointing` | True |
| `aug.rrc_scale` | [0.5, 1.0] |
| `aug.randaugment` | {'n': 2, 'm': 9} |
| `aug.color_jitter` | 0.0 |
| `aug.mixup / aug.cutmix` | 0.0 / 0.0 |
| `aug.random_erasing` | 0.0 |

Preprocessing (`PreprocessSpec`): input 224 px, mean (0.4815, 0.4578, 0.4082), std (0.2686, 0.2613, 0.2758), bicubic resize, `crop_pct` 1.0 → evaluation resizes the short side to 224 px, then centre-crops 224 px. Parameters with a 20-class head: 303,200,276.

Schedule arithmetic for fold 0 (2010 training rows, `drop_last=True`): 251 micro-batches → 63 optimiser steps per epoch; LR warm-up lasts 126 steps; the backbone unfreezes after 126 steps, when the LR is at 1.00× its peak.

Layer-wise LR decay 0.75 yields 26 distinct LR scales over timm's `group_matcher` layer groups: the head trains at 2.0e-05; the bottom group (stem / patch embedding) gets scale 0.000753 = 0.75^25, i.e. LR 1.51e-08 (≈ 1328× smaller than the head).

* **Very low LR + strong LLRD + head warm-up** (the config comment: "CLIP towers are easy to forget"). Fine-tuning
  CLIP with an ordinary recipe is known to underperform; careful recipes (low LR, LLRD, EMA) make it competitive —
  X. Dong et al. (2022), *CLIP Itself is a Strong Fine-tuner*, arXiv:2212.06138.
* **The LLRD arithmetic is extreme here** (lines above): for B/16 the patch embedding trains ≈270× slower than the
  head (LR ≈ 1.1e-7); for L/14 ≈1,300× slower (LR ≈ 1.5e-8) — those layers are effectively frozen. That is a
  deliberate choice, but it is worth ablating (`train.layer_decay=0.8`).
* **`train.weight_decay: 0.1`.** Note that AdamW decay pulls weights towards *zero*, not towards the pretrained
  weights; with long training it slowly erodes pretrained structure.
* **`train.grad_checkpointing: true` + batch 8 × 4 for L/14.** Recompute activations in backward to fit 8 GB.
* **`crop_pct` differs between towers**: 0.9 for B/16, 1.0 for L/14 (both read from timm's pretrained configs).
  Zero-shot uses open_clip's own `preprocess`, not `PreprocessSpec`.

**Design note — warm-up overlaps the frozen phase.** The LR warm-up (`warmup_epochs`) is counted from step 0, including the head-only epochs. When `freeze_epochs` ≥ `warmup_epochs`, the backbone is unfrozen at (or near) the *peak* LR with no ramp of its own; see the multiplier in the schedule line above. To give the backbone its own ramp, increase warm-up, e.g. `--set train.warmup_epochs=4`.

## 5. Adaptation regimes

| regime | command | when it wins |
|---|---|---|
| zero-shot | `uv run python -m compclass.zeroshot --model clip_vitb16` | no training data; also a **data-audit tool** (v2's OOD screen used CLIP zero-shot with distractor prompts) |
| linear probe | `uv run python -m compclass.features --model clip_vitb16` | cheap and strong; robust under label noise |
| full fine-tune + LLRD | `uv run python -m compclass.train --config configs/clip_vitb16.yaml --fold 0` | when it beats the probe on CV — not guaranteed on ~2k noisy images |
| (not implemented) WiSE-FT | average zero-shot and fine-tuned weights | robustness under distribution shift — M. Wortsman et al. 2021, arXiv:2109.01903 |

Zero-shot details (`zeroshot.py`): 6 prompt templates (e.g. "a photo of a {}.", "a {} on a white background."),
some class names rewritten into natural phrases (`PROMPT_NAMES`, e.g. `ic chip` → "integrated circuit chip"); text
embeddings are normalised, averaged over templates and re-normalised. The CLIP paper used 80 templates for ImageNet;
prompt wording is itself a hyperparameter.

## 6. Compute on an 8 GB RTX 4060 Ti

| key | params (M) | input px | batch | grad ckpt | train it/s | train img/s | peak VRAM (MB) | infer img/s | ≈ s/epoch* |
|---|---:|---:|---:|:-:|---:|---:|---:|---:|---:|
| `clip_vitb16` | 85.81 | 224 | 16 | no | 10.06 | 160.9 | 2462 | 768.9 | 12 |
| `clip_vitl14` | 303.2 | 224 | 16 | yes | 1.98 | 31.6 | 5811 | 173.9 | 64 |

Source: `results/gpu_witness_carlpc2.json` (RTX 4060 Ti 8 GB, torch 2.11.0+cu128, CUDA 12.8), produced by `scripts/gpu_witness.py`: fp16 AMP, plain AdamW over *all* parameters (i.e. full fine-tuning), random 20-class inputs at the native size, batch 16, 10 timed steps after 3 warm-up steps; inference = fp16 forward at batch 16. \*Derived, not measured: 2010 fold-0 training images ÷ train img/s — a compute-only lower bound per epoch that ignores JPEG decoding, augmentation, validation and the config's own batch size.

`clip_vitl14` ran out of memory at [{'batch': 16, 'grad_checkpointing': False}]; the row above is the retry with gradient checkpointing.


The `clip_vitl14` config runs at batch 8 with checkpointing, so its VRAM will be below the 5,811 MB measured at batch 16.
Registered but **not** covered by a config file or the GPU witness: `clip_vitl14_336`. They build through `build_model`, but no VRAM/throughput numbers exist for them on the 8 GB card — measure with `uv run python scripts/gpu_witness.py` before planning runs.

## 7. Typical failure modes on small, noisy web data

* **Homonyms in zero-shot.** "switch" (Nintendo Switch, network switch), "transformer" (toys, power-grid
  transformers), "relay" (relay races), "fuse" (brand logos): CLIP knows all senses and the prompt does not
  disambiguate. Templates such as "a photo of a {}, an electronic component." exist to help.
* **Catastrophic forgetting during fine-tuning.** The zero-shot / open-vocabulary ability and robustness are lost
  quickly at normal LRs (see [`../concepts/transfer_learning.md`](../concepts/transfer_learning.md)).
* **Web-caption bias.** CLIP features are strong for product photos (common on the web) and weaker for unusual
  close-ups.
* **Wrong activation.** Loading OpenAI weights into a plain-GELU model gives silently worse results.

## 8. Try this

```bash
# 1. zero-shot on all five folds (nothing is fitted, so every fold is fair)
uv run python -m compclass.zeroshot --model clip_vitb16
uv run python -m compclass.zeroshot --model clip_vitl14
# 2. linear probe vs fine-tune on fold 0
uv run python -m compclass.features --model clip_vitb16 --folds 0
uv run python -m compclass.train --config configs/clip_vitb16.yaml --fold 0
# 3. weaker layer-wise decay (lower layers actually move)
uv run python -m compclass.train --config configs/clip_vitb16.yaml --fold 0 --set train.layer_decay=0.8
# 4. give the backbone its own warm-up after the frozen phase
uv run python -m compclass.train --config configs/clip_vitb16.yaml --fold 0 --set train.warmup_epochs=4
```

## Results on data v2

> **Placeholder — no v2 results exist yet.** See `results/benchmark.csv` (generated from `results/runs.jsonl`) once the benchmark has run. Report mean ± std over the 5 dev folds; never quote the sealed test here.

| method | key | CV macro-F1 | CV accuracy | notes |
|---|---|---:|---:|---|
| — | — | *pending* | *pending* | |


## References

* A. Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* arXiv:2103.00020.
* X. Dong et al. (2022). *CLIP Itself is a Strong Fine-tuner: Achieving 85.7% and 88.0% Top-1 Accuracy with ViT-B and ViT-L on ImageNet.* arXiv:2212.06138.
* M. Wortsman et al. (2021). *Robust fine-tuning of zero-shot models.* arXiv:2109.01903.
