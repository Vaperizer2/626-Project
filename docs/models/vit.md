# ViT — the Vision Transformer

**Registry key:** `vit_b16` (timm `vit_base_patch16_224.augreg_in21k_ft_in1k`) · **Builder:** `_vit` ·
**Config:** `configs/vit_b16.yaml`. The same backbone design underlies DINOv2, CLIP and SigLIP in this repo.

## 1. Core idea and why it was invented

A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer,
G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby (2020). *An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale.* arXiv:2010.11929 (ICLR 2021).

Transformers (A. Vaswani et al. 2017, *Attention Is All You Need*, arXiv:1706.03762) had replaced recurrent
networks in NLP. ViT applies a standard Transformer encoder to an image with minimal changes: cut the image into
16×16 patches, embed each patch linearly as a "token", add a learned position embedding, prepend a learnable
`[CLS]` token and run self-attention. Every token can attend to every other token from the first layer, so the
receptive field is global from the start.

The catch: ViT has **weak inductive bias** — no built-in locality or translation equivariance. Trained on
ImageNet-1k alone it underperforms ResNets; it overtakes them only with large-scale pretraining (ImageNet-21k,
JFT-300M). The `augreg` weights come from A. Steiner, A. Kolesnikov, X. Zhai, R. Wightman, J. Uszkoreit, L. Beyer
(2021), *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers*, arXiv:2106.10270,
which showed that augmentation + regularisation can substitute for part of the data.

## 2. Architecture (ViT-B/16 at 224 px; verified against timm 1.0.30)

```
x 3×224×224
 │ patch embedding: conv16×16 stride 16 (= linear map of each flattened patch) → 14×14 = 196 tokens × 768
 │ prepend [CLS] → 197 tokens;  + learned position embedding (1×197×768)
 │ 12 × Transformer block (pre-norm):
 │      t = t + MHSA(LayerNorm(t))           12 heads × 64 dims, attention over all 197 tokens
 │      t = t + MLP(LayerNorm(t))            768 → 3072 → 768, GELU
 │      (+ stochastic depth on each branch when drop_path_rate > 0)
 │ final LayerNorm → take the [CLS] token (global_pool="token") → head Linear(768 → 20)
 ▼ logits
```

Attention cost grows with the square of the token count N = (input_size / 16)². At 224 px N = 196; at 288 px
N = 324, so attention FLOPs grow ≈ (324/196)² ≈ 2.7×. timm resamples the pretrained position embeddings when
`img_size` changes, so `--set model.kwargs.input_size=288` works for ViTs (verified).

## 3. Pretraining data and objective

Supervised classification on ImageNet-21k (≈14 M images) with the AugReg recipe, then fine-tuned on
ImageNet-1k at 224 px. Normalisation is mean = std = 0.5 (not the ImageNet statistics) — the `PreprocessSpec`
below picks this up automatically from the weights.

## 4. Hyperparameters that matter here

Effective recipe of `configs/vit_b16.yaml` after merging `base.yaml` (computed with `compclass.config.load_config`):

| key | value |
|---|---|
| `model.key` | `vit_b16` |
| `model.pretrained` | True |
| `model.kwargs` | `{'drop_path_rate': 0.1}` |
| `train.epochs / train.patience` | 30 / 8 |
| `train.batch_size × train.accum_steps` | 16 × 2 = 32 |
| `train.lr (peak)` | 0.0001 |
| `train.weight_decay` | 0.05 |
| `train.warmup_epochs` | 2 |
| `train.freeze_epochs` | 1 |
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

Preprocessing (`PreprocessSpec`): input 224 px, mean (0.5, 0.5, 0.5), std (0.5, 0.5, 0.5), bicubic resize, `crop_pct` 0.9 → evaluation resizes the short side to 249 px, then centre-crops 224 px. Parameters with a 20-class head: 85,814,036.

Schedule arithmetic for fold 0 (2010 training rows, `drop_last=True`): 125 micro-batches → 63 optimiser steps per epoch; LR warm-up lasts 126 steps; the backbone unfreezes after 63 steps, when the LR is at 0.51× its peak.

Layer-wise LR decay 0.75 yields 14 distinct LR scales over timm's `group_matcher` layer groups: the head trains at 1.0e-04; the bottom group (stem / patch embedding) gets scale 0.02376 = 0.75^13, i.e. LR 2.38e-06 (≈ 42× smaller than the head).


* **`train.lr: 1.0e-4` with `train.layer_decay: 0.75`.** timm groups ViT-B into 14 groups (patch/position
  embedding + `[CLS]`, 12 blocks, head). Lower layers hold generic features and should move least.
* **`train.batch_size: 16`, `train.accum_steps: 2`.** Effective batch 32 as in `base.yaml`, at half the activation
  memory. With accumulation the loss of each micro-batch is divided by `accum_steps` before `backward()`.
* **`model.kwargs.drop_path_rate: 0.1`.** Stochastic depth, standard for ViT fine-tuning.
* **`train.amp: fp16`.** Attention softmax in fp16 is safe under `torch.autocast`; the `GradScaler` handles
  underflow. `bf16` avoids the scaler.

**Design note — warm-up overlaps the frozen phase.** The LR warm-up (`warmup_epochs`) is counted from step 0, including the head-only epochs. When `freeze_epochs` ≥ `warmup_epochs`, the backbone is unfrozen at (or near) the *peak* LR with no ramp of its own; see the multiplier in the schedule line above. To give the backbone its own ramp, increase warm-up, e.g. `--set train.warmup_epochs=4`.

## 5. Adaptation regimes

| regime | command | expectation |
|---|---|---|
| linear probe on `[CLS]` | `uv run python -m compclass.features --model vit_b16` | supervised `[CLS]` features are tuned to ImageNet-1k classes; probes are usually behind DINOv2/CLIP |
| full fine-tune + LLRD | default config | the standard ViT recipe |
| higher resolution | `--set model.kwargs.input_size=288` | more tokens → finer detail for small components, ≈2.7× attention cost |

## 6. Compute on an 8 GB RTX 4060 Ti

| key | params (M) | input px | batch | grad ckpt | train it/s | train img/s | peak VRAM (MB) | infer img/s | ≈ s/epoch* |
|---|---:|---:|---:|:-:|---:|---:|---:|---:|---:|
| `vit_b16` | 85.81 | 224 | 16 | no | 11.05 | 176.8 | 2232 | 908.5 | 11 |

Source: `results/gpu_witness_carlpc2.json` (RTX 4060 Ti 8 GB, torch 2.11.0+cu128, CUDA 12.8), produced by `scripts/gpu_witness.py`: fp16 AMP, plain AdamW over *all* parameters (i.e. full fine-tuning), random 20-class inputs at the native size, batch 16, 10 timed steps after 3 warm-up steps; inference = fp16 forward at batch 16. \*Derived, not measured: 2010 fold-0 training images ÷ train img/s — a compute-only lower bound per epoch that ignores JPEG decoding, augmentation, validation and the config's own batch size.


## 7. Typical failure modes on small, noisy web data

* **Fine-tuning instability.** A high LR or no LLRD can destroy pretrained attention patterns in the first epochs:
  validation macro-F1 drops right after unfreezing and never recovers. Lower `train.lr` or `train.layer_decay`.
* **Data hunger.** With weak inductive bias and ~2k images, ViT relies entirely on its pretraining; heavy
  augmentation (RandAugment m=9 plus Mixup) can underfit.
* **Patch-grid aliasing for tiny objects.** A small SMD component may occupy only a few 16-px patches at 224 px.
* **Shortcut tokens.** Watermarks and text overlays on web images are easy to attend to.

## 8. Try this

```bash
uv run python -m compclass.train --config configs/vit_b16.yaml --fold 0
# LR sensitivity
uv run python -m compclass.train --config configs/vit_b16.yaml --fold 0 --set train.lr=1.0e-4
uv run python -m compclass.train --config configs/vit_b16.yaml --fold 0 --set train.lr=3.0e-5
# no layer-wise decay
uv run python -m compclass.train --config configs/vit_b16.yaml --fold 0 --set train.layer_decay=null
# higher resolution (position embeddings are interpolated)
uv run python -m compclass.train --config configs/vit_b16.yaml --fold 0 --set model.kwargs.input_size=288
```

## Results on data v2

> **Placeholder — no v2 results exist yet.** See `results/benchmark.csv` (generated from `results/runs.jsonl`) once the benchmark has run. Report mean ± std over the 5 dev folds; never quote the sealed test here.

| method | key | CV macro-F1 | CV accuracy | notes |
|---|---|---:|---:|---|
| — | — | *pending* | *pending* | |


## References

* A. Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* arXiv:2010.11929.
* A. Vaswani et al. (2017). *Attention Is All You Need.* arXiv:1706.03762.
* A. Steiner et al. (2021). *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers.* arXiv:2106.10270.
