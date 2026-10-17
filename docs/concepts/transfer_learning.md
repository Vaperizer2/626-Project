# Transfer learning: probe, fine-tune, and the knobs in between

Every model in this repo except `scratch_cnn` and the classical baseline starts from weights learned on a much
larger dataset. This page explains the adaptation regimes, why fine-tuning can *hurt*, and what each knob in
`configs/base.yaml` does in `src/compclass/train.py`.

## 1. Why transfer works

Early layers of a vision network learn generic features (oriented edges, colour blobs, textures); later layers
become increasingly specific to the pretraining task. J. Yosinski, J. Clune, Y. Bengio, H. Lipson (2014),
*How transferable are features in deep neural networks?*, arXiv:1411.1792, measured this directly by transplanting
layers between networks. S. Kornblith, J. Shlens, Q. V. Le (2019), *Do Better ImageNet Models Transfer Better?*,
arXiv:1805.08974, found that ImageNet accuracy predicts transfer accuracy well for fine-tuning, less so for
fixed features. With ~2,000 training images per fold, learning all of this from scratch is hopeless — that is
the point of [`../models/scratch_cnn.md`](../models/scratch_cnn.md).

## 2. Adaptation regimes

```
             nothing trained          only a linear head          everything trained
             ───────────────────────────────────────────────────────────────────────►
zero-shot ──── linear probe ──── head-only fine-tune ──── partial ──── full fine-tune (+LLRD)
(CLIP/SigLIP)  (features.py)     (freeze_epochs ≥ epochs)  (LLRD≪1)    (train.py default)
```

| regime | code path | trains | strengths | weaknesses |
|---|---|---|---|---|
| zero-shot | `compclass.zeroshot` | nothing | no data needed; audits data | limited by prompts and homonyms |
| linear probe | `compclass.features` | logistic regression on frozen, cached features (`StandardScaler` + `LogisticRegression`, C ∈ {0.01, 0.1, 1, 10} chosen by grouped 3-fold CV inside the training fold) | minutes; no augmentation or LR tuning; the backbone cannot memorise label noise | no augmentation; features fixed |
| head-only | `train.py` with `--set train.freeze_epochs=<epochs>` | the new head, with augmentation | probe + augmentation | BatchNorm statistics still update (see below) |
| partial | strong `train.layer_decay` (e.g. 0.65) | all, but lower layers ~frozen | protects generic features | one more hyperparameter |
| full fine-tune | `train.py` default | all parameters | highest ceiling on in-distribution data | forgetting; needs careful LR |

**When does each win?** Rules of thumb (verify on CV for *this* dataset — that is what the benchmark is for):
strong self-supervised or vision-language features (DINOv2, CLIP, SigLIP) + few, noisy labels → the probe is
often within a few points of fine-tuning and sometimes ahead; supervised ImageNet CNNs → fine-tuning usually wins
clearly; distribution shift at deployment (internet images in the demo) → regimes that change the backbone less are
often more robust. A. Kumar, A. Raghunathan, R. Jones, T. Ma, P. Liang (2022), *Fine-Tuning can Distort Pretrained
Features and Underperform Out-of-Distribution*, arXiv:2202.10054, show that full fine-tuning with a *random* head
distorts features, and propose **LP-FT**: first fit the linear head, then fine-tune everything. `freeze_epochs` is a
light version of LP-FT (it trains the head with SGD for a few epochs rather than fitting the optimal probe).

## 3. Catastrophic forgetting

A network trained on a new task overwrites what it knew (M. McCloskey, N. Cohen 1989, *Catastrophic Interference
in Connectionist Networks*; J. Kirkpatrick et al. 2017, *Overcoming catastrophic forgetting in neural networks*,
arXiv:1612.00796). For transfer learning the "old task" is the general visual knowledge we want to keep. Symptoms
in this repo's per-epoch print-out: validation macro-F1 *drops* right after the backbone unfreezes, or peaks early
and decays. Mitigations available as config knobs:

1. Lower `train.lr`.
2. `train.layer_decay` (LLRD) — lower layers move less.
3. `train.freeze_epochs` — do not let a random head send large gradients into the backbone.
4. Fewer epochs / early stopping (`train.patience`) — keeps the best checkpoint, not the last.
5. Lower `train.weight_decay` — AdamW's decoupled decay pulls weights towards **zero**, not towards their pretrained
   values. (Decay towards the pretrained weights, "L2-SP", is X. Li, Y. Grandvalet, F. Davoine 2018, *Explicit
   Inductive Bias for Transfer Learning with Convolutional Networks*, arXiv:1802.01483 — not implemented here.)

## 4. Layer-wise learning-rate decay (LLRD)

Discriminative fine-tuning — different LRs for different layers — comes from J. Howard, S. Ruder (2018),
*Universal Language Model Fine-tuning for Text Classification*, arXiv:1801.06146; the geometric "layer decay"
form used for Transformers was popularised by K. Clark et al. (2020), *ELECTRA*, arXiv:2003.10555, and
H. Bao et al. (2021), *BEiT*, arXiv:2106.08254.

In `train.py`, `make_optimizer` calls `timm.optim.create_optimizer_v2(..., layer_decay=train.layer_decay)` when the
model exposes `group_matcher` (all timm models; not `scratch_cnn`). timm splits parameters into L layer groups
(bottom = stem/patch embedding, top = head) and gives group i the scale `layer_decay^(L-1-i)`; `train.py` multiplies
each group's LR by that scale, and the warm-up/cosine `LambdaLR` multiplies all groups equally. Group counts and the
resulting smallest LR for each config (computed with timm 1.0.30):

| config | `train.lr` | `layer_decay` | LR groups | bottom-group scale | bottom-group LR |
|---|---:|---:|---:|---:|---:|
| `clip_vitb16.yaml` | 3.0e-05 | 0.65 | 14 | 0.003697 | 1.11e-07 |
| `clip_vitl14.yaml` | 2.0e-05 | 0.75 | 26 | 0.000753 | 1.51e-08 |
| `convnext_tiny.yaml` | 2.0e-04 | 0.8 | 20 | 0.01441 | 2.88e-06 |
| `dinov2_vitb14.yaml` | 5.0e-05 | 0.75 | 14 | 0.02376 | 1.19e-06 |
| `dinov2_vits14.yaml` | 1.0e-04 | 0.75 | 14 | 0.02376 | 2.38e-06 |
| `siglip2_b16.yaml` | 3.0e-05 | 0.65 | 14 | 0.003697 | 1.11e-07 |
| `siglip_b16.yaml` | 3.0e-05 | 0.65 | 14 | 0.003697 | 1.11e-07 |
| `swin_t.yaml` | 2.0e-04 | 0.8 | 14 | 0.05498 | 1.10e-05 |
| `vit_b16.yaml` | 1.0e-04 | 0.75 | 14 | 0.02376 | 2.38e-06 |


Because the number of groups differs between architectures (14 for ViT-B and Swin-T, 20 for ConvNeXt-T, 26 for
ViT-L/14), **the same `layer_decay` value means different things for different families**. Compare "bottom-group LR",
not the raw decay factor. ResNet-50, EfficientNet-V2-S and MobileNet-V3 configs do not use LLRD; with
`--set train.layer_decay=0.7` they would get 18, 40 and 18 groups respectively.

## 5. The rest of the recipe — what each `base.yaml` knob does in `train.py`

| YAML key | default | what the code does | what changing it does |
|---|---|---|---|
| `train.lr` | 3.0e-4 | peak LR after warm-up (head LR when LLRD is on) | the most important knob; too high → forgetting, too low → underfit in 30 epochs |
| `train.weight_decay` | 0.05 | AdamW decoupled decay; timm skips 1-D params (biases, norm weights) | regularises; pulls towards zero (see §3) |
| `train.warmup_epochs` | 2 | linear 0→1 LR multiplier over `warmup_epochs × optimiser steps/epoch`, then cosine to 0 at the last epoch, stepped per optimiser step | avoids large early updates while AdamW's moment estimates are poor (P. Goyal et al. 2017, arXiv:1706.02677) |
| `train.freeze_epochs` | 0 | `set_backbone_trainable(model, epoch >= freeze_epochs)`: only `get_classifier()` params get gradients | LP-FT-like head warm-up |
| `train.layer_decay` | null | §4 | |
| `train.label_smoothing` | 0.1 | `CrossEntropyLoss(label_smoothing=…)`, or passed into the Mixup targets | targets become 0.9 + 0.1/20 on the true class; reduces over-confidence, mildly robust to label noise (C. Szegedy et al. 2015, arXiv:1512.00567; R. Müller, S. Kornblith, G. Hinton 2019, arXiv:1906.02629) |
| `aug.mixup`, `aug.cutmix` | 0.0 | when either α > 0, timm `Mixup` mixes pairs of images *and* labels per batch and the loss switches to `SoftTargetCrossEntropy` | strong regulariser (H. Zhang et al. 2017, arXiv:1710.09412; S. Yun et al. 2019, arXiv:1905.04899); can underfit with small data; `aug.mix_prob` (not in base.yaml) defaults to 1.0 |
| `aug.randaugment` | `{n: 2, m: 9}` | torchvision `RandAugment(num_ops=n, magnitude=m)` | E. D. Cubuk et al. 2019, arXiv:1909.13719; `null` disables |
| `aug.rrc_scale`, `aug.rrc_ratio` | [0.5, 1.0], [0.75, 1.333] | `RandomResizedCrop` area / aspect range | smaller min scale → more aggressive crops; can crop the component out |
| `aug.hflip`, `aug.vflip` | 0.5, 0.0 | flip probabilities | components are mostly flip-invariant; text on ICs is not |
| `aug.color_jitter` | 0.0 | `ColorJitter(b, c, s, hue=min(cj,0.5)/4)` | careful: colour is a real cue (resistor bands, LEDs) |
| `aug.random_erasing` | 0.0 | `RandomErasing(p, value="random")` after normalisation | occlusion robustness (Z. Zhong et al. 2017, arXiv:1708.04896) |
| `train.batch_size`, `train.accum_steps` | 32, 1 | loss ÷ `accum_steps`; `optimizer.step()` every `accum_steps` micro-batches → effective batch = product | same optimiser behaviour at lower VRAM; **BatchNorm still sees micro-batch statistics** |
| `train.amp` | fp16 | `torch.autocast` + `GradScaler` (fp16 only) | faster on tensor-core GPUs and roughly half the activation memory (P. Micikevicius et al. 2017, arXiv:1710.03740); `bf16` needs no scaler; `none` for debugging |
| `train.grad_clip` | 1.0 | `clip_grad_norm_` after `scaler.unscale_` | caps rare huge updates |
| `train.grad_checkpointing` | false | `model.set_grad_checkpointing(True)` | recompute activations in backward: much less VRAM, slower (T. Chen et al. 2016, arXiv:1604.06174); needed for ViT-L on 8 GB |
| `train.epochs`, `train.patience` | 30, 8 | early stopping on validation macro-F1; final metrics come from `best.pt` | |
| `model.kwargs.drop_path_rate` | — | stochastic depth inside timm blocks | regulariser for deep residual nets |
| `model.kwargs.input_size` | native | builder → timm `img_size` (ViTs resample position embeddings) | works for ViT/Swin/DINOv2/CLIP/SigLIP and `scratch_cnn`; **raises TypeError for ResNet/ConvNeXt/EfficientNet/MobileNet** (known issue) |

The optimiser is AdamW (I. Loshchilov, F. Hutter 2017, *Decoupled Weight Decay Regularization*, arXiv:1711.05101);
the schedule is cosine annealing without restarts (I. Loshchilov, F. Hutter 2016, *SGDR*, arXiv:1608.03983).

## 6. Two interactions worth knowing

1. **Warm-up overlaps the frozen phase.** The LR schedule starts at step 0, including the frozen epochs. With
   `freeze_epochs: 2` and `warmup_epochs: 2` (CLIP, SigLIP, DINOv2 configs) the backbone is unfrozen exactly when
   the LR reaches its peak, so the backbone itself gets no warm-up. With `freeze_epochs: 1` (ViT, Swin, ConvNeXt, CNNs)
   it unfreezes at ≈0.5× peak. Each model page lists the exact multiplier for fold 0.
2. **BatchNorm in frozen epochs.** Freezing sets `requires_grad=False` but the model is in `train()` mode, so BN
   running statistics (ResNet, EfficientNet, MobileNet, scratch CNN) keep adapting during `freeze_epochs`.
   LayerNorm models are unaffected.

## 7. Try this

```bash
# probe vs fine-tune on the same backbone and fold
uv run python -m compclass.features --model dinov2_vits14 --folds 0
uv run python -m compclass.train --config configs/dinov2_vits14.yaml --fold 0
# LP-FT strength: no head warm-up vs 5 epochs of head warm-up (with its own warm-up after)
uv run python -m compclass.train --config configs/vit_b16.yaml --fold 0 --set train.freeze_epochs=0
uv run python -m compclass.train --config configs/vit_b16.yaml --fold 0 --set train.freeze_epochs=5 train.warmup_epochs=7
# forgetting: a deliberately too-high LR
uv run python -m compclass.train --config configs/clip_vitb16.yaml --fold 0 --set train.lr=3.0e-4 train.layer_decay=null
# compare everything you ran
uv run python -c "import pandas as pd; d=pd.read_json('results/runs.jsonl', lines=True); print(d.groupby(['method','name'])[['accuracy','macro_f1']].agg(['mean','std','count']))"
```
