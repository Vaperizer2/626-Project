# Experiment log (v2)

Chronological record of decisions and results for the rebuilt pipeline. The pre-v2 log is archived at
[`legacy/docs/EXPERIMENT_LOG.md`](legacy/docs/EXPERIMENT_LOG.md); its numbers are **not comparable** (duplicate
leakage between train and val/test — see [`legacy/README.md`](legacy/README.md)).

## Evaluation contract

1. **Data:** `data/v2/manifest.json` (version `v2.0`). Every result records the manifest `fingerprint`; a
   result produced on a different fingerprint is not comparable.
2. **Model selection:** grouped 5-fold cross-validation on the `dev` split only. Groups are near-duplicate
   `cluster`s, so copies of one image never sit on both sides of a fold. Report the mean ± std of the five
   fold scores (primary metric: macro-F1; secondary: accuracy).
3. **Sealed test:** the `test` split is read only through `compclass.data.get_test(..., confirm=True)` /
   `evaluate --sealed-test`, **once**, for the final chosen model(s), after all choices are frozen.
   Any later look at test is logged here as such.
4. **Machine-readable record:** every run appends one JSON line to `results/runs.jsonl` (config hash, fold,
   params, train time, peak VRAM, throughput, metrics). This log holds the human interpretation.

## Entries

| Date | Experiment | CV macro-F1 (mean ± std) | CV accuracy | Notes |
|---|---|---:|---:|---|
| 2026-10-17 | Package scaffold (`src/compclass`), GPU witness on carlpc2 | — | — | Infrastructure only; no dataset results yet. |
| 2026-10-17 | Dataset v2 (`tools/build_dataset_v2.py`, fingerprint `431315e87d2612cb`) | — | — | 3,530 raw → 2,943 unique → 2,930 rows (13 conflicting-label groups excluded); 2,787 clusters; 816 CLIP OOD-suspect (27.8 %); test 418 / dev 2,512. `data/v2/leakage_report.json`. |
| 2026-10-17 | Phase 1: probes / zero-shot / HOG, 5 folds (`results/phase1/raw/runs.jsonl`) | best 0.8238 ± 0.0119 (SigLIP So400m probe) | 0.8252 | Zero-shot So400m 0.7758; CLIP-L/14@336 probe 0.7211; DINOv2-L probe 0.6670; ResNet-50 probe 0.5149; HOG+SVM 0.2059. In-scope acc. of best probe 0.8612. |
| 2026-10-17 | Phase 1 label audit + cleaning ablation (So400m probe) | — | keep 0.8252 / drop issues 0.8276 / drop OOD 0.7922 | 69 confident label issues (`results/phase1/label_issues_v2.csv`). In-scope acc. 0.8612 → 0.8628 (drop issues) / 0.8523 (drop OOD) → training data kept unchanged. |
| 2026-10-17 | Phase 1b: hflip-TTA probes (`results/phase1b/runs.jsonl`) | So400m 0.8189 (−0.50, 1/5 folds); SigLIP2 So400m 0.8201 (+0.57, 5/5) | 0.8209 / 0.8208 | Inconsistent sign, <1 pt → TTA not adopted. |
| 2026-10-17 | Phase 2, OLD schedule (warm-up not restarted at unfreeze), folds 0–2 (`results/phase2/raw_oldschedule/`) | So400m partial 0.8206; SigLIP2-B 0.7199; CLIP-B 0.6908; … scratch 0.3662 | So400m partial 0.8216 | Kept only as the schedule ablation (`schedule = "old (bug)"` in `results/benchmark.csv`). |
| 2026-10-18 | Phase 2, FIXED schedule, main fine-tune table (`results/phase2/runs_fixed.jsonl`) | SigLIP2-B 0.7329 ± 0.0267; CLIP-B 0.7020; DINOv2-B 0.6683; ViT-B 0.6541; ConvNeXt-T 0.6408; Swin-T 0.6393; EffNetV2-S 0.6079; ResNet-50 0.5801; MobileNetV3-L 0.5475; scratch 0.3552 | 0.7345 … 0.3654 | 5 folds each; So400m partial v2 (76M trainable) 0.8124 ± 0.0232 on folds 0–2 (probe on same folds 0.8216). Fine-tune > probe for CLIP-B/DINOv2-B/ConvNeXt-T/ResNet-50 (+3.6 to +6.5 pt, 5/5 folds); < probe for SigLIP2-B (−2.5, 0/5). |
| 2026-10-18 | Ablations: LR-schedule fix (10 families × folds 0–2), legacy-split leakage replay | schedule Δ +0.19 pt mean, 14/30 folds improved (sign test p = 0.86) | leakage: +5.5 / +11.0 / +13.6 pt (SigLIP / ConvNeXt-T / ResNet-50 probes) | Schedule fix = null result (noise floor ±1.6 pt); `results/schedule_ablation_folds012.csv`. Legacy split: 198 of 853 eval rows duplicate train; `results/leakage_simulation.csv`. |
| 2026-10-18 | Phase 3: final model fit (`compclass.final fit`, `exports/final/fit_report.json`) | OOF 0.8256 | OOF 0.8264 (probe-only 0.8248) | SigLIP So400m probe (C = 0.01) + zero-shot head (w = 0.25, T_zs = 50, fixed a priori); temperature 0.8977 (OOF ECE 0.0501 → 0.0295); abstain threshold 0.5 (OOF coverage 0.846 at 0.909 accepted). Fusion gain +0.16 pt = noise. |
| 2026-10-18 | **Sealed test — the one look** (`results/final/final_test_v2.json`, 418 images) | test 0.8337 | test 0.8373 (probe-only 0.8445) | In-scope (n = 306) acc. 0.8693 / F1 0.8681; coverage 0.871 at 0.909 accepted acc.; ECE 0.0343; weakest class resistor (recall 0.545). Fused vs probe-only within ±1.8 pt SE. Test split is now spent. |
| 2026-10-18 | Phase 3: ONNX export (`exports/final/export_report.json`) | — | — | 1,637.1 MB (external data); onnxruntime parity max \|Δp\| 6.3e-6, 100 % argmax agreement, 892.4 ms/img CPU; OpenCV DNN segfaults (exit −11) → demo uses onnxruntime. |
| 2026-10-18 | Comparative report (`docs/report/REPORT.md`, `scripts/make_report_figures.py`) | — | — | Figures/tables regenerated from repo files only; web holdout (`data/web_holdout_v1`) not yet evaluated (placeholder section). |
| 2026-10-18 | Web holdout v1 — one look (`results/final/final_test_web_holdout_v1.json`, 407 Wikimedia photos) | — | acc. 0.8206 / macro-F1 0.8159 (probe-only 0.7789) | Coverage 0.853 at 0.891 accepted; ECE 0.050; weakest led / switch / transformer. Zero-shot head +4.2 pt out-of-source. Holdout now spent. |
| 2026-10-18 | Original Kaggle download (`scripts/eval_original_kaggle.py`, `results/kaggle_original/summary.json`) | — | all 3,530 files 0.9552 (incl. training rows) | Never-seen (v2 test) files 0.8296 (n = 493); training-row files 0.9811; conflicting duplicates 0.3704 (n = 27). Legacy-protocol replay on legacy-test files 0.8772 (n = 521) vs old best 0.7132; leaked copies 1.000 (113), clean 0.8431 (408). |
| 2026-10-18 | Deliverables: IEEE paper (`docs/paper/main.pdf`), interactive explorer (`docs/explorer/index.html`), GPUI desktop app (`apps/gpui-demo`) | — | — | GPUI headless parity 64/64 argmax vs PyTorch; window not launched in the build sandbox (no display). |


## 2026-10-20 to 2026-10-30 — delivery and sharing

October 20 added the ITRI626 fixed-split A/B/C report/notebook, A/B ONNX exports and the shared C/deployed graph.
The desktop app moved to GPUI Kit 0.7.3 with Classify/Demo/Benchmark, model comparison and Bench themes.
Portable glibc-2.28 Linux packaging and a NixOS/Wayland launcher followed. October 30 published the source privately
at https://github.com/Vaperizer2/626-Project and prepared a model/demo release, runbook and agent handoff.
This delivery work does not change the recorded research scores or reopen held-out model selection.

## 2026-10-31 — distilled students for single-image speed (branch `wip/distill`)

Hypothesis: a student 5-20x cheaper than the frozen So400m-384 trunk, trained to reproduce its pooled embedding,
keeps the deployed head and zero-shot fusion and lands within 0.5 pt of its development-CV accuracy. It does not on
the data available. Students were distilled only on 59,466 unlabeled images disjoint from every `data/v2` and
`data/web_holdout_v1` image (`tools/distill/pool.py`: 25,193 component photos from 11 Kaggle datasets, 34,273
Caltech-256 and COCO val2017 images). A pool image was dropped when its CLIP-B/32 cosine to any v2 or holdout image
reached 0.85 over flips and rotations, or its pHash came within 10 bits. Loss: standardised-feature MSE + cosine +
KL on the zero-shot text head. KL on the deployed probe head was left out because that head was fitted on all
development labels.

Numbers from the speed harness (`tools/speed`, legion RTX 3060 Laptop, batch 1, against `baseline_fp32`:
CV accuracy 0.8268 ± 0.0151, about 110 ms):

| Candidate | Params | Input | CV accuracy | Δ vs baseline | McNemar (B wrong / B right) | Batch-1 ms | Speedup |
|---|---:|---:|---:|---:|---|---:|---:|
| SigLIP2 B/16, CUDA fp32 | 93.8 M | 256 | 0.7397 ± 0.0188 | −8.72 pt | 269 / 50, p < 1e-4 | 11.28 | 10.5x |
| SigLIP2 B/16, TensorRT fp16 | 93.8 M | 256 | 0.7397 ± 0.0187 | −8.72 pt | 269 / 50, p < 1e-4 | 2.28 | 48x |
| MobileCLIP2-S0, CUDA fp32 | 12.0 M | 256 | 0.6433 ± 0.0326 | −18.35 pt | 520 / 59, p < 1e-4 | 4.27 | 26x |
| MobileCLIP2-S0, TensorRT fp16 | 12.0 M | 256 | 0.6425 ± 0.0321 | −18.43 pt | 521 / 58, p < 1e-4 | 7.26 | 16x |

All four are REJECT. The B/16 student reaches 0.926 cosine to the teacher on held-out pool images but 0.844 on
development images (0.859 in-scope). Fidelity is worst on classes the pool covers thinly (switch 0.79, fuse 0.81)
and best on capacitor and breadboard (0.88). It scores below SigLIP2 B/16's own linear probe (0.760, Phase 1).
A one-fold diagnostic (`tools/distill/run_fold_diag.sh 0`, unofficial) adds the 2,010 development images outside
fold 0, label-free. Fold-0 accuracy rises from 0.727 to 0.747 (teacher 0.801) and fold-0 cosine from 0.846 to
0.869. A pool-only control with the same schedule stays at 0.727. In-domain images help but close about a third
of the gap.

## 2026-10-31 to 2026-11-03 — fast inference for the desktop app (tools/speed*)

Goal: make the deployed SigLIP So400m-384 fused model faster on the laptop's RTX 3060 (6 GB, 80 W cap) and lose
at most 0.5 pt of 5-fold dev CV accuracy against 0.8268. Every number in the table comes from the frozen speed harness
(`tools/speed`): batch-1 latency as a paired ratio against the fp32 graph, 5-fold dev CV, McNemar against the
baseline. Model choices used only the dev folds; the sealed test and the web holdout stay spent.

| Candidate | Batch-1 speed vs deployed | CV accuracy | Δ | Verdict |
|---|---:|---:|---:|---|
| 308 px, INT8 SmoothQuant, weights and activations (`compose_int8_res308`) | 9.2x (11.9 ms) | 0.8268 | 0.00 pt | shipped as the app's Fast model |
| 294 px, INT8 | about 10x | 0.8240 | −0.28 pt | not shipped |
| 308 px, token merging at blocks 13 and 20, INT8 | 10.8x, 11.4x with a fused engine | about 0.824 | about −0.3 pt (4 readings, −0.16 to −0.40, one of them fp16) | not shipped |
| 294 px, MLP pruned to 3840, INT8 (speed2) | 11.3x | 0.8224 | −0.44 pt | acceptable, not shipped |
| speed2 shape + full-rank QAT distillation, Q5 (speed3) | 11.2x (9.95 ms) | 0.8260 | −0.08 pt (McNemar 60/58, p 0.93) | acceptable, best so far, not yet shipped |

Q5 was picked from five built variants scored on the same dev folds, so its −0.08 pt is optimistic. Against the
speed2 winner it is +0.36 pt (McNemar 33/42, p 0.36), not beyond noise. LoRA QAT (Q1) and a closed-form fc2 refit
(C1) land at −0.40 to −0.48 pt; distilling on the dev images themselves is worse (−0.84 pt). Q4 (per-token loss)
was only screened, by the fake-quant screen rather than the harness: 84 fixed-head flips against Q5's 91, within
noise. It was not built.

Findings that hold beyond these candidates:
- TensorRT 10.16.1 miscompiles these graphs after two onnxruntime rewrites (TransposeOptimizer: wrong outputs at
  batch > 1; MatMulAddFusion: broken token merging with INT8). A manifest's `session_config` turns both off
  (docs/TEMPLATE.md); the shipped Fast model hits neither.
- Latency is not linear in tokens: on the 30 SMs it steps with GEMM tiles and CTA waves. The token counts that paid
  off were 448, 384, 256 and 128 (`tools/speed4`).
- Batching gains about 10 %, because the GPU is power-capped.
- The same token-merging graph builds a fused TensorRT engine about 1 time in 4; `tools/speed4/engprobe.sh` tells.

## 2026-11-03 — the demo's hard cases (tools/hardcases)

Question: can the deployed model get the app's five "Hard cases" demo pictures right without losing accuracy? They
fail as LEDs → diode, a small transformer → inductor coil, DIP switches → ic chip, SMD power inductors → diode, and an
Arduino board on a breadboard → breadboard. These five pictures come from the web holdout, so the goal itself (the
five confusion pairs, and the ideas tried for them) was taken from holdout pictures. After that, every candidate was
ranked on grouped 5-fold dev CV only; no demo, holdout or test picture was scored until the recommendation had been
written down and the notes hashed, and then the final candidates were run on the demo stack once. Dev holds few of
these errors (9 in 2,512 images), so the main proxy was the calibrated probability on the wrong partner class over the
638 dev images of the five classes (baseline 2.32 %). 41 scored variants of 25 hypotheses; the pick rule was fixed
after H0 to H19 were scored, before H20 to H23. Log and notes in `626-artifacts/hardcases/` on the laptop, code in
`tools/hardcases/`.

| Change | Dev CV | Δ | McNemar (worse / better), p | Hard-pair errors | Partner mass |
|---|---:|---:|---|---:|---:|
| deployed: 384 crop of a 427 short-side resize | 0.8268 | — | — | 9 | 2.32 % |
| whole picture squashed to 384 × 384, head refit | 0.8428 | +1.59 pt | 45 / 85, 6e-4 | 8 | 1.93 % |
| short side 384, centre crop (no border cut on the short side) | 0.8424 | +1.55 pt | 27 / 66, 1e-4 | 9 | 2.13 % |
| squash plus a nearest-neighbour term | 0.8503 | +2.35 pt | 45 / 104, < 1e-4 | 6 | 1.82 % |

The squash gain comes from non-square pictures (62 % of dev, 0.807 → 0.836); square ones do not move, every fold
improves, and the zero-shot head alone improves without training (0.762 → 0.775). The nearest-neighbour term was one
of seven changes stacked on squash; corrected for that, its own p is about 0.11. Other whole-picture views did as well
on accuracy, not better: letterbox without a significant drop in partner mass, and the mean of crop and squash at two
passes per picture. Nothing else helped: the zero-shot weight and temperature, subtype prompts (zero-shot alone falls
to 0.67–0.73), flip TTA or training, class-balanced weights, pair specialists, augmented views, prototypes, or a
ConvNeXt-T ensemble. A SigLIP 2 ensemble was not significant at twice the latency. After 41 scored variants (25
hypotheses) the dev numbers are optimistic.

The one look at the demo stack, with the recommended squash head: no hard picture flipped. Three are less confidently
wrong, the small transformer is more confidently wrong (0.965 → 0.987), and the Arduino picture is now accepted as
breadboard at 0.52 (the deployed model abstains at 0.44; 0.52 clears both its 0.50 threshold and the refit head's
0.45). In the "Not a component / abstains" section both out-of-scope pictures now abstain (1 of 2 before) and the
in-scope coin-cell holder is answered correctly, so the section scores 3 of 3 instead of 1 of 3. The one-per-class
section stays 20 of 20. The study recommends shipping squash without the nearest-neighbour term; it is not shipped,
because it changes the shipped model, its golden file and the Fast model's recipe, which is the owner's call.

## 2026-11-03 to 2026-11-04 — decisions and delivery

On 2026-11-03 the owner kept the current Fast model (`compose_int8_res308`) as the app's Fast. Q5 is not shipped. The
squash preprocessing from the hard-case study is not shipped either; whether to ship it is still open.

The demo stack grew from 28 to 68 pictures. The new Random section holds 40 holdout photos, two per class, drawn with
seed 20261103 from the holdout photos not already in the stack, with nobody looking at the answers first. Deployed
gets 32 of them right, in line with its 309 of 407 (75.9 %) on the whole holdout. The stack is a presentation set:
no model was chosen or tuned on it. `notebooks/ITRI626_training.ipynb` adds the training code, one section per model
(A, B, C, Deployed, Fast).

On 2026-11-04 `main` took PR #1 (merge `e131c87`), PR #3 (`347f339`), PR #4 (`5b73308`) and PR #2 (`4666364`). The GPU
PC `carlpc2` is gone. GPU work runs on the laptop, and heavy CPU work on Boat.dev CPU sandboxes. None of this changes
a recorded research score.

## 2026-11-03 to 2026-11-04 — scratch CNN (model A) improvement study

Development-fold study of the from-scratch CNN on an RTX 4070 laptop; full record in
`docs/SCRATCH_CNN_IMPROVEMENT_LOG.md` and `docs/SCRATCH_CNN_IMPROVEMENT_REPORT.md`, run records in
`results/runs_win.jsonl`. Selection used development folds only.

| Date | Experiment | CV macro-F1 (mean ± std) | CV accuracy | Notes |
|---|---|---:|---:|---|
| 2026-11-03 | Model A, 200-epoch schedule (`b0afa2aa4d60`, `train.epochs=200 train.patience=40`), 5 folds | 0.3903 (+0.035 ± 0.009 vs committed 0.3552, 5/5 folds) | 0.3986 | Resolution, mixup, width, depth, residual blocks, label smoothing, TTA, init, regularisers, EMA, squash: no gain. |
| 2026-11-03 | Model A + distillation from C (`configs/scratch_cnn_distill.yaml`, `fbe64b2765cb`): cross-fitted per-fold SigLIP So400m probe teacher, alpha 0.9, T 2, 300 ep, 5 folds | 0.4361 (+0.046 ± 0.007 vs 200-ep A, 5/5 folds) | 0.4459 | Separate variant: no pretrained weights or extra images, but soft targets from a pretrained model. 500 epochs: +0.004 ± 0.004, not adopted. |
| 2026-11-04 | **Sealed test — one look for the two final recipes** (`results/scratch_cnn_improvement/`, fold-0 training, 418 images) | — | — | A 200 ep: test acc 0.3995 / macro-F1 0.3924 (committed course run 0.3541 / 0.3361). A + distillation: 0.4115 / 0.3933. Paired bootstrap distilled − plain: macro-F1 +0.001 [−0.044, +0.045]. Schedule gain carries over; distillation gain not resolvable on the test split. Test split spent for both recipes. |
| 2026-11-04 | **Wikimedia holdout — one look for the same two checkpoints** (`tools/eval_holdout_cnn.py`, `results/scratch_cnn_improvement/web_holdout_*`, 407 images, files restored from the developer's archive, fingerprint `5458af44e043b3aa`) | — | — | A 200 ep: acc 0.2826 / macro-F1 0.2570 [0.216, 0.293]. A + distillation: 0.3170 / 0.2976 [0.252, 0.334]. Paired bootstrap distilled − plain: macro-F1 +0.040 [+0.003, +0.079], 14/20 classes. Deployed model on the same photos (2026-10-18): 0.8206 / 0.8159. Source shift costs the scratch CNN 10-14 pt macro-F1 vs 2 pt for the pretrained model. Holdout spent for both recipes; B and C not scored. |

## 2026-11-04 — linear-probe sweep of 2025 backbones (laptop RTX 4070, `results/probes_win.jsonl`)

Question: does any backbone released since the October benchmark carry better frozen features for these 20 classes
than SigLIP So400m/14@384? Same protocol as Phase 1: eval-preprocessed features extracted once (fp16 autocast; models
above 1 B parameters loaded in fp16 with batch 8-16), standardise + multinomial logistic regression per fold, C by
grouped 3-fold CV on the training rows, grouped 5-fold dev CV, out-of-fold probabilities in `results/oof/`.
Paired against the SigLIP probe fold by fold. Backbones registered in `src/compclass/models/registry.py`
(section 5). Sealed test and web holdout untouched.

| Backbone (probe) | CV macro-F1 (mean ± SD) | CV acc. | Δ vs SigLIP probe (paired ± SE) | folds better | feature pass (s) |
|---|---:|---:|---:|---:|---:|
| MetaCLIP 2 ViT-bigG/14@378 (1.84 B, fp16) | 0.8496 ± 0.0081 | 0.8515 | +0.0256 ± 0.0033 | 5/5 | ~5,000 (first pass 16:06–17:29; probe re-fitted from the cache) |
| PE-Core L/14@336 | 0.8298 ± 0.0131 | 0.8308 | +0.0058 ± 0.0039 | 4/5 | 625 |
| SigLIP 2 So400m/16@512 | 0.8258 ± 0.0061 | 0.8268 | +0.0018 ± 0.0040 | 2/5 | 663 |
| DFN-5B ViT-H/14@224 | 0.8247 ± 0.0151 | 0.8264 | +0.0007 ± 0.0024 | 2/5 | 734 |
| SigLIP So400m/14@384 (model C, re-run here) | 0.8240 ± 0.0116 | 0.8256 | +0.0000 ± 0.0000 | 0/5 | cached (teacher build) |
| AIMv2 H/14@336 | 0.7978 ± 0.0086 | 0.7990 | -0.0262 ± 0.0024 | 0/5 | 839 |
| EVA-02 L/14 CLIP@336 | 0.7974 ± 0.0153 | 0.7994 | -0.0266 ± 0.0049 | 0/5 | 391 |
| AIMv2 L/14@336 | 0.7847 ± 0.0215 | 0.7858 | -0.0393 ± 0.0064 | 0/5 | 419 |
| DINOv3 ViT-H+/16@256 | 0.6917 ± 0.0155 | 0.6942 | -0.1323 ± 0.0073 | 0/5 | 994 |
| DINOv3 ViT-L/16@256 | 0.6504 ± 0.0146 | 0.6553 | -0.1735 ± 0.0071 | 0/5 | 440 |

Findings: (1) MetaCLIP 2 ViT-bigG/14 is the first backbone to beat the SigLIP probe beyond noise, +2.6 ± 0.3 pt on
every fold (0.8496 vs 0.8240), and also above the deployed fused model's OOF 0.8264. (2) The other image-text
encoders land within ±0.6 pt of SigLIP; resolution (SigLIP 2 at 512 px) adds nothing; self-supervised (DINOv3) and
autoregressive (AIMv2) features trail by 3-17 pt, as DINOv2 did in October. (3) Cost: the bigG feature pass took
83 min for 2,930 images on the 4070 (about 1.7 s/image, with the 8 GB card spilling into shared memory), against
about 1 s/image for So400m on a CPU; its fp16 weights are 3.7 GB. PE-Core G/14@448 (1.88 B) was queued but not run:
the sweep was stopped by the agent harness for low system memory after the bigG pass.
Follow-up (2026-11-04 evening): PE-Core G/14@448 was started a second time at the author's request and stopped by
the author after 5.5 h, still inside its feature pass (one batch of 8 took about 62 s with 0.8 GB paged to system
memory); no result, no cache. Cheap combinations of cached features were checked on CPU: MetaCLIP + PE-Core L
concatenated +0.26 ± 0.35 pt vs MetaCLIP alone, MetaCLIP + SigLIP +0.30 ± 0.28, all three +0.08 ± 0.33, averaged
out-of-fold probabilities −0.3 to +0.2; nothing beyond noise. The author then dropped MetaCLIP 2 bigG (size) and
SigLIP 2 (no gain) and chose to develop PE-Core L.

## 2026-11-05 — PE-Core L/14@336: zero-shot head and fusion, the deployed recipe on a new backbone

Question: given the deployed model's recipe (probe + the backbone's own zero-shot text head, `compclass.final`), does
PE-Core L match or beat SigLIP So400m like for like? Zero-shot: `compclass.zeroshot --model pe_core_l14_336`
(open_clip `PE-Core-L-14-336`/`meta`, six templates, `results/oof/zeroshot_pe_core_l14_336.npz`). Fusion checked
out-of-fold on the dev folds with the cached timm features: probe = `fit_probe` per fold; zero-shot cosine from the
same features passed through PE-Core's own 1024×1024 projection head (taken from the timm checkpoint; SigLIP has no
projection, so this step did not exist before) against the text weights; fused = log p_probe + w · log softmax(T · cos).
The timm-feature + projection pipeline agrees with open_clip's own image tower on 95.1 % of top-1 zero-shot decisions
and the same accuracy (0.7910 vs 0.7914), so one forward pass serves both heads, as in the deployed graph. SigLIP was
re-run through the identical code as the control. Sealed test and web holdout untouched.

| Model (grouped 5-fold dev CV, out-of-fold) | macro-F1 | acc. | paired Δ (± SE), folds better |
|---|---:|---:|---|
| SigLIP probe (model C) | 0.8240 | 0.8256 | reference |
| SigLIP zero-shot alone | 0.7623 | 0.7615 | |
| SigLIP fused, deployed constants T 50, w 0.25 (= the deployed model) | 0.8255 | 0.8268 | +0.15 ± 0.19 vs its probe, 4/5 |
| PE-Core L probe | 0.8298 | 0.8308 | +0.43 ± 0.47 vs SigLIP fused, 4/5 |
| PE-Core L zero-shot alone | 0.7898 | 0.7910 | +2.8 pt over SigLIP zero-shot |
| **PE-Core L fused, deployed constants T 50, w 0.25** | **0.8351** | **0.8368** | **+0.96 ± 0.23 vs SigLIP fused, 5/5**; +0.53 ± 0.37 vs its probe, 4/5 |
| PE-Core L fused, T 50, w 0.5 (best of a 12-point grid, optimistic) | 0.8370 | 0.8388 | +1.15 ± 0.18 vs SigLIP fused, 5/5 |

Per-fold macro-F1, SigLIP fused: 0.800 0.834 0.824 0.834 0.836; PE-Core L fused (T 50, w 0.25): 0.812 0.849 0.835
0.841 0.838. The grid (T ∈ {25, 50, 100} × w ∈ {0.1, 0.25, 0.5, 1}) is flat between +0.3 and +0.7 pt for PE-Core and
between −0.1 and +0.3 pt for SigLIP; w = 1 at T = 100 hurts both.

Findings: (1) With the pre-registered constants, the PE-Core L fused model beats the deployed SigLIP fused model by
+1.0 ± 0.2 pt on every fold, the first like-for-like improvement over the deployed model that clears noise
(MetaCLIP 2 bigG was larger but was dropped for size). (2) The gain comes from both parts: a slightly better probe
(+0.6) and a clearly better zero-shot head (0.790 vs 0.762), which fuses with more benefit (+0.5 vs +0.15).
(3) PE-Core L is 316 M parameters, 1.2 GB fp32, at 336 px (576 tokens vs 729): about 0.6× the deployed compute.
(4) Open_clip's PE-Core logit scale is 100, not SigLIP's; T = 50 still works, T = 100 with w ≥ 0.5 does not.
Next (author's call): calibration + abstain threshold via a key-generalised `compclass.final fit`, ONNX export with the
projection baked in, speed vs Deployed, then whether it earns one sealed-test look.

### 2026-11-05 — control for the input view: the PE-Core edge was the preprocessing, not the backbone

The author noticed that the two backbones were compared under different views: SigLIP's recipe resizes the short side
to 427 and crops 384 (10 % border cut, crop fraction 0.9) while PE-Core's resizes the short side to 336 and crops 336
(nothing cut). The hard-case study (2026-11-03) had already shown SigLIP gains about 1.6 pt without the border cut.
Control: dev features re-extracted for SigLIP at crop fraction 1.0 and squashed, and for PE-Core squashed
(`tools/views/extract_views.py`, 70-90 s per pass on the 4070; caches `results/cache/features/*-crop100.npy`,
`*-squash.npy`), then the same OOF probe / zero-shot / fusion scoring for every backbone × view
(`tools/views/view_fusion.py`; PE-Core heads from `tools/views/pe_core_heads.py`). Deployed constants T 50, w 0.25.

| Backbone | view | probe F1 | zero-shot F1 | fused F1 | fused acc |
|---|---|---:|---:|---:|---:|
| SigLIP So400m@384 | deployed (short 427, crop 384) | 0.8240 | 0.7623 | 0.8255 | 0.8268 |
| SigLIP So400m@384 | short 384, crop 384 (crop 1.0) | 0.8330 | 0.7643 | **0.8403** | **0.8420** |
| SigLIP So400m@384 | squash 384 × 384 | 0.8339 | 0.7758 | 0.8383 | 0.8404 |
| PE-Core L@336 | default (short 336, crop 336 = crop 1.0) | 0.8298 | 0.7898 | 0.8351 | 0.8368 |
| PE-Core L@336 | squash 336 × 336 | 0.8323 | 0.7897 | 0.8354 | 0.8368 |

Paired per fold, fused models: SigLIP crop 1.0 vs SigLIP deployed +1.48 ± 0.25 pt F1 (5/5), reproducing the
hard-case study's +1.55 in this pipeline; SigLIP squash vs deployed +1.29 ± 0.24 (5/5); PE-Core squash vs PE-Core
default +0.04 ± 0.27 (3/5), no gain. Same view: PE-Core vs SigLIP at crop 1.0 **−0.52 ± 0.27 pt (1/5)**; squash vs
squash −0.29 ± 0.31 (2/5). Best view each: SigLIP crop 1.0 beats PE-Core by 0.48 ± 0.32.

Findings: (1) The +1.0 pt of the previous entry was the view, not the backbone: PE-Core's default already keeps the
whole short side, which is worth about 1.5 pt to SigLIP. Like for like, SigLIP So400m is 0.3-0.5 pt ahead of PE-Core L,
within noise. (2) PE-Core's zero-shot head stays clearly better (0.790 vs 0.764-0.776) but its probe is slightly weaker,
and the two cancel. (3) PE-Core does not gain from squashing; SigLIP gains from either whole-frame view. (4) The
cheapest real improvement to the deployed model is therefore the one the hard-case study already found: keep SigLIP,
drop the border cut (+1.5 pt OOF, 5/5 folds, a preprocessing constant and a head refit). PE-Core L remains the
smaller/faster equal, 0.6× compute, 1.2 GB, at about −0.5 pt.

## 2026-11-05 — efficiency study, step 1: small image-text backbones vs SigLIP So400m at the same view

Author's study: a smaller, cheaper model for the 20 classes, compared with SigLIP So400m under the deployed recipe.
Step 1 is the frozen linear probe on twelve backbones of 6-123 M parameters (registry section 6, all registered at
crop 1.0), same protocol as the 2026-11-04 sweep, paired per fold against the SigLIP So400m probe fitted on its
crop-1.0 cache (0.8330) and against PE-Core L/14@336. Compute is measured GFLOPs per image (torch FlopCounterMode,
batch 1), since hybrids and CNNs have no token count. `tools/views/small_sweep_summary.py`; records in
`results/probes_win.jsonl`, OOF probabilities in `results/oof/`. 28 min of GPU for the twelve, 1-3 min each.

| Backbone (probe, crop 1.0) | params | px | GFLOPs/img | compute vs SigLIP | CV macro-F1 (± SD) | CV acc | Δ vs SigLIP probe (± SE) | folds | Δ vs PE-Core L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PE-Core L/14@336 | 316 M | 336 | 352 | 0.58× | 0.8298 ± 0.0131 | 0.8308 | -0.32 ± 0.54 pt | 1/5 | +0.00 ± 0.00 pt |
| MobileCLIP2-S3@256 | 123 M | 256 | 30 | 0.05× | 0.7920 ± 0.0080 | 0.7930 | -4.09 ± 0.42 pt | 0/5 | -3.77 ± 0.46 pt |
| MobileCLIP2-B@224 | 86 M | 224 | 34 | 0.06× | 0.7776 ± 0.0141 | 0.7795 | -5.54 ± 0.62 pt | 0/5 | -5.22 ± 0.52 pt |
| SigLIP 2 B/16@384 | 93 M | 384 | 100 | 0.17× | 0.7727 ± 0.0213 | 0.7743 | -6.03 ± 0.83 pt | 0/5 | -5.71 ± 0.55 pt |
| PE-Core B/16@336 | 93 M | 336 | 77 | 0.13× | 0.7654 ± 0.0109 | 0.7683 | -6.75 ± 0.69 pt | 0/5 | -6.44 ± 0.73 pt |
| MobileCLIP2-S2@256 | 35 M | 256 | 16 | 0.03× | 0.7579 ± 0.0141 | 0.7604 | -7.50 ± 0.70 pt | 0/5 | -7.19 ± 0.96 pt |
| DFN-2B B/16@224 | 86 M | 224 | 34 | 0.06× | 0.7542 ± 0.0212 | 0.7564 | -7.88 ± 1.08 pt | 0/5 | -7.56 ± 1.10 pt |
| PE-Core B/16@224 | 93 M | 224 | 34 | 0.06× | 0.7531 ± 0.0089 | 0.7564 | -7.99 ± 0.52 pt | 0/5 | -7.67 ± 0.60 pt |
| MetaCLIP B/16@224 | 86 M | 224 | 34 | 0.06× | 0.7526 ± 0.0123 | 0.7548 | -8.03 ± 0.39 pt | 0/5 | -7.71 ± 0.38 pt |
| ConvNeXt-B CLIP@320 | 88 M | 320 | 63 | 0.10× | 0.7274 ± 0.0135 | 0.7317 | -10.56 ± 0.51 pt | 0/5 | -10.24 ± 0.90 pt |
| PE-Core S/16@384 | 24 M | 384 | 25 | 0.04× | 0.7190 ± 0.0144 | 0.7209 | -11.39 ± 0.46 pt | 0/5 | -11.07 ± 0.84 pt |
| EVA-02 B/16 CLIP@224 | 86 M | 224 | 34 | 0.06× | 0.7093 ± 0.0174 | 0.7118 | -12.36 ± 0.78 pt | 0/5 | -12.05 ± 0.93 pt |
| PE-Core T/16@384 | 6 M | 384 | 6 | 0.01× | 0.6226 ± 0.0310 | 0.6298 | -21.04 ± 1.34 pt | 0/5 | -20.72 ± 1.64 pt |

Findings: (1) Every backbone under 300 M parameters is 4 pt or more below SigLIP So400m on every fold; the gap to
the 2023 base models seen in October (6-8 pt) holds for the 2025 ones. PE-Core L at 316 M remains the only cheaper
backbone within noise. (2) The best small model is MobileCLIP2-S3 (123 M, 30 GFLOPs, 5 % of SigLIP's compute) at
0.792, −4.1 ± 0.4 pt; MobileCLIP2-B (86 M, 34 GFLOPs) 0.778 is the best of the ViT-B class, ahead of DFN, MetaCLIP
and EVA-02 B/16 at the same cost. (3) Resolution matters for small ViTs: PE-Core B/16 gains 1.2 pt from 224 to
336 px, SigLIP 2 B/16 at 384 scores 0.773 against October's 0.760 at 224. (4) Below 35 M the curve falls fast:
PE-Core S 0.719, PE-Core T 0.623. (5) The ConvNeXt CLIP (0.727) trails the ViTs at equal size, so the TensorRT
friendliness of a CNN would cost 5 pt. Step 2 (zero-shot head + fusion) and step 3 (fine-tuning the best small
backbone, where the 2023 results say fine-tuning did not close the gap either) are the author's call; candidates
MobileCLIP2-S3, MobileCLIP2-B and PE-Core B/16@336.

### 2026-11-05 — efficiency study, step 2: the deployed recipe (zero-shot head + fusion) on the three best small backbones

`compclass.zeroshot` on MobileCLIP2-S3, MobileCLIP2-B and PE-Core B/16 (open_clip towers; records in
`results/probes_win.jsonl`, OOF similarities in `results/oof/zeroshot_<key>.npz`), text weights and each backbone's own
projection from `tools/views/heads.py`, fused table from `tools/views/fusion_table.py`: probe + w · log softmax(T · cos)
with the deployed T 50 / w 0.25, out-of-fold, every backbone at crop 1.0, paired against SigLIP So400m fused at crop 1.0.
The timm-feature + projection zero-shot agrees with open_clip's own pipeline on 98 % of decisions for the MobileCLIPs;
86 % for PE-Core B because open_clip runs it at 224 px and our cache is at 336.

| Backbone (crop 1.0) | GFLOPs | probe F1 | zero-shot F1 | fused F1 | fused acc | fused vs own probe | fused vs SigLIP fused (0.8403) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SigLIP So400m@384 | 604 | 0.8330 | 0.7643 | 0.8403 | 0.8420 | +0.73 ± 0.17 (5/5) | reference |
| PE-Core L/14@336 | 352 | 0.8298 | 0.7898 | 0.8351 | 0.8368 | +0.53 ± 0.37 (4/5) | −0.52 ± 0.27 (1/5) |
| MobileCLIP2-S3@256 | 30 | 0.7920 | 0.7253 | 0.8013 | 0.8025 | +0.93 ± 0.35 (5/5) | −3.89 ± 0.15 (0/5) |
| MobileCLIP2-B@224 | 34 | 0.7776 | 0.7050 | 0.7846 | 0.7866 | +0.70 ± 0.20 (5/5) | −5.57 ± 0.37 (0/5) |
| PE-Core B/16@336 | 77 | 0.7654 | 0.7053 | 0.7758 | 0.7783 | +1.04 ± 0.18 (5/5) | −6.44 ± 0.64 (0/5) |

Findings: (1) The zero-shot head helps every backbone by 0.5-1.0 pt, 5/5 folds for the small ones, so the deployed
recipe transfers; it does not close the gap: the small text towers know the 20 names less well (0.705-0.725 vs
0.764 SigLIP, 0.790 PE-Core L). (2) Under the full recipe MobileCLIP2-S3 reaches 0.801 F1 / 0.803 acc at 5 % of
SigLIP's compute, −3.9 ± 0.2 pt; PE-Core L 0.835 at 58 %, −0.5 ± 0.3. Those two are the efficient frontier; the
ViT-B models are dominated by MobileCLIP2-S3 on both axes. (3) Step 3 (fine-tuning MobileCLIP2-S3, partial and
full) and step 4 (ONNX export and measured latency vs Deployed and Fast) remain; selection on dev folds only.


## 2026-11-05 — Efficiency study, step 3: fine-tuning MobileCLIP2-S3 (development folds)

Question: does updating the backbone of the best small candidate beat its frozen linear probe (0.7920 CV macro-F1,
crop 1.0) or the fused recipe on it (0.8013)? Two configs, one recipe each, chosen before looking at folds:
`configs/mobileclip2_s3_full.yaml` (all 123M parameters) and `configs/mobileclip2_s3_partial.yaml` (stem and stages
0–3 frozen via the new `model.kwargs.trainable_stages` of the registry builder; stage 4 + final conv + head, 61M).
Windows run log `results/runs_win.jsonl`, run dirs `runs/win/`.

Two recipe problems surfaced on fold 0 before the recorded runs and are part of the evidence:
(1) batch 32 at 256 px peaks at 7.8 GB on the 8 GB card and pages to system memory (17 s/step); 16 × 2 accumulation
with gradient checkpointing peaks at 2.9 GB at ~38 img/s. (2) With a random head at the CLIP-safe LR (5e-5, layer
decay 0.75) fold 0 reached only 0.3955 after 30 epochs (record `a2f89f7ff4fe`): train acc 0.60, and validation
swung 0.15–0.40 from epoch to epoch because the FastViT BatchNorm running statistics drifted on small batches. The
recorded runs therefore use LP-FT (Kumar et al. 2022): `train.head_init` loads the fold's own fitted probe, folded
into the Linear head (`tools/views/probe_heads.py`; the folded layer reproduces the probe's F1 to 4 decimals), and
`train.freeze_bn` keeps BN statistics frozen (affine parameters still train). Both are new, additive options in
`compclass.train`. A one-step smoke run scores exactly the probe (0.7829 on fold 0) before any update.

| Run | trainable | epochs (best) | min/fold | peak VRAM | CV macro-F1 (± SD) | CV acc | vs S3 probe | vs S3 fused | vs SigLIP fused | folds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mobileclip2_s3_full `e33f614a2fe3` | 122.9 M | 13.4 (4.4) | 9.6 | 2895 MB | 0.7923 ± 0.0202 | 0.7934 | +0.03 ± 0.79 (3/5) | −0.91 ± 0.97 (2/5) | −4.80 ± 1.00 (0/5) | 0.774, 0.769, 0.807, 0.797, 0.815 |
| mobileclip2_s3_partial `974ca5091a8d` | 60.6 M | 9.6 (2.6) | 2.4 | 1496 MB | 0.7890 ± 0.0227 | 0.7918 | −0.30 ± 0.72 (2/5) | −1.24 ± 0.65 (1/5) | −5.13 ± 0.67 (0/5) | 0.756, 0.786, 0.785, 0.801, 0.817 |

Findings: fine-tuning ties the probe (full) or loses to it (partial) and never reaches the fused probe; the best
epoch is 0–5 in eight of ten runs, train accuracy passes 0.95 within four epochs, and per-fold differences of ±2 pt
are the fold-to-fold noise of 500-image validation sets (folds 2–4 gained, 0–1 lost, for both recipes). Same answer
as October's SigLIP partial fine-tunes: on 2,000 training images the frozen features plus linear heads are the
ceiling for this backbone too, so the deployable small model is the fused probe on the frozen tower, not a
fine-tuned network. Summary: `tools/views/finetune_summary.py` → `results/efficiency/step3_finetune_summary.md`.

## 2026-11-05 — Efficiency study, step 4: exported graphs and measured latency (RTX 4070 Laptop, 8 GB)

Graphs: `tools/views/export_candidate.py` exports a candidate as one ONNX graph (tower + probe head refitted on all
dev rows + zero-shot head, fused; uncalibrated), fp32 and fp16, in inference form (timm `reparameterize_model`,
which folds FastViT's training branches into single convolutions, 1593 → 785 modules, identical outputs). Parity:
ORT CPU vs PyTorch max |logit| gap ≤ 6e-6 fp32 / ≤ 0.07 fp16, argmax equal; on 16 real dev images the graph agrees
with the numpy recipe on cached features 16/16 for all three (max gap 0.013–0.023). The check caught one bug: the
SigLIP graph was first built at crop 0.9 while its head was fitted on crop-1.0 features (gap 1.26) — fixed.
Sidecars tracked under `exports/candidates/`, `.onnx` files git-ignored.

Latency, batch 1, same machine, two runtimes (`results/efficiency/step4_latency.md`, JSON next to it):

| Tower (fused recipe) | px | params | GFLOPs | fp16 graph | PyTorch eager fp16, median ms | batch-8 img/s | ORT CUDA EP fp16, median ms | vs SigLIP@384 (ORT) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SigLIP So400m (Deployed trunk) | 384 | 428 M | 604 | 821 MB | 30.4 | 35 | 32.7 | 1.00× |
| SigLIP So400m at 308 px (Fast trunk, fp16 not INT8) | 308 | 428 M | 401 | – | 18.2 | 58 | – | – |
| PE-Core L/14 | 336 | 316 M | 352 | 607 MB | 18.6 | 63 | 26.1 | 0.80× |
| MobileCLIP2-S3 | 256 | 123 M | 30 | 240 MB | 10.6 | 376 | 9.0 | 0.27× |
| ConvNeXt-T (model B, scale) | 224 | 28 M | 9 | – | 3.9 | 1393 | – | – |

Method: PyTorch — static GPU input, fp16 autocast, CUDA events, 5 rounds × 60 runs interleaved across models
(`tools/views/latency.py`); ORT — IO binding, ORT_ENABLE_ALL, warm-up ≥ 20 runs and 10 s, 5 × 60 interleaved
(`tools/views/ort_latency.py`, run from a throw-away venv with onnxruntime-gpu 1.31.1 + NVIDIA CUDA 13 wheels so
`uv.lock` is untouched). The team lead's Fast model (INT8 under TensorRT on the 3060) is not reproducible here;
its trunk at fp16 is the 308-px row, an upper bound on its latency on this card.

Findings: (1) MobileCLIP2-S3 runs the deployed recipe in 9 ms, 3.6× faster than SigLIP under ORT and 10× more
throughput at batch 8, in a 240 MB fp16 file instead of 821 MB; the price is 3.9 pt macro-F1 (0.801 vs 0.840 OOF).
(2) PE-Core L is only 20 % faster than SigLIP under ORT (39 % in eager PyTorch): its attention-pooling graph does not
fuse as well, so its −0.5 pt costs more latency than the GFLOPs suggested. (3) Before reparameterisation
MobileCLIP2-S3 took 23 ms at batch 1 in eager PyTorch, slower than PE-Core L, despite 12× fewer FLOPs: a hybrid's
batch-1 cost is kernel launches, not arithmetic, so FLOPs alone mis-rank the candidates and measured latency is the
axis for the write-up. (4) The efficient frontier on this card is therefore SigLIP at 384 (0.840 / 33 ms) and
MobileCLIP2-S3 (0.801 / 9 ms); PE-Core L (0.835 / 26 ms) sits close to the line but is dominated once Fast's INT8
trunk is counted. Conclusion of the study: frozen features + fused linear heads is the right recipe at every size,
fine-tuning adds nothing at this data scale, and the choice is 4 pt of macro-F1 for a 3.6× latency and 3.4× size cut.

Follow-up (same day): the Fast trunk's resolution without its INT8. SigLIP So400m at 308 px, crop 1.0, fp16: fused
OOF 0.8222 F1 / 0.8244 acc, −1.81 ± 0.21 pt vs 384 px (0/5 folds), which matches the team lead's recorded Fast
fit (0.826 OOF acc, INT8 trunk, exports/fast/fit_report.json), so the Fast model's accuracy cost is the resolution,
not the quantisation. Under ORT CUDA fp16 in one session: SigLIP@384 31.9 ms, SigLIP@308 20.4 ms, MobileCLIP2-S3
8.9 ms. The small model is therefore 2.3× faster than the Fast trunk at fp16 for 2.1 pt less (0.801 vs 0.822);
Fast's INT8 engine under TensorRT will be faster than 20 ms, by an amount only measurable with its release files.
Tooling: `extract_views.py key@px`, `fusion_table.py --siglip-view TAG`, `export_candidate.py --px`.

## 2026-11-05 — Efficiency study: packaged "Small" model and write-up

`tools/views/package_candidate.py --model mobileclip2_s3 --out exports/mobileclip2_s3` builds the small model the way
`compclass.final` builds the deployed one (helpers reused): OOF of the exact pipeline on the dev folds 0.8025 acc /
0.8022 macro-F1 (in-scope 0.837; probe-only 0.793), temperature 1.100 by NLL (ECE 0.025 → 0.032: the OOF probabilities
were already near-calibrated), abstain threshold 0.55 (first grid step with ≥ 90 % accepted accuracy), probe refitted
on all 2,512 dev rows (C 0.01); one ONNX graph in inference form with the projection, both heads and the temperature
in-graph, fp32 479 MB (fp16 copy 240 MB); parity on 32 real dev images vs ORT CPU: max |Δp| 4.8e-7, argmax 32/32.
`tools/gen_manifests.py` gained a row for it and `--only ID` (a machine without every model's graph can still write
its own manifest); `exports/mobileclip2_s3/model.json` and the index line in `exports/models.json` are its output,
head id `small`, title "Small (fused MobileCLIP2-S3, 256 px)", default list unchanged. `packaging/bundle.txt` lists
the files; weights licence note `apps/gpui-demo/packaging/licenses/MODEL-WEIGHTS-MobileCLIP2.txt` (Apple
`apple-amlr`, research/education only, unlike SigLIP's Apache-2.0). Write-up: `docs/EFFICIENCY_STUDY.md` with
`docs/figures/efficiency_frontier.png` (`tools/views/efficiency_figure.py`). Sealed test and Wikimedia untouched.

## 2026-11-05 — THE one sealed-test look at the Small model (exports/mobileclip2_s3)

`tools/views/test_small.py --confirm`, the exact exported pipeline (head refitted on all dev rows, T 1.10, abstain
0.55) on the 418 sealed test rows, features at the registry view (crop 1.0): **accuracy 0.8062, macro-F1 0.8024**
(probe only 0.7990 / 0.7953; in-scope n 306: 0.8333 / 0.8306; coverage at the 0.55 threshold 0.785 with accepted
accuracy 0.912; ECE 0.060). The OOF estimate was 0.8025 / 0.8022, so the development folds predicted the test to
within 0.4 pt. Paired against the deployed model's recorded test predictions (`results/final/final_test_v2_preds.npz`,
0.8373 / 0.8337, same 418 rows): Δ accuracy −3.1 pt, 95 % paired bootstrap [−6.0, −0.2]; Δ macro-F1 −3.1 pt
[−6.2, −0.2]; McNemar 26 deployed-right/Small-wrong vs 13 the other way, agreement 85.6 %. Files:
`results/final_test_mobileclip2_s3.json` (+ `_preds.npz`). The test split is now spent for the Small model; the
Wikimedia holdout was not scored.

## 2026-11-05 — The deployed model refitted and exported at crop 1.0 (`exports/final_crop100`, not a replacement)

`tools/views/deployed_crop100.py`: compclass.final's fit_head and FusedClassifier on the crop-1.0 SigLIP cache (the
only change is eval preprocessing: resize short side to 384, no centre crop), exported next to the shipped
`exports/final`, which is untouched. OOF of the exact pipeline: accuracy 0.8420 / macro-F1 0.8410 (recorded deployed
0.8264 / 0.8256, +1.6 pt), in-scope 0.873, probe-only 0.835, temperature 0.888, ECE 0.055 → 0.042, abstain threshold
0.45 (the 90 %-accepted-accuracy rule reaches its target one grid step earlier than at crop 0.9; the team lead may
prefer to keep 0.50 for continuity), final C 0.003. Graph: model.onnx + model.onnx.data (1,637 MB fp32, external data
as for `final`), parity on 32 dev images vs ORT CPU 7e-6 / 32 of 32. Manifest `deployed_crop100` ("Deployed at crop
1.0"), listed in `exports/models.json`, default list unchanged, no bundle.txt entry (1.7 GB more; the team lead's call
whether it replaces `final`). Not scored on the sealed test.

## 2026-11-05 — Course pipeline: model D (frozen MobileCLIP2-S3 hybrid + linear head), one test look

The user decided to present the small model as a fourth main model of the ITRI626 comparison. The brief (§6) requires
main models to be trained and evaluated on the same split, so D is a new head, not the packaged Small model (whose
head was fitted on all dev rows): `scripts/itri626_experiments.py --steps head_D reference_D` trains
`nn.Linear(3072, 20)` on the standardised cached MobileCLIP2-S3 features of the fixed split's 2,010 training rows with
the same code, weight-decay grid {1e-4 … 1e-1} and validation rule as model C's head (all four values tie; 1e-4 kept),
best epoch 11 of 31, validation 0.7749 acc / 0.7724 macro-F1, and scores the 418 test rows once: **accuracy 0.7727,
macro-F1 0.7703** (95 % bootstrap 0.732–0.811 / 0.726–0.806). Paired on the same rows: C − D +5.0 pt [1.4, 8.6], 39 vs
18 discordant, McNemar p = 7.5e-3; D − B +13.6 pt [9.1, 17.9], 80 vs 23, p = 1.5e-8; D − A +37.3 pt. R3 reference row =
the packaged Small model recomputed from cached features and asserted equal to its recorded look (0.8062). A, B, C, R1,
R2 entries of metrics.json and profile.json verified unchanged. Profile (`--profile-only`, RTX 4070 Laptop PC, not the
report's 4060 Ti PC): D 15.2 ms GPU b1 / 3.10 ms per image b32 / 111.5 ms CPU; C's tower timed on the same PC with the
same procedure as the anchor: 37.9 / 27.9 / 1,563 ms, so D is 2.5× / 9× / 14× faster. The fused Small model's test
look (0.806) and this one are two different heads on the same tower; both are recorded, neither rescored.

Report integration: the team lead places report text himself, so `docs/itri626_report/`, both notebooks, the helpers
and `results/itri626/figures/` are untouched on `Meiring`. `docs/model_d_inserts/INSERTS.md` lists each location and
the proposed text, with ready-made table rows and four-model figures beside it. A full trial build (report, notebook,
figures with D throughout) is parked on the local branch `Meiring-four-model-draft` (2bd71b3, not for merge); in it
the body grew from 15 to 17 pages, 15.4 with the hyperparameter table moved to the appendix. Open: the lecturer's
approval of a conv-attention hybrid as a fourth family.

Follow-up (same day): the notebooks carry model D (three-models notebook with a `WITH_D` switch, figures to
`results/itri626/figures_four_models/`; training notebook section 3b), and `docs/model_d_inserts/` now also holds
`FULL_REPORT_WITH_MODEL_D.md`, the whole report as Markdown with the additions marked (`<ins>`/`<del>`), built by
`build_full_report.py` from the three-model report and the four-model trial build. The report itself is unchanged.
