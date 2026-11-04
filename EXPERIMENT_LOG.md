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

On 2026-11-03 the owner kept today's Fast model (`compose_int8_res308`) as the app's Fast. Q5 is not shipped. The
squash preprocessing from the hard-case study is not shipped either; whether to ship it is still open.

The demo stack grew from 28 to 68 pictures. The new Random section holds 40 holdout photos, two per class, drawn with
seed 20261103 from the holdout photos not already in the stack, with nobody looking at the answers first. Deployed
gets 32 of them right, in line with its 309 of 407 (75.9 %) on the whole holdout. The stack is a presentation set:
no model was chosen or tuned on it. `notebooks/ITRI626_training.ipynb` adds the training code, one section per model
(A, B, C, Deployed, Fast).

On 2026-11-04 `main` took PR #1 (merge `d4d5113`), PR #3 (`4bebc2d`), PR #4 (`d089596`) and PR #2 (`5d16657`). The GPU
PC `carlpc2` is gone. GPU work runs on the laptop, and heavy CPU work on Boat.dev CPU sandboxes. None of this changes
a recorded research score.
