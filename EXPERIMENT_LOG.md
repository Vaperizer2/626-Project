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
