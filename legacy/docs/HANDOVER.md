# Handover — 2026-08-13

## Latest continuation — 2026-08-13

The authenticated compute-machine state was synchronized back into this local
copy. The validation-only Optuna follow-up ran on `ai-pc-04` for the eight
unique candidates formed from the top three overall and the best model from
each family: ConvNeXt-Tiny, ConvNeXt-Base, ConvNeXt-Small, ResNet152,
DenseNet121, EfficientNet-V2-S, MobileNet-V3-Large, and Swin-B. It used five
nominal trials per model, 15 epochs, batch size 32, four data workers, and
searched learning rate, weight decay, label smoothing, class-weight power,
and backbone-freeze epochs.

ConvNeXt-Tiny remained the tuned leader at validation accuracy 0.7567 and
macro-F1 0.742929, only 0.000229 above the logged reference 0.7427. The
strong-recipe ConvNeXt-Base result at macro-F1 0.7473 remains the best
validation result overall. The tuner never constructed the test loader, so no
new test labels were read. The final artifacts are synchronized under
`notebooks/outputs/metrics/family_hparam_tuning_v3.*` and
`notebooks/outputs/weights/family_hparam_v3_*.pt`.

The background research deliverables are [`DATASET_IMPROVEMENT_RESEARCH.md`](DATASET_IMPROVEMENT_RESEARCH.md),
which recommends deduplication and group/source-disjoint resplitting before
more architecture work, and [`OPENCV_RESISTANCE_RESEARCH.md`](OPENCV_RESISTANCE_RESEARCH.md),
which recommends an abstaining OpenCV resistor-band decoder before adding a
learned secondary model.

## Current state

The requested CNN checkpoint is complete. Three scratch-CNN variants finished on
the frozen raw split. The width-64 no-hidden-layer variant was skipped at the
user's request; it was interrupted during epoch 1 before a checkpoint was
written. The updated notebooks and benchmark runner were authorized, uploaded,
checksum-verified, and compiled on `ai-pc-04`. The seven-model family/size sweep
ran in the background on the RTX 5080. ResNet34 completed with validation
accuracy 0.6820 / macro-F1 0.6588 at epoch 8 (gap 0.2848, not flagged), and
ResNet101 completed as the current validation leader at accuracy 0.7222 /
macro-F1 0.7079 (gap 0.2671, not flagged). DenseNet121 completed at accuracy
0.7050 / macro-F1 0.6783 (gap 0.2807, not flagged). EfficientNet-B2 then
completed at accuracy 0.7146 / macro-F1 0.7004 (gap 0.2489, not flagged), and
MobileNet-V3-Large completed at accuracy 0.6954 / macro-F1 0.6725 (gap 0.2833,
not flagged), and ConvNeXt-Tiny completed as the validation leader at accuracy
0.7605 / macro-F1 0.7427 (gap 0.2296, not flagged). The runner initially paused at
Swin-T because the shared freeze helper omitted the `head` parameter name; notebooks
01–03 were patched, uploaded, smoke-tested, and the resumed Swin-T trial completed
at accuracy 0.7299 / macro-F1 0.7086 (gap 0.2418, not flagged). The validation-selected
ConvNeXt-Tiny then scored 0.6979 accuracy / 0.6789 macro-F1 on the held-out raw test,
so the >80% target was not reached. The deferred-test larger phase is now active on
ConvNeXt-Small after ResNet152 completed at validation accuracy 0.7414 /
macro-F1 0.7257 (gap 0.2516, not flagged), EfficientNet-V2-S completed at
validation accuracy 0.7222 / macro-F1 0.7082 (gap 0.2692, not flagged), and
ConvNeXt-Small completed at validation accuracy 0.7414 / macro-F1 0.7266 (gap
0.2500, not flagged). Swin-V2-Tiny then completed at validation accuracy 0.7222 /
macro-F1 0.7060 (gap 0.2560, not flagged). Phase 2 therefore did not beat the
global ConvNeXt-Tiny validation leader, and its test remained deferred. The largest-model
phase then completed validation-only: ConvNeXt-Base reached accuracy 0.7605 / macro-F1
0.7407 (gap 0.2317), EfficientNet-V2-M reached 0.7126 / 0.6909 (gap 0.2792), and
Swin-B reached 0.7414 / 0.7234 (gap 0.2303). ConvNeXt-Base led that phase but still
trailed ConvNeXt-Tiny's 0.7427 validation macro-F1, so phase 3 also deferred its test.
The strong-regularisation recipe phase then put ConvNeXt-Base first at validation
accuracy 0.7605 / macro-F1 0.7473 (gap 0.2329, not flagged), ahead of ConvNeXt-Tiny
at macro-F1 0.7366. That new validation leader was evaluated once on the raw test:
accuracy 0.7132 / macro-F1 0.6969 / weighted-F1 0.7113, still below 80%. Four
pre-specified ensembles were validation-only and all trailed the single strong
ConvNeXt-Base. The accuracy-focused ConvNeXt-Base recipe reached validation
macro-F1 0.7317 (gap 0.2090), so it was not test-promoted. No model has been
promoted to production. The final training-only curation trial is complete: four
reviewed `train-*` images were removed from training only, leaving validation and
test membership fixed. It reached accuracy 0.7567 / macro-F1 0.7421 at epoch 16
(gap 0.2351), below the strong ConvNeXt-Base validation leader at macro-F1 0.7473.
No additional test labels were read. The bounded search is complete for this
contract; the best permitted raw-test result remains 0.7132 accuracy, below the
80% target. The final logs are `outputs/metrics/model_family_benchmark_phase6_accuracy.log`
and `outputs/metrics/model_family_benchmark_phase7_train_clean.log`.

The working repository is `/home/carlai04/SEB` on `ai-pc-04`; the local mirror
contains the source and documentation updates.

## Evaluation contract

- Raw dataset fingerprint: `78637c8898aa10f8`
- Raw split fingerprint: `45f4831c25e45119`
- Split sizes: train 2,438 / validation 522 / test 523
- Model selection: validation macro-F1 only
- Held-out test: evaluated only after validation selection
- Objective: 80–90% accuracy without overfitting

## Results to carry forward

| Model | Validation | Held-out raw test |
|---|---:|---:|
| ResNet18 baseline | best macro-F1 0.6747 | accuracy 0.6367; macro-F1 0.6149 |
| ResNet50 | accuracy 0.7471; macro-F1 0.7274 | accuracy 0.6845; macro-F1 0.6640 |
| EfficientNet-B0 | accuracy 0.7069; macro-F1 0.6860 | not selected |
| Family benchmark — ResNet34 | accuracy 0.6820; macro-F1 0.6588 | deferred until all validation trials finish |
| Family benchmark — ResNet101 | accuracy 0.7222; macro-F1 0.7079 | deferred until all validation trials finish |
| Family benchmark — DenseNet121 | accuracy 0.7050; macro-F1 0.6783 | deferred until all validation trials finish |
| Family benchmark — EfficientNet-B2 | accuracy 0.7146; macro-F1 0.7004 | deferred until all validation trials finish |
| Family benchmark — MobileNet-V3-Large | accuracy 0.6954; macro-F1 0.6725 | deferred until all validation trials finish |
| Family benchmark — ConvNeXt-Tiny | accuracy 0.7605; macro-F1 0.7427 | deferred until all validation trials finish |
| Family benchmark — Swin-T | accuracy 0.7299; macro-F1 0.7086 | deferred until all validation trials finish |
| Family benchmark — selected ConvNeXt-Tiny | accuracy 0.7605; macro-F1 0.7427 | accuracy 0.6979; macro-F1 0.6789; weighted-F1 0.6960 |
| Larger phase — ResNet152 | accuracy 0.7414; macro-F1 0.7257 | deferred until phase-2 validation selection |
| Larger phase — EfficientNet-V2-S | accuracy 0.7222; macro-F1 0.7082 | deferred until phase-2 validation selection |
| Larger phase — ConvNeXt-Small | accuracy 0.7414; macro-F1 0.7266 | deferred until phase-2 validation selection |
| Larger phase — Swin-V2-Tiny | accuracy 0.7222; macro-F1 0.7060 | deferred until phase-2 validation selection |
| Larger phase — selected ConvNeXt-Small | accuracy 0.7414; macro-F1 0.7266 | test deferred; global leader remains ConvNeXt-Tiny |
| Largest phase — ConvNeXt-Base | accuracy 0.7605; macro-F1 0.7407 | test deferred; phase winner, global leader remains ConvNeXt-Tiny |
| Largest phase — EfficientNet-V2-M | accuracy 0.7126; macro-F1 0.6909 | test deferred |
| Largest phase — Swin-B | accuracy 0.7414; macro-F1 0.7234 | test deferred |
| Largest phase — selected ConvNeXt-Base | accuracy 0.7605; macro-F1 0.7407 | test deferred; global leader remains ConvNeXt-Tiny |
| Strong recipe — ConvNeXt-Tiny | accuracy 0.7548; macro-F1 0.7366 | test deferred; gap 0.2382 |
| Strong recipe — ConvNeXt-Base | accuracy 0.7605; macro-F1 0.7473 | selected validation leader; gap 0.2329 |
| Strong recipe — selected ConvNeXt-Base | accuracy 0.7605; macro-F1 0.7473 | accuracy 0.7132; macro-F1 0.6969; weighted-F1 0.7113 |
| Best ensemble — strong ConvNeXt-Base + Swin-T | accuracy 0.7586; macro-F1 0.7415 | test deferred; below single strong Base |
| Accuracy-focused recipe — ConvNeXt-Base | accuracy 0.7471; macro-F1 0.7317 | test deferred; gap 0.2090 |
| Training-only reviewed curation — ConvNeXt-Base | accuracy 0.7567; macro-F1 0.7421 | test deferred; gap 0.2351; four training exclusions applied |
| Best completed scratch CNN: width 32 | accuracy 0.2375; macro-F1 0.1817 | accuracy 0.2275; macro-F1 0.1803 |

The CNN result is a clear negative result: capacity/dropout changes did not
make scratch training competitive with pretrained backbones. Do not promote
the width-32 checkpoint merely because it won its small comparison.

The evaluation-only model error audit is also complete. On the raw test split,
ResNet50 was uniquely correct on 247 images while the width-32 scratch CNN was
uniquely correct on 8. ResNet50's lowest test recalls were fuse (0.364),
resistor (0.440), diode (0.455), transistor (0.458), and relay (0.522).

The user-session contract has been carried into notebooks 01–04. They accept
`COMPCLASS_WORK_DIR`, write `outputs/metrics/session_manifest.json`, use the
raw split manifest, and record validation-only selection. Notebook 02 now
reports test metrics only for its validation-selected row. The new
`12_model_family_benchmark.py` completed the registered family/size and recipe
phases with per-epoch progress checkpoints and skips only matching completed
checkpoints on restart. It supports the larger model families, the
`robust-v2-tta`, `strong-regularized`, and `accuracy-regularized` recipes, a
separate artifact prefix, and an optional training-only curation manifest. The
new `13_ensemble_benchmark.py` evaluates pre-specified TTA-logit ensembles on
validation and leaves test evaluation opt-in.
The new `14_model_family_walkthrough.ipynb` presents all 14 candidates in
family/size order, gives each model and family an explanatory markdown block,
runs one validation-only model per cell, and compiles the resulting statistics
into a styled table plus CSV/JSON summary.

## Checkpoints and reports

Remote artifacts are under `/home/carlai04/SEB/notebooks/`:

- `outputs/metrics/cnn_trials.json`
- `outputs/weights/small_cnn_original.pt`
- `outputs/weights/small_cnn_high_dropout.pt`
- `outputs/weights/sized_cnn_width32.pt`
- `outputs/weights/resnet50_full.pt`
- `outputs/weights/efficientnet_b0_full.pt`
- `outputs/weights/family_benchmark_resnet34.pt`
- `outputs/weights/family_benchmark_resnet101.pt`
- `outputs/weights/family_benchmark_densenet121.pt`
- `outputs/weights/family_benchmark_efficientnet_b2.pt`
- `outputs/weights/family_benchmark_mobilenet_v3_large.pt`
- `outputs/weights/family_benchmark_convnext_tiny.pt`
- `outputs/weights/family_benchmark_swin_t.pt`
- `outputs/weights/family_benchmark_resnet152.pt`
- `outputs/weights/family_benchmark_efficientnet_v2_s.pt`
- `outputs/weights/family_benchmark_convnext_small.pt`
- `outputs/weights/family_benchmark_swin_v2_t.pt`
- `outputs/weights/family_benchmark_convnext_base.pt`
- `outputs/weights/family_benchmark_efficientnet_v2_m.pt`
- `outputs/weights/family_benchmark_swin_b.pt`
- `outputs/metrics/model_error_audit.json`
- `outputs/metrics/model_error_audit.csv`
- `outputs/figures/model_error_audit_confusion.png`
- `outputs/metrics/session_manifest.json` (created by the updated notebooks)
- `outputs/metrics/model_family_benchmark.json` (completed seven-model sweep)
- `outputs/metrics/model_family_benchmark_phase2.json` (deferred-test larger-model phase, when run)
- `outputs/metrics/model_family_benchmark_phase3.json` (deferred-test largest-model phase, when run)
- `outputs/metrics/model_family_benchmark_phase4_strong.json` (strong-recipe validation phase)
- `outputs/metrics/model_family_benchmark_final_test.json` (one test evaluation of the strong-recipe validation winner)
- `outputs/metrics/ensemble_benchmark_phase5*.json` (validation-only ensemble comparisons)
- `outputs/metrics/model_family_benchmark_phase6_accuracy.json` (accuracy-focused validation phase)
- `outputs/metrics/model_family_benchmark_phase7_train_clean.json` (training-only reviewed-exclusion phase)
- `outputs/metrics/model_walkthrough_*_summary.csv` / `.json` (guided 14-model walkthrough summary)
- `outputs/metrics/family_hparam_tuning_v3.json` (validation-only Optuna report)
- `outputs/metrics/family_hparam_tuning_v3.log` and `outputs/metrics/family_hparam_v3.db` (tuner log and study)
- `outputs/weights/family_benchmark_*.pt` (completed/progress checkpoints)
- `outputs/weights/family_recipe_*.pt` (recipe-phase checkpoints)
- `outputs/weights/family_hparam_v3_*.pt` (best checkpoint for each tuned candidate)
- `data/curation/reviewed_exclusions_v1.json`
- `data/splits/split_manifest_raw.json`

The checkpoint payloads retain their training histories. The aggregate history
is also in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), and the CNN report is
[`notebooks/outputs/metrics/cnn_trials.json`](notebooks/outputs/metrics/cnn_trials.json).

## Resume commands

```bash
ssh -F /home/vaper/.ssh/config ai-pc-04
cd /home/carlai04/SEB/notebooks
../.venv/bin/python -m py_compile 12_model_family_benchmark.py
../.venv/bin/python 12_model_family_benchmark.py
```

Inspect the report without training:

```bash
../.venv/bin/python -c 'import json; print(json.dumps(json.load(open("outputs/metrics/model_family_benchmark.json")), indent=2))'
../.venv/bin/python -c 'import json; print(json.dumps(json.load(open("outputs/metrics/cnn_trials.json")), indent=2))'
../.venv/bin/python 15_family_hyperparam_tuning.py --profile paper --trials 5 --epochs 15 --patience 4 --batch-size 32 --num-workers 4 --report family_hparam_tuning_v3.json --artifact-prefix family_hparam_v3
```

The family benchmark skips matching completed checkpoints and can be narrowed
with `COMPCLASS_BENCHMARK_MODELS=...`. For a training-only reviewed exclusion
trial, pass `--training-exclusions data/curation/reviewed_exclusions_v1.json`;
only `train-*` entries are applied, and the manifest hash is part of the
checkpoint fingerprint. Do not use the held-out test labels to choose a new
candidate list; append every completed trial to `EXPERIMENT_LOG.md`.

## Recommended next work

1. Use the strong-recipe ConvNeXt-Base checkpoint as the current raw-test model:
   it was selected without test tuning and scored 0.7132 accuracy / 0.6969
   macro-F1 / 0.7113 weighted-F1.
2. For future self-training, use notebook 14 for the guided family walkthrough,
   or start from notebooks 01–03 / the resumable runner for a more compact run;
   keep `session_manifest.json`, the frozen split fingerprints, validation-only
   selection, and separate artifact prefixes. Notebook 04 remains an explicitly
   educational toy lab and is not a production benchmark.
3. Treat further gains as a new, explicitly registered data-quality project.
   Do not reopen the held-out test for more tuning unless a new validation phase
   first beats macro-F1 0.7473 without exceeding the gap guard.
4. Do not start another broad architecture sweep on the legacy image-level
   split. Follow `DATASET_IMPROVEMENT_RESEARCH.md`: make a versioned,
   byte/perceptual-deduplicated, group/source-disjoint dataset and a fresh
   untouched holdout first.
5. For the OpenCV demo, implement the hybrid resistor-band pipeline described
   in `OPENCV_RESISTANCE_RESEARCH.md`; return an abstention/uncertainty state
   when bands cannot be localized or decoded reliably.
