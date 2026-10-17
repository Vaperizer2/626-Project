# Notebook and OpenCV Demo Audit

**Audit date:** 2026-08-16  
**Repository commit inspected:** `0f81bd55c6082d13157bc478be979d7ff9fe10e8`  
**Scope:** read-only inspection of notebooks, experiment scripts, saved metrics,
checkpoints, figures, reports, dependency metadata, and searches for existing
OpenCV/demo code. No training was run, no compute host was accessed, and no
existing file was changed.

## Executive decision

The repository has enough archived evidence to build a good final comparative
notebook, but it does not yet have a defensible live-camera demo. The current
best legacy-split checkpoint is the strong-regularized ConvNeXt-Base: validation
accuracy/macro-F1 `0.7605/0.7473`, followed by one held-out legacy-test result of
`0.7132/0.6969`. That is below the stated 80% accuracy target, and the later data
audit found exact training copies in 24.5% of validation and 22.0% of test rows.
These scores must therefore be labeled **legacy, image-level, and
duplicate-contaminated**, not deployment estimates.

The newer exact-content-grouped five-fold experiment is the stronger internal
comparison. Its best single model is ConvNeXt-Base at OOF accuracy/macro-F1
`0.6402/0.6288`; the all-14 majority vote reaches `0.6574/0.6472`. The legacy
test was deliberately deferred. This result is less flattering but more useful
for deciding what remains to be proved.

The smallest honest live deliverable is one OpenCV window, one guided ROI, one
already-trained classifier, calibrated accept/abstain logic, temporal
stability, and clear status overlays. A controlled four-band resistor decoder
can be a second, gated route after a stable `resistor` decision. It must report
a **visual nominal value and tolerance**, never an electrical measurement. A
frame-wide detector, an all-model ensemble, a learned resistance regressor,
and a separate desktop GUI framework are outside the minimum viable scope.

## Evidence inspected

| Area | Principal evidence | Audit reading |
|---|---|---|
| Primary comparative | `notebooks/01_primary_comparative.ipynb`, `primary_metrics.json` | Defines HOG→LinearSVM, SmallCNN, and ResNet18. The saved notebook is only partly executed: setup/data/HOG have outputs, while SmallCNN, ResNet18, and final results do not. The HOG output also records repeated `max_iter=2000` convergence warnings. |
| Exploration/tuning | Notebooks 02–04 and scripts 05–13, 15–16 | Useful experiment archive, but spread across notebooks, scripts, subprocess wrappers, and multiple run contracts. Notebook 04 is correctly marked educational rather than benchmark evidence. |
| Family walkthrough | `notebooks/14_model_family_walkthrough.ipynb`, `model_walkthrough_robust_v2_tta_summary.{json,csv}` | Contains a short explanation and one `run_model(...)` call per model. It has no executed cells; every row in the saved summary is `not run`. It is a launcher, not a final result notebook. |
| Legacy benchmark | `model_family_benchmark*.json`, `family_hparam_tuning_v3.json`, `EXPERIMENT_LOG.md`, `HANDOVER.md` | ConvNeXt-Base strong recipe is the recorded legacy-split leader. Tuning did not improve it. No model reached the target. |
| Grouped benchmark | `CROSS_FOLD_AUGMENTATION_REPORT.md`, `cross_fold_augmentation.{json,csv}`, OOF tensor and log | Five-fold, exact-byte-grouped comparison over legacy train+validation. Best single model: ConvNeXt-Base; best aggregation: all-14 majority vote. Legacy test remains deferred. |
| Data quality | `DATASET_IMPROVEMENT_RESEARCH.md`, split manifests, curation/audit outputs | The 3,483-image scrape has mixed visual domains, label noise, cross-label duplicates, and substantial exact-copy leakage. There is no independent target-camera set. |
| OpenCV/value work | `OPENCV_RESISTANCE_RESEARCH.md`, repository-wide source/dependency search | Design research exists, but there is no `cv2` import, `VideoCapture`, `imshow`, camera calibration, ROI/detection/segmentation implementation, demo entry point, or OpenCV package in `pyproject.toml`/`uv.lock`. |
| Model artifacts | `notebooks/outputs/weights/` (about 5.3 GiB) | State-dict checkpoints exist, including the selected 350 MB ConvNeXt-Base and a 17 MB MobileNet-V3-Large. They are training artifacts, not versioned deployment exports. |

## Current gaps

### P0 — claims and evaluation contract

1. **There is no single authoritative run contract.** Archived artifacts use at
   least three fingerprint pairs: old primary metrics
   (`468da67462c84f85/ae4fdfb77f68462f`), current notebook smoke artifacts
   (`78637c8898aa10f8/1c82cf8d3c6ba77d`), and family reports
   (`78637c8898aa10f8/45f4831c25e45119`). The latter two appear to identify the
   same 2,438/522/523 membership through different fingerprint algorithms, but
   that equivalence is not recorded in the artifacts.
2. **The legacy validation and test sets are not content-independent.** Exact
   training copies occur in 128/522 validation rows and 115/523 test rows;
   cross-label duplicate groups also exist. Legacy metrics may be retained for
   historical traceability but cannot support a camera-performance claim.
3. **The legacy test is no longer an unseen final holdout.** It has been used in
   several selected evaluations and error/curation audits. The grouped OOF run
   correctly leaves it untouched, but a fresh target-domain holdout is still
   required.
4. **Saved outputs are not notebook-complete.** Notebook 01 ends before the two
   neural baselines/results are executed. Notebook 14 has no executions, and
   its saved summary contains fourteen `not run` rows. A reviewer cannot obtain
   the reported final story by opening one notebook and reading top to bottom.
5. **Some archived metrics are smoke tests.** `playground_metrics.json`,
   `optuna_results.json`, and the current `session_manifest.json` describe
   64-pixel CPU smoke runs or a 40/40/40 profile slice. They must not appear in
   a paper table beside full paper-profile results.

### P0 — live-demo safety

1. **There is no OpenCV or camera implementation.** Even dependency resolution
   for OpenCV is absent.
2. **Classifier confidence is uncalibrated.** Softmax scores, per-row confidence,
   margins, and flip agreement are archived, but there is no temperature
   scaling, reliability diagram, ECE/Brier report, risk–coverage curve, or
   frozen acceptance threshold.
3. **There is no abstention/OOD contract.** The model always chooses one of 20
   classes. It has never been validated on empty frames, hands, multiple
   objects, printed images/screens, unsupported components, severe blur, or
   target-camera backgrounds. A low softmax score alone is not an OOD detector.
4. **Training and demo domains do not match.** Training data mix product photos,
   diagrams, collages, adverts, and scenes; there is no camera/session/product
   identity metadata and no camera-domain validation or holdout.
5. **The classifier does not localize objects.** It is a whole-image classifier.
   A live full frame with clutter is outside its demonstrated input contract;
   a guided or user-selected ROI is required for the MVP.
6. **No latency or resource measurement exists on the demo machine.** The
   selected checkpoint is about 350 MB. The grouped all-14 ensemble requires
   many large checkpoints and is not a smallest viable real-time path.
7. **Resistance decoding is unimplemented and unlabeled.** The current resistor
   class contains 167 heterogeneous images and has no band sequence, tolerance,
   body box/mask, physical identity, or measured-value labels. The existing
   classifiers are resistor gates only.

### P1 — reproducibility and artifact provenance

1. `uv.lock` is a good base, but run artifacts do not consistently record the
   exact package versions, Python/platform, CUDA/cuDNN, device, driver, OpenCV
   build, or deterministic-algorithm flags.
2. Seeding covers Python, NumPy, and Torch, but the inspected helper does not
   enable deterministic algorithms or record cuDNN determinism/benchmark state.
3. Family checkpoints record model/settings, class order, image size, and split
   fingerprints, but the selected checkpoint omits the normalization constants,
   preprocessing version/hash, pretrained weight enum/hash, code commit,
   dependency lock hash, calibration object, and decision thresholds.
4. Several reports/checkpoints contain absolute paths from `/home/carlai04/SEB`;
   the current session manifest contains an absolute local path. Artifact
   identities should be content hashes plus repository-relative paths.
5. There are two data/output roots (`data/` versus `notebooks/data/`, and
   `outputs/` versus `notebooks/outputs/`). The root `outputs/` is empty while
   the notebook-local ignored output tree contains the evidence.
6. The checkpoint directory contains many exploratory copies. There is no small
   promotion bundle containing exactly one model, class map, preprocessing,
   calibration policy, license/provenance note, and checksum manifest.
7. There are no automated inference-contract tests: BGR→RGB conversion, class
   ordering, tensor shape/range, checkpoint compatibility, empty/corrupt frame,
   threshold boundary, temporal reset, and graceful camera shutdown are
   currently unchecked.

### P2 — presentation

1. The final narrative is distributed across README, handover, experiment log,
   research reports, multiple notebooks, and machine-readable outputs.
2. Per-model explanations in notebook 14 are useful but too shallow for a final
   submission: setup, actual training configuration, saved results, limitations,
   and model-specific interpretation are not separate visible sections.
3. Learning curves, per-class tables, normalized confusion matrices, model size,
   latency, calibration, and selective-performance views are not presented under
   one consistent contract.
4. “No overfit flag” only means the configured train–validation accuracy gap did
   not exceed 0.30. Gaps around 0.23 are still large and must not be described as
   evidence of good generalization.

## Results that the final notebook may carry forward

The notebook should display provenance beside every result. It must not merge
the following tiers into one sortable leaderboard.

| Evidence tier | Result | Permitted interpretation |
|---|---|---|
| Historical primary | HOG+LinearSVM test accuracy/macro-F1 `0.3111/0.2606`; SmallCNN `0.2309/0.1632`; ResNet18 `0.6565/0.6295` in `primary_metrics.json` | Historical paper-profile comparison under the older fingerprint only. Do not compare numerically with current-contract reports without rerunning. |
| Legacy raw v1 | ResNet18 validation macro-F1 `0.6747`, test `0.6149`; ResNet50 validation/test `0.7274/0.6640`; EfficientNet-B0 validation `0.6860` | Reproducible image-level legacy benchmark, known to contain duplicate leakage. |
| Legacy raw v1, selected recipe | Strong ConvNeXt-Base validation/test accuracy `0.7605/0.7132`, macro-F1 `0.7473/0.6969` | Best archived single checkpoint for a prototype gate; not production-promoted and not a camera estimate. |
| Exact-content-grouped development OOF | ConvNeXt-Base accuracy/macro-F1 `0.6402/0.6288`; all-14 majority vote `0.6574/0.6472` | Preferred internal model-comparison evidence. Still shares sources/products and label noise; test deferred. |
| Walkthrough summary | Fourteen `not run` rows | Structure only; not evidence. |

## Final notebook outline

### Notebook rule

Create one final notebook whose saved state is readable without rerunning. It
may reuse shared data/training/evaluation helpers, but every model must have its
own visible **Setup**, **Explanation**, **Training**, **Results**, and
**Interpretation** subsections. A bare `run_model("name")` cell is not a model
section. The default mode should load immutable archived artifacts; training
must require an explicit `RUN_TRAINING=True`/confirmation and write to a new,
never-overwriting run directory.

### 0. Title, question, and executive result

- State the 20-class task and the intended guided-ROI camera use case.
- State the two questions separately: comparative image classification and
  live camera feasibility.
- Lead with the selected legacy checkpoint, grouped-OOF reality check, target
  shortfall, and “not production-promoted” status.
- Put the dataset leakage and domain-shift warning above the first score table.

### 1. How to run and how not to leak

- Exact `uv sync --frozen` and kernel commands.
- `MODE=load_archived|train_new`; default `load_archived`.
- One canonical working/data/output root.
- Validation-only model/threshold selection; fresh camera holdout labels sealed.
- Explain why smoke, historical-primary, legacy-raw, grouped-OOF, and camera
  results are separate evidence tiers.

### 2. Reproducibility manifest

Generate and display the complete metadata block specified later in this audit.
Fail early on an unexpected dataset/split/checkpoint/class-map hash. Show dirty
worktree state rather than silently treating it as a clean release.

### 3. Dataset, ontology, and quality audit

- Class names/order and train/validation/test counts.
- Representative images including failure modes: diagrams, collages,
  multi-object scenes, irrelevant images, and resistor subtypes.
- Exact-duplicate and cross-label counts; explain grouped splitting.
- Explicit target ontology, plus `ambiguous` and `out_of_scope` policy for new
  camera data.
- State that the legacy test is historical and a new camera holdout is needed.

### 4. Shared preprocessing and evaluation protocol

- RGB conversion, resize policy, ImageNet normalization, and train-only
  augmentation, rendered as data examples.
- Loss, class weighting, optimizer, scheduler, early stopping, TTA, fold logic,
  and selection metric.
- Metrics: accuracy, macro/weighted F1, per-class precision/recall/F1,
  confusion matrix, train–validation gap, parameter count, artifact size,
  wall time, and target-device latency.
- Calibration/abstention metrics are separate from classification metrics.

### 5–23. Separate model sections

Each numbered section below must contain these five headings in this exact
order:

1. **Setup** — model constructor/weight enum, input size, parameter count,
   model-specific hyperparameters, checkpoint/report hashes, and evidence tier.
2. **Explanation** — architecture intuition, why it is included, and expected
   accuracy/compute trade-off.
3. **Training** — an explicit cell showing the call and all resolved settings;
   either trains into a new run directory or loads the archived history. It
   must never overwrite the archived checkpoint.
4. **Results** — learning curve where applicable, validation/OOF metrics,
   per-class result, confusion matrix, model size, and latency placeholder or
   measurement. Test appears only where the selection contract permits it.
5. **Interpretation** — strengths, weak classes, generalization gap, data risks,
   whether it advances, and suitability for a live guided-ROI demo.

| Section | Model | Model-specific emphasis and current evidence to reconcile |
|---:|---|---|
| 5 | HOG + Linear SVM | Transparent classical baseline; disclose the archived convergence warning and old-contract test macro-F1 `0.2606`. |
| 6 | SmallCNN | Scratch-learning baseline; explain low data efficiency and old-contract test macro-F1 `0.1632`. Controlled scratch variants belong in an appendix table, not separate promoted models. |
| 7 | ResNet18 | First transfer baseline; explain the large training gap and legacy validation/test macro-F1 `0.6747/0.6149`. |
| 8 | ResNet50 | Historical family step; legacy validation/test macro-F1 `0.7274/0.6640`. |
| 9 | EfficientNet-B0 | Historical efficiency step; validation macro-F1 `0.6860`, test correctly not selected. |
| 10 | ResNet34 | Grouped OOF accuracy/macro-F1 `0.5537/0.5422`; contrast with legacy validation `0.6820/0.6588`. |
| 11 | ResNet101 | Grouped OOF `0.5845/0.5724`; legacy validation `0.7222/0.7079`. |
| 12 | ResNet152 | Grouped OOF `0.6014/0.5912`; legacy validation `0.7414/0.7257`. More depth is not a reliable route around data quality. |
| 13 | DenseNet121 | Grouped OOF `0.5733/0.5604`; legacy validation `0.7050/0.6783`; emphasize feature reuse versus memory traffic. |
| 14 | EfficientNet-B2 | Grouped OOF `0.5588/0.5468`; legacy validation `0.7146/0.7004`. |
| 15 | EfficientNet-V2-S | Grouped OOF `0.5855/0.5736`; legacy validation `0.7222/0.7082`. |
| 16 | EfficientNet-V2-M | Grouped OOF `0.5818/0.5708`; legacy validation `0.7126/0.6909`; larger was not better. |
| 17 | MobileNet-V3-Large | Grouped OOF `0.5311/0.5171`; legacy validation `0.6954/0.6725`; checkpoint is about 17 MB and is the explicit latency/size challenger for the demo. |
| 18 | ConvNeXt-Tiny | Grouped OOF `0.6088/0.5966`; legacy validation `0.7605/0.7427`; tuned result improved only `0.00023` macro-F1 over its logged reference. |
| 19 | ConvNeXt-Small | Grouped OOF `0.6291/0.6181`; legacy validation `0.7414/0.7266`. |
| 20 | ConvNeXt-Base | Best grouped single model at `0.6402/0.6288`; strong legacy recipe `0.7605/0.7473`, selected legacy test `0.7132/0.6969`; about 350 MB. This is the accuracy candidate, not an automatic live winner. |
| 21 | Swin-T | Grouped OOF `0.6061/0.5955`; legacy validation `0.7299/0.7086`; document the earlier head-freezing bug and fixed run provenance. |
| 22 | Swin-V2-T | Grouped OOF `0.6017/0.5891`; legacy validation `0.7222/0.7060`. |
| 23 | Swin-B | Grouped OOF `0.6176/0.6066`; legacy validation `0.7414/0.7234`; explain cost and why attention did not beat ConvNeXt-Base. |

### 24. Controlled recipe and tuning appendix

- Scratch-CNN variants, regularization trial, curated training-only trial,
  Optuna runs, and ensemble phase reports.
- Clearly mark invalidated, smoke, interrupted, deferred-test, and
  non-comparable runs.
- Show that the strong recipe helped ConvNeXt-Base, the training-only removal
  of four reviewed rows did not beat it, and bounded tuning did not materially
  improve the leader.

### 25. Aggregation experiment

- Describe the exact-content-grouped five-fold design and TTA.
- Compare mean probability, mean logits, majority vote, and top-three
  ConvNeXt aggregation.
- Interpret the all-14 majority result as a validation/OOF result, not a live
  deployment choice. Report its storage/latency cost and keep its test deferred.

### 26. Calibration, abstention, and model selection for camera use

- Compare ConvNeXt-Base (accuracy candidate) with MobileNet-V3-Large (size and
  latency candidate) on **camera validation**, not the scrape.
- Fit temperature scaling on camera validation logits only.
- Plot calibration and risk–coverage curves; select thresholds under a
  pre-registered accepted-error constraint.
- Freeze model, preprocessing, calibrator, thresholds, temporal policy, and ROI
  contract before the camera holdout is opened.

### 27. Error analysis and limitations

- Weak classes/confusions, especially fuse/resistor/diode/transistor/relay.
- Duplicate leakage, label/ontology noise, repeated validation adaptation,
  source-domain mismatch, and absence of a fresh holdout.
- Explain that nominal resistor decoding is not resistance measurement.

### 28. Reproducible conclusion and artifact index

- One table of claims with evidence tier and allowed use.
- One selected **prototype** bundle only after camera acceptance.
- Relative paths and SHA-256 for notebook, run manifest, model, class map,
  preprocessing, calibrator, thresholds, metrics, and camera calibration.
- Explicit verdict: accepted for guided demo, accepted with limitations, or
  blocked; never infer promotion from the best legacy score.

## Required reproducibility metadata

The final notebook should emit one machine-readable run manifest and render its
important fields. At minimum record:

### Code and environment

- UTC start/end time, notebook name/version, Git commit, branch, and dirty diff
  flag/hash.
- Python implementation/version; OS/kernel/architecture; CPU/RAM.
- Exact versions of NumPy, Pillow, scikit-image, scikit-learn, Torch,
  torchvision, OpenCV, Jupyter/nbformat, and the `uv.lock` SHA-256.
- Device name, CUDA runtime, GPU driver, cuDNN, compute capability, and peak
  memory where applicable.
- All resolved environment/profile flags; never record secrets.

### Randomness and execution

- Python/NumPy/Torch/CUDA seeds and per-model/fold seed derivation.
- `torch.use_deterministic_algorithms`, cuDNN deterministic/benchmark, TF32,
  DataLoader worker seed function, generator seed, worker count, and batch size.
- Cell execution order and a hash of the resolved training/evaluation code.

### Data contract

- Dataset name/source/license/version and canonical repository-relative root.
- Immutable content-manifest SHA-256, decoded-image count, class order/counts,
  exclusions/relabels, and corrupt/unsupported-file report.
- Split/fold assignment SHA-256, grouping key, seed, sizes, exact/near-duplicate
  audit, conflicting-label groups, and holdout-access policy.
- Image decoder/library version and the serialized train/evaluation transforms.

### Model and optimization

- Stable model ID, family/variant, exact constructor, pretrained-weight enum
  and hash, parameter count, input shape, channel order, mean/std, and dtype.
- Loss/class weights/label smoothing; optimizer; learning rate; weight decay;
  scheduler; freeze/unfreeze plan; epochs; patience; best epoch; selection metric;
  augmentation; TTA; and training history.
- Checkpoint/report SHA-256 and repository-relative artifact path.

### Calibration and demo

- Calibration method, fitting split, temperature/parameters, ECE/Brier/NLL,
  risk–coverage curve artifact, top-score and margin thresholds, temporal
  window/consensus, quality thresholds, and rejection reason taxonomy.
- Camera make/model/device ID, OpenCV backend/build, requested and actual
  resolution/FPS, codec, focus/exposure/white-balance state, calibration/profile
  hash, ROI coordinates, capture session, lighting condition, distance, and
  physical component ID.

## Smallest viable live OpenCV GUI pipeline

### Scope

Use a single Python process and an OpenCV HighGUI window. Start with one object
inside a fixed center guide rectangle on a plain background. Do not add Tk/Qt,
database logging, network calls, object detection, model ensembles, or automatic
resistor localization to the first live slice.

### Startup

1. Load a promotion-bundle manifest and verify the SHA-256 of the checkpoint,
   class list, preprocessing definition, calibrator, and threshold policy.
2. Reconstruct the declared model with pretrained download disabled, load the
   state dict, set `eval()`, and run one known-tensor/self-test.
3. Open `cv2.VideoCapture(camera_index, backend)`; verify `isOpened()`, request
   resolution/FPS, then record the actual properties returned by the camera.
4. If camera opening, checkpoint verification, class order, or self-test fails,
   show/print a specific error and exit nonzero without entering the loop.

### Per-frame loop

```text
read BGR frame
  -> reject failed/stale frame
  -> draw fixed ROI guide
  -> crop ROI
  -> quality gates: size, brightness/clipping, blur, occupancy/background
  -> BGR to RGB exactly once
  -> resize + tensor + normalization from bundle
  -> classifier inference at a capped rate (display may run faster)
  -> temperature-scaled probabilities
  -> top-1 score + top-1/top-2 margin + OOD/quality gate
  -> temporal consensus over recent valid predictions
  -> ACCEPTED_CLASS or explicit abstention reason
  -> optional resistor route only after stable accepted `resistor`
  -> overlay status, label/top alternatives, confidence, FPS/latency, controls
  -> `q`/Esc exits; release camera and destroy windows in `finally`
```

Inference can run every Nth frame in the same loop; reuse the most recent result
with its age shown. That is simpler than adding threads and is adequate until a
measured GUI responsiveness failure justifies concurrency.

### Minimum overlay and controls

- ROI rectangle; status color; accepted label or `UNSURE`.
- Calibrated top score, runner-up/margin, rejection reason, inference latency,
  display FPS, and age of the displayed result.
- Short corrective hints: `move object into box`, `move closer`, `hold still`,
  `improve lighting`, or `unsupported/uncertain object`.
- `q`/Esc quit and `r` reset temporal state. No frame saving by default.

### Optional controlled resistor route

Only after the component classifier stably accepts `resistor`, pass the same
tight ROI to a classical decoder:

```text
quality gate -> body mask/contour -> rotated rectangle -> deskewed body crop
-> robust 1-D band profile -> evaluate both directions and 4/5-band layouts
-> standards lookup -> unique visual nominal value + tolerance, or abstain
```

The MVP may support only a single axial four-band resistor on a neutral
background. Five/six-band, SMD, cement, wire-wound, printed, packaged,
occluded, multiple, or poorly localized resistors must return `unsupported` or
`ambiguous`. `nominal_ohms` remains null unless one standards-valid decoding is
uniquely accepted.

## Confidence and abstention behavior

### Decision policy

Do not hard-code “80% softmax means confident.” Fit the policy on camera
validation after the model is frozen:

1. Temperature-scale logits.
2. Reject failed quality/OOD gates before considering class confidence.
3. Require calibrated top probability `p1 >= tau_p` and margin
   `p1 - p2 >= tau_margin`.
4. Require temporal agreement, initially tested as at least 4 of the latest 5
   valid inferences agreeing on the same class. Reset the window after an ROI
   change, long frame gap, quality rejection, or user reset.
5. Select `tau_p` and `tau_margin` to maximize validation coverage subject to
   the pre-registered accepted-error limit; then freeze and test once.

The 4-of-5 rule is an initial validation candidate, not a claimed optimal
threshold.

### User-visible states

| State | Required behavior |
|---|---|
| `NO_CAMERA` / `FRAME_ERROR` | No prediction; actionable error; clean exit or bounded retry. |
| `LOW_QUALITY` | No class/value; state the failed check (blur, dark, clipped, too small, unstable). |
| `OUT_OF_SCOPE` | No forced 20-class label; explain unsupported/empty/multiple object. |
| `UNCERTAIN` | Show up to two candidates only as suggestions; do not call either a detection. |
| `STABILIZING` | Show progress while temporal consensus accumulates; do not accept yet. |
| `ACCEPTED_CLASS` | Show calibrated class result and confidence; never imply a value measurement. |
| `AMBIGUOUS_RESISTOR` / `UNSUPPORTED_RESISTOR` | Keep `nominal_ohms=null`; show why and how to recapture. |
| `ACCEPTED_VISUAL_NOMINAL` | Show ordered bands, nominal Ω, tolerance, specification interval, and `visual estimate—not meter reading`. |

Do not leave a stale accepted label looking current after an object leaves the
ROI. Expire or visibly gray it after the configured age and reset on sustained
quality/OOD failure.

## Camera-domain demo acceptance criteria

Thresholds and the exact supported label set must be registered before the
camera holdout is opened. “Looks good on a webcam” is not acceptance.

### Dataset and protocol gates

1. Define supported classes and object/ROI rules. If all 20 classes are claimed,
   every class is evaluated; otherwise the GUI must label the demo as a scoped
   subset.
2. Collect target-camera clips across independent physical items, at least
   three capture sessions/days, intended backgrounds, two lighting levels,
   supported distances, rotations, and natural hand placement.
3. Group all frames by physical item and capture session. No item/session may
   cross training, calibration/threshold-validation, and final holdout.
4. Include negative/OOD clips: empty ROI, hands, multiple components, printed
   photos/screens, unsupported electronics, severe blur/glare/occlusion, and
   objects leaving the ROI.
5. Use clips only to prevent adjacent frames being counted as independent
   successes. Report both per-clip and per-physical-item results with bootstrap
   confidence intervals.
6. Freeze bundle hashes, preprocessing, calibrator, thresholds, temporal policy,
   camera settings, and acceptance criteria before one final holdout run.

### Quantitative gates for a guided-ROI component demo

| Criterion | Pass condition on untouched camera holdout |
|---|---|
| Accepted prediction quality | Selective top-1 accuracy **≥95% overall** and macro precision **≥90%** across supported classes. |
| Per-class floor | Accepted precision **≥80% for every claimed class**; publish support and interval. |
| Useful coverage | **≥70%** of in-scope, quality-compliant clips reach `ACCEPTED_CLASS`; lower coverage may pass only with an explicitly narrower demo claim. |
| OOD safety | False acceptance on negative/OOD clips **≤5%** overall and no negative category above **10%**. |
| Calibration | Camera-holdout ECE **≤0.05** as a supporting metric; the selective risk/coverage result remains primary. |
| Stability | Once accepted on a stationary valid object, label changes in **≤2%** of subsequent valid frames; no accepted label persists more than **0.5 s** after sustained object removal/invalid quality. |
| Responsiveness | Median time to a stable decision **≤1.0 s**, p95 **≤2.0 s** for valid in-scope clips. |
| Performance | On the named demo machine: display **≥15 FPS**, inference throughput **≥5 FPS**, p95 inference latency **≤200 ms**, and no UI stall over 1 s in a 15-minute run. |
| Robustness | No crash or resource leak in the 15-minute run; camera disconnect/read failure produces a clear state and cleanup. |
| Reproducibility | A second clean environment reproduces class mapping/self-test exactly and holdout aggregate metrics within the predeclared tolerance. |

These are promotion gates for a controlled demonstration, not a claim of broad
20-class real-world recognition. Confidence intervals and sample support must be
shown; a tiny holdout cannot establish the percentages above.

### Additional gates for resistor value display

| Criterion | Pass condition on physically/session-disjoint supported resistors |
|---|---|
| Localization/quality | Every displayed value has a valid body crop and ordered band overlay; localization failures abstain. |
| Accepted exact code | **≥98%** exact ordered-band-code accuracy among accepted decodes. |
| Accepted nominal value | **≥98%** exact nominal-value accuracy and **≤2%** confidently wrong values. A wrong order of magnitude is an automatic fail regardless of aggregate score. |
| Coverage | **≥70%** of quality-compliant, supported four-band clips yield a value; all others return an explicit non-value state. |
| Direction/tolerance | **≥99%** direction correctness and **≥95%** tolerance-band accuracy among accepted decodes. |
| Repeatability | Same physical resistor across supported sessions/lighting returns the same accepted code in **≥95%** of accepted clips. |
| Semantics | UI always says `visual nominal`, shows tolerance/interval, and never labels the output a measured resistance. |

## Completion order

1. Freeze a canonical data/artifact root and fingerprint vocabulary.
2. Assemble the final notebook in archived-load mode with all model sections and
   provenance assertions; do not retrain merely to fill presentation cells.
3. Define the scoped camera ontology and collect grouped camera validation/OOD
   data plus a sealed holdout.
4. Build the one-window guided-ROI classifier demo and measure ConvNeXt-Base
   versus MobileNet-V3-Large on the target machine.
5. Calibrate, select abstention/temporal thresholds, freeze a single promotion
   bundle, and run the camera acceptance protocol once.
6. Add the controlled classical resistor decoder only if resistor value display
   is required; retain explicit abstention and validate it separately.

Until those gates pass, the correct status is **research-complete for archived
classification experiments; final notebook and live camera demo not yet
accepted**.
