# Agent-controlled post-camera-data experiment plan

## Decision

Run exactly six validation-only arms: three fixed ImageNet-pretrained backbones
(`resnet18`, `convnext_base`, and `swin_b`) under two data conditions. The first
condition trains on the cleaned legacy development pool; the second adds only
the camera-training partition. Both conditions use the same camera validation
partition. Evaluate the locked camera test partition once, for one
validation-selected checkpoint, after all six arms finish.

This is the smallest matrix that separates architecture choice from the value
of the new camera data. A three-run matrix without the web-only controls could
rank models, but could not show that camera training helped.

`swin_b` is the fixed third finalist. It was the strongest non-ConvNeXt model in
the grouped cross-fold report (OOF macro-F1 0.6066, fold standard deviation
0.0126), while ConvNeXt-Base was the best single model (0.6288). On the earlier
fixed split, ConvNeXt-Base also beat Swin-B (0.7407 versus 0.7234 validation
macro-F1); its strong-recipe checkpoint reached 0.7473 validation macro-F1 and
0.6969 legacy-test macro-F1. The historical test results are context only and
must not affect the post-camera selection.

## Data contract required before any run

Create and freeze one versioned camera release and two version-2 split
manifests:

- `data/splits/post_camera_web_only_v1.json`: cleaned legacy train+validation
  rows in `train`, target-camera selection rows in `val`, and the locked
  target-camera rows in `test`.
- `data/splits/post_camera_adapt_v1.json`: identical except that the fixed
  camera-training rows are added to `train`.

The two manifests must have byte-for-byte identical `val` and `test` members.
The legacy 523-image test set is an already-consumed audit set and appears in
neither manifest. Camera allocation is group-stratified 60/20/20 by class at
the connected-component level of `physical_item_id` and `capture_session_id`;
all frames from a burst/video and all views of one physical item stay in one
partition. If any class has fewer than five independent item/session groups,
collect more data rather than falling back to a frame-level split.

Before freezing the manifests:

1. Quarantine unreadable files, unknown/ambiguous labels, and conflicting
   exact-duplicate labels.
2. Remove exact SHA-256 duplicates and review perceptual-near-duplicate groups
   across legacy, camera, and any external sources. No exact or near duplicate
   may cross train/validation/test.
3. Record immutable file hashes, class counts, item counts, session counts,
   camera ID, capture date, and manifest SHA-256. Never overwrite an image in
   place after hashing it.
4. Keep camera-test labels operationally sealed until the final command. The
   current family runner constructs the test dataset but does not iterate or
   score it under `--defer-test`; access control, not just operator discipline,
   should protect the labels.
5. Verify that the composed dataset root used by notebook 01 contains every
   path in both manifests. The notebook's dataset/checkpoint fingerprint is
   based on label, filename, and file size rather than full file contents, so
   the separately recorded manifest and image hashes are mandatory provenance.

Do not start if the manifests differ in validation/test membership, a group
crosses partitions, a camera-test label has been used for a choice, or the
camera release lacks the required provenance fields.

## Fixed experiment matrix

| Arm | Training rows | Validation rows | Model | Registered learning rate |
|---|---|---|---|---:|
| W-R18 | cleaned legacy development only | camera validation | ResNet18 | 1e-3 |
| W-CNB | cleaned legacy development only | camera validation | ConvNeXt-Base | 3e-4 |
| W-SWB | cleaned legacy development only | camera validation | Swin-B | 4e-4 |
| C-R18 | same legacy rows + camera train | same camera validation | ResNet18 | 1e-3 |
| C-CNB | same legacy rows + camera train | same camera validation | ConvNeXt-Base | 3e-4 |
| C-SWB | same legacy rows + camera train | same camera validation | Swin-B | 4e-4 |

ResNet18 is the naive architecture baseline: the registered, unmodified
pretrained backbone, not the old `primary_resnet18.pt` checkpoint. It receives
the same recipe and data rules as the finalists so architecture comparisons
are not confounded by a weaker training recipe.

Everything except the registered per-backbone learning rate is fixed:

- seed 42, 224x224 RGB input, batch size 16, pretrained torchvision weights;
- `strong-regularized` transforms on training rows only: resize, horizontal
  flip, rotation up to 20 degrees, mild colour jitter, 5% translation,
  0.95-1.05 scale, ImageNet normalization, and random erasing at p=0.15;
- deterministic resize and ImageNet normalization for validation/test;
- two-view original-plus-horizontal-flip logit averaging at validation/test;
- AdamW, registered learning rate, five times registered weight decay,
  class-weight power 0.5, label smoothing 0.12, and two frozen-backbone epochs;
- maximum 15 epochs, early-stopping patience 4, minimum improvement 1e-4 in
  validation macro-F1, train/validation accuracy-gap guard 0.30;
- no Optuna, learning-rate search, recipe search, ensembles, camera-specific
  transform tuning, or per-model resampling.

## Commands using the existing runner

Run from `notebooks/`. Use new report names and artifact prefixes so historical
checkpoints cannot be loaded or overwritten.

Web-only control:

```bash
cd notebooks
COMPCLASS_WORK_DIR="$PWD" \
COMPCLASS_SPLIT_MANIFEST=post_camera_web_only_v1.json \
COMPCLASS_SEED=42 COMPCLASS_BATCH_SIZE=16 COMPCLASS_USE_PRETRAINED=1 \
../.venv/bin/python 12_model_family_benchmark.py \
  --models resnet18,convnext_base,swin_b \
  --epochs 15 --patience 4 --profile paper \
  --recipe strong-regularized --max-train-val-gap 0.30 --defer-test \
  --report post_camera_web_only_v1.json \
  --artifact-prefix post_camera_web_only_v1
```

Camera-adapted condition:

```bash
cd notebooks
COMPCLASS_WORK_DIR="$PWD" \
COMPCLASS_SPLIT_MANIFEST=post_camera_adapt_v1.json \
COMPCLASS_SEED=42 COMPCLASS_BATCH_SIZE=16 COMPCLASS_USE_PRETRAINED=1 \
../.venv/bin/python 12_model_family_benchmark.py \
  --models resnet18,convnext_base,swin_b \
  --epochs 15 --patience 4 --profile paper \
  --recipe strong-regularized --max-train-val-gap 0.30 --defer-test \
  --report post_camera_adapt_v1.json \
  --artifact-prefix post_camera_adapt_v1
```

After selection, run exactly one of the following shapes, substituting the
winning model and its already-used manifest/prefix. All training arguments must
remain identical so the runner loads the matching completed checkpoint instead
of fitting a new one.

```bash
cd notebooks
POSTCAM_MODEL=convnext_base
POSTCAM_MANIFEST=post_camera_adapt_v1
POSTCAM_PREFIX=post_camera_adapt_v1
COMPCLASS_WORK_DIR="$PWD" \
COMPCLASS_SPLIT_MANIFEST="$POSTCAM_MANIFEST.json" \
COMPCLASS_SEED=42 COMPCLASS_BATCH_SIZE=16 COMPCLASS_USE_PRETRAINED=1 \
../.venv/bin/python 12_model_family_benchmark.py \
  --models "$POSTCAM_MODEL" \
  --epochs 15 --patience 4 --profile paper \
  --recipe strong-regularized --max-train-val-gap 0.30 \
  --report post_camera_final_test_v1.json \
  --artifact-prefix "$POSTCAM_PREFIX"
```

Before allowing that command to score test, confirm from its console output
that it says `Loaded completed ...`; abort if it begins an epoch.

## Metrics and checkpoint selection

For every arm, retain the runner's best-epoch checkpoint and report:

- primary: camera-validation macro-F1;
- secondary: accuracy and weighted-F1;
- diagnostics: best epoch, train accuracy at that epoch, train/validation
  accuracy gap, parameter count, training time, per-class support/recall, and
  confusion matrix;
- adaptation effect: camera-adapted minus web-only validation macro-F1,
  accuracy, and per-class recall for each backbone;
- practical tie-breaker, measured only for otherwise eligible finalists:
  batch-1 p50/p95 latency and peak memory on the intended deployment device.

The existing JSON contains the first three headline metrics, best epoch, gap,
parameter count, and history. Produce per-class diagnostics without changing
the selection rule and never use them to tune a second round.

Selection is mechanical:

1. Exclude any arm with a train/validation accuracy gap above 0.30. If every
   arm is excluded, stop with “no promotable checkpoint”; do not pick the least
   bad arm.
2. Among eligible arms, select the highest camera-validation macro-F1.
3. Treat models within 0.01 absolute macro-F1 as tied; choose the one with lower
   batch-1 p95 latency, then fewer parameters. This makes ResNet18 a real
   practical baseline rather than a ceremonial run.
4. Do not require the adapted arm to win. A web-only arm may be selected if
   camera training provides no validation benefit.
5. Evaluate only the selected best-epoch checkpoint once on camera test. Report
   test accuracy, macro-F1, weighted-F1, per-class support/recall, confusion
   matrix, and the train/test accuracy gap. Do not change the winner afterward.

## ElectroCom61 implication

ElectroCom61 is excluded from all six primary arms. Its reported addition hurt
ConvNeXt, so it cannot be treated as generic augmentation or mixed into the
camera arm without confounding the question. The likely possibilities are
source/label mismatch, class-ratio distortion, crop artifacts, or negative
transfer; more volume is not evidence of more target-domain information.

The current cross-fold implementation appends selected external rows to every
fold's training loader, but its SHA-256 grouping applies only to the legacy
development rows. It does not itself prevent an external exact/near duplicate
from matching a fold-validation image. Therefore any later ElectroCom61
retest must first deduplicate it against all camera and legacy validation/test
groups, must use a separate artifact/report prefix, and must be a single
pre-registered ConvNeXt-Base data ablation. It is not part of this plan and is
not triggered merely because the primary matrix misses a target.

## Stopping rules

- Complete all six validation arms unless a manifest/leakage invariant fails,
  pretrained weights are unavailable, or an arm repeatedly crashes for a
  reproducible technical reason. Do not prune a model because an early epoch
  looks weak.
- Per arm, stop at 15 epochs or after four consecutive epochs without at least
  1e-4 validation macro-F1 improvement; select the best epoch, not the last.
- Stop the experiment after the one locked camera-test evaluation. No seed
  reruns, hyperparameter rescue, ElectroCom61 retry, ensemble, or test-informed
  threshold/transform change belongs to this experiment version.
- If the winner fails the gap guard on the final train/test comparison, or the
  camera-test macro-F1 is more than 0.05 below its camera-validation macro-F1,
  record a failed promotion and return to data/provenance review under a new
  pre-registration. Do not reopen the current camera test for tuning.
