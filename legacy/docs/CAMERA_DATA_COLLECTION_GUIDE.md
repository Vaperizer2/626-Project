# Camera data collection guide

This guide defines a new camera-native dataset for the repository's existing
20-class component classifier and the planned camera/OpenCV path. It does not
modify or extend the legacy Kaggle split. That split contains cross-partition
duplicates and its test labels have already been inspected, so it remains a
historical audit set only.

The new dataset is single-label image classification data. A physical object,
capture session, acquisition source, exact duplicate, or accepted near-duplicate
cluster must never cross a train, validation, or target-holdout boundary.

## 1. Canonical labels and labeling rule

Use these exact lowercase labels:

```text
battery holder
breadboard
capacitor
connector
crystal oscillator
diode
fuse
heat sink
ic chip
inductor coil
jumper wire
led
microcontroller board
potentiometer
power supply module
relay
resistor
switch
transformer
transistor
```

Label the intended, dominant object. A complete assembly takes precedence over
its visible parts: for example, a framed microcontroller board remains
`microcontroller board`, not `ic chip`. A tight crop of a chip removed from the
board is `ic chip`. If two different mapped classes are equally plausible,
mark the image `ambiguous` and exclude it from supervised splits; do not force a
label.

`resistor` includes axial, SMD, and power resistors for primary classification.
Only images marked `subtype=axial_banded` and passing the stricter band-visibility
checks may enter a future resistor-band decoder dataset. A visually decoded
nominal value is not an electrical measurement.

## 2. Folder and filename conventions

Keep the canonical camera dataset separate from `data/raw/` until a new
manifest-aware loader exists. Do not rely on class folders as ground truth and
do not run the repository's current image-level splitter over these frames.

Recommended logical layout (the sealed holdout should live in restricted
storage, outside the ordinary development checkout):

```text
camera_data/
  v001/
    incoming/
      <source_id>/<session_id>/<object_id>/
    images/
      <source_id>/<session_id>/<object_id>/<sample_id>.jpg
    quarantine/
    metadata/
      captures.csv
      label_reviews.csv
    manifests/
      train.csv
      validation.csv
      dataset_seal.json

restricted_target_holdout/
  v001/
    images/<source_id>/<session_id>/<object_id>/<sample_id>.jpg
    metadata/captures.csv
    metadata/label_reviews.csv
    manifest.csv
    holdout_seal.json
```

Identifiers are opaque, stable, lowercase ASCII tokens:

- `source_id`: acquisition stream, normally a site + camera + collector setup,
  such as `src03`. A material camera/site/operator change creates a new source.
- `session_id`: one staged capture run, such as `src03_s0012`.
- `object_id`: one physical specimen, such as `src03_o0048`. Keep the same ID
  across sessions and never reuse it for another specimen.
- `frame_id`: monotonically increasing within a session/object, such as `f0007`.
- `sample_id`: `<session_id>_<object_suffix>_<frame_id>`, for example
  `src03_s0012_o0048_f0007`.

The filename is `<sample_id>.<ext>`. Do not put the class or split in the
filename: labels can be corrected, and holdout labels must be blind. Prefer
camera-original JPEG at quality 90 or better, or PNG when the camera supplies
it. Do not repeatedly recompress files. Never rename an accepted image; replace
it with a new sample ID and retain the old row as excluded.

## 3. Metadata CSV schema

Use UTF-8, comma-separated CSV with one row per image, an ISO-8601 UTC timestamp,
period as decimal separator, blank for unknown optional values, and no formulas.
The canonical header is:

```csv
dataset_version,sample_id,relative_path,sha256,source_id,session_id,object_id,visible_object_ids,frame_id,captured_at_utc,camera_id,camera_model,width_px,height_px,codec,exposure_us,iso,white_balance_kelvin,focus_mode,azimuth_deg,elevation_deg,roll_deg,distance_cm,frame_fill_pct,light_setup,illuminance_lux,background_id,background_type,occlusion_pct,occlusion_bin,target_instance_count,other_class_present,proposed_label,final_label,subtype,split,duplicate_cluster_id,quality_status,exclude_reason,review_status,license_status,notes
```

### Required fields

| Field | Rule |
|---|---|
| `dataset_version` | Immutable release ID, initially `camera_v001`. |
| `sample_id`, `relative_path` | Unique and stable; path is relative, never machine-specific. |
| `sha256` | Hash of original file bytes after ingestion. |
| `source_id`, `session_id`, `object_id`, `frame_id` | Required grouping keys. |
| `visible_object_ids` | Pipe-separated physical IDs visible in the frame; normally the primary `object_id` only. |
| `captured_at_utc` | ISO-8601 UTC, for example `2026-08-16T08:45:12Z`. |
| `camera_id`, `camera_model` | Stable device/profile ID and human-readable model. |
| `width_px`, `height_px`, `codec` | Values verified after decoding, not copied from the extension. |
| `azimuth_deg`, `elevation_deg`, `roll_deg` | Approximate camera pose relative to the object's marked canonical front. |
| `distance_cm`, `frame_fill_pct` | Lens-to-object distance and approximate percent of shorter frame dimension occupied by the target. |
| `light_setup` | One of `diffuse_neutral`, `diffuse_warm`, `diffuse_cool`, `side`, `backlit`, or `low`. |
| `background_id`, `background_type` | Reusable background ID and one of `neutral_light`, `neutral_dark`, `bench`, or `clutter`. |
| `occlusion_pct`, `occlusion_bin` | Estimated target area hidden; bin is `none`, `mild`, or `moderate`. |
| `target_instance_count` | Count of intended-class instances in the frame. |
| `other_class_present` | `true` when another mapped class is visibly present. |
| `proposed_label`, `final_label` | Proposed collector label and adjudicated canonical label. `final_label` stays blank until review finishes. |
| `split` | Blank during collection, then exactly `train`, `validation`, or `target_holdout`. |
| `duplicate_cluster_id` | Shared by exact and accepted near-duplicate variants; a singleton still gets an ID. |
| `quality_status` | `pending`, `accepted`, `quarantined`, or `excluded`. |
| `review_status` | `pending`, `agreed`, `adjudicated`, or `rejected`. |
| `license_status` | Normally `project_owned`; otherwise record explicit permission before inclusion. |

Camera controls (`exposure_us`, `iso`, `white_balance_kelvin`, `focus_mode`) and
`illuminance_lux` may be blank only when the device cannot report them. Record
the actual setting rather than the intended preset. `subtype` is optional but
strongly recommended; examples include `electrolytic`, `blade`, `smd`, and
`axial_banded`.

Use an append-only `label_reviews.csv` so independent decisions are not
overwritten:

```csv
sample_id,review_round,reviewer_id,reviewed_at_utc,decision,proposed_final_label,reason_code,comment
```

`decision` is `accept`, `relabel`, `ambiguous`, or `exclude`. Reviewer IDs must
be pseudonymous. Model predictions, confidence, and loss do not belong in this
file and must not be visible during ground-truth review.

## 4. Minimum release counts

Counts below are accepted images after review, corrupt-file removal, and
duplicate clustering. Capture at least 15% extra to absorb exclusions.

| Partition | Exclusive sources | Physical objects per class | Sessions per object | Accepted frames per session | Minimum images per class |
|---|---:|---:|---:|---:|---:|
| Train | at least 3 | 15 | 2 | 12 | 360 |
| Validation | at least 1 new source | 5 | 2 | 12 | 120 |
| Sealed target holdout | at least 1 new target source | 5 | 2 | 12 | 120 |
| **Total** | **at least 5** | **25** | **2** | **12** | **600** |

This is a minimum of 12,000 accepted images across 20 classes. Prefer two
validation sources and more physical objects rather than more burst frames.
If a class misses any object, session, source, or image minimum, label the
release a pilot; do not present its holdout metric as the final target-domain
result.

Each session must re-stage the object and change at least one meaningful
condition (day/time, camera position, lighting, or background). Restarting the
camera without re-staging does not create an independent session. Frames from a
video burst count as one view unless pose or conditions materially change.

## 5. Capture protocol

### Before each session

1. Clean the lens; set the native resolution to at least 1280×720; disable
   beauty filters, portrait blur, overlays, and watermarks.
2. Synchronize the clock to UTC and create the source, session, and object IDs.
3. Fix or record focus, exposure, white balance, resolution, and zoom. Avoid
   silent auto-mode changes within a condition block.
4. Capture a gray/color reference and a focus target once for the session, as
   calibration records rather than labeled component frames.
5. Mark a canonical object front so angle metadata is repeatable. Randomize the
   class capture order to avoid class being confounded with time or lighting.

### Required 12-frame session matrix

One image may satisfy several rows. Do not take 12 near-identical burst frames.

| Dimension | Per-session requirement |
|---|---|
| Angle | At least six distinct views: canonical front, rear, left, right, overhead, and one low/oblique view. Record approximate azimuth/elevation/roll. Include the underside when it is a normal visible feature. |
| Distance/scale | Four near frames (target fills 70–85%), four medium (40–60%), and four far (20–30%). Keep the whole target in frame unless the intended deployment includes partial crops. |
| Light | Eight frames under the session's primary diffuse light and four under a challenge setup. Across an object's two sessions, use at least two light setups. Avoid clipped highlights on glossy packages and resistor bands. |
| Background | At least two backgrounds per session. Across the class, cover neutral light, neutral dark, realistic bench, and controlled clutter. Do not use class names, packaging, or datasheets as backgrounds. |
| Occlusion | Seven `none` (0%), three `mild` (1–15%), and two `moderate` (16–35%). Occlude different regions with neutral, deployment-realistic objects. |
| Instances | Nine single-target frames, two frames with 2–3 same-class targets, and one frame with a non-target distractor when the intended label remains dominant. |

For multi-object frames, record every known specimen in `visible_object_ids`.
All connected specimens must remain in the same split. Do not mix holdout and
development objects in one photograph.

### Quality gates during capture

- Keep the shorter image side at 720 px or more and the target's longest extent
  at 160 px or more in the original frame.
- Retake accidental motion blur, missed focus, severe compression, clipped
  highlights, crushed shadows, or an unintentionally cropped object.
- Keep exposure and white balance varied only as the protocol calls for; do not
  repair challenge images into clean images with post-processing.
- Save the original frame. Derived crops may be generated later and must point
  back to the original `sample_id` and remain in the same split.
- For axial-banded resistors, include clean views where the entire body, both
  ends, and every band are visible. Record `subtype=axial_banded`; band count,
  ordered colors, tolerance, body geometry, and meter readings belong in a
  separate task-specific annotation, not in the 20-class label.

## 6. Inclusion and exclusion rules

### Include only when all are true

- The file decodes successfully and its hash, dimensions, and capture metadata
  are recorded.
- Two humans can identify the dominant target under the class policy without
  using the folder name, filename, or model prediction.
- The target is large and sharp enough to show class-defining shape/features.
- Occlusion is 35% or less and does not hide the features needed to distinguish
  the class.
- Any additional class is subordinate to the intended target. Components
  naturally mounted on a board/module do not change the assembly label.
- The image is a genuine camera capture owned or explicitly licensed for the
  project.

### Exclude or quarantine when any are true

- Corrupt/unreadable file; accidental blur; severe under/overexposure; target
  too small; defining features hidden; or more than 35% occlusion.
- Wrong class, unresolved class ambiguity, or two different classes with no
  clearly dominant target.
- Screenshot, diagram, datasheet, collage, advertisement, rendered/synthetic
  image, packaging-only image, or visible class-name text that gives away the
  label.
- Identifiable people, private information, unlicensed content, or device
  watermarks/overlays.
- Exact duplicate or effectively identical burst frame. Keep the best original
  and cluster all legitimate alternate crops/recompressions together.
- Post-capture edits beyond orientation normalization and lossless metadata
  handling. Augmentations belong in training, never in the canonical dataset.

Use these `exclude_reason`/review reason codes: `corrupt`, `blur`, `exposure`,
`too_small`, `occluded`, `wrong_class`, `ambiguous`, `multi_class_no_primary`,
`out_of_scope`, `duplicate`, `privacy`, `license`, and `synthetic_or_edited`.
Never delete a reviewed row; mark it excluded and move the file to quarantine.

## 7. All-class collection checklist

Every row must reach **360 train / 120 validation / 120 sealed-holdout accepted
images** and the object/session/source minimums above. The variety notes are
coverage requirements, not new labels.

| Done | Exact label | Required variety and boundary check |
|---|---|---|
| [ ] | `battery holder` | Multiple cell formats and contact styles; loaded and unloaded; exclude a loose battery without a holder. |
| [ ] | `breadboard` | Mini/full-size, colors, bare and sparsely wired; the board must remain dominant. |
| [ ] | `capacitor` | Electrolytic, ceramic, film, and SMD packages; do not use a full board as a capacitor image. |
| [ ] | `connector` | Headers, terminals, USB/coax and board connectors; connector body must be the intended object. |
| [ ] | `crystal oscillator` | Metal-can crystals, oscillator modules, and package sizes; record subtype to expose ontology ambiguity. |
| [ ] | `diode` | Axial, rectifier, signal, and SMD forms; LED packages belong to `led`. |
| [ ] | `fuse` | Glass, ceramic, blade, resettable, and SMD forms; distinguish fuse holders and resistor-like axial bodies. |
| [ ] | `heat sink` | Extruded, finned, stamped, and small package sinks; with/without an attached device while the sink dominates. |
| [ ] | `ic chip` | DIP, SOIC, QFP/QFN and similar packages; exclude an entire microcontroller board. |
| [ ] | `inductor coil` | Radial, SMD, toroidal, and air-core forms; distinguish coupled windings/transformers. |
| [ ] | `jumper wire` | Single/bundled and male/female combinations; exclude generic power/data cables. |
| [ ] | `led` | Through-hole, SMD, and simple arrays; the emitter must dominate, not a complete module/board. |
| [ ] | `microcontroller board` | Multiple board families, sizes, and connector layouts; capture the whole assembly. |
| [ ] | `potentiometer` | Rotary, trimmer, and slider forms; distinguish ordinary switches and fixed resistors. |
| [ ] | `power supply module` | DC-DC, regulator, and AC-DC modules; exclude a standalone transformer or complete unrelated board. |
| [ ] | `relay` | Electromechanical and solid-state forms, varied packages and contact layouts; record subtype. |
| [ ] | `resistor` | Axial banded, SMD, and power forms; flag axial-banded images separately for the planned decoder. |
| [ ] | `switch` | Toggle, pushbutton, slide, rocker, rotary, and DIP forms; distinguish potentiometers. |
| [ ] | `transformer` | Mains, pulse, ferrite, and toroidal forms; distinguish single-winding inductors. |
| [ ] | `transistor` | BJT/MOSFET-style discrete packages including TO and SMD families; exclude multi-pin ICs and full boards. |

Prioritize extra physical specimens—not repeated frames—for the repository's
historically weak/confused classes: `fuse`, `resistor`, `diode`, `transistor`,
and `relay`.

## 8. Source-, session-, and object-disjoint splits

Split only after technical QC, label adjudication, and duplicate clustering.
Use a saved manifest, never an image-level random split.

1. Assign each acquisition source wholly to one partition. A source ID may not
   appear in another partition.
2. Assign each physical `object_id` wholly to one partition, including all of
   its sessions, frames, crops, and recompressions.
3. Assign each `session_id` wholly to its object's partition.
4. Build exact SHA-256 groups and conservative near-duplicate/source-family
   clusters. Treat connected clusters as indivisible, even when this changes
   the nominal 60/20/20 image ratio.
5. Allocate groups to meet the per-class object and source counts, targeting
   60% train, 20% validation, and 20% sealed holdout by physical object.
6. Before freezing, assert zero overlap between partitions for `source_id`,
   `session_id`, `object_id`, `sample_id`, `sha256`, and
   `duplicate_cluster_id`; resolve contradictory labels within a cluster.
7. Save relative paths, final labels, group IDs, hashes, per-class counts, and a
   hash of each manifest. A changed membership creates a new dataset version;
   never overwrite a released manifest.

Validation is for model, augmentation, threshold, abstention, and error-analysis
decisions. It may be inspected, but any label/membership correction creates a
new validation manifest/version. The sealed target holdout is not validation.

## 9. Sealed target holdout procedure

1. **Pre-register the target.** Before collection, write down the intended
   camera/site/operator conditions, 20 labels, per-class minimum, quality rules,
   metrics, abstention treatment, and the maximum number of final evaluations.
2. **Use new provenance.** The holdout uses at least one target-domain source,
   five new physical specimens per class, and two newly staged sessions per
   specimen. None may have appeared in train, validation, the legacy web set,
   demos, calibration tuning, or exploratory captures.
3. **Separate custody.** A holdout custodian who is not choosing models stores
   the images, labels, and review log in access-restricted storage. Developers
   receive only aggregate class/count and capture-contract summaries.
4. **Review before sealing.** Two independent reviewers label every image with
   predictions hidden; an adjudicator resolves disagreements. Perform QC and
   duplicate checks before, not after, the seal.
5. **Seal immutably.** Freeze images and manifest, calculate every file SHA-256
   plus the manifest SHA-256, and record dataset version, counts, timestamp,
   custodian, review completion, and hashes in `holdout_seal.json`. Make the
   directory read-only and retain an offline copy.
6. **Freeze the candidate.** Before access, record the checkpoint hash, source
   commit, environment/lockfile hash, preprocessing, class order, thresholds,
   abstention rule, and metric script hash. No manual checking of holdout frames
   is allowed during the run.
7. **Run once.** The custodian runs the frozen pipeline and returns the
   pre-registered aggregate and per-class metrics plus a signed evaluation
   record. Do not reveal per-image labels, predictions, or a confusion-driven
   sample gallery until the result is formally accepted or the holdout is
   retired.
8. **Retire after feedback.** Once images, labels, or error patterns influence
   development, rename the set as an audit set. A subsequent headline result
   requires a newly collected and sealed holdout version. Do not silently drop
   hard or mislabeled holdout rows after seeing model output; document a defect,
   retire the version, correct the policy, and reseal a replacement.

## 10. Label review workflow

1. **Ingest/QC:** copy the original, decode it, compute SHA-256, verify metadata,
   detect exact/near duplicates, and quarantine technical failures.
2. **First review:** reviewer A sees the image without path/class hints and
   records `accept`, `relabel`, `ambiguous`, or `exclude` plus a reason.
3. **Independent review:** reviewer B reviews 100% of validation and holdout,
   all duplicate/conflicting clusters, all weak/confusable classes, and at
   least a stratified random 20% of training. Reviewer B cannot see reviewer A's
   answer or any model output.
4. **Adjudication:** a third reviewer resolves disagreements using this guide.
   Review the whole duplicate/object cluster together. Update `final_label`,
   `review_status`, and `quality_status`; preserve both original review rows.
5. **Coverage audit:** verify every class's source/object/session counts and the
   angle, distance, light, background, occlusion, and instance quotas. Fill
   deficits with new specimens or sessions rather than copying frames.
6. **Split/leakage audit:** run all disjointness checks in section 8 and inspect
   every contradictory cluster before manifests are frozen.
7. **Release:** freeze the dataset and manifests, publish per-class counts and
   exclusion reasons, and record their hashes. Training corrections create a
   new version; validation/holdout changes require a new evaluation contract.

Model-ranked errors may seed a separate training-data review queue after the
first release, but they are not ground truth and must never drive holdout
filtering. Keep all decisions append-only and auditable.

## 11. Session close-out checklist

- [ ] All 20 class names use the canonical spelling.
- [ ] Every file has one metadata row and a unique sample ID.
- [ ] Object, session, and source IDs match the staged capture.
- [ ] Twelve non-burst views per object/session cover the required matrix.
- [ ] Calibration frames are separated from labeled component frames.
- [ ] Files decode; SHA-256, dimensions, and codec are recorded.
- [ ] Exclusions are quarantined with reason codes, not deleted.
- [ ] No development object/source appears in the sealed holdout.
- [ ] Independent label review is complete at the required rate.
- [ ] Manifest counts and all six split-overlap checks pass before sealing.
