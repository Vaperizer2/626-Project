# Dataset v2 audit and release contract

**Audit date:** 2026-08-16  
**Scope:** local repository only; read-only inspection and CPU hashing/metric recomputation. No training was run, CUDA was not used, and no compute rig or network resource was accessed.

## Decision

Do not run another architecture or augmentation sweep on the current 70/15/15 split. Preserve it as `legacy_raw_v1` for historical diagnostics, but do not call its test partition unseen. Dataset v2 should be built from one canonical source tree, adjudicated and clustered before splitting, with a grouped development split and a separately collected, sealed camera holdout.

The blocking fact is content leakage, not ordinary class imbalance: 243 of 1,045 legacy validation/test rows have an exact training copy. Exact-hash grouping has already produced materially more credible out-of-fold (OOF) development results, but it does not address resized/cropped copies, source families, physical items, or capture sessions.

## Current evidence

### Manifest and content snapshot

The following was recomputed from the current `notebooks/data/splits/split_manifest_raw.json`, resolving every row against `notebooks/` and hashing file bytes with SHA-256:

| Check | Current result |
|---|---:|
| Manifest version / recorded dataset fingerprint | 2 / `78637c8898aa10f8` |
| Manifest SHA-256 | `46d022f8d61cc7fed9419759c59256c8deb653f4ee868c2fe32f1c9c28d3e221` |
| Rows / classes | 3,483 / 20 |
| Train / validation / test | 2,438 / 522 / 523 |
| Missing manifest files | 0 |
| Unique byte SHA-256 values | 2,903 |
| Exact duplicate groups / rows in them | 527 / 1,107 |
| Redundant rows | 580 |
| Exact duplicate groups crossing splits | 243 |
| Validation rows with an exact train copy | 128 / 522 (24.5%) |
| Test rows with an exact train copy | 115 / 523 (22.0%) |
| Exact duplicate groups with conflicting labels | 12 |

The two notebook manifests, `split_manifest.json` and `split_manifest_raw.json`, are byte-identical. The root `data/splits/split_manifest.json` is a different, older absolute-path artifact (SHA-256 `ff316978aa19e4fb547ff07e65c19cd0bf2d997d461256d2af5290a90f35a846`). The two full raw-data trees are separate copies, although their visible files match by relative path; their source ZIPs both hash to `431426d2543396904b09e9cc09c30aa58b9c3cd0367aaadb2ba86d8bc822d459`. The notebook tree also contains six hidden checkpoint JPEGs. The effective data and output roots still depend on the working directory.

Other current evidence:

- Accepted class totals range from 114 (`connector`) to 251 (`power supply module`), only 2.20x imbalance. The legacy test has just 17-38 rows per class, so its class estimates are noisy even before leakage is considered.
- The current suffix allowlist silently omits 47 non-hidden, Pillow-readable files: 23 `.gif`, 16 `.img`, 4 `.image`, 3 `.view`, and 1 `.php`. Actual codecs do not reliably match suffixes; even accepted rows include JPEG, WebP, PNG, AVIF, and MPO content.
- `reviewed_exclusions_v1.json` records 30 obvious contaminants from a model-ranked 120-row review queue: 4 train, 13 validation, and 13 test. This 25% yield shows that review is valuable, but it is not a dataset-wide noise estimate.
- The present curation record supports exclusion only. It has no relabel/ambiguous/out-of-scope decision, duplicate cluster, source, license, physical item, capture session, independent reviewer, or adjudication fields.
- The current notebook fingerprint hashes label, basename, and file size; a same-size replacement can evade it. Its split fingerprint includes absolute paths and therefore changes across machines. A stale manifest can be overwritten automatically instead of failing closed.
- The optional external-data path in `notebooks/16_cross_fold_augmentation_benchmark.py` filters declared source/split/status/label, but it does not validate `source_sha256` or check external content/groups against development or holdout content.

### Evaluation status

The legacy test is not an untouched holdout. The experiment log records multiple test evaluations, a per-row error audit, contact-sheet review, and a post-hoc curated-test diagnostic. The August 14 cross-fold run did not load the test partition, but that statement is scoped to that run and does not restore historical blindness.

The log also has a stale summary: decision 2 still names ResNet50 at 0.6845 as the raw-test leader, while later entries and decision 10 record the strong ConvNeXt-Base result at 0.7132. This does not change the dataset conclusion, but v2 reports should be generated from machine-readable results rather than manually duplicated leader text.

The completed grouped run supplies useful development evidence:

- Development pool: legacy train + validation, 2,960 rows.
- Five `StratifiedGroupKFold` folds grouped by exact byte SHA-256: 2,527 groups, including 406 duplicate groups and 10 conflicting-label groups.
- Best single model: ConvNeXt-Base, 0.6288 OOF macro-F1.
- Best aggregation: 14-model majority vote, 0.6472 OOF macro-F1 and 0.6574 accuracy.
- Grouping assertions prevent exact copies from crossing a fold, but no current code prevents perceptual/source/item/session leakage.

These OOF values should not be directly subtracted from the older single-split scores as a clean estimate of leakage cost: the training/evaluation procedures differ. They do show that exact-content grouping is practical and must become a dataset invariant.

## Prioritized cleanup queue

| Priority | Work item | Completion gate |
|---|---|---|
| P0 | Freeze the existing manifests and metrics as `legacy_raw_v1`; retire its test as a final holdout. | New code cannot select or claim a final result from the legacy test; historical reports remain reproducible. |
| P0 | Select one canonical root: future v2 paths are repository-relative under `data/v2/`; `notebooks/data/` is never another source of truth. | Running from root or `notebooks/` resolves the same manifest and yields the same full fingerprints. |
| P0 | Build the v2 occurrence manifest and compute byte/pixel hashes for every candidate. | All rows pass the schema, decode, path, digest, and count checks below. |
| P0 | Adjudicate the 12 exact-content cross-label groups first, then apply one decision to the whole duplicate cluster. | No accepted duplicate cluster contains conflicting canonical labels. |
| P0 | Collapse exact duplicates and approved near/source/item links into indivisible split groups before assigning a split. | Zero path, byte hash, pixel hash, duplicate-cluster, or split-group overlap. |
| P0 | Close the external-manifest hole before any external augmentation is used. | Every external row is rehashed; collisions and linked groups are checked against train, validation, and sealed holdout. |
| P1 | Write the 20-class labeling policy, including dominant-object rules and explicit `ambiguous`, `multi_object`, and `out_of_scope` outcomes. | Two reviewers can independently reach a recordable decision without seeing model predictions. |
| P1 | Convert the 120 high-loss rows, cross-model disagreements, low-confidence rows, and all members of flagged clusters into an adjudication queue. Add a random per-class sample. | Decisions are cluster-aware, versioned, and include keep/relabel/exclude/ambiguous plus reason and reviewer fields. |
| P1 | Quarantine and review the 47 readable files excluded only by suffix. | Each is explicitly accepted and normalized or excluded with a reason; none is silently ignored. |
| P1 | Generate near-duplicate candidates and recover source/product provenance where possible. | Every candidate edge is accepted/rejected; unresolved candidates block split release. |
| P2 | Collect camera data under the handoff below, reserving whole acquisition batches for the sealed holdout before model inspection. | Camera/item/session provenance is complete and all holdout groups are disjoint from development. |
| P2 | Re-run one fixed baseline on v2 train/validation, then perform one preregistered holdout evaluation. | Dataset checks pass before training; the final evaluator alone opens holdout labels. |

## Exact dataset v2 contract

### 1. Artifacts and authority

Dataset v2 consists of five immutable, machine-readable artifacts:

1. `data/v2/manifest.jsonl` - one row per source occurrence or camera capture.
2. `data/v2/duplicate_edges.jsonl` - reviewed near-duplicate/source-identity edges and decisions.
3. `data/v2/splits.json` - saved group-to-split assignments and generator metadata.
4. `data/v2/audit.json` - counts, fingerprints, leakage checks, class coverage, and all warnings/failures.
5. `holdout_labels.jsonl` - a sealed evaluator-only mapping of holdout `sample_id` to adjudicated label and review fields; it is not present in the ordinary training checkout.

The repository root is located independently of CWD (for example by walking to `pyproject.toml`). All stored paths are POSIX paths relative to that root. Absolute paths, `..`, symlinks escaping the root, and duplicate roots are forbidden. An artifact is never rewritten in place after release; any content, label, provenance, grouping, or split change creates a new semantic version.

### 2. Manifest row

Every JSONL row has these fields; `null` is allowed only where stated:

| Field | Rule |
|---|---|
| `dataset_version` | Exact release string, for example `2.0.0`. |
| `sample_id` | `sha256(source_id + "\0" + source_item_id + "\0" + capture_id)`, full lowercase hex. `capture_id` is empty for a non-camera source. |
| `asset_path` | Unique repository-relative POSIX path. |
| `byte_sha256` | SHA-256 of file bytes, full lowercase hex. |
| `pixel_sha256` | SHA-256 of the canonical decoded pixels defined below. |
| `byte_size`, `width`, `height`, `decoded_format`, `mode` | Recomputed from the asset, never trusted from an incoming manifest. |
| `source_type` | One of `kaggle_dataset`, `external_dataset`, `web`, `camera`. |
| `source_id` | Stable dataset release, URL, or collection-batch identifier. For the legacy set use the Kaggle dataset identifier plus downloaded archive SHA-256. |
| `source_item_id` | Provider record ID/URL, or the legacy source-relative filename when no better identifier survives. |
| `source_uri`, `license_id`, `acquired_at_utc` | Required strings; use explicit `unknown` for unrecoverable legacy values, never silent nulls. |
| `original_label` | Source folder/annotation, preserved exactly. |
| `canonical_label` | One of the frozen 20 labels, or `null` when excluded/ambiguous/out-of-scope or while holdout labels are sealed. |
| `decision` | One of `keep`, `relabel`, `exclude`, `ambiguous`, `out_of_scope`, `pending`. |
| `reason_code` | Controlled value; required unless `decision=keep`. |
| `reviewer_ids`, `adjudicator_id`, `reviewed_at_utc` | Required for validation/holdout and for any non-`keep` decision. Model names/predictions are not reviewer IDs. |
| `physical_item_id`, `source_entity_id`, `capture_session_id`, `collection_batch_id`, `camera_id` | Conditional provenance. All five are required for camera rows except `source_entity_id`, which may be `unknown`; non-camera rows use `unknown` where recovery fails. |
| `duplicate_cluster_id`, `split_group_id`, `split` | Builder outputs under the rules below. `split` is `train`, `val`, `holdout`, or `excluded`. |
| `legacy_v1_split` | `train`, `val`, or `test` for migrated legacy rows; otherwise `null`. It is provenance only and never controls v2 membership. |

`sample_id` identifies an occurrence; multiple occurrences may intentionally share content hashes. JSON serialization used for fingerprints is UTF-8, sorted by `sample_id`, with sorted keys and separators `,` and `:`.

### 3. Canonical decoding and deduplication

For each asset:

1. Compute `byte_sha256` before decoding.
2. Decode with Pillow, apply `ImageOps.exif_transpose`, fully load the first frame, and convert to RGB. Animated/multi-frame input is rejected or explicitly normalized by a separately versioned rule; it is never silently reduced.
3. Compute `pixel_sha256` over `uint32_be(width) || uint32_be(height) || RGB_bytes_in_row_major_order`.
4. Create mandatory duplicate edges between rows sharing `byte_sha256` or `pixel_sha256`.
5. Generate review candidates with a versioned `dhash64-v1` implementation (EXIF-transposed RGB, grayscale 9x8 LANCZOS resize, row-major adjacent-pixel comparisons); Hamming distance <= 6 is a candidate, not an automatic merge. Add candidates from matching source/product IDs and manual review.
6. Store every reviewed edge as `{left_sample_id, right_sample_id, method, score, decision, reviewer_id}` where `decision` is `same_entity` or `different`.
7. `duplicate_cluster_id` is `dup:` plus the lexicographically smallest `sample_id` in the connected component formed by mandatory edges and reviewed `same_entity` edges. Singleton rows get their own cluster ID.

No duplicate row is selected by filename. Within an accepted same-label cluster, choose the representative deterministically by: accepted decision, largest pixel area, highest byte size, then smallest `sample_id`. Keep all occurrence rows for provenance, but only the representative is eligible for model input. Any duplicate cluster with more than one accepted canonical label fails release until adjudicated.

### 4. Group contract

`split_group_id` is computed as the connected component over eligible representatives with an edge for any of:

- same `duplicate_cluster_id`;
- same known `physical_item_id`;
- same known `capture_session_id`;
- same known `source_entity_id` (product/page/burst/source family);
- an explicit reviewer decision that two rows must remain together.

The ID is `grp:` plus SHA-256 of the newline-joined, sorted member `sample_id` values. A capture session may legitimately contain multiple labels; only a contradictory label inside a duplicate cluster is an automatic label failure. Derived crops, rotations, recompressions, and augmentations inherit the parent's `sample_id` lineage and group and are training-time transforms, not new independently split samples.

### 5. Split contract

- All legacy rows, including the old test rows, enter only the v2 development candidate pool after curation. `legacy_v1_split` is retained for traceability; no old membership is preserved as a v2 holdout.
- Sealed holdout rows come only from collection batches marked `intended_use=holdout` before labels or model predictions are inspected. The entire `collection_batch_id`, capture sessions, physical items, and linked groups stay in holdout. Holdout labels live in a separate sealed file; ordinary loaders receive `canonical_label=null` for these rows.
- Remaining eligible development rows are sorted by `sample_id`, then assigned with `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`: fold 0 is `val`, folds 1-4 are `train`. The saved assignments, not rerunning scikit-learn, are authoritative. Record Python/scikit-learn versions and the exact input fingerprint.
- Each eligible representative appears exactly once. All members of one `split_group_id` have exactly one split. Train augmentation is generated only inside the training loader.
- Train and validation must each contain every canonical label and at least five independent split groups per label. A final headline holdout must contain every label and meet a preregistered per-class physical-item target; until that target exists and is met, report it as a pilot, not a final benchmark.
- Model/recipe/threshold/curation choices may use train and validation only. The final evaluator opens holdout labels once for the frozen pipeline and appends the result; it never feeds per-row holdout errors back into v2 decisions.

### 6. Fingerprints and immutability

- `content_fingerprint`: full SHA-256 of sorted `{sample_id, asset_path, byte_sha256, pixel_sha256}` records.
- `dataset_fingerprint`: full SHA-256 of sorted `{sample_id, canonical_label, decision, provenance IDs, duplicate_cluster_id, split_group_id}` records plus `content_fingerprint`.
- `split_fingerprint`: full SHA-256 of sorted `{sample_id, split_group_id, split}` records plus `dataset_fingerprint`.

All three full hashes are recorded in `splits.json`, `audit.json`, checkpoints, and reports. A mismatch fails closed. A builder writes to a new staging directory, validates it, then promotes it as a new version; it never silently rebuilds an existing released manifest.

## Automated checks implementable now

A single CPU-only `validate_dataset_v2` command can implement these with the existing standard library, Pillow, scikit-learn, and scikit-image dependencies:

1. **Schema:** required fields, types, enums, UTC timestamps, exact class vocabulary, and conditional camera/provenance fields.
2. **Paths:** relative/canonical paths only; no escape, case-fold collision, repeated path, repeated `sample_id`, missing file, or unmanifested non-hidden file.
3. **Decode:** full decode succeeds; dimensions/format/mode match; animated/multi-frame policy is explicit; minimum-side, aspect-ratio, blur, and clipping measurements are reported under a versioned quality policy.
4. **Hashes:** recompute byte/pixel hashes and all three release fingerprints; mismatch is fatal.
5. **Dedup:** rebuild mandatory clusters; fail on unreviewed near candidates, contradictory accepted labels, multiple eligible representatives, or a representative-selection mismatch.
6. **Provenance:** camera conditional fields must be present; legacy `unknown` fields are counted; public-release mode fails on unknown/unsupported licenses.
7. **Groups:** independently rebuild connected components and IDs; fail if duplicate, item, session, source-entity, or reviewed-same edges cross split groups.
8. **Splits:** every eligible representative appears once; exclusions appear zero times; no path/hash/cluster/group overlaps; every group has one split; class/group support gates pass.
9. **Holdout:** every holdout row belongs to a preregistered holdout batch; no physical item, session, source entity, byte hash, pixel hash, or accepted near cluster appears in development; ordinary loaders cannot see holdout labels.
10. **Curation:** every decision references a real `sample_id`; reviewer/adjudicator requirements pass; decisions propagate across duplicate clusters; before/after counts and reasons reconcile exactly.
11. **External data:** incoming files are rehashed rather than trusting supplied hashes, deduplicated against all v2 partitions, and admitted only to their declared allowed split.
12. **Determinism:** two builds from identical inputs produce byte-identical manifest, split, and audit artifacts. The command exits nonzero on any failed release gate and writes all counts/candidate queues to `audit.json`.

The first useful implementation should stop at this validator and manifest builder. A service, database, or new dependency is unnecessary for 3,483 rows.

## Class risks

The grouped OOF figures below were recomputed from the saved 14-model logits and targets using the report's majority-vote rule. They are development diagnostics, not camera-domain estimates.

| Risk tier | Class evidence | Required cleanup |
|---|---|---|
| Highest error | `resistor`: legacy ResNet50 test recall 0.440; grouped OOF recall 0.465 / F1 0.537. | Review resistor -> crystal oscillator/diode/transistor errors; define dominant-object and multi-component rules. |
| Highest error | `fuse`: 0.364 legacy recall; 0.488 OOF recall / 0.543 F1. | Review fuse <-> relay conflicts, automotive fuse-box scenes, diagrams, and unrelated images. |
| Highest error | `relay`: 0.522 legacy recall; 0.519 OOF recall / 0.507 F1. | Adjudicate relay <-> fuse/crystal oscillator/power-supply groups and scene labels. |
| Highest error | `transistor`: 0.458 legacy recall; 0.526 OOF recall / 0.532 F1. | Review transistor -> crystal oscillator and resistor confusions; separate package similarity from wrong labels. |
| Elevated error | `connector`: 0.647 legacy recall; 0.536 OOF recall / 0.559 F1. `diode`: 0.455 legacy recall; 0.611 OOF recall / 0.577 F1. | Sample both randomly as well as by model error; low supports and recipe sensitivity make either single estimate unreliable. |
| Highest leakage | Exact train copies in legacy test: power supply module 21/38 (55.3%), LED 13/26 (50.0%), crystal oscillator 16/35 (45.7%), jumper wire 15/38 (39.5%), inductor coil 13/37 (35.1%). | Do not use current per-class scores as evidence of generalization; deduplicate/group before balancing. |
| Ontology | Primitive parts, assemblies, accessories, diagrams, ads, and multi-object scenes share one single-label ontology. Frequent grouped-OOF confusions include power supply module -> microcontroller board (15), LED -> switch (12), jumper wire -> breadboard (12), and inductor coil -> relay (11). | Freeze an image-scope policy and explicit ambiguous/multi-object/out-of-scope decisions before relabeling. |

The class count imbalance is secondary. Do not oversample or scrape more examples until cluster-adjusted counts are known; heavily duplicated classes may have many rows but few independent entities.

## Camera-data handoff

Deliver one directory containing `capture_manifest.jsonl`, `images/`, and optional `calibration/`. The collector supplies one manifest row per original frame:

```json
{
  "capture_id": "uuid",
  "asset_path": "images/uuid.jpg",
  "canonical_label": "resistor",
  "physical_item_id": "item-uuid",
  "source_entity_id": "sku-or-unknown",
  "capture_session_id": "session-uuid",
  "collection_batch_id": "batch-uuid",
  "intended_use": "dev_candidate",
  "camera_id": "device-and-lens-id",
  "captured_at_utc": "2026-08-16T12:00:00Z",
  "resolution": [1920, 1080],
  "focus": "auto-or-recorded-value",
  "exposure": "auto-or-recorded-value",
  "white_balance": "auto-or-recorded-value",
  "distance_mm": 250,
  "lighting_id": "setup-id",
  "background_id": "setup-id",
  "annotator_id": "person-or-team-id"
}
```

- Reuse `physical_item_id` for every view of the same physical component, including later sessions. Use a new `capture_session_id` whenever setup/time changes; never treat video frames as independent items.
- Mark whole batches `dev_candidate` or `holdout` before model review. Do not mix one physical item/session across uses, and do not expose holdout labels to ordinary training notebooks.
- Preserve originals. Ingestion computes hashes, dimensions, blur/clipping measurements, and v2 IDs; the collector should not hand-edit these derived fields.
- Vary item identity, manufacturer/package, view, scale, background, and lighting. Record rather than discard controlled failures; quality policy decides `keep`, `exclude`, or `out_of_scope` with a reason.
- For colour-sensitive resistor work, also record camera calibration/profile and keep exposure/white balance/lighting stable enough to audit; component classification and resistance-band labels remain separate tasks.

## Evidence map

- `DATASET_IMPROVEMENT_RESEARCH.md`, especially lines 20-74 (layout/counts), 76-107 (split/curation), 109-148 (audit/class/shift), and 150-193 (leakage).
- `CROSS_FOLD_AUGMENTATION_REPORT.md`, lines 8-12 (OOF result/test scope), 52-66 (grouped folds), and 93-98 (caveats).
- `EXPERIMENT_LOG.md`, lines 7-14 (legacy contract), 17-68 (test use and model history), and 70-88 (decisions).
- `notebooks/01_primary_comparative.ipynb`, code cells 4-7: suffix filtering and decoding, weak dataset/split fingerprints, automatic manifest rebuild, stratified image-level splitting, and CWD-derived paths.
- `notebooks/16_cross_fold_augmentation_benchmark.py`, lines 73-86 and 176-229 (content hashes and grouped folds), 110-173 (external manifest ingestion), and 375-452 (development/test policy).
- `notebooks/data/splits/split_manifest_raw.json`, `notebooks/data/splits/split_meta.json`, and `notebooks/data/curation/reviewed_exclusions_v1.json`.
- `notebooks/outputs/metrics/cross_fold_augmentation.json` and `cross_fold_augmentation_oof.pt` for the grouped-run and recomputed class metrics; `model_error_audit.json` for legacy ResNet50 class recalls.
