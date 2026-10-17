# Dataset Improvement Research

**Local inspection date:** 2026-08-13  
**Scope:** repository evidence first; no web access was used. Counts below describe the frozen raw benchmark under `notebooks/data/`, because the recorded runs use that working directory.

## Executive summary

The next improvement should be a **dataset rebuild, not another model sweep**. The repository already shows substantial label/content noise, but the more urgent finding is split contamination:

- The frozen manifest contains **3,483 accepted images in 20 classes** and a stratified 2,438/522/523 train/validation/test split.
- A local read-only exact-content scan (file size followed by SHA-256) found **527 duplicate-content groups**, covering 1,107 rows and leaving only **2,903 unique byte hashes**. There are **580 redundant copies**.
- **128/522 validation rows (24.5%)** and **115/523 test rows (22.0%)** have an exact byte-identical copy in training. Path overlap is zero, so a path-only leakage check would miss this.
- There are **12 exact-content groups with contradictory class labels**; five cross split boundaries. Of the evaluation rows duplicated in training, 3 validation and 2 test rows conflict with the training label.
- The archived ResNet50 audit scores **0.6845 test accuracy on all 523 rows but 0.6005 on the 408-row slice without an exact training copy**. This is a diagnostic slice, not a replacement benchmark, but it demonstrates material score inflation.
- The existing manual review is high-value but narrow: 30 obvious contaminants were found among 120 indexed high-loss rows. It does not address the hundreds of duplicate clusters, near duplicates, provenance, or ambiguous/multi-object images.
- The current test set has been evaluated, error-audited, and manually reviewed repeatedly. It should now be treated as a **legacy development/audit set**, not a pristine final holdout.

Accordingly, the highest-priority plan is: (1) quarantine the legacy benchmark, (2) deduplicate and adjudicate before splitting, (3) create group/source-disjoint train and validation sets, and (4) collect a new target-domain holdout that remains unseen until one pre-registered final evaluation.

## 1. Data layout and class counts

### Two independent data/output roots

The checkout has two full working trees rather than links:

- `data/` and `outputs/` at repository root.
- `notebooks/data/` and `notebooks/outputs/` under the notebook working directory.

The two raw trees contain the same 3,530 non-hidden class-folder files and are separate copies (no shared inodes), roughly 1.14 GB each. `notebooks/data/` also has six hidden checkpoint JPEGs under the diode tree; the current loader correctly skips hidden path components. The split assignments in root `data/splits/split_manifest.json` and notebook `notebooks/data/splits/split_manifest_raw.json` are equivalent after paths are normalized, but their formats differ:

- Root `data/splits/split_manifest.json` is the older bare split dictionary with machine-specific absolute paths.
- `notebooks/data/splits/split_manifest.json` and `split_manifest_raw.json` are byte-identical version-2 manifests with relative paths.
- The README names `data/splits/split_manifest_raw.json`, but that file is absent at repository root.
- The reviewed curation manifest exists only at `notebooks/data/curation/reviewed_exclusions_v1.json`.

This makes the effective dataset, split file, and output location depend on the process working directory. The README's normal launch path (`cd notebooks`) and the archived `session_manifest.json` make `notebooks/` the de facto source for recorded experiments, but the repository does not enforce that choice.

### Accepted benchmark counts

The loader in `notebooks/01_primary_comparative.ipynb` (cells 4–5) accepts only `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, and `.tiff`, skips hidden directories, and validates accepted files with Pillow. The version-2 raw manifest records these counts:

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| battery holder | 154 | 33 | 33 | 220 |
| breadboard | 99 | 21 | 21 | 141 |
| capacitor | 83 | 18 | 17 | 118 |
| connector | 80 | 17 | 17 | 114 |
| crystal oscillator | 162 | 35 | 35 | 232 |
| diode | 103 | 23 | 22 | 148 |
| fuse | 101 | 22 | 22 | 145 |
| heat sink | 113 | 24 | 24 | 161 |
| ic chip | 114 | 24 | 25 | 163 |
| inductor coil | 169 | 36 | 37 | 242 |
| jumper wire | 174 | 37 | 38 | 249 |
| led | 121 | 26 | 26 | 173 |
| microcontroller board | 108 | 23 | 23 | 154 |
| potentiometer | 113 | 24 | 24 | 161 |
| power supply module | 176 | 37 | 38 | 251 |
| relay | 109 | 24 | 23 | 156 |
| resistor | 117 | 25 | 25 | 167 |
| switch | 131 | 28 | 28 | 187 |
| transformer | 101 | 22 | 21 | 144 |
| transistor | 110 | 23 | 24 | 157 |
| **Total** | **2,438** | **522** | **523** | **3,483** |

The largest class has 251 accepted rows and the smallest 114, a 2.20× ratio. Stratification keeps label proportions similar across splits, but some per-class test supports are only 17–25, so per-class estimates are noisy.

### Files outside the accepted set

There are 47 non-hidden files inside class folders that the suffix whitelist ignores: 23 `.gif`, 16 `.img`, 4 `.image`, 3 `.view`, and 1 `.php`. All 47 opened successfully with the installed Pillow build (actual formats included GIF, JPEG, AVIF, WebP, and PNG). They should be quarantined and reviewed rather than silently discarded or automatically added.

The accepted files are heterogeneous even at the codec level. Pillow identifies the 3,483 manifest rows as 1,742 JPEG, 954 WebP, 472 PNG, 312 AVIF, and 3 MPO images; therefore filename suffix is not a reliable content-format identifier. Dimensions range from a 19×19 test image to multi-megapixel images, with several extreme aspect ratios. This is consistent with a mixed web scrape rather than a controlled acquisition process.

**Primary evidence:** `notebooks/data/splits/split_manifest_raw.json`, `notebooks/data/splits/split_meta.json`, `notebooks/01_primary_comparative.ipynb` cells 4–7, `notebooks/outputs/metrics/session_manifest.json`, and `README.md`.

## 2. Frozen split and curation behavior

### What is currently frozen

The current notebook code:

1. Enumerates validated images by immediate parent class directory.
2. Uses two stratified `train_test_split` calls with seed 42 and ratios 70/15/15.
3. Writes a version-2 manifest with relative paths, dataset fingerprint, seed, and ratios.
4. Reuses the manifest when those fields match and every listed path exists.

The recorded raw dataset fingerprint is `78637c8898aa10f8`; archived remote runs report split fingerprint `45f4831c25e45119` and sizes 2,438/522/523. Model-family checkpoints and reports carry these identifiers, and the benchmark runner selects on validation macro-F1 before optional test evaluation (`notebooks/12_model_family_benchmark.py`). These are useful safeguards.

### Why “frozen” is weaker than it appears

- **No group-aware split:** stratification is by class only. Duplicate images, near duplicates, product/source families, and capture sessions can be divided across all three splits.
- **Automatic rewrite:** if the sample fingerprint is stale, `build_or_load_splits` rebuilds and overwrites the manifest rather than failing closed and requiring a new dataset version.
- **Weak dataset fingerprint:** `_fingerprint` hashes label, basename, and file size—not file bytes or a stable relative path. A same-size content replacement can evade detection.
- **Machine-dependent split signature:** `_split_fingerprint` hashes resolved absolute paths. The same checked-in manifest produces `45f4831c25e45119` under `/home/carlai04/SEB/notebooks` but `1c82cf8d3c6ba77d` in this local checkout. Membership is the same; the identifier is not portable.
- **No manifest integrity assertions:** loading does not assert unique content, disjoint duplicate groups, exact class-count expectations, or equality between manifest rows and the current enumerated `(path, label)` set.
- **CWD-dependent source:** running from root versus `notebooks/` selects different duplicated data, split, curation, and output trees.

### Curation paths

`reviewed_exclusions_v1.json` contains 30 conservative exclusions: 4 train, 13 validation, and 13 test. Reasons include vehicles, people, food, game/cartoon art, logos/advertisements, and component mislabels.

There are two materially different ways the repository uses it:

- Setting `COMPCLASS_CURATION_MANIFEST` makes the notebook preserve raw membership and then filter exclusions from **all** splits. This avoids moving rows between partitions, but a validation/test score after model-guided review is a curated diagnostic, not an untouched benchmark (`notebooks/07_curated_matched_retrain.py`; `notebooks/08_raw_test_curation_eval.py`).
- `12_model_family_benchmark.py --training-exclusions ...` applies only `train-*` entries, hashes the manifest into checkpoint metadata, and leaves validation/test membership fixed. The archived trial removed four rows (2,438 → 2,434) and reached validation macro-F1 0.7421 versus 0.7473 for the prior strong ConvNeXt-Base. This four-image result is too small to conclude that broader cleaning is ineffective, especially because the validation split itself is duplicate-contaminated.

The raw-test curation diagnostic is appropriately labeled: removing 13 reviewed test contaminants changed ResNet18 accuracy from 0.6367 to 0.6529. It must not be presented as unbiased generalization performance.

## 3. Existing error-audit evidence

The repository already contains unusually useful audit material:

| Artifact | What it contains | Dataset use |
|---|---|---|
| `notebooks/outputs/metrics/resnet18_sample_audit.csv` | Per-row split, path, label, prediction, correctness, confidence, margin, loss, and flip agreement for all 3,483 rows | Strong seed list for review; model-dependent |
| `resnet18_high_loss_indexed.csv` and three `resnet18_high_loss_*_indexed.jpg` sheets | Top 40 loss rows from each split, with stable review IDs | 120-row manual review queue |
| `resnet18_review_candidates.csv` / `resnet18_audit_summary.json` | High-confidence wrong predictions and full per-class evaluation | Useful disagreement/error signals |
| `reviewed_exclusions_v1.json` | 30 reviewed obvious contaminants with reasons | Exclusions only; no relabel/adjudication schema |
| `model_error_audit.csv` / `.json` | Per-row ResNet50 and scratch-CNN predictions plus per-class metrics/confusions | Cross-model disagreement and hard-example mining |
| `model_error_audit_confusion.png` | Normalized validation/test confusion matrices | Class-pair review priorities |
| `EXPERIMENT_LOG.md` and family benchmark JSON files | Validation/test history, invalidated runs, and test-deferral decisions | Evaluation-governance history |

The contact sheets visibly mix clean product photos with schematics, datasheets, collages, e-commerce graphics, black/near-empty-looking tiles, multi-component scenes, and unrelated content. The 30 exclusions are 25% of the 120 reviewed high-loss rows (10% of the train queue and 32.5% of each validation/test queue), confirming that model-ranked review is productive. It is not a prevalence estimate: training loss is suppressed by memorization, and only the top-loss tail was examined.

For ResNet50, the lowest archived raw-test recalls are fuse 0.364, resistor 0.440, diode 0.455, transistor 0.458, and relay 0.522. Common confusions include fuse → resistor, relay/transistor → crystal oscillator, resistor → diode/inductor coil, and microcontroller board → ic chip. These are the first classes and class pairs to audit, but errors may reflect ontology ambiguity and scene composition as well as insufficient data.

Missing audit fields include source URL/provider, acquisition date, license, physical/product identity, capture session, duplicate-cluster ID, image type (photo/diagram/ad/collage), annotator identity, independent second annotation, adjudicated replacement label, and an explicit `ambiguous`/`out_of_scope` status.

## 4. Label noise and likely distribution shift

### Strong evidence of label/content noise

1. The reviewed manifest records 30 unambiguous contaminants in only 120 high-loss rows.
2. The exact-byte scan found 12 duplicate groups assigned to different classes. Examples include the same image labeled fuse and relay, diode and LED, diode and transistor, and crystal oscillator and inductor coil. Several within-train contradictory pairs match reviewed irrelevant-image exclusions.
3. The contact sheets show images where the folder label is not the salient object, multi-object boards/scenes, diagrams rather than components, and unrelated advertisements/artwork.
4. The ontology mixes primitive parts (`resistor`, `diode`), assemblies (`microcontroller board`, `power supply module`), accessories (`battery holder`, `jumper wire`), and scenes that may legitimately contain several labeled classes. A single folder label is sometimes under-specified rather than simply wrong.

### Distribution-shift assessment

There is no true target-domain shift benchmark in the repository. All partitions are random image-level samples from the same Kaggle web scrape, and no source/session metadata exists. Consequently, the project cannot currently measure performance on a target camera, vendor catalog, lab bench, or newly collected web source.

There are warning signs, not proof, of unstable mixture proportions:

- Validation-to-test accuracy drops recur for pretrained models: ResNet50 0.7471 → 0.6845, selected ConvNeXt-Tiny 0.7605 → 0.6979, and strong ConvNeXt-Base 0.7605 → 0.7132.
- The visual domains are highly mixed: isolated product photos, circuit diagrams, screenshots, ads, collages, and unrelated images.
- Duplicate prevalence varies sharply by class. Exact training copies account for 55.3% of power-supply-module test rows, 50.0% of LED, 45.7% of crystal oscillator, 39.5% of jumper wire, and 35.1% of inductor coil, but 0% of microcontroller board and under 10% for several weak classes. Thus aggregate and per-class scores reflect different amounts of memorization opportunity.

The score gaps could also arise from finite sample size, differing duplicate clusters, and repeated validation adaptation. A new source- or session-disjoint holdout is required to distinguish ordinary sampling variation from deployment shift.

## 5. Leakage risks

### Critical: exact duplicate content across partitions

The local scan resolved every row in `notebooks/data/splits/split_manifest_raw.json`, grouped candidate files by byte size, then compared SHA-256. It did not modify the dataset.

| Finding | Count |
|---|---:|
| Manifest rows | 3,483 |
| Unique exact-content hashes | 2,903 |
| Duplicate-content groups | 527 |
| Rows belonging to duplicate groups | 1,107 |
| Redundant copies | 580 |
| Duplicate groups spanning multiple splits | 243 |
| Validation rows with an exact train copy | 128 / 522 (24.5%) |
| Test rows with an exact train copy | 115 / 523 (22.0%) |
| Cross-label exact-content groups | 12 |

The archived per-row audits expose the impact:

| Model / split | Published-slice accuracy | Accuracy after excluding rows with an exact train copy | Exact-train-copy slice accuracy |
|---|---:|---:|---:|
| ResNet18 validation | 0.6973 | 0.6066 (n=394) | 0.9766 (n=128) |
| ResNet18 test | 0.6367 | 0.5392 (n=408) | 0.9826 (n=115) |
| ResNet50 validation | 0.7471 | 0.6726 (n=394) | 0.9766 (n=128) |
| ResNet50 test | 0.6845 | 0.6005 (n=408) | 0.9826 (n=115) |

For same-label exact training copies, both pretrained audits are 100% correct; for the 3 validation and 2 test copies whose training duplicate has a conflicting label, they are 0% correct and highly confident. The post-hoc non-copy slices remain contaminated by possible resized/cropped near duplicates and source-family overlap, and the models were not retrained on a deduplicated training set. They are diagnostic evidence only—not corrected benchmark results. There is no archived per-row output for the strong ConvNeXt-Base, so its clean-slice score cannot be derived locally.

### High: near-duplicate and source-family leakage

Exact hashing is a lower bound. Web scrapes commonly contain resized, recompressed, cropped, watermarked, or alternate-angle variants. The present manifest has no perceptual-hash cluster, source URL, product ID, or capture-session group, so those variants can also cross partitions.

### High: adaptive reuse and inspection of the test set

Although newer runners defer test evaluation until validation selection, the same 523-row test set has supported multiple model evaluations, a full per-sample error audit, high-loss contact sheets, manual exclusions, and a curated-test diagnostic. Later experiment phases were planned after earlier test results showed the target was missed. The repository documents these decisions honestly, but the set is no longer unseen for future model or dataset choices.

### High when enabled: filtering validation/test after model-guided review

Filtering reviewed validation or test rows is useful for estimating contamination, but it conditions evaluation membership on observed errors/content. `COMPCLASS_CURATION_MANIFEST` can filter all partitions, so those resulting metrics must remain diagnostic. Training-only exclusions are safer.

### Medium: split identity and source-of-truth drift

CWD-dependent data copies, automatic stale-manifest rebuilding, non-content dataset hashes, and absolute-path split signatures can cause accidental re-splits, mismatched checkpoint reuse, or apparently different split IDs for identical membership.

## 6. Prioritized dataset-improvement plan

### P0 — Restore a trustworthy evaluation contract

1. **Retire the current test as a final holdout.** Keep it and all current metrics under a clearly named `legacy_raw_v1` contract for historical comparison and audit only. Do not use its labels, errors, or curated score to choose new models or cleaning rules.
2. **Choose one canonical data root.** Make dataset, split, curation, and output paths independent of CWD. After byte-for-byte verification, remove or link the duplicate tree in a separate implementation change. Preserve the legacy manifests unchanged.
3. **Create an immutable content manifest.** For every candidate file record stable relative path, SHA-256 bytes, decoded format, dimensions, class, provenance/source when recoverable, license/status, and dataset version. Make a changed manifest create a new version rather than overwrite the old split.
4. **Cluster before splitting.** Collapse exact hashes and then build conservative perceptual/source clusters for recompressions, crops, watermarks, product families, and capture sessions. Assert that every cluster belongs to exactly one partition. Manually adjudicate all 12 current cross-label exact groups first.

### P1 — Curate the ontology and labels

5. **Write a labeling policy before more review.** Define whether the target is the dominant object, any object present, or the image's intended catalog category. Specify handling for multi-component scenes, diagrams, boards/modules, logos/ads, and non-component images. Add explicit `ambiguous`, `multi_object`, and `out_of_scope` states; do not force every row into one of 20 classes.
6. **Turn exclusions into adjudications.** Review by duplicate/source cluster, not isolated file. Record `keep`, `relabel`, `exclude`, or `ambiguous`, old/new label, reason code, annotator, and adjudicator. Use at least two independent reviewers for the new validation and holdout labels, with disagreements resolved without model predictions visible.
7. **Use existing audits as queues, not truth.** Start with the 120 indexed high-loss rows, all cross-label duplicate groups, cross-model disagreements, low-confidence rows, and the weak classes (fuse, resistor, diode, transistor, relay). Randomly sample each class and image type as well so easy systematic noise is not missed.
8. **Quarantine the 47 suffix-excluded but readable files.** Review and normalize format before deciding whether they belong in the new version. Do not expand the whitelist without content and animation-frame policy.

### P1 — Build group-disjoint development splits

9. **Split only after curation and clustering.** Use duplicate/source/product/capture group as the indivisible unit, with class-balance constraints. A grouped splitter such as `GroupShuffleSplit` or `StratifiedGroupKFold` is a starting mechanism; the saved group assignment, not a random seed alone, is the contract.
10. **Add hard leakage gates.** Fail dataset construction on repeated paths, exact-hash overlap, perceptual-cluster overlap, contradictory labels within an unresolved cluster, missing files, unexpected class order/count drift, or non-portable paths. Save a machine-readable leakage report with every dataset release.
11. **Create a fresh validation set.** Use it for model, recipe, threshold, and active-learning decisions. Report both macro-F1 and per-class support/recall, plus results by image type/source group and an explicit out-of-scope rejection metric if that label is introduced.

### P2 — Collect data for the intended deployment distribution

12. **Define the target domain first.** A webcam/lab-bench classifier, catalog-image classifier, and general web-image classifier require different data. Record camera/source/session/product identity at acquisition.
13. **Collect a new untouched holdout.** Use new physical components, capture sessions, vendors, or sources that cannot duplicate development data. Label it independently and seal labels until one pre-registered pipeline is frozen. This is the only credible set for a new headline performance claim.
14. **Acquire clean diversity, not merely more scrape volume.** Prioritize weak/underrepresented classes and confusing pairs, but vary manufacturers, package styles, backgrounds, scale, lighting, and viewpoint. For modules/scenes, collect examples consistent with the new ontology. Keep repeated views grouped by physical item/session.
15. **Re-run a small data ablation before architecture work.** Compare the same strong baseline on: legacy raw, exact-deduplicated/grouped, adjudicated training data, and the final curated/grouped dataset. Promote a dataset version based on validation gains and audit quality; touch the fresh holdout once.

### Release gates for the next dataset version

- Zero path, exact-hash, or accepted near-duplicate-cluster overlap across train/validation/holdout.
- Every cross-label duplicate cluster adjudicated.
- Portable content and split fingerprints independent of checkout path.
- One canonical source tree and one immutable split manifest per version.
- Counts and exclusion/relabel reasons published per class, source, and image type.
- No holdout image or label used for curation, active learning, thresholding, or model selection.
- A reproducible audit report generated before any training begins.

## Conclusion

The repository's own artifacts correctly identify web-scrape noise and show that four training exclusions do not change the current validation result. Local content analysis adds a more fundamental result: roughly one quarter of validation and one fifth of test are exact copies of training content, with contradictory labels in some clusters. The frozen split is reproducible as membership but is not independent as content. Further architecture tuning on it will optimize a contaminated contract. The shortest credible route to improvement is to deduplicate and adjudicate at cluster level, replace image-level splitting with group/source splitting, and reserve a newly collected target-domain holdout.

## Sources

General claims and recommended APIs are supported by these official references/original papers; repository-specific counts and conclusions above come from local files.

- torchvision `ImageFolder` documentation: <https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html>
- scikit-learn `train_test_split` documentation: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>
- scikit-learn grouped splitters: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html> and <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html>
- PyTorch reproducibility notes: <https://pytorch.org/docs/stable/notes/randomness.html>
- Northcutt, Athalye, and Mueller, “Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks” (original paper): <https://arxiv.org/abs/2103.14749>
- Koh et al., “WILDS: A Benchmark of in-the-Wild Distribution Shifts” (original paper): <https://proceedings.mlr.press/v139/koh21a.html>
